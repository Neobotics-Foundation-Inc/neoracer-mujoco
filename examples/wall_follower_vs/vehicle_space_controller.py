"""
Vehicle-space wall follower: a pure controller that selects a steering angle
from one dense LiDAR scan.

Frame: +X forward, +Y left, +Z up. Every angle (rad or deg) is + = left.

Each control tick, step() does these operations:

1. It converts the scan to obstacle points (x, y) in the car frame.
2. It uses a fixed grid of candidate steering angles. The grid does not
   depend on the number of beams N in the scan.
3. For each candidate, it computes the circular arc that the car center
   follows at that steering angle (bicycle model, see arc_table()).
4. It finds the safe arc length of each candidate. This is the arc length
   that the car travels before an obstacle point comes within the clearance
   radius of the arc.
5. It groups adjacent usable candidates into regions and scores each region.
6. It keeps the region that it tracks unless a new region has a much higher
   score. It then limits the rate of change of the target steering angle.
7. It sets the throttle from the safe arc length of the target path and from
   the steering angle. The steer command is the target steering angle.

The controller converts the scan to points once and then does a geometric
test for each candidate. A wall to the side of the car is far from the
straight arc, so more samples on that wall do not block the straight arc.
Thus the result does not change much when N changes.
"""

import functools
import math
from dataclasses import dataclass
from typing import assert_never

import numpy as np
import numpy.typing as npt
from scipy.spatial import KDTree

from neoracer_mujoco import contract
from neoracer_mujoco.control.steering import ControlCommand
from neoracer_mujoco.scene import Scene

type FloatArray = npt.NDArray[np.float64]  # shape (N,): one value per beam
type CandidateArray = npt.NDArray[np.float64]  # shape (n_candidates,)
type ArcArray = npt.NDArray[np.float64]  # shape (n_candidates, n_samples)


@dataclass(frozen=True)
class CarGeometry:
    """The car dimensions that the bicycle model uses. Read them from the
    scene with car_geometry()."""

    wheelbase_m: float  # m; front axle to rear axle
    steer_max_rad: float  # rad; steer_servo ctrlrange upper limit


def car_geometry(scene: Scene) -> CarGeometry:
    """Read the wheelbase and the steering limit from a compiled car scene.

    The steer_servo command is the steering angle of the bicycle model.
    neoracer.xml converts it to the two front wheel angles (Ackermann
    geometry), so R = wheelbase / tan(steer) applies directly to the
    steer_servo command.
    At full lock this R is 0.68 m. verify_kinematics.py measures about
    0.76 m on the simulated car, probably because the tires slip.
    """
    front = scene.spec.body("fl_wheel_body")
    rear = scene.spec.body("rl_wheel_body")
    if front is None or rear is None:
        raise ValueError("the scene has no fl_wheel_body or rl_wheel_body")
    model = scene.model
    return CarGeometry(
        wheelbase_m=float(
            model.bind_scalar(front).pos[0] - model.bind_scalar(rear).pos[0]
        ),
        steer_max_rad=contract.actuator_limits(model, scene.car).steer_rad,
    )


@dataclass(frozen=True)
class VsParams:
    """Tuning values of the controller. car comes from the model."""

    car: CarGeometry
    n_candidates: int = 81  # steering angles in the grid, from -max to +max
    clearance_radius_m: float = 0.21  # m; car half-width, swept along each arc
    planning_horizon_m: float = 3.0  # m; arc length of each candidate path
    # m between two samples on an arc. It is less than half of
    # clearance_radius_m, so an obstacle cannot pass between two samples.
    arc_sample_ds_m: float = 0.1
    # deg; the median filter replaces each range with the median of the
    # ranges in this angular window, so one bad beam cannot open or block a path.
    outlier_window_deg: float = 2.0
    # m; if two adjacent obstacle points are this close or closer, the
    # controller adds their midpoint, so a wall cannot pass between them.
    # Points farther apart can be two different surfaces, for example the two
    # sides of an opening. The controller does not join them.
    max_segment_gap_m: float = 0.5
    # m; a candidate joins the region of a peak if its safe arc length is
    # within this distance of the safe arc length at the peak.
    region_margin_m: float = 0.3
    w_clearance: float = 1.0  # score per m of mean safe arc length in a region
    w_forward: float = 0.01  # score removed per deg of region center from 0
    w_continuity: float = 0.02  # score removed per deg from the last target
    w_width: float = 0.05  # score per candidate in a region
    max_slew_deg_per_s: float = 90.0  # deg/s; rate limit of the target steering
    # A new region must have a score this many times the score of the tracked
    # region before the controller changes to it.
    region_switch_margin: float = 1.15
    # N·m per wheel; 3 % of the fl_motor ctrlrange of 1 N·m.
    max_throttle_nm: float = 0.03
    steer_throttle_cut: float = 0.6  # fraction of throttle removed at full steer
    # N·m per wheel; minimum throttle while the target path is usable, so the
    # car has enough torque to complete a sharp turn.
    min_throttle_nm: float = 0.015
    # Candidates on each side of the target that target_path_clearance() reads.
    path_window_candidates: int = 2

    def __post_init__(self) -> None:
        # arc_table() divides by n_candidates - 1. extract_regions() divides
        # by a sum of safe arc lengths that are each at least
        # clearance_radius_m. The throttle must stay in [0, max_throttle_nm].
        # An odd count puts one candidate at 0 deg, the straight path.
        if self.n_candidates < 3 or self.n_candidates % 2 == 0:
            raise ValueError(
                f"n_candidates must be odd and >= 3, got {self.n_candidates}"
            )
        if not self.planning_horizon_m > 0:
            raise ValueError(
                f"planning_horizon_m must be > 0, got {self.planning_horizon_m}"
            )
        if not self.clearance_radius_m > 0:
            raise ValueError(
                f"clearance_radius_m must be > 0, got {self.clearance_radius_m}"
            )
        if not 0 < self.arc_sample_ds_m <= self.clearance_radius_m / 2:
            raise ValueError(
                "arc_sample_ds_m must be in (0, clearance_radius_m / 2], got "
                f"{self.arc_sample_ds_m}"
            )
        if not 0 <= self.steer_throttle_cut <= 1:
            raise ValueError(
                f"steer_throttle_cut must be in [0, 1], got {self.steer_throttle_cut}"
            )
        if not 0 <= self.min_throttle_nm <= self.max_throttle_nm:
            raise ValueError(
                "min_throttle_nm must be in [0, max_throttle_nm], got "
                f"{self.min_throttle_nm}"
            )
        if self.path_window_candidates < 1:
            raise ValueError(
                f"path_window_candidates must be >= 1, got {self.path_window_candidates}"
            )


@dataclass(frozen=True)
class ArcTable:
    """The candidate grid and the sampled arc of each candidate. The arrays
    are read-only."""

    steer_deg: CandidateArray  # deg; candidate steering angles, ascending
    s_m: npt.NDArray[np.float64]  # m; arc length of each sample, shape (n_samples,)
    x_m: ArcArray  # m; car center x at each sample
    y_m: ArcArray  # m; car center y at each sample
    # m; arc length at which each arc turns 180 deg, or the horizon if less.
    cap_m: CandidateArray
    valid: npt.NDArray[np.bool_]  # shape (n_candidates, n_samples): s_m <= cap_m
    # Flat index (candidate * n_samples + sample) of each valid sample, in the
    # order of the rows of sample_tree. Shape (n_valid,).
    valid_flat: npt.NDArray[np.intp]
    # A k-d tree of the (x_m, y_m) points of the valid samples. A k-d tree is
    # a search structure that finds all points within a radius quickly.
    sample_tree: KDTree
    straight_index: int  # index of the candidate nearest to 0 deg
    window_deg: float  # deg; half-width of the window of target_path_clearance()


@functools.cache
def arc_table(params: VsParams) -> ArcTable:
    """Compute the arc of each candidate once per params.

    For a steering angle delta, the curvature is kappa = tan(delta) / L, with
    L the wheelbase. The car center starts at the origin and moves along +X.
    At arc length s it is at
        x(s) = sin(kappa * s) / kappa
        y(s) = (1 - cos(kappa * s)) / kappa
    For delta = 0 the path is the straight line x(s) = s, y(s) = 0.

    Each arc stops at 180 deg of turn. After 180 deg the arc goes back
    toward the start and can pass around a wall without touching it. Then a
    full-lock turn next to a wall looks open, but the car only drives in a
    small circle.
    """
    steer_max_deg = math.degrees(params.car.steer_max_rad)
    steer_deg = np.linspace(-steer_max_deg, steer_max_deg, params.n_candidates)
    horizon_m = params.planning_horizon_m
    # ceil, so the spacing is at most arc_sample_ds_m. The 1e-9 keeps an exact
    # multiple (3.0 / 0.1 = 29.999...) from getting one more sample.
    n_samples = math.ceil(horizon_m / params.arc_sample_ds_m - 1e-9) + 1
    s = np.linspace(0.0, horizon_m, n_samples)
    arc_x = np.zeros((len(steer_deg), n_samples))
    arc_y = np.zeros_like(arc_x)
    cap_m = np.zeros(len(steer_deg))
    for i, deg in enumerate(steer_deg):
        delta = math.radians(deg)
        if abs(delta) < 1e-6:
            arc_x[i] = s
            arc_y[i] = 0.0
            cap_m[i] = horizon_m
        else:
            kappa = math.tan(delta) / params.car.wheelbase_m
            arc_x[i] = np.sin(kappa * s) / kappa
            arc_y[i] = (1.0 - np.cos(kappa * s)) / kappa
            cap_m[i] = min(horizon_m, math.pi / abs(kappa))
    valid = s[None, :] <= cap_m[:, None]
    valid_flat = np.flatnonzero(valid)
    step_deg = (2.0 * steer_max_deg) / (params.n_candidates - 1)
    for array in (steer_deg, s, arc_x, arc_y, cap_m, valid, valid_flat):
        array.setflags(write=False)
    return ArcTable(
        steer_deg=steer_deg,
        s_m=s,
        x_m=arc_x,
        y_m=arc_y,
        cap_m=cap_m,
        valid=valid,
        valid_flat=valid_flat,
        sample_tree=KDTree(
            np.column_stack([arc_x.ravel()[valid_flat], arc_y.ravel()[valid_flat]])
        ),
        straight_index=int(np.argmin(np.abs(steer_deg))),
        window_deg=params.path_window_candidates * step_deg,
    )


def turn_radius_m(params: VsParams, steer_deg: float) -> float:
    """Turn radius (m) of the car center at steer_deg; inf for 0 deg."""
    delta = math.radians(steer_deg)
    if abs(delta) < 1e-6:
        return math.inf
    return abs(params.car.wheelbase_m / math.tan(delta))


def _median_filter(
    params: VsParams, angles_rad: FloatArray, ranges_m: FloatArray
) -> FloatArray:
    """Replace each range with the median of the ranges within
    outlier_window_deg of its angle. The window is an angle, not a beam
    count, so a sparse scan has fewer neighbors in the window. A range with
    no other range in its window does not change."""
    window_rad = math.radians(params.outlier_window_deg)
    out = ranges_m.copy()
    # Limit: searchsorted needs ascending angles. If a sensor gives another
    # order, sort the scan by angle before step().
    lo = np.searchsorted(angles_rad, angles_rad - window_rad, side="left")
    count = np.searchsorted(angles_rad, angles_rad + window_rad, side="right") - lo
    # np.median works on rows of equal length, so the filter takes the beams
    # with the same window length together. A scan has only a few window
    # lengths: one or two in the middle, and shorter ones at the two ends.
    for length in np.unique(count[count > 1]):
        beams = np.flatnonzero(count == length)
        windows = ranges_m[lo[beams, None] + np.arange(length)]
        out[beams] = np.median(windows, axis=1)
    return out


def obstacle_points(
    params: VsParams, angles_rad: FloatArray, ranges_m: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """The obstacle points (x, y) in m that safe_arc_lengths() tests.

    The points are the valid returns of the filtered scan, plus the midpoint
    of each pair of adjacent points that are max_segment_gap_m apart or
    closer. A range that is not finite or not > 0 is "no return" and gives
    no point.
    """
    filtered = _median_filter(params, angles_rad, ranges_m)
    valid = np.isfinite(filtered) & (filtered > 0.0)
    r = filtered[valid]
    theta = angles_rad[valid]
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    if len(x) < 2:
        return x, y
    bridge = np.hypot(np.diff(x), np.diff(y)) <= params.max_segment_gap_m
    mx = (x[:-1] + x[1:])[bridge] / 2.0
    my = (y[:-1] + y[1:])[bridge] / 2.0
    return np.concatenate([x, mx]), np.concatenate([y, my])


def safe_arc_lengths(
    params: VsParams, angles_rad: FloatArray, ranges_m: FloatArray
) -> CandidateArray:
    """The safe arc length (m) of each candidate: the arc length at which an
    obstacle point first comes within clearance_radius_m of the arc. If no
    point does, the value is the cap of that arc (see arc_table())."""
    table = arc_table(params)
    x, y = obstacle_points(params, angles_rad, ranges_m)
    # A point farther than the horizon plus the clearance radius cannot touch
    # any arc, so the search below does not use it.
    reach = params.planning_horizon_m + params.clearance_radius_m
    keep = np.hypot(x, y) <= reach
    x, y = x[keep], y[keep]
    if len(x) == 0:
        return table.cap_m.copy()
    blocked = np.zeros(table.valid.size, dtype=bool)
    blocked[_blocked_samples(params, table, x, y)] = True
    blocked = blocked.reshape(table.valid.shape)
    # argmax gives the first blocked sample of each arc, but it also gives 0
    # for an arc with no blocked sample. np.where gives the cap to those arcs.
    first = np.argmax(blocked, axis=1)
    return np.where(blocked.any(axis=1), table.s_m[first], table.cap_m)


def _blocked_samples(
    params: VsParams, table: ArcTable, x: FloatArray, y: FloatArray
) -> npt.NDArray[np.intp]:
    """The flat index (see ArcTable.valid_flat) of each valid sample that
    has an obstacle point (x, y) at a squared distance of at most
    clearance_radius_m ** 2. An index can occur more than once.

    The k-d trees find the candidate pairs with a radius that is 1e-9 larger
    than clearance_radius_m. The tree can round a distance differently, so
    the larger radius makes sure that it finds every pair. This function
    then calculates the squared distance of each pair with the same float
    operations as a direct test of all pairs, so it gives the same result,
    bit for bit.
    """
    search_m = params.clearance_radius_m * (1.0 + 1e-9)
    pairs = table.sample_tree.sparse_distance_matrix(
        KDTree(np.column_stack([x, y])), search_m, output_type="ndarray"
    )
    sample, point = pairs["i"], pairs["j"]
    flat = table.valid_flat[sample]
    dx = table.x_m.ravel()[flat] - x[point]
    dy = table.y_m.ravel()[flat] - y[point]
    clearance2 = params.clearance_radius_m * params.clearance_radius_m
    return flat[dx * dx + dy * dy <= clearance2]


@dataclass(frozen=True)
class Region:
    """Adjacent usable candidates, first to last (indices into the grid)."""

    first: int
    last: int
    start_deg: float  # deg; steering angle of candidate first
    end_deg: float  # deg; steering angle of candidate last
    mean_safe_m: float  # m; mean safe arc length of the candidates
    center_deg: float  # deg; mean steering angle, weighted by safe arc length


def _split_run_into_plateaus(
    values: CandidateArray, margin_m: float
) -> list[tuple[int, int]]:
    """Split one run of usable candidates (local indices 0 to len(values) - 1)
    into plateaus. Each plateau is a contiguous range of indices around a
    local maximum. Returns the first and the last index of each plateau.

    The highest peak goes first. Each peak extends to each side while the
    next value is within margin_m of the peak value and no other plateau has
    that index. Then each index that no plateau has goes to the nearest
    plateau. A single slope up to one peak and down again thus gives one
    plateau that is symmetric about the peak.
    """
    n = len(values)
    if n == 1:
        return [(0, 0)]
    is_peak = np.ones(n, dtype=bool)
    is_peak[1:] &= values[1:] >= values[:-1]
    is_peak[:-1] &= values[:-1] >= values[1:]
    peak_order = sorted(np.flatnonzero(is_peak), key=lambda k: -values[k])

    claimed = [False] * n
    clusters: list[list[int]] = []
    for k in peak_order:
        if claimed[k]:
            continue
        floor_m = values[k] - margin_m
        lo = k
        while lo - 1 >= 0 and not claimed[lo - 1] and values[lo - 1] >= floor_m:
            lo -= 1
        hi = k
        while hi + 1 < n and not claimed[hi + 1] and values[hi + 1] >= floor_m:
            hi += 1
        cluster = list(range(lo, hi + 1))
        for c in cluster:
            claimed[c] = True
        clusters.append(cluster)

    for k in range(n):
        if claimed[k]:
            continue
        nearest = min(clusters, key=lambda cl: min(abs(k - c) for c in cl))
        nearest.append(k)
        claimed[k] = True
    return [(min(cl), max(cl)) for cl in clusters]


def extract_regions(params: VsParams, safe_m: CandidateArray) -> tuple[Region, ...]:
    """Group the usable candidates into regions.

    A candidate is usable if its safe arc length is at least one clearance
    radius. Each run of adjacent usable candidates splits into one region per
    plateau (see _split_run_into_plateaus), so two openings in one run stay
    two regions.
    """
    steer_deg = arc_table(params).steer_deg
    usable = safe_m >= params.clearance_radius_m
    n = len(safe_m)
    regions: list[Region] = []
    i = 0
    while i < n:
        if not usable[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and usable[j + 1]:
            j += 1
        for lo, hi in _split_run_into_plateaus(
            safe_m[i : j + 1], params.region_margin_m
        ):
            first, last = i + lo, i + hi
            w = safe_m[first : last + 1]
            regions.append(
                Region(
                    first=first,
                    last=last,
                    start_deg=float(steer_deg[first]),
                    end_deg=float(steer_deg[last]),
                    mean_safe_m=float(w.mean()),
                    center_deg=float((w * steer_deg[first : last + 1]).sum() / w.sum()),
                )
            )
        i = j + 1
    return tuple(regions)


def target_path_clearance(
    params: VsParams, safe_m: CandidateArray, target_deg: float
) -> float:
    """The smallest safe arc length (m) of the candidates within
    path_window_candidates grid steps of target_deg. This is the clearance of
    the path that the car drives, not of the straight path."""
    table = arc_table(params)
    in_window = np.abs(table.steer_deg - target_deg) <= table.window_deg
    return float(safe_m[in_window].min())


def score_region(params: VsParams, region: Region, prev_target_deg: float) -> float:
    """Score of a region. A higher score is better."""
    return (
        params.w_clearance * region.mean_safe_m
        + params.w_width * (region.last - region.first + 1)
        - params.w_forward * abs(region.center_deg)
        - params.w_continuity * abs(region.center_deg - prev_target_deg)
    )


@dataclass(frozen=True)
class TrackedRegion:
    """The region that the controller selected at the last tick with a path."""

    score: float
    center_deg: float  # deg


@dataclass(frozen=True)
class VsState:
    """Controller state from one tick to the next. Start a run with VsState()."""

    target_steer_deg: float = 0.0  # deg; target steering angle of the last tick
    tracked: TrackedRegion | None = None  # None until the first region is found


@dataclass(frozen=True)
class Driving:
    """At least one region exists. The car drives toward the selected region."""

    command: ControlCommand
    target_steer_deg: float  # deg; slew-limited target steering angle
    region: Region  # the selected region
    straight_clear_m: float  # m; safe arc length of the 0 deg candidate
    path_clear_m: float  # m; target_path_clearance() at target_steer_deg


@dataclass(frozen=True)
class NoPath:
    """No candidate is usable. The throttle is 0 and the steering stays at the
    last target."""

    command: ControlCommand
    target_steer_deg: float  # deg; the target of the last tick
    straight_clear_m: float  # m; safe arc length of the 0 deg candidate


type VsResult = Driving | NoPath


def _throttle_nm(params: VsParams, path_clear_m: float, target_deg: float) -> float:
    """Throttle (N·m per wheel). It decreases when the target path is short
    and when the steering angle is large. If the target path is usable, the
    throttle is at least min_throttle_nm."""
    steer_max_deg = math.degrees(params.car.steer_max_rad)
    clear_frac = min(1.0, path_clear_m / params.planning_horizon_m)
    steer_frac = min(1.0, abs(target_deg) / steer_max_deg)
    scale = clear_frac * (1.0 - params.steer_throttle_cut * steer_frac)
    throttle = params.max_throttle_nm * scale
    if path_clear_m >= params.clearance_radius_m:
        return max(throttle, params.min_throttle_nm)
    return throttle


def _steer_rad(params: VsParams, target_deg: float) -> float:
    """Steer command (rad, + = left) for a target steering angle in deg."""
    steer_max_deg = math.degrees(params.car.steer_max_rad)
    return target_deg / steer_max_deg * params.car.steer_max_rad


def step(
    params: VsParams,
    state: VsState,
    angles_rad: FloatArray,
    ranges_m: FloatArray,
    dt: float,
) -> tuple[VsState, VsResult]:
    """One control tick.

    angles_rad: beam angles (rad, + = left), ascending.
    ranges_m: range of each beam (m); a value that is not > 0 is no return.
    dt: time since the last tick (s), > 0.
    """
    if not dt > 0:
        raise ValueError(f"dt must be > 0, got {dt}")
    table = arc_table(params)
    safe_m = safe_arc_lengths(params, angles_rad, ranges_m)
    regions = extract_regions(params, safe_m)
    straight_clear_m = float(safe_m[table.straight_index])

    if not regions:
        held = state.target_steer_deg
        return state, NoPath(
            command=ControlCommand(throttle=0.0, steer=_steer_rad(params, held)),
            target_steer_deg=held,
            straight_clear_m=straight_clear_m,
        )

    scored = [(r, score_region(params, r, state.target_steer_deg)) for r in regions]
    best = max(scored, key=lambda t: t[1])
    # Keep the tracked region unless the best region has a much higher score.
    # This stops the target from switching between two similar openings.
    tracked = state.tracked
    if (
        tracked is None
        or tracked.score <= 0
        or best[1] >= tracked.score * params.region_switch_margin
    ):
        chosen, chosen_score = best
    else:
        chosen, chosen_score = min(
            scored, key=lambda t: abs(t[0].center_deg - tracked.center_deg)
        )

    max_step_deg = params.max_slew_deg_per_s * dt
    delta = max(
        -max_step_deg, min(max_step_deg, chosen.center_deg - state.target_steer_deg)
    )
    target_deg = state.target_steer_deg + delta
    path_clear_m = target_path_clearance(params, safe_m, target_deg)

    new_state = VsState(
        target_steer_deg=target_deg,
        tracked=TrackedRegion(score=chosen_score, center_deg=chosen.center_deg),
    )
    return new_state, Driving(
        command=ControlCommand(
            throttle=_throttle_nm(params, path_clear_m, target_deg),
            steer=_steer_rad(params, target_deg),
        ),
        target_steer_deg=target_deg,
        region=chosen,
        straight_clear_m=straight_clear_m,
        path_clear_m=path_clear_m,
    )


def label(result: VsResult) -> str:
    """Short name of the result case, for logs."""
    match result:
        case Driving():
            return "driving"
        case NoPath():
            return "no_path"
        case _:
            assert_never(result)
