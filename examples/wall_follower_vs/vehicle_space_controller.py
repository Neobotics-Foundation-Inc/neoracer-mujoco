"""
Vehicle-space wall follower -- experimental, NOT part of PR #27.

Replaces the two prior experimental attempts (see design notes at the bottom
of this docstring) with a controller whose steering-candidate count and
obstacle-inflation geometry are BOTH independent of LiDAR sample count N.
The same 81 fixed steering candidates and the same physical clearance/
horizon are evaluated whether the scan has 24 or 1080 points -- only the
density of Cartesian obstacle points describing the environment changes
with N.

KINEMATIC-TRAJECTORY REVISION (loop_corridor.xml Stage A/B follow-up)

Candidates used to represent instantaneous straight HEADINGS ("if the car's
center travels in a straight line along this direction..."). That model let
the controller believe a +60deg direction was "safe" whenever a straight ray
at +60deg was clear, even though the car cannot instantaneously reorient --
it can only curve onto a new heading along an Ackermann arc set by its
current steering angle and wheelbase. On loop_corridor.xml's first chamfered
corner this produced a real wedge: the sharp turn wasn't selected until
forward clearance was ~0.12-0.22m, by which point the car was already closer
to the wall than its own minimum turning radius (~0.68m theoretical, ~0.75m
empirical -- see design notes at WHEELBASE_M) permits recovering from.

Candidates now represent STEERING ANGLES (equivalently, curvatures), not
headings: each one is a full circular-arc trajectory computed from the
vehicle's actual Ackermann geometry, collision-checked along its entire
swept path (not just its endpoint or its centerline ray). A "+60deg desired
direction" is no longer treated as reachable just because a straight ray
that way is open -- it has to survive the geometry of actually curving onto
it. This makes early turn commitment an emergent property of the collision
geometry (a moderate-curvature arc reaches the wall later than a straight
one) rather than something bolted on as a track-specific heuristic.

PIPELINE
    dense LiDAR (theta, r)
        -> Cartesian vehicle-space obstacle points (Step 1, unchanged)
        -> fixed 81-candidate STEERING-ANGLE grid, independent of N (Step 2)
        -> per-candidate circular ARC trajectory from Ackermann geometry
           (Step 3)
        -> collision check along the SWEPT ARC (not a straight tube) against
           the same Cartesian obstacle points -> safe_arc_length (Step 4)
        -> contiguous safe-steering regions ("plateaus"), not per-candidate
           argmax (Step 4/6, unchanged algorithm, now over steering degrees)
        -> region scoring + slew-limited, hysteretic target STEERING ANGLE
           (Step 5/8, unchanged algorithm, now over steering degrees)
        -> NORMAL/NO_PATH/RECOVERING state machine (Step 9, unchanged)
        -> speed scheduler (Step 10, unchanged) + DIRECT steering-angle
           output, no intermediate P-controller (Step 11 -- see design notes
           at VehicleSpaceController._steer_angle for why A-vs-B was decided
           in favor of B)

WHY THE PRIOR TWO ATTEMPTS FAILED (do not repeat)

  1. Old per-ray-candidate controller: treated every LiDAR ray as its own
     steering candidate, so candidate count and the per-candidate blocking
     check both scaled with N -- O(N^2) work per tick (~51ms at N=1080) and,
     because a single grazing ray could veto a whole direction, the
     candidate set became noisier (not more precise) as N grew, exactly
     backwards from what better sensor resolution should buy you.

  2. Gap/shadow-union controller: fixed the compute cost by turning each
     LiDAR hit into an angular "shadow" and unioning shadows to find gaps in
     the union's complement. At high N, dense samples along a single
     continuous wall union into one enormous shadow -- long side walls can
     blot out the entire forward FOV even though the car has 3m of clear
     road directly ahead, because the union operation has no notion of
     "this obstacle is off to the side, not in front."

  The failure in both cases is the same at its root: they let the RAW BEAM
  COUNT drive either the candidate set or the obstacle representation. This
  controller fixes that by converting the scan to Cartesian points ONCE and
  then asking a purely geometric question per fixed candidate -- "does any
  obstacle point fall inside the physical clearance tube ahead of the car
  along this heading?" A wall 0.5m to the side has a large *lateral*
  coordinate in every candidate's rotated frame; it only threatens headings
  that actually point toward it. Density of samples along that wall does
  not change its lateral coordinate, so it cannot make the forward heading
  look more blocked no matter how many points describe it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# --- vehicle kinematics (verified from assets/neoracer.xml -- NOT assumed) --
# Wheelbase: front/rear axle x-offsets from the car body origin are both
# 0.1439m (fl_wheel_body / rl_wheel_body "pos" attributes in neoracer.xml),
# so L = 2*0.1439 = 0.2878m.
WHEELBASE_M = 0.2878
# Track width: front-wheel y-offset is 0.1175m each side -> W = 0.235m.
# Not used directly here (the steer_input joint's Ackermann polycoef
# constraints already convert this one command into the correct fl_steer/
# fr_steer inner/outer angles -- see neoracer.xml's <equality> block), but
# recorded since it's part of the same kinematic derivation.
TRACK_WIDTH_M = 0.235
# steer_servo actuator ctrlrange in neoracer.xml is [-0.4, 0.4] rad. Per
# neoracer.xml's own comment above its <equality> block, this IS the
# standard bicycle-model steering angle delta (the polycoef fl/fr formulas
# are exactly fl_steer=atan(L/(L/tan(delta)-W/2)), fr_steer=atan(L/(L/tan(delta)+W/2))
# in terms of this one command), so R = L / tan(delta) applies directly to
# ctrl[4] without averaging fl_steer_pos/fr_steer_pos.
STEER_ANGLE_MAX_RAD = 0.4
STEER_MAX_DEG = math.degrees(STEER_ANGLE_MAX_RAD)  # ~22.918deg
# Theoretical minimum turning radius: R = L / tan(delta_max) ~= 0.681m.
# Empirically verified (see design notes / diagnostics/verify_kinematics.py):
# constant low forward torque + full steering lock traced a path whose
# least-squares circle fit gives R ~= 0.75m -- ~10% larger than theory, the
# expected direction and magnitude for tire slip/suspension compliance, and
# close enough to confirm the model behaves approximately Ackermann.
MIN_TURN_RADIUS_M = WHEELBASE_M / math.tan(STEER_ANGLE_MAX_RAD)

# --- fixed physical parameters (Step 2/3) -----------------------------------
# 81 candidates, independent of LiDAR N -- mandatory per design spec. Each
# candidate is now a STEERING ANGLE in degrees (was a heading in degrees).
N_CANDIDATES = 81
CANDIDATE_HEADINGS_DEG = np.linspace(-STEER_MAX_DEG, STEER_MAX_DEG, N_CANDIDATES)
assert len(CANDIDATE_HEADINGS_DEG) == 81

CLEARANCE_RADIUS_M = 0.21  # car physical half-width, swept along each arc
PLANNING_HORIZON_M = 3.0  # forward lookahead cap (arc length, meters)

# --- arc trajectory generation (Step 3) --------------------------------------
# Meters between arc samples -- independent of LiDAR beam count, per spec.
# 0.1m gives 31 samples over the 3.0m horizon, still well under half of
# CLEARANCE_RADIUS_M (0.21m) -- consecutive samples can't be far enough
# apart to let a collision pass between them undetected. (0.05m/61 samples
# was tried first and measured ~2x slower for no detection-accuracy
# difference at this margin -- a profiling choice, not a behavior tune.)
ARC_SAMPLE_DS_M = 0.1

# --- outlier / geometry robustness (Step 4/7), physical angle-based --------
OUTLIER_WINDOW_DEG = 2.0  # local median filter neighborhood
MAX_SEGMENT_GAP_M = 0.5  # adjacent samples closer than this get a midpoint
# added between them, so a wall surface passing between two sparse samples
# still registers as an obstacle point. Two samples farther apart than this
# are assumed to belong to different surfaces (e.g. across an opening) and
# are NOT bridged -- bridging them would fabricate a wall across a gap.

# --- region extraction (Step 6) ---------------------------------------------
REGION_MARGIN_M = 0.3  # candidates within this of a region's own local peak
# are considered part of the same "wide safe region" as that peak.
# CLEARANCE_RADIUS_M: a candidate offering less than one clearance-radius of
# travel is not a usable path -- e.g. an oblique candidate grazing along a
# corridor narrower than the clearance tube itself can register a nominal
# ~0.1m of "safe" travel through near-parallel rays without ever
# representing a direction the car could actually drive.
MIN_PROGRESS_M = CLEARANCE_RADIUS_M

# --- target scoring (Step 5) -------------------------------------------------
W_CLEARANCE = 1.0
W_FORWARD = 0.01  # per-degree penalty for being off center
W_CONTINUITY = 0.02  # per-degree penalty for moving from the previous target
W_WIDTH = 0.05  # per-candidate-in-region bonus (wider region = more robust)

# --- temporal continuity (Step 8), physical units ---------------------------
MAX_SLEW_DEG_PER_S = 90.0  # hard cap on target heading rate of change
REGION_SWITCH_MARGIN = 1.15  # a new region must score >=15% better than the
# currently-tracked one to be adopted -- hysteresis against flicker between
# two similarly-good regions.

# --- no-path state machine (Step 9) -----------------------------------------
NO_PATH_HOLD_S = 0.5  # how long to coast on the last known-good heading
# before the controller admits it has genuinely lost the path and starts
# slowing toward a stop.

# --- speed scheduler (Step 10) ----------------------------------------------
BASE_SPEED_CAP = 0.03  # racecar_core speed fraction upper bound (mandatory)
SPEED_CLEARANCE_REF_M = (
    PLANNING_HORIZON_M  # selected-path clearance that earns full speed
)
SPEED_STEER_PENALTY = 0.6  # fraction of speed shed at full-deflection steering

# Stage A fix (traced wedge on loop_corridor.xml seed 0): the scheduler used
# to key off the dead-ahead (0deg) candidate's clearance even while steering
# hard toward a completely different, actually-clear heading -- exactly when
# the car most needed torque to complete a turn, dead-ahead clearance
# collapsed toward zero (because that's WHY it was turning) and starved
# speed toward zero with it, producing a self-reinforcing wedge. The
# scheduler now keys off the clearance of the PATH ACTUALLY BEING DRIVEN
# (get_target_path_clearance()), not the dead-ahead candidate.
# Window scaled to 2 candidate steps, not a fixed absolute degree value:
# the old heading-degree grid's step was 2deg (window_deg=4 covered ~5
# candidates); the new steering-degree grid's step is much finer
# (~0.573deg, since the whole grid now spans +/-STEER_MAX_DEG ~=22.9deg
# instead of +/-80deg), so reusing the old absolute 4deg would cover ~15
# candidates -- a much wider smoothing than originally intended. Each
# candidate here is also already a full swept-arc collision result (not a
# single raw ray), so it needs less smoothing to begin with; 2 steps keeps
# "a small neighborhood around the selected candidate," proportioned to
# the new domain, without over-averaging across meaningfully different arcs.
_STEER_STEP_DEG = (2.0 * STEER_MAX_DEG) / (N_CANDIDATES - 1)
PATH_CLEARANCE_WINDOW_DEG = 2.0 * _STEER_STEP_DEG

# Applying the torque floor only when the selected path clears this bar
# reuses MIN_PROGRESS_M -- the same threshold extract_regions() already uses
# to call a candidate "usable" -- instead of inventing a second one.
TORQUE_FLOOR_MIN_CLEARANCE_M = MIN_PROGRESS_M
# Bounded 3-value diagnostic on the failing seed (0.006, 0.010, 0.015 -- not
# a sweep): all three kept the commanded speed correctly pinned at the floor
# value through the turn (verified in the trace -- speed no longer collapses
# toward ~0.0003-0.0004 the way it did before this fix), but NONE of the
# three actually cleared the corner -- the car still wedges at the same
# position with the same ~0.17 lap fraction regardless of floor value. This
# means torque starvation was a real, now-fixed defect, but was NOT the
# (sole) cause of the wedge -- see the Stage A follow-up report for the
# deeper cause found once torque starvation was ruled out (a target-
# selection/turn-radius timing issue, not fixed here). Kept at 0.015 (the
# largest of the three, i.e. the most propulsion without exceeding
# BASE_SPEED_CAP) since it is at least as good as the smaller values and
# none showed any downside.
TORQUE_FLOOR_SPEED = 0.015

# --- steering output (Step 11) -----------------------------------------------
# Architecture A vs B (kinematic-trajectory revision):
#   A: trajectory candidate -> desired heading -> P controller -> steering
#   B: trajectory candidate -> steering command directly
# A made sense under the OLD heading-candidate model, where a candidate was
# an abstract direction and a P law was the only thing converting "desired
# direction" into a physical steering command. Under the NEW model, a
# candidate IS already a physical steering angle (that's what its arc was
# generated from) -- there is no separate "desired heading" left to control
# toward. Running the selected steering angle back through a P-controller
# error law (error = angle/max, output = kp*error) would just re-derive the
# same value scaled by kp (kp=1.0 makes it an identity map anyway) while
# adding a second, redundant control loop stacked on a quantity that is
# already in exact physical units. B is implemented: the selected
# candidate's own steering degrees maps directly (scaled + sign-flipped for
# racecar_core's convention) to the output angle -- see
# VehicleSpaceController._steer_angle(). STEER_KP/STEER_KD (the old P/D
# gains) are removed entirely, not just zeroed, since there is no error
# signal left for them to act on.
CONTROL_HZ = 30.0


def lidar_to_points(
    angles_rad: np.ndarray, ranges_m: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a physical LiDAR scan to vehicle-space Cartesian obstacle
    points (Step 1). +X forward, +Y left (matches neoracer.xml's body frame).

    Invalid samples (non-finite, <=0, i.e. no return) are dropped -- they
    carry no evidence of an obstacle in that direction, so they must not be
    converted into a fabricated point. Angle ordering is not required by
    this function (mask-then-index preserves whatever order was given), but
    callers should pass scans in physical angle order per the design spec.
    """
    ranges_m = np.asarray(ranges_m, dtype=float)
    angles_rad = np.asarray(angles_rad, dtype=float)
    valid = np.isfinite(ranges_m) & (ranges_m > 0.0)
    r = ranges_m[valid]
    theta = angles_rad[valid]
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def _angular_median_filter(
    angles_rad: np.ndarray, ranges_m: np.ndarray, window_deg: float = OUTLIER_WINDOW_DEG
) -> np.ndarray:
    """Median-filter ranges over a fixed PHYSICAL angular neighborhood
    (Step 7), not a fixed beam count -- at low N the window collapses to
    fewer (or zero extra) neighbors automatically, since the filter is
    defined in degrees, not indices. A lone long-range or short-range spike
    surrounded by consistent neighbors is pulled to the local median instead
    of being able to single-handedly clear or block a heading.
    """
    n = len(ranges_m)
    if n < 3:
        return ranges_m.copy()
    window_rad = math.radians(window_deg)
    out = ranges_m.copy()
    # angles_rad is assumed sorted ascending (physical angle order).
    lo = np.searchsorted(angles_rad, angles_rad - window_rad, side="left")
    hi = np.searchsorted(angles_rad, angles_rad + window_rad, side="right")
    for i in range(n):
        if hi[i] - lo[i] <= 1:
            continue  # no neighbors inside the window -- nothing to filter
        out[i] = np.median(ranges_m[lo[i] : hi[i]])
    return out


def _obstacle_points_with_midpoints(
    angles_rad: np.ndarray, ranges_m: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Build the Cartesian obstacle point set used for clearance checks:
    every valid LiDAR return, PLUS a midpoint inserted between adjacent
    valid samples that are close enough together to plausibly be the same
    continuous surface (Step 4). This is a deliberately simple stand-in for
    full point-to-segment distance: it catches a wall surface that would
    otherwise pass between two sparse samples without adding a real
    line-clip solver. Samples farther apart than MAX_SEGMENT_GAP_M are left
    unconnected -- e.g. across a doorway/opening -- so a gap is never
    bridged into a fake wall.
    """
    filtered = _angular_median_filter(angles_rad, ranges_m)
    x, y = lidar_to_points(angles_rad, filtered)
    if len(x) < 2:
        return x, y
    dx = np.diff(x)
    dy = np.diff(y)
    seg_len = np.hypot(dx, dy)
    bridge = seg_len <= MAX_SEGMENT_GAP_M
    mx = (x[:-1] + x[1:])[bridge] / 2.0
    my = (y[:-1] + y[1:])[bridge] / 2.0
    return np.concatenate([x, mx]), np.concatenate([y, my])


def _build_arc_table(
    steer_deg_grid: np.ndarray, wheelbase_m: float, horizon_m: float, ds_m: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Precompute each candidate's circular-arc trajectory ONCE at import
    time (Step 3): the arc geometry depends only on wheelbase + steering
    angle + horizon, none of which change tick to tick, so this is pure
    setup cost, not part of the per-tick budget.

    For steering angle delta, curvature kappa = tan(delta)/L (bicycle
    model -- see WHEELBASE_M's derivation for why ctrl[4]/delta applies
    directly). The vehicle center's position at arc length s along that
    curvature, starting at the origin heading along +X:
        x(s) = sin(kappa*s) / kappa
        y(s) = (1 - cos(kappa*s)) / kappa
    which is the standard circular-arc parametrization (radius R=1/kappa,
    center at (0, R)); as kappa -> 0 this reduces to the straight line
    x(s)=s, y(s)=0, handled as an explicit special case to avoid a 0/0.

    180-DEGREE SWEEP CAP (found via synthetic test C, a geometric
    correctness fix, not a tuned parameter): at the tightest candidate
    curvature (R ~= MIN_TURN_RADIUS_M ~= 0.68m), horizon_m=3.0m of arc
    length sweeps kappa*s ~= 253 degrees -- past a full half-circle. Beyond
    180 degrees the arc's own x(s) turns negative and the path heads back
    toward where the vehicle started, i.e. it can loop all the way around
    behind a nearby wall corner without ever coming within clearance of it,
    which made a maximal-lock turn read as "fully open to the horizon" when
    it was actually just circling away from the wall in a tiny useless
    loop -- not a reachable, useful escape for a forward-driving reactive
    controller. cap_m per candidate = min(horizon_m, pi/|kappa|), i.e. the
    arc length at which kappa*s reaches 180 degrees; straight (kappa~=0) is
    uncapped (never wraps).
    """
    n_samples = round(horizon_m / ds_m) + 1
    s = np.linspace(0.0, horizon_m, n_samples)
    arc_x = np.zeros((len(steer_deg_grid), n_samples))
    arc_y = np.zeros_like(arc_x)
    cap_m = np.zeros(len(steer_deg_grid))
    for i, deg in enumerate(steer_deg_grid):
        delta = math.radians(deg)
        if abs(delta) < 1e-6:
            arc_x[i] = s
            arc_y[i] = 0.0
            cap_m[i] = horizon_m
        else:
            kappa = math.tan(delta) / wheelbase_m
            arc_x[i] = np.sin(kappa * s) / kappa
            arc_y[i] = (1.0 - np.cos(kappa * s)) / kappa
            cap_m[i] = min(horizon_m, math.pi / abs(kappa))
    return s, arc_x, arc_y, cap_m


ARC_S_M, ARC_X_M, ARC_Y_M, ARC_CAP_M = _build_arc_table(
    CANDIDATE_HEADINGS_DEG, WHEELBASE_M, PLANNING_HORIZON_M, ARC_SAMPLE_DS_M
)
# (candidates, samples) mask of which samples are within each candidate's
# own 180-degree-sweep cap -- precomputed once, reused every tick.
ARC_VALID_MASK = ARC_S_M[None, :] <= ARC_CAP_M[:, None]


def safe_arc_lengths(
    angles_rad: np.ndarray,
    ranges_m: np.ndarray,
    clearance_radius_m: float = CLEARANCE_RADIUS_M,
    planning_horizon_m: float = PLANNING_HORIZON_M,
) -> np.ndarray:
    """Step 4: for every fixed candidate STEERING ANGLE, how far along that
    candidate's actual Ackermann arc can the vehicle travel before any
    obstacle point comes within clearance_radius_m of the swept path?
    Candidates with no such point along the whole horizon score the full
    horizon.

    This replaces the old straight-tube safe_travel_distances(): a point is
    now checked against the CURVED path the car would actually follow to
    reach that heading, not a straight ray toward it. A +60deg candidate
    whose straight ray is open but whose actual arc clips a nearby wall
    before reaching that heading is correctly scored as reaching only as
    far as the clip point, not the full horizon.
    """
    x, y = _obstacle_points_with_midpoints(angles_rad, ranges_m)
    n_cand = ARC_X_M.shape[0]
    # No-threat fallback is each candidate's own 180-degree-sweep cap (see
    # _build_arc_table), not a flat planning_horizon_m -- a tight-curvature
    # candidate that loops back on itself past 180deg cannot be scored as
    # "safe to the horizon" using arc length alone (see module docstring).
    out = np.minimum(ARC_CAP_M, planning_horizon_m)
    if len(x) == 0:
        return out

    # Points farther from the origin than any arc could possibly reach
    # (horizon + clearance_radius, generously rounded) can't threaten any
    # candidate -- drop them before the per-candidate loop.
    reach = planning_horizon_m + clearance_radius_m
    keep = np.hypot(x, y) <= reach
    x, y = x[keep], y[keep]
    if len(x) == 0:
        return out

    n_s = ARC_X_M.shape[1]
    clearance2 = clearance_radius_m * clearance_radius_m
    for c in range(n_cand):
        dx = ARC_X_M[c][:, None] - x[None, :]  # (S, M)
        dy = ARC_Y_M[c][:, None] - y[None, :]
        within = (dx * dx + dy * dy) <= clearance2
        within &= ARC_VALID_MASK[c][:, None]  # ignore the looped-back tail
        # argmax gives the first True index per point, but returns 0 (a
        # false positive) for a point that never threatens at all -- mask
        # those out to the out-of-range sentinel n_s before taking the min,
        # rather than copying a fancy-indexed subset (measured ~2x faster).
        first_s_idx = np.where(within.any(axis=0), np.argmax(within, axis=0), n_s)
        c_first = int(first_s_idx.min())
        if c_first < n_s:
            out[c] = ARC_S_M[c_first]
    return out


def _split_run_into_plateaus(values: np.ndarray, margin_m: float) -> list[list[int]]:
    """Partition one contiguous run of usable candidates (local indices
    0..len(values)-1) into plateaus, each centered on a local maximum, so a
    monotonic rise-to-a-peak-then-fall (e.g. approaching a single open
    corridor head-on) becomes ONE symmetric region instead of a one-pass
    left-to-right merge dragging the whole slope in (that asymmetric-merge
    bug was caught by test_straight_centered_corridor reporting a target
    ~-19deg for a perfectly symmetric scene -- see design notes).

    Each local maximum expands outward while neighbors are within margin_m
    of ITS OWN peak and not already claimed by a higher peak; any indices
    left over (dips between plateaus that don't reach any peak's margin)
    are folded into the index-nearest claimed plateau, so every usable
    candidate ends up in exactly one region.
    """
    n = len(values)
    if n == 1:
        return [[0]]
    is_peak = np.ones(n, dtype=bool)
    is_peak[1:] &= values[1:] >= values[:-1]
    is_peak[:-1] &= values[:-1] >= values[1:]
    peak_order = sorted(np.flatnonzero(is_peak), key=lambda k: -values[k])

    claimed = [False] * n
    clusters: list[list[int]] = []
    for k in peak_order:
        if claimed[k]:
            continue
        peak_val = values[k]
        lo = k
        while (
            lo - 1 >= 0
            and not claimed[lo - 1]
            and values[lo - 1] >= peak_val - margin_m
        ):
            lo -= 1
        hi = k
        while (
            hi + 1 < n and not claimed[hi + 1] and values[hi + 1] >= peak_val - margin_m
        ):
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
    return [sorted(cl) for cl in clusters]


def extract_regions(
    safe_dist: np.ndarray,
    heading_deg: np.ndarray = CANDIDATE_HEADINGS_DEG,
    margin_m: float = REGION_MARGIN_M,
    min_progress_m: float = MIN_PROGRESS_M,
) -> list[dict]:
    """Step 6: group contiguous candidates into navigable regions instead of
    picking a lone per-candidate argmax (Step 4/5's "one wide free corridor,
    not 81 independent decisions"). Usable candidates (safe_dist >=
    min_progress_m) are first grouped by plain contiguity into runs, then
    each run is split into one or more peak-centered plateaus (see
    _split_run_into_plateaus) so multiple distinct openings within one
    usable run are not incorrectly fused into a single region.
    """
    usable = safe_dist >= min_progress_m
    n = len(safe_dist)
    regions = []
    i = 0
    while i < n:
        if not usable[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and usable[j + 1]:
            j += 1
        run_values = safe_dist[i : j + 1]
        for local_cluster in _split_run_into_plateaus(run_values, margin_m):
            idx = np.array([i + k for k in local_cluster])
            regions.append(
                {
                    "start_deg": float(heading_deg[idx[0]]),
                    "end_deg": float(heading_deg[idx[-1]]),
                    "indices": idx,
                    "mean_safe_m": float(safe_dist[idx].mean()),
                    "min_safe_m": float(safe_dist[idx].min()),
                    "width_deg": float(heading_deg[idx[-1]] - heading_deg[idx[0]]),
                    "n_candidates": len(idx),
                }
            )
        i = j + 1
    return regions


def _region_center_deg(
    region: dict, safe_dist: np.ndarray, heading_deg: np.ndarray
) -> float:
    """Clearance-weighted centroid of a region (Step 6's second option)."""
    idx = region["indices"]
    w = safe_dist[idx]
    total = w.sum()
    if total <= 0:
        return (region["start_deg"] + region["end_deg"]) / 2.0
    return float((w * heading_deg[idx]).sum() / total)


def get_target_path_clearance(
    safe_dist: np.ndarray,
    target_deg: float,
    heading_deg: np.ndarray = CANDIDATE_HEADINGS_DEG,
    window_deg: float = PATH_CLEARANCE_WINDOW_DEG,
) -> float:
    """Conservative clearance of the path the controller actually intends to
    drive (Stage A fix) -- the MINIMUM safe_travel_distance among candidates
    within window_deg of target_deg, e.g. if target_deg=+60, this reads the
    +60 path's own clearance, not the (possibly fully blocked) 0deg one.

    Minimum, not a percentile: at this window's scale (window_deg=4,
    candidate spacing=2deg) at most ~5 candidates fall in the window, so any
    percentile above the 0th is effectively picking among the 1-2 lowest
    values anyway. Minimum keeps this consistent with the worst-case
    philosophy safe_travel_distances() itself already uses (one threatening
    point anywhere in a candidate's tube sets that candidate's whole
    distance) instead of introducing a different statistic here.
    """
    in_window = np.abs(heading_deg - target_deg) <= window_deg
    if not np.any(in_window):
        # target_deg is slew-limited and can sit slightly off the candidate
        # grid's own alignment -- fall back to the single nearest candidate
        # rather than an empty window.
        nearest = int(np.argmin(np.abs(heading_deg - target_deg)))
        return float(safe_dist[nearest])
    return float(safe_dist[in_window].min())


def _score_region(region: dict, center_deg: float, prev_target_deg: float) -> float:
    return (
        W_CLEARANCE * region["mean_safe_m"]
        + W_WIDTH * region["n_candidates"]
        - W_FORWARD * abs(center_deg)
        - W_CONTINUITY * abs(center_deg - prev_target_deg)
    )


@dataclass
class VehicleSpaceResult:
    speed: float
    angle: float  # racecar_core convention: -1..1, + = right
    target_heading_deg: float  # selected candidate's STEERING ANGLE, degrees
    # (kinematic-trajectory revision -- see module docstring; the name is
    # kept for minimal churn, but this is no longer an abstract heading).
    state: str  # "NORMAL" | "NO_PATH" | "RECOVERING"
    safe_dist: np.ndarray = field(default_factory=lambda: np.zeros(0))
    regions: list = field(default_factory=list)
    chosen_region: dict | None = None
    forward_clear_m: float = 0.0  # straight (delta~=0) candidate's safe arc
    # length; diagnostic/comparison only, never drives the speed scheduler.
    selected_path_clear_m: float = 0.0  # safe arc length of the path actually
    # being driven (see get_target_path_clearance) -- the speed scheduler's input.
    no_path_ticks: int = 0

    @property
    def curvature_1_per_m(self) -> float:
        """kappa = tan(delta)/L for the selected steering angle. 0 for a
        straight (delta~=0) candidate."""
        delta = math.radians(self.target_heading_deg)
        if abs(delta) < 1e-6:
            return 0.0
        return math.tan(delta) / WHEELBASE_M

    @property
    def turn_radius_m(self) -> float:
        """R = 1/|kappa|; +inf for a straight (delta~=0) candidate."""
        kappa = self.curvature_1_per_m
        return float("inf") if kappa == 0.0 else abs(1.0 / kappa)


class VehicleSpaceController:
    """Stateful controller. Call compute(angles_rad, ranges_m, dt) once per
    30 Hz decision tick, in physical LiDAR angle order (radians, +left)."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._prev_target_deg = 0.0
        self._prev_region_score = -math.inf
        self._prev_region_center = 0.0
        self._no_path_ticks = 0
        self._state = "NORMAL"

    def compute(
        self, angles_rad: np.ndarray, ranges_m: np.ndarray, dt: float
    ) -> VehicleSpaceResult:
        safe_dist = safe_arc_lengths(angles_rad, ranges_m)
        regions = extract_regions(safe_dist)
        # "forward_clear_m" here is the STRAIGHT (delta~=0) candidate's own
        # safe arc length -- kept only as a diagnostic/comparison value (see
        # Step 11's requested straight-vs-selected logging), never used to
        # drive the speed scheduler (that's selected_path_clear_m below).
        straight_idx = int(np.argmin(np.abs(CANDIDATE_HEADINGS_DEG)))
        forward_clear_m = float(safe_dist[straight_idx])

        if not regions:
            # --- NO_PATH state (Step 9) -----------------------------------
            self._no_path_ticks += 1
            hold_ticks = NO_PATH_HOLD_S * CONTROL_HZ
            if self._no_path_ticks <= hold_ticks:
                state = "NO_PATH"
                target_deg = self._prev_target_deg  # coast on last heading
                speed = 0.0
            else:
                state = "RECOVERING"
                target_deg = self._prev_target_deg
                speed = 0.0
            angle = self._steer_angle(target_deg, dt)
            # No region exists, so there is no selected path -- computed here
            # only as a diagnostic (e.g. for trial logging), never used to
            # raise speed above 0.0 while NO_PATH/RECOVERING.
            selected_path_clear_m = get_target_path_clearance(safe_dist, target_deg)
            return VehicleSpaceResult(
                speed,
                angle,
                target_deg,
                state,
                safe_dist,
                regions,
                None,
                forward_clear_m,
                selected_path_clear_m,
            )

        self._no_path_ticks = 0

        # --- region scoring + hysteresis (Step 5/6/8) ----------------------
        scored = [
            (
                r,
                _region_center_deg(r, safe_dist, CANDIDATE_HEADINGS_DEG),
            )
            for r in regions
        ]
        scored = [(r, c, _score_region(r, c, self._prev_target_deg)) for r, c in scored]
        best_region, best_center, best_score = max(scored, key=lambda t: t[2])

        # Keep tracking the previously-chosen region unless a candidate
        # region beats it by REGION_SWITCH_MARGIN (prevents flicker between
        # two similarly-good openings).
        if (
            best_score >= self._prev_region_score * REGION_SWITCH_MARGIN
            or math.isinf(self._prev_region_score)
            or self._prev_region_score <= 0
        ):
            chosen_region, chosen_center, chosen_score = (
                best_region,
                best_center,
                best_score,
            )
        else:
            # find the region closest to the previously-tracked center and
            # keep it if it is still valid this tick.
            prev_match = min(scored, key=lambda t: abs(t[1] - self._prev_region_center))
            chosen_region, chosen_center, chosen_score = prev_match

        self._prev_region_score = chosen_score
        self._prev_region_center = chosen_center

        # --- slew limit (Step 8) -------------------------------------------
        max_step_deg = MAX_SLEW_DEG_PER_S * dt
        delta = chosen_center - self._prev_target_deg
        delta = max(-max_step_deg, min(max_step_deg, delta))
        target_deg = self._prev_target_deg + delta
        self._prev_target_deg = target_deg

        state = "NORMAL"
        selected_path_clear_m = get_target_path_clearance(safe_dist, target_deg)
        speed = self._speed(selected_path_clear_m, target_deg, state)
        angle = self._steer_angle(target_deg, dt)

        return VehicleSpaceResult(
            speed,
            angle,
            target_deg,
            state,
            safe_dist,
            regions,
            chosen_region,
            forward_clear_m,
            selected_path_clear_m,
        )

    def _speed(self, path_clear_m: float, target_deg: float, state: str) -> float:
        """Turn-aware speed scheduler (Stage A fix). path_clear_m is the
        clearance of the path actually being driven (get_target_path_
        clearance()), NOT the dead-ahead candidate -- see design notes at
        this module's SPEED_* constants for why that distinction is the fix.
        """
        clear_frac = min(1.0, path_clear_m / SPEED_CLEARANCE_REF_M)
        steer_frac = min(1.0, abs(target_deg) / STEER_MAX_DEG)
        scale = clear_frac * (1.0 - SPEED_STEER_PENALTY * steer_frac)
        speed = BASE_SPEED_CAP * scale
        # Torque floor: only while there IS a genuinely valid selected path
        # (state has an actual target to drive, and that path clears the
        # same bar extract_regions() already calls "usable") -- never a
        # universal minimum throttle. Prevents the scheduler's own
        # clearance/steering scaling from starving torque to near-zero
        # exactly when a sharp-but-valid turn needs it most (the traced
        # wedge failure).
        if (
            state in ("NORMAL", "RECOVERING")
            and path_clear_m >= TORQUE_FLOOR_MIN_CLEARANCE_M
        ):
            speed = max(speed, TORQUE_FLOOR_SPEED)
        return max(0.0, min(BASE_SPEED_CAP, speed))

    def _steer_angle(self, target_deg: float, dt: float) -> float:
        """Direct steering-command mapping (Architecture B -- see design
        notes at this module's "Step 11" constants block for why A was
        rejected). target_deg IS the selected candidate's physical steering
        angle in degrees already (slew-limited + hysteretic upstream in
        compute()); this only rescales it to the actuator's [-1, 1] command
        range and flips sign for racecar_core's convention. dt is accepted
        (unused) to keep the same call signature as the prior P/D version
        without perturbing compute()'s call sites.
        """
        del dt
        steering = max(-1.0, min(1.0, target_deg / STEER_MAX_DEG))
        # target_deg convention: + = left (Step 1's Cartesian frame). racecar_core's
        # angle convention is + = right (see racecar_core_adapter.py's own
        # negation for the same reason) -- negate exactly once, here.
        return -steering
