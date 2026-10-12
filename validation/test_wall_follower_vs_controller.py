"""
Tests of the vehicle-space wall follower (examples/wall_follower_vs/) on
synthetic scans. The tests use no physics and no unseeded random numbers.

Frame: +X forward, +Y left. A steering angle is + = left.
"""

from typing import assert_never

import numpy as np
import pytest

from examples.wall_follower_vs.dense_lidar import DenseLidarParams, Scan
from examples.wall_follower_vs.synthetic_scenes import (
    Segment,
    cast_scan,
    corridor_walls,
    gap_ring_scan,
)
from examples.wall_follower_vs.vehicle_space_controller import (
    Driving,
    NoPath,
    VsParams,
    VsState,
    arc_table,
    car_geometry,
    extract_regions,
    safe_arc_lengths,
    score_region,
    step,
)
from neoracer_mujoco import assets
from neoracer_mujoco.control.wall_following import Side

PARAMS = VsParams(car=car_geometry(assets.load()))
TABLE = arc_table(PARAMS)
DT = 1.0 / 30.0  # s; one tick at 30 Hz
RESOLUTIONS = [24, 270, 540, 1080]


def _scan(walls: list[Segment], n: int) -> Scan:
    return cast_scan(walls, DenseLidarParams(n_beams=n))


def _mirror(walls: list[Segment]) -> list[Segment]:
    """The same walls, mirrored about the X axis (y -> -y)."""
    return [((x1, -y1), (x2, -y2)) for (x1, y1), (x2, y2) in walls]


def _random_walls(rng: np.random.Generator) -> list[Segment]:
    """Three wall segments with random end points in front of and beside the car."""

    def point() -> tuple[float, float]:
        return float(rng.uniform(-1.0, 4.0)), float(rng.uniform(-2.0, 2.0))

    return [(point(), point()) for _ in range(3)]


def _best_center_deg(angles: np.ndarray, ranges: np.ndarray) -> float:
    """Center (deg) of the region with the best score_region() for a
    controller whose last target is 0 deg."""
    regions = extract_regions(PARAMS, safe_arc_lengths(PARAMS, angles, ranges))
    assert regions, "expected at least one usable arc"
    best = max(regions, key=lambda r: score_region(PARAMS, r, prev_target_deg=0.0))
    return best.center_deg


def _forward_blocked_side_open(open_side: Side, wall_x_m: float = 1.0) -> list[Segment]:
    """A wall at x = wall_x_m in front of the car and on one side. The side
    open_side stays open, so an arc can turn into it."""
    match open_side:
        case "right":
            return [((wall_x_m, -0.1), (wall_x_m, 2.0))]
        case "left":
            return _mirror(_forward_blocked_side_open("right", wall_x_m))
        case _:
            assert_never(open_side)


@pytest.mark.parametrize("n", RESOLUTIONS)
def test_straight_corridor_prefers_straight(n):
    target_deg = _best_center_deg(*_scan(corridor_walls(), n))
    assert abs(target_deg) < 2.0, (
        f"N={n}: expected about 0 deg, got {target_deg:.2f} deg"
    )


@pytest.mark.parametrize("open_side", ["left", "right"])
@pytest.mark.parametrize("n", RESOLUTIONS)
def test_side_opening_prefers_arc_toward_it(n, open_side: Side):
    safe_m = safe_arc_lengths(PARAMS, *_scan(_forward_blocked_side_open(open_side), n))
    straight_m = safe_m[TABLE.straight_index]
    best = int(np.argmax(safe_m))
    assert straight_m < 1.5, f"N={n}: expected the wall to block the straight arc"
    assert safe_m[best] > straight_m + 0.5, (
        f"N={n}: expected a turning arc to go at least 0.5 m farther than straight "
        f"({safe_m[best]:.2f} m vs {straight_m:.2f} m)"
    )
    sign = 1.0 if open_side == "left" else -1.0
    assert sign * TABLE.steer_deg[best] > 0, (
        f"N={n}: the best arc turns away from the {open_side} opening "
        f"({TABLE.steer_deg[best]:.2f} deg)"
    )


@pytest.mark.parametrize("n", RESOLUTIONS)
def test_mirrored_scene_mirrors_safe_arc_lengths(n):
    # Law: if the scene is mirrored in y, the safe arc lengths are mirrored
    # over the candidate grid. The tolerance is one arc sample, for rounding
    # at the clearance radius.
    rng = np.random.default_rng(n)
    scenes = [_forward_blocked_side_open("right")] + [
        _random_walls(rng) for _ in range(10)
    ]
    for walls in scenes:
        safe_m = safe_arc_lengths(PARAMS, *_scan(walls, n))
        mirrored_m = safe_arc_lengths(PARAMS, *_scan(_mirror(walls), n))
        np.testing.assert_allclose(
            mirrored_m, safe_m[::-1], atol=PARAMS.arc_sample_ds_m
        )


@pytest.mark.parametrize("n", RESOLUTIONS)
def test_mirrored_scene_mirrors_steer(n):
    # Law: from the start state, a mirrored scene gives the same result case
    # and a steer command of the opposite sign.
    rng = np.random.default_rng(1000 + n)
    for _ in range(10):
        walls = _random_walls(rng)
        _, result = step(PARAMS, VsState(), *_scan(walls, n), DT)
        _, mirrored = step(PARAMS, VsState(), *_scan(_mirror(walls), n), DT)
        assert type(mirrored) is type(result), f"walls={walls}"
        assert mirrored.command.steer == pytest.approx(
            -result.command.steer, abs=1e-3
        ), f"walls={walls}"


@pytest.mark.parametrize("n", RESOLUTIONS)
def test_negotiable_corner_finds_feasible_arc(n):
    # A wall 0.8 m ahead with the right side open. The straight arc is
    # blocked, but an arc of moderate curvature reaches the full horizon.
    safe_m = safe_arc_lengths(
        PARAMS, *_scan(_forward_blocked_side_open("right", wall_x_m=0.8), n)
    )
    assert safe_m.max() > PARAMS.planning_horizon_m - 0.1, (
        f"N={n}: expected an arc within 0.1 m of the horizon, got {safe_m.max():.2f} m"
    )


@pytest.mark.parametrize("n", RESOLUTIONS)
def test_inside_min_turn_radius_no_arc_escapes(n):
    # A corner 0.3 m ahead is much closer than the minimum turn radius
    # (about 0.68 m). No arc, not even at full lock, can turn away in time.
    walls = [
        ((0.3, -2.0), (0.3, 0.05)),  # wall almost straight ahead
        ((0.3, 0.05), (0.5, 2.0)),  # corner to the left
    ]
    safe_m = safe_arc_lengths(PARAMS, *_scan(walls, n))
    assert safe_m.max() < 1.0, (
        f"N={n}: expected every safe arc length below 1.0 m, got {safe_m.max():.2f} m"
    )


@pytest.mark.parametrize("n", RESOLUTIONS)
def test_side_wall_only_keeps_straight_clear(n):
    safe_m = safe_arc_lengths(PARAMS, *_scan([((-2.0, 0.5), (12.0, 0.5))], n))
    assert safe_m[TABLE.straight_index] >= PARAMS.planning_horizon_m - 1e-6, (
        f"N={n}: a side wall shortened the straight arc to "
        f"{safe_m[TABLE.straight_index]:.2f} m"
    )


@pytest.mark.parametrize("n", RESOLUTIONS)
def test_narrow_corridor_has_no_region(n):
    # The corridor half-width of 0.15 m is less than the clearance radius.
    safe_m = safe_arc_lengths(PARAMS, *_scan(corridor_walls(half_width_m=0.15), n))
    assert extract_regions(PARAMS, safe_m) == (), (
        f"N={n}: expected no usable arc in a corridor narrower than the car"
    )


def test_target_changes_little_with_resolution():
    walls = _forward_blocked_side_open("right", wall_x_m=1.5)
    targets_deg = [_best_center_deg(*_scan(walls, n)) for n in RESOLUTIONS]
    spread_deg = max(targets_deg) - min(targets_deg)
    assert spread_deg <= 3.0, (
        f"the target changed by {spread_deg:.2f} deg over N={RESOLUTIONS}: {targets_deg}"
    )


def test_dense_side_walls_do_not_shorten_straight_arc():
    straight_m = [
        safe_arc_lengths(PARAMS, *_scan(corridor_walls(half_width_m=0.4), n))[
            TABLE.straight_index
        ]
        for n in RESOLUTIONS
    ]
    assert max(straight_m) - min(straight_m) < 1e-6, (
        f"the straight safe arc length (m) changed with N: {straight_m}"
    )
    assert straight_m[0] >= PARAMS.planning_horizon_m - 1e-6


@pytest.mark.parametrize("n", [270, 1080])
def test_single_outlier_changes_target_at_most_one_slew_step(n):
    angles, ranges = _scan(corridor_walls(half_width_m=0.5), n)
    ranges = ranges.copy()
    ranges[len(ranges) // 2] = 0.08  # m; one false close return straight ahead
    state, first = step(PARAMS, VsState(), angles, ranges, DT)
    _, second = step(PARAMS, state, angles, ranges, DT)
    max_step_deg = PARAMS.max_slew_deg_per_s * DT
    assert abs(second.target_steer_deg - first.target_steer_deg) <= max_step_deg + 1e-6


def test_no_path_keeps_state_and_recovers_toward_closer_option():
    lidar = DenseLidarParams(n_beams=270)
    state = VsState()
    angles, ranges = gap_ring_scan(
        lidar, gap_centers_deg=[-8.0], gap_half_width_deg=5.0
    )
    for _ in range(15):
        state, result = step(PARAMS, state, angles, ranges, DT)
    prev_target_deg = result.target_steer_deg

    # A barrier 0.03 m away in every direction blocks every arc.
    blocked = gap_ring_scan(
        lidar, gap_centers_deg=[], gap_half_width_deg=0.0, barrier_range_m=0.03
    )
    for _ in range(5):
        new_state, result = step(PARAMS, state, *blocked, DT)
        assert isinstance(result, NoPath)
        assert result.command.throttle == 0.0
        assert new_state == state, "a NoPath tick must not change the state"

    reopened = gap_ring_scan(
        lidar, gap_centers_deg=[-9.0, 18.0], gap_half_width_deg=4.0
    )
    _, result = step(PARAMS, state, *reopened, DT)
    assert isinstance(result, Driving)
    assert abs(result.target_steer_deg - (-9.0)) < abs(
        result.target_steer_deg - 18.0
    ), (
        f"expected a target nearer to -9 deg (the last target was "
        f"{prev_target_deg:.1f} deg), got {result.target_steer_deg:.1f} deg"
    )
