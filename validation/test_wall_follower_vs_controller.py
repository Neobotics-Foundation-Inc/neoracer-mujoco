"""
Tests of the vehicle-space wall follower (examples/wall_follower_vs/) on
synthetic scans. The tests use no physics and no unseeded random numbers.

Frame: +X forward, +Y left. A steering angle is + = left.
"""

import dataclasses
from copy import replace
from typing import assert_never

import numpy as np
import pytest

from examples.wall_follower_vs.dense_lidar import DenseLidarParams, Scan, scan_angles
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
    _blocked_samples,
    _median_filter,
    arc_table,
    car_geometry,
    extract_regions,
    obstacle_points,
    safe_arc_lengths,
    score_region,
    step,
)
from neoracer_mujoco import assets
from neoracer_mujoco.control.wall_following import Side

PARAMS = VsParams(car=car_geometry(assets.load()))
TABLE = arc_table(PARAMS)
DT = 1.0 / 30.0
"""One tick at 30 Hz (s)."""
RESOLUTIONS = [24, 270, 540, 1080]


def _scan(walls: list[Segment], n: int) -> Scan:
    return cast_scan(walls, DenseLidarParams(n_beams=n), PARAMS.car.lidar_xy_m)


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


def test_a_straight_corridor_keeps_the_straight_arc():
    """At every resolution, the target in a straight corridor is straight
    ahead, and the side walls leave the straight arc clear to the horizon,
    the same at every resolution."""
    straight_m = []
    for n in RESOLUTIONS:
        target_deg = _best_center_deg(*_scan(corridor_walls(), n))
        assert abs(target_deg) < 2.0, f"N={n}: target {target_deg:.2f} deg"
        safe_m = safe_arc_lengths(PARAMS, *_scan(corridor_walls(half_width_m=0.4), n))
        straight_m.append(safe_m[TABLE.straight_index])
    assert max(straight_m) - min(straight_m) < 1e-6, straight_m
    assert straight_m[0] >= PARAMS.planning_horizon_m - 1e-6


def test_a_blocked_road_with_one_side_open_turns_into_the_opening():
    """With a wall 0.8 m ahead and one side open, at every resolution, the
    straight arc is blocked, and the longest arc turns toward the opening
    and reaches within 0.1 m of the horizon."""
    for n in RESOLUTIONS:
        for open_side in ("left", "right"):
            case = f"N={n}, {open_side} open"
            walls = _forward_blocked_side_open(open_side, wall_x_m=0.8)
            safe_m = safe_arc_lengths(PARAMS, *_scan(walls, n))
            best = int(np.argmax(safe_m))
            assert safe_m[TABLE.straight_index] < 1.5, case
            assert safe_m[best] > PARAMS.planning_horizon_m - 0.1, case
            sign = 1.0 if open_side == "left" else -1.0
            assert sign * TABLE.steer_deg[best] > 0, case


def test_mirrored_scene_mirrors_safe_arc_lengths():
    """If the scene is mirrored in y, the safe arc lengths are mirrored over
    the candidate grid. The tolerance is one arc sample, for rounding at the
    clearance radius."""
    for n in RESOLUTIONS:
        rng = np.random.default_rng(n)
        scenes = [_forward_blocked_side_open("right")] + [
            _random_walls(rng) for _ in range(10)
        ]
        for walls in scenes:
            safe_m = safe_arc_lengths(PARAMS, *_scan(walls, n))
            mirrored_m = safe_arc_lengths(PARAMS, *_scan(_mirror(walls), n))
            np.testing.assert_allclose(
                mirrored_m,
                safe_m[::-1],
                atol=PARAMS.arc_sample_ds_m,
                err_msg=f"N={n}, walls={walls}",
            )


def test_mirrored_scene_mirrors_steer():
    """From the start state, a mirrored scene gives the same result case and a
    steer command of the opposite sign."""
    for n in RESOLUTIONS:
        rng = np.random.default_rng(1000 + n)
        for _ in range(10):
            walls = _random_walls(rng)
            _, result = step(PARAMS, VsState(), *_scan(walls, n), DT)
            _, mirrored = step(PARAMS, VsState(), *_scan(_mirror(walls), n), DT)
            assert type(mirrored) is type(result), f"N={n}, walls={walls}"
            assert mirrored.command.steer == pytest.approx(
                -result.command.steer, abs=1e-3
            ), f"N={n}, walls={walls}"


def test_inside_min_turn_radius_no_arc_escapes():
    """A corner 0.3 m ahead is much closer than the minimum turn radius (about
    0.68 m). No arc, not even at full lock, can turn away in time."""
    walls = [
        ((0.3, -2.0), (0.3, 0.05)),  # wall almost straight ahead
        ((0.3, 0.05), (0.5, 2.0)),  # corner to the left
    ]
    for n in RESOLUTIONS:
        safe_m = safe_arc_lengths(PARAMS, *_scan(walls, n))
        assert safe_m.max() < 1.0, f"N={n}: longest safe arc {safe_m.max():.2f} m"


def test_narrow_corridor_has_no_region():
    """The corridor half-width of 0.15 m is less than the clearance radius."""
    for n in RESOLUTIONS:
        safe_m = safe_arc_lengths(PARAMS, *_scan(corridor_walls(half_width_m=0.15), n))
        assert extract_regions(PARAMS, safe_m) == (), f"N={n}"


def test_target_changes_little_with_resolution():
    walls = _forward_blocked_side_open("right", wall_x_m=1.5)
    targets_deg = [_best_center_deg(*_scan(walls, n)) for n in RESOLUTIONS]
    spread_deg = max(targets_deg) - min(targets_deg)
    assert spread_deg <= 3.0, (
        f"the target changed by {spread_deg:.2f} deg over N={RESOLUTIONS}: {targets_deg}"
    )


def test_single_outlier_changes_target_at_most_one_slew_step():
    max_step_deg = PARAMS.max_slew_deg_per_s * DT
    for n in (270, 1080):
        angles, ranges = _scan(corridor_walls(half_width_m=0.5), n)
        ranges = ranges.copy()
        ranges[len(ranges) // 2] = 0.08  # m; one false close return straight ahead
        state, first = step(PARAMS, VsState(), angles, ranges, DT)
        _, second = step(PARAMS, state, angles, ranges, DT)
        change = abs(second.target_steer_deg - first.target_steer_deg)
        assert change <= max_step_deg + 1e-6, f"N={n}"


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
        assert result.command.target == 0.0
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


def test_params_reject_a_grid_without_a_straight_path_or_horizon():
    for bad in [{"n_candidates": 80}, {"n_candidates": 1}, {"planning_horizon_m": 0.0}]:
        with pytest.raises(ValueError):
            dataclasses.replace(PARAMS, **bad)


def test_arc_spacing_is_at_most_arc_sample_ds():
    """The arc samples are never farther apart than arc_sample_ds_m."""
    for horizon, ds in [(3.0, 0.1), (0.24, 0.1), (1.0, 0.07)]:
        params = replace(PARAMS, planning_horizon_m=horizon, arc_sample_ds_m=ds)
        s = arc_table(params).s_m
        assert np.max(np.diff(s)) <= ds + 1e-12
        assert s[-1] == pytest.approx(horizon)


def test_a_single_beam_points_forward():
    assert scan_angles(DenseLidarParams(n_beams=1)).tolist() == [0.0]


def _random_scan(rng: np.random.Generator, n: int) -> Scan:
    """n beams at random ascending angles over 270 deg, with random ranges.
    Some ranges are no return (-1, 0, inf) or not a number."""
    angles = np.sort(rng.uniform(-np.radians(135.0), np.radians(135.0), n))
    ranges = rng.uniform(0.05, 4.0, n)
    bad = rng.random(n)
    ranges[bad < 0.1] = -1.0
    ranges[(bad >= 0.1) & (bad < 0.12)] = 0.0
    ranges[(bad >= 0.12) & (bad < 0.13)] = np.inf
    ranges[(bad >= 0.13) & (bad < 0.14)] = np.nan
    return angles, ranges


def test_median_filter_equals_median_of_each_window():
    """The filter gives, bit for bit, the np.median of the ranges in the
    angular window of each beam. A beam alone in its window keeps its range."""
    rng = np.random.default_rng(7)
    window_rad = np.radians(PARAMS.outlier_window_deg)
    for n in [1, 2, 24, 270, 1080]:
        angles, ranges = _random_scan(rng, n)
        expected = ranges.copy()
        for i, theta in enumerate(angles):
            in_window = np.abs(angles - theta) <= window_rad
            if in_window.sum() > 1:
                expected[i] = np.median(ranges[in_window])
        np.testing.assert_array_equal(
            _median_filter(PARAMS, angles, ranges), expected, strict=True
        )


def test_obstacle_points_are_in_the_car_frame_wherever_the_lidar_is():
    """The scan's ranges are from the LiDAR, but a wall across the path 0.4 m
    ahead of the car center gives points at x = 0.4 m, wherever the LiDAR is
    mounted. The median filter's window holds one beam here, so each point is
    its own beam's."""
    wall = [((0.4, -0.3), (0.4, 0.3))]
    for lidar_xy_m in [(0.0, 0.0), (-0.0442, 0.0), (0.1, -0.05)]:
        car = replace(PARAMS.car, lidar_xy_m=lidar_xy_m)
        params = replace(PARAMS, car=car, outlier_window_deg=0.1)
        angles, ranges = cast_scan(wall, DenseLidarParams(n_beams=360), lidar_xy_m)
        x, y = obstacle_points(params, angles, ranges)
        assert len(x) > 10, lidar_xy_m
        np.testing.assert_allclose(x, 0.4, atol=1e-9, err_msg=str(lidar_xy_m))
        assert np.all(np.abs(y) <= 0.3 + 1e-9), lidar_xy_m


def _all_pairs_safe_arc_lengths(angles: np.ndarray, ranges: np.ndarray) -> np.ndarray:
    """safe_arc_lengths() by a direct test of every (sample, point) pair."""
    x, y = obstacle_points(PARAMS, angles, ranges)
    dx = TABLE.x_m[:, :, None] - x[None, None, :]  # (candidates, samples, points)
    dy = TABLE.y_m[:, :, None] - y[None, None, :]
    clearance2 = PARAMS.clearance_radius_m * PARAMS.clearance_radius_m
    blocked = ((dx * dx + dy * dy) <= clearance2).any(axis=2) & TABLE.valid
    return np.array(
        [
            TABLE.s_m[np.argmax(row)] if row.any() else cap
            for row, cap in zip(blocked, TABLE.cap_m, strict=True)
        ]
    )


def test_safe_arc_lengths_equal_a_test_of_all_pairs():
    """The k-d tree search gives, bit for bit, the safe arc lengths of a
    direct test of every pair of arc sample and obstacle point."""
    rng = np.random.default_rng(11)
    for n in [24, 90, 270]:
        for _ in range(5):
            angles, ranges = _random_scan(rng, n)
            np.testing.assert_array_equal(
                safe_arc_lengths(PARAMS, angles, ranges),
                _all_pairs_safe_arc_lengths(angles, ranges),
                strict=True,
            )
        walls = _random_walls(rng)
        np.testing.assert_array_equal(
            safe_arc_lengths(PARAMS, *_scan(walls, n)),
            _all_pairs_safe_arc_lengths(*_scan(walls, n)),
            strict=True,
        )


def test_blocked_samples_include_points_at_the_clearance_radius():
    """For points at the clearance radius from a sample, to the last bit, the
    k-d tree search gives the same blocked samples as a test of all pairs."""
    rng = np.random.default_rng(3)
    rows = rng.choice(len(TABLE.valid_flat), 200)
    flat = TABLE.valid_flat[rows]
    heading = rng.uniform(0.0, 2.0 * np.pi, len(flat))
    radius_m = PARAMS.clearance_radius_m * (
        1.0 + rng.integers(-3, 4, len(flat)) * 1e-16
    )
    x = TABLE.x_m.ravel()[flat] + radius_m * np.cos(heading)
    y = TABLE.y_m.ravel()[flat] + radius_m * np.sin(heading)

    dx = TABLE.x_m.ravel()[TABLE.valid_flat][:, None] - x[None, :]
    dy = TABLE.y_m.ravel()[TABLE.valid_flat][:, None] - y[None, :]
    clearance2 = PARAMS.clearance_radius_m * PARAMS.clearance_radius_m
    expected = TABLE.valid_flat[((dx * dx + dy * dy) <= clearance2).any(axis=1)]
    assert np.array_equal(np.unique(_blocked_samples(PARAMS, TABLE, x, y)), expected)
