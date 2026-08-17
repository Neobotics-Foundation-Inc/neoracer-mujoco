"""
Synthetic kinematic-trajectory tests for the vehicle-space wall follower
(kinematic-trajectory revision, loop_corridor.xml Stage A/B follow-up).
Pure logic + deterministic raycasting -- no MuJoCo, no randomness.

Replaces the earlier straight-heading-candidate test suite: candidates now
represent physical STEERING ANGLES (Ackermann arcs), not instantaneous
headings, so the old assertions (built against a straight-ray/tube model)
no longer describe what the controller does. See
examples/wall_follower_vs/vehicle_space_controller.py's module docstring
for the full architecture change and why it was made.

Experimental -- NOT part of PR #27, does not touch racecar_core_adapter.py,
assets/neoracer.xml, or examples/ultimate_wall_follower/.
"""

import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "examples", "wall_follower_vs"
    ),
)

import synthetic_scenes as scenes  # noqa: E402
import vehicle_space_controller as vsc  # noqa: E402

RESOLUTIONS = [24, 270, 540, 1080]
STRAIGHT_IDX = int(np.argmin(np.abs(vsc.CANDIDATE_HEADINGS_DEG)))


def _raw_target_deg(angles_rad, ranges_m):
    """Selected steering angle with NO temporal/hysteresis state -- the raw
    geometric answer for one isolated scan."""
    safe_dist = vsc.safe_arc_lengths(angles_rad, ranges_m)
    regions = vsc.extract_regions(safe_dist)
    assert regions, "expected at least one feasible arc"
    best = max(regions, key=lambda r: r["mean_safe_m"] + 0.05 * r["n_candidates"])
    return vsc._region_center_deg(
        best, safe_dist, vsc.CANDIDATE_HEADINGS_DEG
    ), safe_dist


# --- A: straight corridor -> straight trajectory wins -----------------------


@pytest.mark.parametrize("n", RESOLUTIONS)
def test_a_straight_corridor_prefers_straight(n):
    angles, ranges = scenes.cast_scan(scenes.corridor_walls(), n)
    target, _ = _raw_target_deg(angles, ranges)
    assert abs(target) < 2.0, f"N={n}: expected ~0deg steering, got {target:.2f}deg"


# --- B/C: wall ahead with an opening to one side -----------------------------


def _forward_blocked_side_open(open_side: str, wall_x: float = 1.0) -> list:
    """A wall spanning dead-ahead and the CLOSED side, leaving open_side
    clear -- e.g. open_side="right" blocks y in [-0.1, 2.0] (ahead + left),
    leaving y < -0.1 (right) open for a curving arc to sweep into."""
    assert open_side in ("left", "right")
    if open_side == "right":
        return [((wall_x, -0.1), (wall_x, 2.0))]
    return [((wall_x, 0.1), (wall_x, -2.0))]


@pytest.mark.parametrize("n", RESOLUTIONS)
def test_b_right_opening_prefers_right_arc_before_critical(n):
    angles, ranges = scenes.cast_scan(_forward_blocked_side_open("right"), n)
    safe_dist = vsc.safe_arc_lengths(angles, ranges)
    straight_clear = safe_dist[STRAIGHT_IDX]
    best_idx = int(np.argmax(safe_dist))
    best_clear = safe_dist[best_idx]
    assert straight_clear < 1.5, f"N={n}: expected straight to be blocked near the wall"
    assert best_clear > straight_clear + 0.5, (
        f"N={n}: expected a curving arc to clear substantially farther than "
        f"straight ({best_clear:.2f}m vs {straight_clear:.2f}m)"
    )
    # right turn = negative steering degrees in this module's convention
    # (+ = left, see lidar_to_points()'s docstring).
    assert vsc.CANDIDATE_HEADINGS_DEG[best_idx] < 0, (
        f"N={n}: best arc should curve right (negative), got "
        f"{vsc.CANDIDATE_HEADINGS_DEG[best_idx]:.2f}deg"
    )


@pytest.mark.parametrize("n", RESOLUTIONS)
def test_c_left_opening_prefers_left_arc_before_critical(n):
    angles, ranges = scenes.cast_scan(_forward_blocked_side_open("left"), n)
    safe_dist = vsc.safe_arc_lengths(angles, ranges)
    straight_clear = safe_dist[STRAIGHT_IDX]
    best_idx = int(np.argmax(safe_dist))
    best_clear = safe_dist[best_idx]
    assert straight_clear < 1.5
    assert best_clear > straight_clear + 0.5
    assert vsc.CANDIDATE_HEADINGS_DEG[best_idx] > 0, (
        f"N={n}: best arc should curve left (positive), got "
        f"{vsc.CANDIDATE_HEADINGS_DEG[best_idx]:.2f}deg"
    )


# --- D: corner where a turn is physically possible ---------------------------


@pytest.mark.parametrize("n", RESOLUTIONS)
def test_d_negotiable_corner_finds_feasible_arc(n):
    # Forward wall at 0.8m with the whole right side open: farther than
    # test E's 0.3m (physically inside the minimum turn radius, see below)
    # but close enough that straight is clearly blocked. A moderate-
    # curvature arc (not necessarily max lock) should sweep clear through
    # to the full horizon.
    angles, ranges = scenes.cast_scan(
        _forward_blocked_side_open("right", wall_x=0.8), n
    )
    safe_dist = vsc.safe_arc_lengths(angles, ranges)
    assert safe_dist.max() > vsc.PLANNING_HORIZON_M - 0.1, (
        f"N={n}: expected a feasible arc reaching the full horizon through "
        f"a negotiable corner, best={safe_dist.max():.2f}m"
    )


# --- E: already inside the minimum turning radius ----------------------------


@pytest.mark.parametrize("n", RESOLUTIONS)
def test_e_inside_min_turn_radius_no_arc_escapes(n):
    # Replicates the traced loop_corridor.xml wedge geometry: a wall corner
    # much tighter than MIN_TURN_RADIUS_M (~0.68m), positioned close enough
    # that no candidate curvature -- including max lock -- can complete the
    # turn within the clearance radius. No candidate should reach anywhere
    # near the full planning horizon.
    walls = [
        ((0.3, -2.0), (0.3, 0.05)),  # wall almost dead ahead, very close
        ((0.3, 0.05), (0.5, 2.0)),  # sharp corner immediately to the left
    ]
    angles, ranges = scenes.cast_scan(walls, n)
    safe_dist = vsc.safe_arc_lengths(angles, ranges)
    assert safe_dist.max() < 1.0, (
        f"N={n}: expected no arc to escape a sub-min-turn-radius corner, "
        f"best={safe_dist.max():.2f}m"
    )


# --- F: side wall only -> straight remains safe ------------------------------


@pytest.mark.parametrize("n", RESOLUTIONS)
def test_f_side_wall_only_keeps_straight_clear(n):
    walls = [((-2.0, 0.5), (12.0, 0.5))]
    angles, ranges = scenes.cast_scan(walls, n)
    safe_dist = vsc.safe_arc_lengths(angles, ranges)
    assert safe_dist[STRAIGHT_IDX] >= vsc.PLANNING_HORIZON_M - 1e-6, (
        f"N={n}: a side-only wall must not reduce the straight arc's clearance, "
        f"got {safe_dist[STRAIGHT_IDX]:.2f}m"
    )


# --- G: narrow corridor -> impossible-curvature candidates rejected ---------


@pytest.mark.parametrize("n", RESOLUTIONS)
def test_g_narrow_corridor_no_feasible_arc(n):
    angles, ranges = scenes.cast_scan(scenes.narrow_unsafe_corridor(), n)
    safe_dist = vsc.safe_arc_lengths(angles, ranges)
    regions = vsc.extract_regions(safe_dist)
    assert regions == [], (
        f"N={n}: expected no feasible arc in a sub-clearance-width corridor"
    )


# --- H: resolution invariance ------------------------------------------------


def test_h_resolution_invariance_steering_spread():
    walls = _forward_blocked_side_open("right", wall_x=1.5)
    targets = []
    for n in RESOLUTIONS:
        angles, ranges = scenes.cast_scan(walls, n)
        target, _ = _raw_target_deg(angles, ranges)
        targets.append(target)
    spread = max(targets) - min(targets)
    assert spread <= 3.0, (
        f"selected steering-angle spread across N too large: {targets} (spread={spread:.2f}deg)"
    )


# --- I: dense same-wall samples must not alter feasibility ------------------


def test_i_dense_side_walls_do_not_reduce_straight_feasibility():
    vals = []
    for n in RESOLUTIONS:
        angles, ranges = scenes.cast_scan(scenes.corridor_walls(half_width_m=0.4), n)
        safe_dist = vsc.safe_arc_lengths(angles, ranges)
        vals.append(safe_dist[STRAIGHT_IDX])
    assert max(vals) - min(vals) < 1e-6, f"straight-arc clearance varied with N: {vals}"
    assert vals[0] >= vsc.PLANNING_HORIZON_M - 1e-6


# --- J: one LiDAR outlier must not catastrophically change selection -------


@pytest.mark.parametrize("n", [270, 1080])
def test_j_single_outlier_no_catastrophic_switch(n):
    angles, ranges = scenes.cast_scan(scenes.corridor_walls(half_width_m=0.5), n)
    mid = len(ranges) // 2
    ranges = ranges.copy()
    ranges[mid] = 0.08  # spurious close spike
    ctrl = vsc.VehicleSpaceController()
    dt = 1.0 / vsc.CONTROL_HZ
    first = ctrl.compute(angles, ranges, dt)
    second = ctrl.compute(angles, ranges, dt)
    max_step = vsc.MAX_SLEW_DEG_PER_S * dt
    assert abs(second.target_heading_deg - first.target_heading_deg) <= max_step + 1e-6


# --- no-path / recovery (state machine, unchanged logic) --------------------


def test_no_path_then_recovery_prefers_closer_option():
    # Gap centers rescaled into the new steering-angle range (+/-~22.9deg,
    # was +/-80deg for the old heading grid) -- same "close vs far option"
    # shape as the original design-spec example, just within the physically
    # achievable steering range.
    n = 270
    dt = 1.0 / vsc.CONTROL_HZ
    ctrl = vsc.VehicleSpaceController()

    angles, ranges = scenes.gap_ring_scan(
        n, gap_centers_deg=[-8.0], gap_half_width_deg=5.0
    )
    for _ in range(15):
        result = ctrl.compute(angles, ranges, dt)
    prev_target = result.target_heading_deg

    blocked_angles, blocked_ranges = scenes.gap_ring_scan(
        n, gap_centers_deg=[], barrier_range_m=0.03
    )
    for _ in range(5):
        result = ctrl.compute(blocked_angles, blocked_ranges, dt)
        assert result.state == "NO_PATH"
        assert result.speed == 0.0

    reopen_angles, reopen_ranges = scenes.gap_ring_scan(
        n, gap_centers_deg=[-9.0, 18.0], gap_half_width_deg=4.0
    )
    result = ctrl.compute(reopen_angles, reopen_ranges, dt)
    assert abs(result.target_heading_deg - (-9.0)) < abs(
        result.target_heading_deg - 18.0
    ), (
        f"expected recovery toward -9deg (closer to prev {prev_target:.1f}), "
        f"got {result.target_heading_deg:.1f}"
    )


# --- performance --------------------------------------------------------------


def test_performance_n1080_under_hard_budget():
    angles, ranges = scenes.cast_scan(_forward_blocked_side_open("right"), 1080)
    ctrl = vsc.VehicleSpaceController()
    dt = 1.0 / vsc.CONTROL_HZ
    ctrl.compute(angles, ranges, dt)  # warm up
    n_iters = 50
    t0 = time.perf_counter()
    for _ in range(n_iters):
        ctrl.compute(angles, ranges, dt)
    elapsed = time.perf_counter() - t0
    mean_ms = (elapsed / n_iters) * 1000.0
    print(f"\nN=1080 mean tick: {mean_ms:.3f} ms (hard budget 33.3ms, preferred <10ms)")
    assert mean_ms < 33.3, f"exceeded the 30Hz hard budget: {mean_ms:.3f}ms"
