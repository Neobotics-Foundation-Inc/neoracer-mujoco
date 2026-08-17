"""
Deterministic synthetic LiDAR scan generation for testing
vehicle_space_controller.py without MuJoCo (Step 12 of the design spec).

A "scene" is a list of 2D wall segments in the CURRENT vehicle frame
(+X forward, +Y left, origin = car center). cast_scan() raycasts the real
270 deg FOV against those segments to produce a physical (angles_rad,
ranges_m) LiDAR scan at any sample count N -- the same geometry, sampled at
different densities, exactly like a real sensor's resolution changing.
"""

from __future__ import annotations

import math

import numpy as np

FOV_DEG = 270.0
MAX_RANGE_M = 12.0


def make_angles(n: int, fov_deg: float = FOV_DEG) -> np.ndarray:
    """N physical sample angles, ascending, symmetric about straight ahead
    (0 rad), spanning the real FOV -- e.g. N=1080 at fov=270 reproduces the
    Lakibeam L1's 0.25 deg resolution exactly."""
    half = math.radians(fov_deg) / 2.0
    return np.linspace(-half, half, n)


def _seg_intersect(a1, a2, b1, b2):
    """Standard segment-segment intersection; returns the ray-parametric t
    in [0,1] along a1->a2 where it crosses b1->b2, or None."""
    rx, ry = a2[0] - a1[0], a2[1] - a1[1]
    sx, sy = b2[0] - b1[0], b2[1] - b1[1]
    rxs = rx * sy - ry * sx
    if abs(rxs) < 1e-12:
        return None
    qpx, qpy = b1[0] - a1[0], b1[1] - a1[1]
    t = (qpx * sy - qpy * sx) / rxs
    u = (qpx * ry - qpy * rx) / rxs
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return t
    return None


def cast_scan(
    walls: list[tuple[tuple[float, float], tuple[float, float]]],
    n: int,
    fov_deg: float = FOV_DEG,
    max_range: float = MAX_RANGE_M,
) -> tuple[np.ndarray, np.ndarray]:
    """Raycast N physical rays across fov_deg against `walls` (vehicle-frame
    segments). Returns (angles_rad, ranges_m); ranges_m is -1 where nothing
    was hit within max_range (no-return, matching the real sensor and
    lidar_to_points()'s handling of invalid samples)."""
    angles = make_angles(n, fov_deg)
    ranges = np.full(n, -1.0)
    origin = (0.0, 0.0)
    for i, theta in enumerate(angles):
        far = (math.cos(theta) * max_range, math.sin(theta) * max_range)
        best_t = None
        for p1, p2 in walls:
            t = _seg_intersect(origin, far, p1, p2)
            if t is not None and (best_t is None or t < best_t):
                best_t = t
        if best_t is not None:
            ranges[i] = best_t * max_range
    return angles, ranges


def corridor_walls(
    half_width_m: float = 0.5, x_start: float = -2.0, x_end: float = 12.0
) -> list:
    """Two parallel walls, symmetric about y=0 (car centered)."""
    return [
        ((x_start, half_width_m), (x_end, half_width_m)),
        ((x_start, -half_width_m), (x_end, -half_width_m)),
    ]


def offset_corridor_walls(
    left_dist_m: float, right_dist_m: float, x_start: float = -2.0, x_end: float = 12.0
) -> list:
    """Corridor walls at asymmetric distances -- simulates the car sitting
    off-center (left wall at +left_dist_m, right wall at -right_dist_m)."""
    return [
        ((x_start, left_dist_m), (x_end, left_dist_m)),
        ((x_start, -right_dist_m), (x_end, -right_dist_m)),
    ]


def wall_ahead(x_m: float = 1.0, half_span_m: float = 2.0) -> list:
    return [((x_m, -half_span_m), (x_m, half_span_m))]


def opening_walls(side: str, half_width_m: float = 0.5, open_at_x: float = 2.0) -> list:
    """Corridor that continues on one side but has a wall ending (opening)
    on `side` ("left" or "right") at open_at_x."""
    assert side in ("left", "right")
    y_open = half_width_m if side == "left" else -half_width_m
    y_solid = -half_width_m if side == "left" else half_width_m
    return [
        ((-2.0, y_open), (open_at_x, y_open)),  # ends at the opening
        ((-2.0, y_solid), (12.0, y_solid)),  # continues straight
    ]


def isolated_obstacle(
    x_m: float = 1.5,
    y_m: float = 0.0,
    half_size_m: float = 0.08,
    corridor_half_width_m: float = 0.5,
) -> list:
    """A small square pole, approximated as 4 short wall segments, plus
    corridor walls so there's a defined free space around it."""
    s = half_size_m
    box = [
        ((x_m - s, y_m - s), (x_m + s, y_m - s)),
        ((x_m + s, y_m - s), (x_m + s, y_m + s)),
        ((x_m + s, y_m + s), (x_m - s, y_m + s)),
        ((x_m - s, y_m + s), (x_m - s, y_m - s)),
    ]
    return box + corridor_walls(half_width_m=corridor_half_width_m)


def narrow_unsafe_corridor(half_width_m: float = 0.15) -> list:
    """Narrower than 2*CLEARANCE_RADIUS_M (0.21m) -- no candidate should
    ever be reported safe here."""
    return corridor_walls(half_width_m=half_width_m)


def gap_ring_scan(
    n: int,
    gap_centers_deg: list[float],
    gap_half_width_deg: float = 15.0,
    barrier_range_m: float = 1.5,
    fov_deg: float = FOV_DEG,
) -> tuple:
    """Procedural scan (bypasses raycasting): a barrier at barrier_range_m in
    every direction EXCEPT narrow angular gaps centered on gap_centers_deg,
    where the reading is a no-return (open beyond the planning horizon).
    Used to give exact, direct control over where safe openings sit in the
    candidate grid -- e.g. reproducing the design spec's own no-path/
    recovery example (openings at -12 deg and -70 deg) without depending on
    a raycasting engine to happen to produce that exact profile.

    barrier_range_m defaults to 1.5m, not something tiny: a barrier point at
    range r only threatens candidate headings within ~asin(clearance/r) of
    its OWN angle (see vehicle_space_controller.safe_travel_distances). A
    barrier placed too close to the origin threatens nearly every heading
    regardless of angle, which would falsely swallow up any nearby gap --
    call with a small barrier_range_m explicitly only when you want a
    genuinely fully-blocked (no gap survives) scan."""
    angles = make_angles(n, fov_deg)
    ranges = np.full(n, barrier_range_m)
    deg = np.degrees(angles)
    for center in gap_centers_deg:
        in_gap = np.abs(deg - center) <= gap_half_width_deg
        ranges[in_gap] = -1.0
    return angles, ranges
