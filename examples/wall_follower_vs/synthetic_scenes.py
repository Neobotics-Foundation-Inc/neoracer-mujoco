"""
Synthetic LiDAR scans for the tests of vehicle_space_controller.py. The scans
need no MuJoCo and no random numbers.

A scene is a list of 2D wall segments in the car frame
(+X forward, +Y left, origin at the car center). cast_scan() casts the beams
of a DenseLidarParams against the segments from the LiDAR's position. The
same scene thus gives a scan at any beam count N.
"""

import math

import numpy as np

from examples.wall_follower_vs.dense_lidar import DenseLidarParams, Scan, scan_angles

type Point = tuple[float, float]  # m, car frame
type Segment = tuple[Point, Point]


def _seg_intersect(a1: Point, a2: Point, b1: Point, b2: Point) -> float | None:
    """The parameter t in [0, 1] along a1 -> a2 where that segment crosses
    b1 -> b2, or None if the segments do not cross."""
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


def cast_scan(walls: list[Segment], lidar: DenseLidarParams, origin: Point) -> Scan:
    """Cast the beams of lidar against walls from origin (m, the LiDAR's
    position in the car frame). Returns (angles_rad, ranges_m), ranges from
    origin. A range is -1.0 if the beam hits no wall within max_range_m."""
    angles = scan_angles(lidar)
    ranges = np.full(lidar.n_beams, -1.0)
    for i, theta in enumerate(angles):
        far = (
            origin[0] + math.cos(theta) * lidar.max_range_m,
            origin[1] + math.sin(theta) * lidar.max_range_m,
        )
        crossings = [_seg_intersect(origin, far, p1, p2) for p1, p2 in walls]
        hits = [t for t in crossings if t is not None]
        if hits:
            ranges[i] = min(hits) * lidar.max_range_m
    return angles, ranges


def corridor_walls(half_width_m: float = 0.5) -> list[Segment]:
    """Two parallel walls from x = -2 m to x = 12 m, at y = +half_width_m and
    y = -half_width_m."""
    return [
        ((-2.0, half_width_m), (12.0, half_width_m)),
        ((-2.0, -half_width_m), (12.0, -half_width_m)),
    ]


def gap_ring_scan(
    lidar: DenseLidarParams,
    gap_centers_deg: list[float],
    gap_half_width_deg: float,
    barrier_range_m: float = 1.5,
) -> Scan:
    """A scan without ray casting. Each beam reads barrier_range_m, except
    the beams within gap_half_width_deg of a gap center. Those beams have no
    return. The test thus puts each opening at an exact steering angle.

    A barrier point at range r blocks only the arcs that pass within the
    clearance radius of it. If r is small, the point blocks nearly every arc
    and closes the gaps. The default of 1.5 m keeps the gaps open. Use a
    small barrier_range_m only for a scan with no usable path.
    """
    angles = scan_angles(lidar)
    ranges = np.full(lidar.n_beams, barrier_range_m)
    deg = np.degrees(angles)
    for center in gap_centers_deg:
        ranges[np.abs(deg - center) <= gap_half_width_deg] = -1.0
    return angles, ranges
