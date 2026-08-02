"""Step 1: draw the ring of random control points a track is spline'd through.

  draw_control_points  the stage entry point: returns (n, 2) points in loop order
  spread_out_angles    helper, keeps two points from landing at the same angle
  smooth_around_loop   helper, periodic blur used on both radii and points

Polar, not cartesian: each point is an angle plus a radius, so sorting by angle
puts them in loop order for free and the loop always encloses the origin. The
only randomness in a track's shape is here - every later stage is deterministic
given these points.
"""

import numpy as np

from ..settings import TrackSettings


def spread_out_angles(angles, min_gap):
    """Walk a sorted list of angles and push each one forward until it is at
    least `min_gap` past the previous. Two control points at nearly the same
    angle would otherwise sit almost on top of each other and force the loop
    into a hairpin. No randomness."""
    angles = np.sort(angles).copy()
    for i in range(1, len(angles)):
        if angles[i] - angles[i - 1] < min_gap:
            angles[i] = angles[i - 1] + min_gap
    return angles


def smooth_around_loop(values, rounds):
    """Blur a periodic sequence: each entry becomes half itself plus a quarter of
    each neighbor, wrapping at the ends. `rounds` passes. This is a low-pass
    filter, so it removes the sharp jumps that turn into tight corners.

    Works on a 1D sequence of numbers or an (n, 2) sequence of points alike.
    """
    for _ in range(rounds):
        values = 0.5 * values + 0.25 * (np.roll(values, 1, 0) + np.roll(values, -1, 0))
    return values


def draw_control_points(random, settings: TrackSettings):
    """Scatter the (n, 2) control points a track is drawn through.

    Each point starts at a random angle around the origin and a random distance
    out from it, which guarantees they come out in loop order (sorted by angle)
    and that the loop encloses the origin. Then:
      - the radii get smoothed, so neighboring points sit at similar distances
        and the loop does not zigzag in and out;
      - each point gets an independent random shove, which is what makes one
        track look different from the next;
      - the points get smoothed, which rounds off the corners the shove created.
    """
    count = int(random.integers(settings.control_points_min,
                                settings.control_points_max + 1))
    angles = spread_out_angles(random.uniform(0, 2 * np.pi, count),
                               settings.min_angle_between_points)
    radii = random.uniform(settings.ring_radius_min, settings.ring_radius_max, count)
    radii = smooth_around_loop(radii, settings.radius_smoothing_rounds)

    points = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)
    points += random.normal(0.0, settings.point_jitter_meters, points.shape)
    return smooth_around_loop(points, settings.point_smoothing_rounds)
