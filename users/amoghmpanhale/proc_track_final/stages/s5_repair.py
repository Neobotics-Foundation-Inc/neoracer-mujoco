"""Step 5: nudge the control points behind the problems step 4 reported.

  nearest_control_point    maps a failing sample back to the point that moves it
  smooth_sharp_corners     fixes too_curvy: blend a point toward its neighbors
  push_apart_near_touches  fixes too_close: shove two points away from each other
  repair                   the stage entry point: corners first, then near-touches

Nudge rather than redraw, because most rejected tracks are one bad corner away
from valid and a redraw throws away the other fifteen good ones. Nothing here
guarantees a fix - the caller re-splines and re-checks, and gives up on this
candidate after max_repairs_per_attempt passes.

No randomness at all, so a seed that needed three repair passes reproduces
exactly. Per-pass shifts are capped (max_repair_shift) so one bad pair cannot
fling the loop into a new shape; anything larger arrives over several passes.
There is no rule 4 (total length) repair - length is a whole-loop property with
no single control point to blame, so a too-long track is redrawn instead.
"""

import numpy as np

from ..settings import TrackSettings
from .s4_validation import Problems


def nearest_control_point(points, position):
    """Index of the control point closest to a position on the centerline - the
    one whose movement most affects that stretch of track."""
    return int(np.argmin(np.linalg.norm(points - position, axis=1)))


def smooth_sharp_corners(points, too_curvy_samples, center):
    """Blend each control point behind a too-curvy sample toward its two
    neighbors, which opens the corner up. Every offending sample is mapped to a
    control point first, then all of them move at once off the same starting
    positions, so two samples blaming the same corner do not smooth it twice."""
    count = len(points)
    corners = {nearest_control_point(points, center[i]) for i in too_curvy_samples}
    smoothed = points.copy()
    for corner in corners:
        smoothed[corner] = 0.5 * points[corner] + 0.25 * (
            points[(corner - 1) % count] + points[(corner + 1) % count]
        )
    return smoothed


def push_apart_near_touches(points, too_close_pairs, center, half_width,
                            settings: TrackSettings):
    """Push the control points behind each near-touching pair of samples
    directly away from each other, by half the missing clearance each.

    The per-pass shift is capped so one bad pair cannot fling the loop into a
    new shape; a pair needing more than the cap gets it over several passes.
    """
    points = points.copy()
    needed_clearance = 2 * half_width.max() + settings.extra_separation_margin

    for i, j in too_close_pairs:
        first = nearest_control_point(points, center[i])
        second = nearest_control_point(points, center[j])
        if first == second:
            continue  # one control point governs both, pushing it apart is a no-op
        separation = points[first] - points[second]
        distance = np.linalg.norm(separation)
        if distance < 1e-9:
            continue  # no direction to push along
        shortfall = needed_clearance - np.linalg.norm(center[i] - center[j])
        shift = min(0.5 * shortfall, settings.max_repair_shift)
        if shift <= 0:
            continue
        direction = separation / distance
        points[first] += direction * shift
        points[second] -= direction * shift

    return points


def repair(points, problems: Problems, center, half_width, settings: TrackSettings):
    """Return a new set of control points with each reported problem nudged at.

    Corners first, then near-touches: smoothing a corner also drags the track
    sideways, so doing it after the push-apart would partly undo the push.
    Nothing is fixed for certain here - the caller re-splines and re-checks.
    """
    if len(problems.too_curvy) > 0:
        points = smooth_sharp_corners(points, problems.too_curvy, center)
    return push_apart_near_touches(points, problems.too_close, center, half_width,
                                   settings)
