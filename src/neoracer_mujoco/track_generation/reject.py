"""What makes a candidate track unusable, and how to nudge the control points
that caused it.

Finding and fixing live together because they are one feedback loop: the
`Problems` a check reports name the samples a repair acts on, and nothing else
in the package needs that type.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .config import TrackSettings
from .shape import shortest_loop_distance

# Cap on how far one repair pass may move a control point, so one bad pair
# cannot fling the loop into a new shape. Larger corrections arrive over
# several passes. Only push_apart_near_touches reads it.
MAX_REPAIR_SHIFT = 0.3


@dataclass
class Problems:
    """What is wrong with a candidate track. Falsy when nothing is.

    Indices rather than a bool verdict, so a repair can nudge exactly the
    control points behind a failure instead of redrawing the whole track.
    """

    too_curvy: np.ndarray  # sample indices that are too tight to drive
    too_close: list  # (i, j) sample pairs that nearly touch in space
    length_ok: bool

    def __bool__(self):
        return len(self.too_curvy) > 0 or len(self.too_close) > 0 or not self.length_ok


# --- finding the problems --------------------------------------------------
# Checked on the sampled geometry, not on the control points: a well-behaved
# ring of control points still splines into a hairpin now and then.


def find_too_curvy_samples(curvature, half_width, settings: TrackSettings):
    """Sample indices failing either curvature rule.

    1. Too tight to drive: |curvature| over the limit. A cap on how sharp a
       corner the car is asked to take.
    2. Corridor folds onto itself: half_width * |curvature| over the limit. At
       half_width * curvature == 1 the inner edge collapses to a point, so this
       dimensionless ratio is the real geometric constraint; rule 1 is a
       driveability cap on top of it.
    """
    return np.where(
        (np.abs(curvature) > settings.max_curvature)
        | (half_width * np.abs(curvature) > settings.max_width_times_curvature)
    )[0]


def find_near_self_touches(
    center, half_width, distance_along, total_length, settings: TrackSettings
):
    """(i, j) sample pairs far apart travelling around the loop but close
    together in space -- the loop nearly touching itself, meaning two corridors
    overlap or the gap between them is too thin to fit walls into.

    Both halves of that test matter: neighboring samples on a straight are
    close in space and perfectly fine, so pairs near each other along the loop
    are dropped.

    A cKDTree radius query rather than the O(N^2) pairwise distance matrix --
    N is 2048 samples and this runs on every repair pass of every attempt.
    """
    needed_clearance = 2 * half_width.max() + settings.extra_separation_margin
    pairs = cKDTree(center).query_pairs(r=needed_clearance, output_type="ndarray")
    if len(pairs) > 0:
        loop_gap = np.abs(
            shortest_loop_distance(
                distance_along[pairs[:, 1]] - distance_along[pairs[:, 0]], total_length
            )
        )
        pairs = pairs[loop_gap > settings.min_separation_along_loop]
    return [(int(i), int(j)) for i, j in pairs]


def find_problems(
    center, curvature, half_width, distance_along, total_length, settings: TrackSettings
):
    """Run every rejection rule over a candidate track: curvature, near
    self-touch, and total length inside the requested range."""
    return Problems(
        too_curvy=find_too_curvy_samples(curvature, half_width, settings),
        too_close=find_near_self_touches(
            center, half_width, distance_along, total_length, settings
        ),
        length_ok=(
            settings.min_total_length <= total_length <= settings.max_total_length
        ),
    )


# --- nudging the control points behind them --------------------------------
# Nudge rather than redraw: most rejected tracks are one bad corner away from
# valid, and a redraw throws away the other fifteen good ones. No randomness
# here at all, so a seed that needed three repair passes reproduces exactly.


def nearest_control_point(points, position):
    """Index of the control point closest to a position on the centerline --
    the one whose movement most affects that stretch of track."""
    return int(np.argmin(np.linalg.norm(points - position, axis=1)))


def smooth_sharp_corners(points, too_curvy_samples, center):
    """Blend each control point behind a too-curvy sample toward its two
    neighbors, which opens the corner up.

    Every offending sample is mapped to a control point first, then all of them
    move at once off the same starting positions, so two samples blaming the
    same corner do not smooth it twice.
    """
    count = len(points)
    corners = {nearest_control_point(points, center[i]) for i in too_curvy_samples}
    smoothed = points.copy()
    for corner in corners:
        smoothed[corner] = 0.5 * points[corner] + 0.25 * (
            points[(corner - 1) % count] + points[(corner + 1) % count]
        )
    return smoothed


def push_apart_near_touches(
    points, too_close_pairs, center, half_width, settings: TrackSettings
):
    """Push the control points behind each near-touching pair of samples
    directly away from each other, by half the missing clearance each (capped
    at MAX_REPAIR_SHIFT)."""
    points = points.copy()
    needed_clearance = 2 * half_width.max() + settings.extra_separation_margin

    for i, j in too_close_pairs:
        first = nearest_control_point(points, center[i])
        second = nearest_control_point(points, center[j])
        if first == second:
            continue  # one control point governs both; pushing it apart is a no-op
        separation = points[first] - points[second]
        distance = np.linalg.norm(separation)
        if distance < 1e-9:
            continue  # no direction to push along
        shortfall = needed_clearance - np.linalg.norm(center[i] - center[j])
        shift = min(0.5 * shortfall, MAX_REPAIR_SHIFT)
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

    Nothing is fixed for certain -- the caller re-splines and re-checks, and
    gives up on this candidate after max_repairs_per_attempt passes. There is
    no repair for total length: it is a whole-loop property with no single
    control point to blame, so a too-long track is redrawn instead.
    """
    if len(problems.too_curvy) > 0:
        points = smooth_sharp_corners(points, problems.too_curvy, center)
    return push_apart_near_touches(
        points, problems.too_close, center, half_width, settings
    )
