"""What makes a candidate track unusable, and how to nudge the control points
that caused it.

Finding and fixing live together because they are one feedback loop: the
`Problems` a check reports name the samples a repair acts on, and nothing else
in the package needs that type.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.spatial import KDTree

from .config import TrackSettings
from .shape import Centerline, Points, Samples, shortest_loop_distance

type Indices = npt.NDArray[np.intp]  # shape (K,), sample indices
type Pairs = npt.NDArray[np.intp]  # shape (K, 2), (i, j) sample index pairs


@dataclass(frozen=True)
class Problems:
    """What is wrong with a candidate track.

    Indices rather than a bool verdict, so a repair can nudge exactly the
    control points behind a failure instead of redrawing the whole track.
    """

    too_curvy: Indices  # samples that are too tight to drive
    too_close: Pairs  # sample pairs that nearly touch in space
    length_ok: bool
    # The loop turns left overall. A clockwise loop runs the other way around
    # every track, so it is redrawn.
    counterclockwise: bool


# --- finding the problems --------------------------------------------------
# Checked on the sampled geometry, not on the control points: the spline
# between well-spaced control points can still bend too sharply.


def find_too_curvy_samples(
    curvature: Samples, half_width: Samples, settings: TrackSettings
) -> Indices:
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
    line: Centerline, half_width: Samples, settings: TrackSettings
) -> Pairs:
    """(i, j) sample pairs far apart traveling around the loop but close
    together in space -- the loop nearly touching itself, meaning two corridors
    overlap or the gap between them is too thin to fit walls into.

    Both halves of that test matter: neighboring samples on a straight are
    close in space and perfectly fine, so pairs near each other along the loop
    are dropped.

    A KDTree radius query rather than the O(N^2) pairwise distance matrix --
    N is 2048 samples and this runs on every repair pass of every attempt.
    """
    needed_clearance = 2 * half_width.max() + settings.extra_separation_margin
    pairs = KDTree(line.center).query_pairs(r=needed_clearance, output_type="ndarray")
    along = line.distance_along
    loop_gap = np.abs(
        shortest_loop_distance(
            along[pairs[:, 1]] - along[pairs[:, 0]], line.total_length
        )
    )
    return pairs[loop_gap > settings.min_separation_along_loop]


def find_problems(
    line: Centerline, half_width: Samples, settings: TrackSettings
) -> Problems:
    """Run every rejection rule over a candidate track: curvature, near
    self-touch, total length inside the requested range, and direction."""
    return Problems(
        too_curvy=find_too_curvy_samples(line.curvature, half_width, settings),
        too_close=find_near_self_touches(line, half_width, settings),
        length_ok=(
            settings.min_total_length <= line.total_length <= settings.max_total_length
        ),
        # A loop's curvature adds up to +2 pi counterclockwise, -2 pi clockwise.
        counterclockwise=bool(line.curvature.sum() > 0),
    )


# --- nudging the control points behind them --------------------------------
# Nudge rather than redraw: most rejected tracks are one bad corner away from
# valid, and a redraw throws away the other fifteen good ones. No randomness
# here at all, so a seed that needed three repair passes reproduces exactly.


def nearest_control_points(points: Points, positions: Points) -> Indices:
    """For each position on the centerline, the index of the closest control
    point -- the one whose movement most affects that stretch of track."""
    gaps = np.linalg.norm(positions[:, None, :] - points[None, :, :], axis=2)
    return np.argmin(gaps, axis=1)


def smooth_sharp_corners(
    points: Points, too_curvy_samples: Indices, center: Points
) -> Points:
    """Blend each control point behind a too-curvy sample toward its two
    neighbors, which opens the corner up.

    Every offending sample is mapped to a control point first, then all of them
    move at once off the same starting positions, so two samples blaming the
    same corner do not smooth it twice.
    """
    count = len(points)
    corners = np.unique(nearest_control_points(points, center[too_curvy_samples]))
    smoothed = points.copy()
    for corner in corners.tolist():
        smoothed[corner] = 0.5 * points[corner] + 0.25 * (
            points[(corner - 1) % count] + points[(corner + 1) % count]
        )
    return smoothed


def push_apart_near_touches(
    points: Points,
    too_close_pairs: Pairs,
    center: Points,
    half_width: Samples,
    settings: TrackSettings,
) -> Points:
    """Push the control points behind near-touching samples directly away from
    each other, by half the missing clearance each.

    One near touch is hundreds of sample pairs that map to the same two control
    points, so the pairs are grouped by control point pair first, and each group
    is pushed once, by its worst shortfall. All pushes start from the same
    positions, like smooth_sharp_corners. A point can be in several groups, so
    its pushes are added, and then its total move is capped at
    settings.max_repair_shift.
    """
    needed_clearance = 2 * half_width.max() + settings.extra_separation_margin
    i, j = too_close_pairs[:, 0], too_close_pairs[:, 1]
    owner = nearest_control_points(points, center)
    low = np.minimum(owner[i], owner[j])
    high = np.maximum(owner[i], owner[j])
    shortfall = needed_clearance - np.linalg.norm(center[i] - center[j], axis=1)
    # worst[a, b]: the largest shortfall between control points a < b. One
    # control point that governs both samples cannot push them apart.
    count = len(points)
    worst = np.zeros((count, count))
    np.maximum.at(worst, (low, high), shortfall)
    np.fill_diagonal(worst, 0.0)

    move = np.zeros_like(points)
    for a, b in np.argwhere(worst > 0).tolist():
        separation = points[a] - points[b]
        distance = float(np.linalg.norm(separation))
        if distance < 1e-9:
            continue  # no direction to push along
        shift = 0.5 * float(worst[a, b]) * separation / distance
        move[a] += shift
        move[b] -= shift
    length = np.linalg.norm(move, axis=1, keepdims=True)
    scale = np.minimum(1.0, settings.max_repair_shift / np.maximum(length, 1e-12))
    return points + move * scale


def repair(
    points: Points,
    problems: Problems,
    center: Points,
    half_width: Samples,
    settings: TrackSettings,
) -> Points:
    """Return a new set of control points with each reported problem nudged at.

    Corners first, then near-touches: smoothing a corner also drags the track
    sideways, so doing it after the push-apart would partly undo the push.

    Nothing is fixed for certain -- the caller re-splines and re-checks, and
    gives up on this candidate after max_repairs_per_attempt passes. There is
    no repair for total length or direction: they are whole-loop properties
    with no single control point to blame, so such a track is redrawn instead.
    """
    if len(problems.too_curvy) > 0:
        points = smooth_sharp_corners(points, problems.too_curvy, center)
    return push_apart_near_touches(
        points, problems.too_close, center, half_width, settings
    )
