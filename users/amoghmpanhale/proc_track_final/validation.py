"""Stage 4a: decide whether a candidate track is drivable.

Four rejection rules, checked on the finished geometry rather than on the
control points, because a well-behaved ring of control points can still spline
into a hairpin. Reporting *which* samples failed is the point - `repair` uses
those indices to decide what to move.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .centerline import shortest_loop_distance
from .settings import TrackSettings


@dataclass
class Problems:
    """What is wrong with a candidate track. Falsy when nothing is."""

    too_curvy: np.ndarray        # sample indices that are too tight to drive
    too_close: list              # (i, j) sample pairs that nearly touch in space
    length_ok: bool

    def __bool__(self):
        return len(self.too_curvy) > 0 or len(self.too_close) > 0 or not self.length_ok


def find_too_curvy_samples(curvature, half_width, settings: TrackSettings):
    """Sample indices failing either curvature rule.

    1. Too tight to drive: |curvature| over the limit. A cap on how sharp a
       corner the car is asked to take.
    2. Corridor folds onto itself: half_width * |curvature| over the limit. At
       half_width * curvature == 1 the inner edge of the corridor collapses to a
       single point, so this dimensionless ratio is the real geometric
       constraint, and rule 1 is a driveability cap on top of it.
    """
    return np.where(
        (np.abs(curvature) > settings.max_curvature)
        | (half_width * np.abs(curvature) > settings.max_width_times_curvature)
    )[0]


def find_near_self_touches(center, half_width, distance_along, total_length,
                           settings: TrackSettings):
    """(i, j) sample pairs that are far apart travelling around the loop but
    close together in space - the loop nearly touching itself, which means two
    corridors overlapping or a gap too thin to fit walls into.

    Both halves of that test matter: neighboring samples on a straight are close
    in space and perfectly fine, so pairs that are also near each other along
    the loop are dropped.
    """
    needed_clearance = 2 * half_width.max() + settings.extra_separation_margin
    pairs = cKDTree(center).query_pairs(r=needed_clearance, output_type="ndarray")
    if len(pairs) > 0:
        loop_gap = np.abs(shortest_loop_distance(
            distance_along[pairs[:, 1]] - distance_along[pairs[:, 0]], total_length
        ))
        pairs = pairs[loop_gap > settings.min_separation_along_loop]
    return [(int(i), int(j)) for i, j in pairs]


def find_problems(center, curvature, half_width, distance_along, total_length,
                  settings: TrackSettings):
    """Run every rejection rule over a candidate track: curvature, near
    self-touch, and total length inside the requested range."""
    return Problems(
        too_curvy=find_too_curvy_samples(curvature, half_width, settings),
        too_close=find_near_self_touches(center, half_width, distance_along,
                                         total_length, settings),
        length_ok=(settings.min_total_length
                   <= total_length
                   <= settings.max_total_length),
    )
