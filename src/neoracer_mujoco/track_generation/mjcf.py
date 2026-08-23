"""Turn a Track into MJCF scenery: two walls of boxes and a floor, no car.

The scene around the walls -- assets, materials, floor, compiler settings --
lives in scene_template.xml next door, not in this file. This module computes
the wall boxes and fills the template's placeholders in.

Output follows the convention in assets/tracks/*.xml -- scenery that never
names the car -- so `assets.compose` can attach the car to it at runtime.
"""

import itertools
import string
from pathlib import Path

import numpy as np

from .track import Track

_TEMPLATE = Path(__file__).parent / "scene_template.xml"

_FLOOR_MARGIN = 1.0  # meters of floor plane beyond the outermost wall


def _start_transform(track: Track):
    """Build the rigid transform putting sample 0 at the origin facing +X.

    Attaching a car drops it at the origin facing +X, but a Track's sample 0 is
    somewhere out on the loop pointing anywhere. Moving the walls to meet the
    car (rather than the car to meet the walls) keeps Track pure geometry, with
    every field it carries -- distance_along, curvature -- valid as written.

    Returns a function mapping (N, 2) points to (N, 2) transformed points.
    """
    origin = track.center[0]
    heading = np.arctan2(track.tangent[0, 1], track.tangent[0, 0])
    cos, sin = np.cos(-heading), np.sin(-heading)
    rotation = np.array([[cos, -sin], [sin, cos]])

    def transform(points):
        return (points - origin) @ rotation.T

    return transform


def _max_deviation(points, start, end):
    """Farthest any point strictly between `start` and `end` sits from the
    straight chord joining them. `end` may be len(points), meaning the wrap
    back to index 0."""
    first = points[start]
    last = points[end % len(points)]
    chord = last - first
    length = float(np.linalg.norm(chord))
    interior = points[start + 1 : end]
    if length < 1e-12 or len(interior) == 0:
        return 0.0
    offsets = interior - first
    # 2D cross product / |chord| is the perpendicular distance to the chord.
    across = chord[0] * offsets[:, 1] - chord[1] * offsets[:, 0]
    return float(np.abs(across).max() / length)


def _decimate_closed(points, segment_length, max_deviation):
    """Split a closed polyline into as few chords as the two caps allow.

    A track has 2048 samples, so one wall box per sample would be 4096 static
    geoms -- broadphase cost on every step of every RL rollout, for detail no
    1/12-scale car can feel. This walks the polyline greedily instead, ending a
    chord when it has covered `segment_length` of arclength or the fine
    polyline has bowed `max_deviation` away from it. The arclength cap governs
    straights; the deviation cap takes over in corners, where a fixed step
    would cut the apex off (a 0.5 m chord on a 1 m radius misses by 3 cm).

    Returns [0, i1, i2, ..., len(points)]; the final entry wraps back to 0, so
    consecutive pairs are the segments.
    """
    count = len(points)
    hops = np.linalg.norm(np.roll(points, -1, 0) - points, axis=1)

    kept = [0]
    start = 0
    while start < count:
        end = start + 1
        arclength = hops[start]
        while end < count:
            if arclength + hops[end] > segment_length:
                break
            if _max_deviation(points, start, end + 1) > max_deviation:
                break
            arclength += hops[end]
            end += 1
        kept.append(end)
        start = end
    return kept


def _wall_boxes(edge, name_prefix, segment_length, max_deviation, thickness, height):
    """MJCF <geom> lines for one wall: one box per decimated segment, centred on
    its midpoint and yawed to lie along it.

    Each box is over-long by `thickness` at both ends so consecutive boxes
    overlap at their corners instead of leaving a notch for a wheel to catch in.
    """
    kept = _decimate_closed(edge, segment_length, max_deviation)
    lines = []
    for n, (i, j) in enumerate(itertools.pairwise(kept)):
        first, last = edge[i], edge[j % len(edge)]
        span = last - first
        length = float(np.linalg.norm(span))
        if length < 1e-9:
            continue
        middle = 0.5 * (first + last)
        yaw = float(np.arctan2(span[1], span[0]))
        lines.append(
            f'        <geom class="track_wall" name="{name_prefix}_{n:04d}"'
            f' pos="{middle[0]:.5f} {middle[1]:.5f} {height / 2:.5f}"'
            f' euler="0 0 {yaw:.5f}"'
            f' size="{length / 2 + thickness:.5f} {thickness / 2:.5f}'
            f' {height / 2:.5f}"/>'
        )
    return lines


def to_mjcf(
    track: Track,
    *,
    wall_height: float = 0.30,
    wall_thickness: float = 0.05,
    segment_length: float = 0.5,
    max_deviation: float = 0.01,
) -> str:
    """MJCF text for `track` as scenery: two walls of boxes and a floor, no car.

    wall_height/wall_thickness  box dimensions, meters. 0.30 m tall matches the
                                wall convention in assets/tracks/*.xml.
    segment_length              longest wall box, meters. Lower it for walls
                                that hug the generated edges more closely, at a
                                proportional cost in geom count.
    max_deviation               how far the true edge may bow away from a wall
                                box, meters. See _decimate_closed.

    Walls come straight off track.left_edge / track.right_edge, which already
    carry the loop's varying half-width, moved by the start transform so the
    car spawns at sample 0 facing +X.
    """
    transform = _start_transform(track)
    left = transform(track.left_edge)
    right = transform(track.right_edge)

    geoms = _wall_boxes(
        left,
        "track_wall_left",
        segment_length,
        max_deviation,
        wall_thickness,
        wall_height,
    )
    geoms += _wall_boxes(
        right,
        "track_wall_right",
        segment_length,
        max_deviation,
        wall_thickness,
        wall_height,
    )

    corners = np.vstack([left, right])
    low = corners.min(axis=0) - _FLOOR_MARGIN
    high = corners.max(axis=0) + _FLOOR_MARGIN
    floor_size = 0.5 * (high - low)
    floor_pos = 0.5 * (high + low)

    # substitute() rather than safe_substitute(): a typo'd placeholder should
    # fail loudly here, not silently emit a "$name" into the MJCF.
    template = string.Template(_TEMPLATE.read_text())
    return template.substitute(
        seed=track.seed,
        total_length=f"{track.total_length:.2f}",
        wall_count=len(geoms),
        floor_size_x=f"{floor_size[0]:.5f}",
        floor_size_y=f"{floor_size[1]:.5f}",
        floor_pos_x=f"{floor_pos[0]:.5f}",
        floor_pos_y=f"{floor_pos[1]:.5f}",
        walls="\n".join(geoms),
    )
