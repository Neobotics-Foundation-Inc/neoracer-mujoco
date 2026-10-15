"""Turn a Track into MJCF scenery: two walls of boxes and a floor, no car.

The scene around the walls -- assets, materials, floor, compiler settings --
lives in scene_template.xml next door, not in this file. This module computes
the wall boxes and fills the template's placeholders in.

Frame: +X forward from the start, +Y left, +Z up. Output follows the convention in assets/tracks/*.xml -- scenery that never
names the car -- so `assets.compose` can attach the car to it at runtime:

    assets.compose(to_spec(track))  # track: a Track from generate_track
"""

import itertools
import string
from dataclasses import dataclass
from importlib import resources

import mujoco
import numpy as np
import numpy.typing as npt

from .shape import Points
from .track import Track, left_edge, right_edge


@dataclass(frozen=True)
class WallParams:
    """Sizes of the generated walls and floor."""

    height: float = 0.30  # m; the height of the walls in assets/tracks/*.xml
    thickness: float = 0.05  # m
    # m; the longest wall box. A lower value follows the edges more closely,
    # with proportionally more geoms.
    segment_length: float = 0.5
    max_deviation: float = 0.01  # m; how far an edge may bow away from a wall box
    floor_margin: float = 1.0  # m of floor beyond the outermost wall
    # None: as few boxes as the two caps above allow. A number: exactly that
    # many boxes in each wall, so that every track compiles to a model of the
    # same size, as a batch with one track per row needs (see split_to).
    boxes_per_wall: int | None = None


def to_start_frame(track: Track, points: Points) -> Points:
    """points moved by the rigid transform that puts sample 0 at the origin
    facing +X.

    Attaching a car drops it at the origin facing +X, but a Track's sample 0 is
    somewhere out on the loop pointing anywhere. Moving the walls to meet the
    car (rather than the car to meet the walls) keeps Track pure geometry, with
    every field it carries -- distance_along, curvature -- valid as written.
    """
    origin = track.center[0]
    heading = np.arctan2(track.tangent[0, 1], track.tangent[0, 0])
    cos, sin = np.cos(-heading), np.sin(-heading)
    rotation = np.array([[cos, -sin], [sin, cos]])
    return (points - origin) @ rotation.T


def _max_deviation(points: Points, start: int, end: int) -> float:
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


def decimate_closed(
    points: Points, segment_length: float, max_deviation: float
) -> list[int]:
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


def split_to(
    points: Points, kept: list[int], count: int, max_deviation: float
) -> list[int]:
    """kept, a list of decimate_closed over points, with the longest segments
    split until there are count segments. Each split takes the sample
    nearest the middle of the segment whose two halves both stay within
    max_deviation of points; the first half is a prefix of the segment,
    which decimate_closed already checked. Raise ValueError if kept already
    has more than count segments, or if no segment can be split."""
    if len(kept) - 1 > count:
        raise ValueError(
            f"this wall needs {len(kept) - 1} boxes, more than boxes_per_wall={count}"
        )
    kept = list(kept)
    while len(kept) - 1 < count:
        for segment in np.argsort(-np.diff(kept), kind="stable"):
            start, end = kept[segment], kept[segment + 1]
            middle = (start + end) // 2
            for k in sorted(range(start + 1, end), key=lambda k: abs(k - middle)):
                if _max_deviation(points, k, end) <= max_deviation:
                    kept.insert(int(segment) + 1, k)
                    break
            else:
                continue
            break
        else:
            raise ValueError(
                f"no wall box can be split within max_deviation={max_deviation}"
            )
    return kept


def _wall_boxes(edge: Points, name_prefix: str, walls: WallParams) -> list[str]:
    """MJCF <geom> lines for one wall: one box per decimated segment, centred on
    its midpoint and yawed to lie along it.

    Each box is over-long by `thickness` at both ends so consecutive boxes
    overlap at their corners instead of leaving a notch for a wheel to catch in.
    """
    kept = decimate_closed(edge, walls.segment_length, walls.max_deviation)
    if walls.boxes_per_wall is not None:
        kept = split_to(edge, kept, walls.boxes_per_wall, walls.max_deviation)
    thickness, height = walls.thickness, walls.height
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
            f'            <geom class="track_wall" name="{name_prefix}_{n:04d}"'
            f' pos="{middle[0]:.5f} {middle[1]:.5f} {height / 2:.5f}"'
            f' euler="0 0 {yaw:.5f}"'
            f' size="{length / 2 + thickness:.5f} {thickness / 2:.5f}'
            f' {height / 2:.5f}"/>'
        )
    return lines


def to_mjcf(track: Track, walls: WallParams = WallParams()) -> str:
    """MJCF text for `track` as scenery: two walls of boxes and a floor, no car.

    Walls come straight off left_edge(track) / right_edge(track), which already
    carry the loop's varying half-width, moved by the start transform so the
    car spawns at sample 0 facing +X.
    """
    left = to_start_frame(track, left_edge(track))
    right = to_start_frame(track, right_edge(track))

    geoms = _wall_boxes(left, "track_wall_left", walls)
    geoms += _wall_boxes(right, "track_wall_right", walls)

    corners = np.vstack([left, right])
    low = corners.min(axis=0) - walls.floor_margin
    high = corners.max(axis=0) + walls.floor_margin
    floor_size = 0.5 * (high - low)
    floor_pos = 0.5 * (high + low)

    # substitute() rather than safe_substitute(): a typo'd placeholder should
    # fail loudly here, not silently emit a "$name" into the MJCF.
    template = string.Template(
        resources.files("neoracer_mujoco.track_generation")
        .joinpath("scene_template.xml")
        .read_text()
    )
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


def lap_distance(track: Track, xy: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """The distance (m) along the centerline, from the start, of the sample
    nearest to each point of xy (..., 2) in the frame of to_mjcf, where the
    car starts at the origin facing +X. It is in [0, total_length).

    Limit: the nearest sample is the right one only while the point is in
    the corridor, which reject.py keeps apart from the rest of the loop."""
    center = to_start_frame(track, track.center)  # (N, 2)
    points = np.asarray(xy, dtype=np.float64)
    gaps = points[..., None, :] - center  # (..., N, 2)
    nearest = np.argmin(np.einsum("...ij,...ij->...i", gaps, gaps), axis=-1)
    return track.distance_along[nearest]


def to_spec(track: Track, walls: WallParams = WallParams()) -> mujoco.MjSpec:
    """The scenery of to_mjcf as an MjSpec, for assets.compose."""
    return mujoco.MjSpec.from_string(to_mjcf(track, walls))
