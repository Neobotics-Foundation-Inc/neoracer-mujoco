"""Procedural racetracks.

    match generate_track(7, settings_for_difficulty(2)):
        case Track() as track:
            scene = assets.compose(to_spec(track))
        case NoValidTrack():
            ...  # try another seed or looser settings

Read track.py first -- it holds the Track and the loop that makes one. That
loop draws a candidate (shape.py), checks it and nudges it (reject.py), and
gives up or returns. config.py holds the knobs; mjcf.py turns a finished Track
into scenery.
"""

from .config import Difficulty, TrackSettings, settings_for_difficulty
from .mjcf import WallParams, lap_distance, pose_along, to_mjcf, to_spec
from .shape import shortest_loop_distance
from .track import (
    Generated,
    NoValidTrack,
    Track,
    generate_track,
    left_edge,
    right_edge,
)

__all__ = [
    "Difficulty",
    "Generated",
    "NoValidTrack",
    "Track",
    "TrackSettings",
    "WallParams",
    "generate_track",
    "lap_distance",
    "left_edge",
    "pose_along",
    "right_edge",
    "settings_for_difficulty",
    "shortest_loop_distance",
    "to_mjcf",
    "to_spec",
]
