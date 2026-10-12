"""Procedural racetracks.

    track = generate_track(7, settings_for_difficulty(2))
    model = assets.compose(to_spec(track))

Read track.py first -- it holds the Track and the loop that makes one. That
loop draws a candidate (shape.py), checks it and nudges it (reject.py), and
gives up or returns. config.py holds the knobs; mjcf.py turns a finished Track
into scenery.
"""

from .config import Difficulty, TrackSettings, settings_for_difficulty
from .mjcf import to_mjcf, to_spec
from .shape import shortest_loop_distance
from .track import Track, TrackGenerationError, generate_track, left_edge, right_edge

__all__ = [
    "Difficulty",
    "Track",
    "TrackGenerationError",
    "TrackSettings",
    "generate_track",
    "left_edge",
    "right_edge",
    "settings_for_difficulty",
    "shortest_loop_distance",
    "to_mjcf",
    "to_spec",
]
