"""Procedural racetracks: `generate_track(seed, difficulty)` returns a `Track`.

Read track.py first -- it holds the Track and the loop that makes one. That
loop draws a candidate (shape.py), checks it and nudges it (reject.py), and
gives up or returns. config.py holds the knobs; mjcf.py turns a finished Track
into scenery. Use `neoracer_mujoco.compose` to put the car on it.
"""

from .config import TrackSettings, settings_for_difficulty
from .mjcf import to_mjcf
from .shape import shortest_loop_distance
from .track import Track, TrackGenerationError, generate_track

__all__ = [
    "Track",
    "TrackGenerationError",
    "TrackSettings",
    "generate_track",
    "settings_for_difficulty",
    "shortest_loop_distance",
    "to_mjcf",
]
