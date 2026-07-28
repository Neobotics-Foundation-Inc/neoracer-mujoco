"""Procedural racetrack generation: `generate_track(seed)` returns a `Track`.

Read track.py first - it holds the Track and the generate/check/repair loop, and
names the module each stage lives in.
"""

from .centerline import shortest_loop_distance
from .settings import TrackSettings, settings_for_difficulty
from .track import Track, TrackGenerationError, generate_track

__all__ = [
    "generate_track",
    "Track",
    "TrackGenerationError",
    "TrackSettings",
    "settings_for_difficulty",
    "shortest_loop_distance",
]
