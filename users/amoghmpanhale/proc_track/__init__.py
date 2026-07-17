"""Procedural racetrack RL environment for the NeoRacer (per-developer scratch).

Step 1 exports: track generation only. ProcTrackEnv arrives in later stages.
"""

from .config import (
    EnvConfig,
    GenConfig,
    env_config_for_phase,
    gen_config_for_phase,
)
from .env import ProcTrackEnv
from .generator import Track, TrackGenerationError, sample_valid_track

__all__ = [
    "GenConfig",
    "EnvConfig",
    "gen_config_for_phase",
    "env_config_for_phase",
    "Track",
    "TrackGenerationError",
    "sample_valid_track",
    "ProcTrackEnv",
]
