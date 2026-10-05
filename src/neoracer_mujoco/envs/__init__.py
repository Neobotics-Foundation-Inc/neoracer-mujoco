"""
Gymnasium environments for the NeoRacer car (issue #7).

Not imported by neoracer_mujoco/__init__.py: gymnasium is an optional `rl`
dependency (see pyproject.toml's [project.optional-dependencies]), so
importing the base neoracer_mujoco package must never require it. Import
this subpackage explicitly:

    from neoracer_mujoco.envs import NeoRacerEnv
"""

from .neoracer_env import NeoRacerEnv, NeoRacerEnvConfig, RewardConfig

__all__ = ["NeoRacerEnv", "NeoRacerEnvConfig", "RewardConfig"]
