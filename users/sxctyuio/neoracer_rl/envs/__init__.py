"""
RL environments for the NeoRacer car.

Not imported by neoracer_mujoco/__init__.py. Each env has its own optional
dependency group (see pyproject.toml), so nothing is imported until used:

    from neoracer_rl.envs import NeoRacerEnv
"""

_EXPORTS = {
    "NeoRacerEnv": ".neoracer_env",
    "NeoRacerEnvConfig": ".neoracer_env",
    "RewardConfig": ".neoracer_env",
    "SpawnRange": ".neoracer_env",
    "TerminationConfig": ".neoracer_env",
    "ProceduralRacerEnv": ".procedural_env",
    "ProceduralEnvConfig": ".procedural_env",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name in _EXPORTS:
        import importlib

        return getattr(importlib.import_module(_EXPORTS[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
