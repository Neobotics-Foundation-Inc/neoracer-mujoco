"""
neoracer-rl's NeoRacerRace as a Stable-Baselines3 VecEnv: each car that the
policy drives is an env of its own, so a plain PPO trains one policy on
every car's experience (parameter sharing).

A race ends for every car at once, when one car meets an end; the others
are truncated (info["TimeLimit.truncated"]), so PPO bootstraps their value
instead of blaming them. The race then starts over from its spawn, and
every respawn_every steps each race draws a new spawn, track and physics.
"""

from typing import Any

import numpy as np
from gymnasium import spaces
from neoracer_rl.envs.neoracer_batch import BatchState, Cpu, NeoRacerRace
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnv,
    VecEnvObs,
    VecEnvStepReturn,
)

from .history import History
from .setup import Setup, race_config
from .sim2real import Sim2Real


class RaceVecEnv(VecEnv):
    render_mode = None

    def __init__(self, race: NeoRacerRace, seed: int, respawn_every: int = 1024):
        self.race = race
        self.respawn_every = respawn_every
        self._rng = np.random.default_rng(seed)
        self.state: BatchState = race.reset(self._rng)
        self._steps = 0
        self._actions = np.zeros((race.nbatch, race.action_dim), np.float32)
        obs = spaces.Box(-np.inf, np.inf, self.state.obs.shape[1:], np.float32)
        act = spaces.Box(-1.0, 1.0, (race.action_dim,), np.float32)
        super().__init__(race.nbatch, obs, act)

    def reset(self) -> VecEnvObs:
        self.state = self.race.reset(self._rng)
        return self.state.obs

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        self.state, out = self.race.step(self.state, self._actions)
        self._steps += 1
        if self._steps % self.respawn_every == 0:
            self.state = self.race.respawn(self.state, self._rng)
        infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]
        for i in np.flatnonzero(out.done):
            code = out.reason[i]
            infos[i] = {
                "terminal_observation": out.obs[i],
                "TimeLimit.truncated": bool(out.truncated[i]),
                "outcome": out.ends[code - 1] if code else "truncated",
                "lap_frac": float(out.lap_frac[i]),
            }
        return self.state.obs, out.reward.astype(np.float32), out.done, infos

    def close(self) -> None:
        self.race.close()

    def get_attr(self, attr_name: str, indices: Any = None) -> list[Any]:
        return [getattr(self, attr_name)] * len(list(self._get_indices(indices)))

    def set_attr(self, attr_name: str, value: Any, indices: Any = None) -> None:
        setattr(self, attr_name, value)

    def env_method(
        self, method_name: str, *args: Any, indices: Any = None, **kwargs: Any
    ) -> list[Any]:
        result = getattr(self, method_name)(*args, **kwargs)
        return [result] * len(list(self._get_indices(indices)))

    def env_is_wrapped(self, wrapper_class: type, indices: Any = None) -> list[bool]:
        return [False] * len(list(self._get_indices(indices)))


def make(setup: Setup, races: int, seed: int, nthread: int) -> VecEnv:
    """races races of setup on the CPU, as the policy sees them: with
    sim2real's noise and delay if setup.sim2real, and setup.history steps
    of memory."""
    config = race_config(setup)
    race = NeoRacerRace(Cpu(nthread=min(nthread, races)), races, config)
    env: VecEnv = RaceVecEnv(race, seed)
    if setup.sim2real:
        env = Sim2Real(env, seed=seed)
    if setup.history:
        env = History(env, setup.history)
    return env
