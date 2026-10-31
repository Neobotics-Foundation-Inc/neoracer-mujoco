"""
Memory for the policy: its last n observations and actions, in the
observation.

    obs_t, obs_t-1, ..., obs_t-(n-1), a_t-1, a_t-2, ..., a_t-n

The actions are what the policy asked for, clipped to [-1, 1], which a real
car's program knows: not what reached the motors after sim2real's noise and
delay. Comparing them with the observations is how the policy can sense the
delay and how strong this car's motor and steering are. At the start of an
episode the past observations are copies of the first and the past actions
are zeros, a car standing still.
"""

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnv,
    VecEnvObs,
    VecEnvStepReturn,
    VecEnvWrapper,
)


class History(VecEnvWrapper):
    def __init__(self, venv: VecEnv, n: int):
        space, act = venv.observation_space, venv.action_space
        assert isinstance(space, spaces.Box) and space.shape is not None
        assert isinstance(act, spaces.Box) and act.shape is not None
        self.n, self.dim, self.adim = n, space.shape[0], act.shape[0]
        size = n * (self.dim + self.adim)
        super().__init__(venv, spaces.Box(-np.inf, np.inf, (size,), np.float32))
        self._obs = np.zeros((venv.num_envs, n, self.dim), np.float32)
        self._actions = np.zeros((venv.num_envs, n, self.adim), np.float32)

    def _stacked(self, obs: np.ndarray, actions: np.ndarray) -> np.ndarray:
        rows = len(obs)
        return np.concatenate([obs.reshape(rows, -1), actions.reshape(rows, -1)], -1)

    def _start(self, rows: np.ndarray, obs: np.ndarray) -> None:
        self._obs[rows] = obs[rows, None]
        self._actions[rows] = 0.0

    def reset(self) -> VecEnvObs:
        obs = np.asarray(self.venv.reset())
        self._start(np.ones(self.num_envs, np.bool_), obs)
        return self._stacked(self._obs, self._actions)

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = np.roll(self._actions, 1, axis=1)
        self._actions[:, 0] = np.clip(actions, -1.0, 1.0)
        self.venv.step_async(actions)

    def step_wait(self) -> VecEnvStepReturn:
        obs, reward, done, infos = self.venv.step_wait()
        obs = np.asarray(obs)
        past = np.roll(self._obs, 1, axis=1)
        for i in np.flatnonzero(done):
            last = past[i : i + 1].copy()
            last[0, 0] = infos[i]["terminal_observation"]
            infos[i]["terminal_observation"] = self._stacked(last, self._actions[i : i + 1])[0]
        past[:, 0] = obs
        self._obs = past
        self._start(done, obs)
        return self._stacked(self._obs, self._actions), reward, done, infos
