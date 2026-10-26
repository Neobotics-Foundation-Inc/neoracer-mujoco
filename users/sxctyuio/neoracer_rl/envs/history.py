"""
Memory for the policy: its last few observations and actions, appended to
the observation.

    new obs = [obs_t, obs_t-1, ..., obs_t-(n-1),  a_t-1, a_t-2, ..., a_t-n]

obs_t is the env's usual observation and stays first, so a policy trained
without memory carries over unchanged (train_ppo.py --init-from copies its
weights onto obs_t; the memory inputs start at zero weight). The actions are
what the policy commanded (clipped to [-1, 1]) -- what a real car's
controller knows -- not what reached the motors after sim2real noise and
delay; comparing them with the observations is how the policy can sense
control delay and how strong this car's motors and steering are.

At the start of an episode (and when a racing car respawns) the past
observations are copies of the first one and the past actions are zeros: a
car standing still.

HistoryWrapper wraps a single-car gym.Env; MultiHistory wraps a racing env
(MultiRacerEnv or MultiRacerSim2Real) and keeps one memory per car.
"""

from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces


def history_space(base: spaces.Box, n: int) -> spaces.Box:
    """Observation space with n observations and n actions of memory."""
    low = np.concatenate([np.tile(base.low, n), np.full(2 * n, -1.0)]).astype(np.float32)
    high = np.concatenate([np.tile(base.high, n), np.full(2 * n, 1.0)]).astype(np.float32)
    return spaces.Box(low, high, dtype=np.float32)


class _Memory:
    def __init__(self, n: int):
        self.n = n
        self.obs: deque = deque(maxlen=n)
        self.act: deque = deque(maxlen=n)

    def start(self, obs: np.ndarray) -> None:
        self.obs.clear()
        self.act.clear()
        for _ in range(self.n):
            self.obs.append(np.asarray(obs, dtype=np.float32))
            self.act.append(np.zeros(2, dtype=np.float32))

    def push(self, obs: np.ndarray, action: np.ndarray) -> None:
        self.obs.appendleft(np.asarray(obs, dtype=np.float32))
        self.act.appendleft(np.clip(np.asarray(action, dtype=np.float32).reshape(2), -1.0, 1.0))

    def vector(self) -> np.ndarray:
        return np.concatenate([*self.obs, *self.act])


class HistoryWrapper(gym.Wrapper):
    """Single car: see the module docstring."""

    def __init__(self, env: gym.Env, n: int = 3):
        super().__init__(env)
        self.n = n
        self.observation_space = history_space(env.observation_space, n)
        self._mem = _Memory(n)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._mem.start(obs)
        return self._mem.vector(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._mem.push(obs, action)
        return self._mem.vector(), reward, terminated, truncated, info


class MultiHistory:
    """Racing: one memory per car. Same API as MultiRacerEnv (reset, step,
    respawn, step_autoreset); a respawned car starts a fresh memory."""

    def __init__(self, env, n: int = 3):
        self.env = env
        self.n_mem = n
        self.observation_space = history_space(env.observation_space, n)
        self.action_space = env.action_space
        self._mems = [_Memory(n) for _ in range(env.n)]

    def __getattr__(self, name):  # model, data, n, effective_control_dt, ...
        return getattr(self.env, name)

    def _vectors(self) -> np.ndarray:
        return np.stack([m.vector() for m in self._mems])

    def reset(self, seed: int | None = None):
        obs, infos = self.env.reset(seed)
        for m, o in zip(self._mems, obs):
            m.start(o)
        return self._vectors(), infos

    def step(self, actions: np.ndarray):
        actions = np.asarray(actions, dtype=np.float32).reshape(self.env.n, 2)
        obs, rewards, term, trunc, infos = self.env.step(actions)
        for m, o, a in zip(self._mems, obs, actions):
            m.push(o, a)
        return self._vectors(), rewards, term, trunc, infos

    def respawn(self, i: int) -> np.ndarray:
        obs = self.env.respawn(i)
        self._mems[i].start(obs[i])
        return self._vectors()

    def step_autoreset(self, actions: np.ndarray):
        # Same rules as multi_racer_env._AutoReset, through this wrapper's
        # step / reset / respawn so the memories stay in step with the cars.
        obs, rewards, term, trunc, infos = self.step(actions)
        dones = term | trunc
        for i in np.where(dones)[0]:
            infos[i]["terminal_observation"] = obs[i]
            infos[i]["TimeLimit.truncated"] = bool(trunc[i] and not term[i])
        if dones.all():
            obs, _ = self.reset()
        elif dones.any():
            for i in np.where(dones)[0]:
                obs = self.respawn(int(i))
        return obs, rewards, dones, infos
