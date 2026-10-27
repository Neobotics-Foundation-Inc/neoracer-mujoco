"""
Sim-to-real on the policy's side of the env: what neoracer-rl's physics
randomization (setup.GENTLE, drawn for each car at each spawn) leaves out.

Per episode of each car: a steering trim, like a misaligned servo horn,
biases on the IMU, a scale error on the wheel speed, and an action delay of
0 to max_delay_steps control steps. Per step: noise on every observation,
LiDAR beams that drop out (-1, no hit), and noise on the action.

It wraps a VecEnv whose observation is task.HeadingFree's. All ranges are
first guesses, not measurements from the real car.
"""

from dataclasses import dataclass

import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnv,
    VecEnvObs,
    VecEnvStepReturn,
    VecEnvWrapper,
)

from .task import GRAVITY


@dataclass(frozen=True)
class Sim2RealConfig:
    steer_trim: float = 0.045
    """The bound of the trim, in the action's steer: about 0.02 rad."""
    max_delay_steps: int = 1
    """0 to 1 steps: 0 to 33 ms at 30 Hz."""
    action_noise: float = 0.02
    lidar_noise_m: float = 0.02
    """Plus lidar_noise_frac of the range."""
    lidar_noise_frac: float = 0.01
    lidar_dropout: float = 0.02
    accel_noise: float = 0.2
    accel_bias: float = 0.2
    gyro_noise: float = 0.02
    gyro_bias: float = 0.01
    gravity_noise: float = 0.02
    steer_noise: float = 0.005
    speed_noise: float = 0.05
    speed_scale: tuple[float, float] = (0.95, 1.05)


class Sim2Real(VecEnvWrapper):
    def __init__(self, venv: VecEnv, config: Sim2RealConfig = Sim2RealConfig(), seed: int = 0):
        super().__init__(venv)
        self.config = config
        self._rng = np.random.default_rng(seed)
        n, c = venv.num_envs, config
        self._sent = np.zeros((c.max_delay_steps + 1, n, 2), np.float32)
        """The actions of the last steps, the newest first."""
        self._trim = np.zeros(n)
        self._delay = np.zeros(n, np.int64)
        self._accel_bias = np.zeros((n, 3))
        self._gyro_bias = np.zeros((n, 3))
        self._speed_scale = np.ones(n)
        self._draw(np.ones(n, np.bool_))

    def _draw(self, rows: np.ndarray) -> None:
        """A new episode's trim, delay, biases and scale for the envs where
        rows, which have sent no action yet."""
        c, rng, k = self.config, self._rng, int(rows.sum())
        self._trim[rows] = rng.uniform(-c.steer_trim, c.steer_trim, k)
        self._delay[rows] = rng.integers(0, c.max_delay_steps + 1, k)
        self._accel_bias[rows] = rng.uniform(-c.accel_bias, c.accel_bias, (k, 3))
        self._gyro_bias[rows] = rng.uniform(-c.gyro_bias, c.gyro_bias, (k, 3))
        self._speed_scale[rows] = rng.uniform(*c.speed_scale, k)
        self._sent[:, rows] = 0.0

    def _noisy(self, obs: np.ndarray, rows: np.ndarray) -> np.ndarray:
        """obs, the observations of the envs rows, with noise."""
        c, rng, n = self.config, self._rng, len(obs)
        out = obs.astype(np.float64)
        lidar = out[:, :8]
        hit = lidar >= 0
        spread = c.lidar_noise_m + c.lidar_noise_frac * lidar
        lidar = np.where(hit, np.maximum(lidar + rng.normal(0, 1, (n, 8)) * spread, 0), lidar)
        out[:, :8] = np.where(rng.random((n, 8)) < c.lidar_dropout, -1.0, lidar)
        out[:, 8:11] += self._accel_bias[rows] + rng.normal(0, c.accel_noise, (n, 3))
        out[:, 11:14] += self._gyro_bias[rows] + rng.normal(0, c.gyro_noise, (n, 3))
        out[:, GRAVITY] += rng.normal(0, c.gravity_noise, (n, 3))
        out[:, 17] += rng.normal(0, c.steer_noise, n)
        out[:, 18] = out[:, 18] * self._speed_scale[rows] + rng.normal(0, c.speed_noise, n)
        return out.astype(np.float32)

    def reset(self) -> VecEnvObs:
        obs = self.venv.reset()
        self._draw(np.ones(self.num_envs, np.bool_))
        return self._noisy(np.asarray(obs), np.arange(self.num_envs))

    def step_async(self, actions: np.ndarray) -> None:
        action = actions + self._rng.normal(0, self.config.action_noise, actions.shape)
        action[:, 1] += self._trim
        self._sent = np.roll(self._sent, 1, axis=0)
        self._sent[0] = action
        self.venv.step_async(self._sent[self._delay, np.arange(self.num_envs)])

    def step_wait(self) -> VecEnvStepReturn:
        obs, reward, done, infos = self.venv.step_wait()
        for i in np.flatnonzero(done):
            last = infos[i]["terminal_observation"][None]
            infos[i]["terminal_observation"] = self._noisy(last, np.array([i]))[0]
        self._draw(done)
        return self._noisy(np.asarray(obs), np.arange(self.num_envs)), reward, done, infos
