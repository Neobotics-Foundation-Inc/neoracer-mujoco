"""
Sim-to-real wrapper for NeoRacerEnv: domain randomization, sensor noise,
actuation latency, and a heading-free observation.

A policy trained on a perfect simulator learns its exact physics and
noise-free sensors, then fails on the real car. This wrapper makes every
episode a slightly different car and degrades what the policy sees, so it
has to learn behavior that works across the whole range -- which the real
car should fall inside.

    from neoracer_rl.envs import NeoRacerEnv
    from neoracer_rl.envs.sim2real import Sim2RealWrapper
    env = Sim2RealWrapper(NeoRacerEnv())

The base env is not changed. Every model parameter is resampled from its
nominal (as-compiled) value at each reset, so randomization never drifts.

Per episode (reset):
    physics        the upstream repo's domain randomization, ported in
                   physics.py: wheel friction, chassis mass, motor torque and
                   steering stiffness, each a uniform factor on the nominal
                   car (Sim2RealConfig.physics; physics.GENTLE_RANGES)
    steer trim     constant steering offset, like a misaligned servo horn
    sensor biases  constant IMU offsets and a wheel-speed scale error
    delay          actions arrive 0..max_delay_steps control steps late

Per step:
    noise on every observation, LiDAR beams dropping out (-1, "no hit"),
    and noise on the commanded action.

Observation: 19-D instead of 20-D. The world-frame quaternion (4) becomes
gravity in the body frame (3). It still tells the policy roll and pitch
(is the car tipping?), but not heading, which the real IMU can't measure
without drift and which the policy would otherwise memorize for this track.
    [0:8]   lidar ranges (m, -1 = no hit)
    [8:11]  imu_accel (m/s^2)
    [11:14] imu_gyro (rad/s)
    [14:17] gravity direction in the body frame (unit vector; (0,0,-1) level)
    [17]    steer_cmd_pos (rad)
    [18]    estimated speed (m/s)

All ranges are first guesses, not measurements from the real car. Measure
what you can (mass, top speed, servo response, LiDAR noise) and narrow them.
"""

import weakref
from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from neoracer_mujoco import contract
from .physics import GENTLE_RANGES, PhysicsRanges, apply_physics, car_parts, sample, set_const

OBS_DIM = 19

type Range = tuple[float, float]


@dataclass(frozen=True)
class Sim2RealConfig:
    """Randomization ranges. Scales multiply the nominal value; the three
    switches turn each group off (e.g. for evaluating on the nominal car)."""

    randomize_physics: bool = True
    sensor_noise: bool = True
    action_delay: bool = True

    # --- physics, sampled per episode: the upstream domain randomization ----
    # (physics.py). Upstream's default randomizes nothing; GENTLE_RANGES is
    # this project's choice: grip ~0.6-1.0 around the realistic drive's 0.8,
    # mass and motor +-10%, steering stiffness +-15%.
    physics: PhysicsRanges = GENTLE_RANGES
    steer_trim_rad: float = 0.02  # +- constant steering offset

    # --- actuation ----------------------------------------------------------
    max_delay_steps: int = 1  # 0..1 steps = 0..34 ms at 30 Hz
    action_noise: float = 0.02  # std, in normalized action units

    # --- sensors (std per step unless noted) --------------------------------
    lidar_noise_m: float = 0.02  # plus lidar_noise_frac of the range
    lidar_noise_frac: float = 0.01
    lidar_dropout: float = 0.02  # probability a beam reads "no hit"
    accel_noise: float = 0.2  # m/s^2
    accel_bias: float = 0.2  # +- m/s^2, per episode
    gyro_noise: float = 0.02  # rad/s
    gyro_bias: float = 0.01  # +- rad/s, per episode
    gravity_noise: float = 0.02  # ~1 degree of roll/pitch error
    steer_noise: float = 0.005  # rad
    speed_noise: float = 0.05  # m/s
    speed_scale: Range = (0.95, 1.05)  # wheel radius / encoder error, per episode


def gravity_in_body(quat: np.ndarray) -> np.ndarray:
    """World gravity direction (0, 0, -1) expressed in the body frame, from a
    (w, x, y, z) body-to-world quaternion: -1 * the third row of R."""
    w, x, y, z = quat
    return -np.array([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)])


class Sim2RealWrapper(gym.Wrapper):
    """See the module docstring."""

    def __init__(self, env: gym.Env, config: Sim2RealConfig = Sim2RealConfig()):
        super().__init__(env)
        self.config = config
        self._model = None
        # Nominal values per model, captured the first time a model is seen --
        # before it is ever randomized. Envs that switch models (one per track)
        # revisit models that still hold the last episode's randomized values,
        # so re-capturing then would compound. Weak keys: replaced tracks free.
        self._nominals: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        self._capture_nominal(env.unwrapped.model)

        base_space = env.observation_space
        low = np.concatenate([base_space.low[0:14], np.full(3, -1.5), base_space.low[18:20]])
        high = np.concatenate([base_space.high[0:14], np.full(3, 1.5), base_space.high[18:20]])
        self.observation_space = spaces.Box(low.astype(np.float32), high.astype(np.float32), dtype=np.float32)

        self._rng = np.random.default_rng()
        self._episode: dict = {}
        self._queue: deque = deque()

    def _capture_nominal(self, m: mujoco.MjModel) -> None:
        """Make m the model to randomize, with its nominal values: copied the
        first time m is seen, reused after that (see __init__)."""
        self._model = m
        self._scratch = None  # an MjData of m for mj_setConst, made when needed
        if m in self._nominals:
            self._parts, self._steer_limit = self._nominals[m]
            return
        self._parts = car_parts(m)  # also gives the wheels contact priority
        from .car_model import car_info

        self._steer_limit = car_info(m).steer_limit
        self._nominals[m] = (self._parts, self._steer_limit)

    # --- Gymnasium API --------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        base = self.env.unwrapped
        if hasattr(base, "select_track"):  # pick the episode's model before randomizing it
            base.select_track(seed)
        if base.model is not self._model:
            self._capture_nominal(base.model)
        self._sample_episode()
        if self.config.randomize_physics:
            self._apply_physics()
        # The base env settles the car after this, so it settles on the new physics.
        obs, info = self.env.reset(seed=seed, options=options)
        self._queue = deque([np.zeros(2, dtype=np.float32)] * self._episode["delay"])
        info["sim2real"] = self.episode_params
        return self._observe(obs), info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).copy()
        if self.config.sensor_noise:
            action += self._rng.normal(0.0, self.config.action_noise, 2).astype(np.float32)
        action[1] += self._episode["trim"]
        self._queue.append(action)
        applied = self._queue.popleft()  # delay 0: this step's action
        obs, reward, terminated, truncated, info = self.env.step(applied)
        return self._observe(obs), reward, terminated, truncated, info

    @property
    def episode_params(self) -> dict[str, float]:
        """This episode's sampled values, for logging."""
        return {k: float(v) if np.isscalar(v) else v.tolist() for k, v in self._episode.items()}

    # --- internals -------------------------------------------------------------

    def _uniform(self, r: Range | float) -> float:
        lo, hi = (-r, r) if np.isscalar(r) else r
        return float(self._rng.uniform(lo, hi))

    def _sample_episode(self) -> None:
        c = self.config
        phys, noise = c.randomize_physics, c.sensor_noise
        physics = sample(c.physics, self._rng) if phys else sample(PhysicsRanges(), self._rng)
        self._physics = physics
        self._episode = dict(
            wheel_friction=physics.wheel_friction,
            chassis_mass=physics.chassis_mass,
            motor_torque=physics.motor_torque,
            steer_stiffness=physics.steer_stiffness,
            # Trim is stored in normalized action units (action[1] * steer limit = rad).
            trim=self._uniform(c.steer_trim_rad) / self._trim_limit() if phys else 0.0,
            delay=int(self._rng.integers(0, c.max_delay_steps + 1)) if c.action_delay else 0,
            accel_bias=self._rng.uniform(-c.accel_bias, c.accel_bias, 3) if noise else np.zeros(3),
            gyro_bias=self._rng.uniform(-c.gyro_bias, c.gyro_bias, 3) if noise else np.zeros(3),
            speed_scale=self._uniform(c.speed_scale) if noise else 1.0,
        )

    def _trim_limit(self) -> float:
        """rad of steering per unit of steering action."""
        drive = getattr(self.env.unwrapped, "_drive", None)
        return drive.steer_limit_rad if drive is not None else self._steer_limit

    def _apply_physics(self) -> None:
        m = self.env.unwrapped.model
        apply_physics(m, self._parts, self._physics, getattr(self.env.unwrapped, "_drive", None))
        if self._scratch is None:
            self._scratch = mujoco.MjData(m)
        set_const(m, self._scratch)

    def _observe(self, obs: np.ndarray) -> np.ndarray:
        """20-D base observation -> 19-D noisy, heading-free observation."""
        lidar = obs[0:8].astype(np.float64)
        accel, gyro = obs[8:11].astype(np.float64), obs[11:14].astype(np.float64)
        gravity = gravity_in_body(obs[14:18].astype(np.float64))
        steer, speed = float(obs[18]), float(obs[19])

        if self.config.sensor_noise:
            c, e, rng = self.config, self._episode, self._rng
            hit = lidar >= 0
            lidar = np.where(hit, lidar + rng.normal(0, 1, 8) * (c.lidar_noise_m + c.lidar_noise_frac * lidar), lidar)
            lidar = np.where(hit, np.maximum(lidar, 0.0), lidar)
            lidar = np.where(rng.random(8) < c.lidar_dropout, -1.0, lidar)
            accel = accel + e["accel_bias"] + rng.normal(0, c.accel_noise, 3)
            gyro = gyro + e["gyro_bias"] + rng.normal(0, c.gyro_noise, 3)
            gravity = gravity + rng.normal(0, c.gravity_noise, 3)
            steer += rng.normal(0, c.steer_noise)
            speed = speed * e["speed_scale"] + rng.normal(0, c.speed_noise)

        out = np.concatenate([lidar, accel, gyro, gravity, [steer], [speed]])
        return np.clip(out, self.observation_space.low, self.observation_space.high).astype(np.float32)
