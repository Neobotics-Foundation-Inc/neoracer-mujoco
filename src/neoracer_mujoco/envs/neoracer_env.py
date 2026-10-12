"""
Gymnasium environment for the NeoRacer MuJoCo car.

This module gives the NeoRacer simulator the Gymnasium Env API. It uses the
toolbox modules (assets, sim, sensors, collision, contract) and adds no
physics, sensors or actuator mappings. The classical controllers in control/
drive the same simulator.

Frame: +X forward, +Y left, +Z up (see assets/neoracer.xml).

Track: assets/tracks/loop_corridor.xml only.

Action space -- Box([-1,-1],[1,1]), float32:
    action[0] = normalized throttle, + = forward
    action[1] = normalized steer,    + = LEFT

    The signs are the ones in contract.py: + torque = forward, + steer = left.
    racecar_core uses + angle = right, so this env does not use
    racecar_core_adapter.MujocoDrive, which negates the steer. step() clamps
    the action and scales it by the actuator limits of the compiled model
    (contract.actuator_limits).

Observation space -- Box(20,), float32, built only from sensors.SensorReadings
(no raw qpos/qvel):
    [0:8]   lidar_ranges   m, -1 = no hit (sensors.lidar_ranges)
    [8:11]  imu_accel      m/s^2, body frame, gravity included
    [11:14] imu_gyro       rad/s, body frame
    [14:18] imu_quat       orientation quaternion (w,x,y,z), world frame
    [18]    steer_cmd_pos  rad, virtual steering input
    [19]    estimated_speed  m/s, from motor_encoder_rad_s: the mean of the
            4 wheel speeds, like the one drive motor of the real car. It is
            not the true speed imu_linvel.

Control rate: 30 Hz, the target LiDAR scan rate of the real hardware.
__init__ computes the physics steps per action from model.opt.timestep.

Reward (RewardConfig, all weights explicit and UNTUNED, see _reward):
    r_t = progress_weight * delta_theta_t  -  contact_penalty * contact_t
    plus one-time terminal adjustments:
        + lap_bonus        on successful lap completion
        - rollover_penalty on rollover termination
        - stuck_penalty    on stuck termination
    delta_theta_t is the signed net change in cumulative angle (radians)
    around the loop's center this tick (see Progress).

Termination (terminated=True) vs. truncation (truncated=True):
    terminated: rollover (upright_cos below the threshold), stuck (no net
    progress in a window of time), or a full lap (cumulative angle >= 2 pi
    in the forward direction of the track, which is +). See _termination.
    truncated: the episode reached max_episode_steps.
    A wall contact does not end the episode. The env records it in info and
    the reward has a penalty for it. If the car is stuck against a wall, the
    stuck check ends the episode.
"""

import os
from dataclasses import dataclass
from typing import Any, Literal, TypedDict, assert_never

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
import numpy.typing as npt
from gymnasium import spaces

from .. import assets, collision, contract, sensors, sim
from ..control.steering import ControlCommand

# The spawn offset, the progress around the world origin, the lap and the
# reward are specific to the geometry of this one track.
_TRACK = os.path.join(assets.TRACK_DIR, "loop_corridor.xml")

# Spawn offset (m) on loop_corridor.xml: its outer wall is at y=-3 and its
# inner wall at y=-2, so the 1.0 m wide bottom straight is centered at
# y=-2.5. At the origin, the car would spawn inside the inner wall.
_LOOP_SPAWN_OFFSET = (0.0, -2.5, 0.0)

OBS_DIM = 20
type Observation = npt.NDArray[np.float32]  # shape (OBS_DIM,), see module docstring

_ACTION_LOW = np.array([-1.0, -1.0], dtype=np.float32)
_ACTION_HIGH = np.array([1.0, 1.0], dtype=np.float32)

type TerminationReason = Literal["rollover", "stuck", "lap_complete"]


class EnvInfo(TypedDict, closed=True):
    """The info dict of reset() and step(). gymnasium requires a dict."""

    progress_rad: float  # rad; cumulative signed progress in this episode
    lap_frac: float  # laps; abs(progress_rad) / 2 pi
    speed_ms: float  # m/s; estimated_speed, observation [19]
    upright_cos: float  # cos(tilt) of the car body
    contact: bool  # a car geom touched a wall in this step
    episode_time_s: float  # s
    termination_reason: TerminationReason | None


def _compose_yaw_perturbation(q: np.ndarray, yaw: float) -> np.ndarray:
    """q * dq, where dq is a rotation of yaw (rad) about +Z. q and the
    result are quaternions (w, x, y, z)."""
    half = yaw / 2.0
    dq = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])
    out = np.empty(4)
    mujoco.mju_mulQuat(out, q, dq)
    return out


@dataclass(frozen=True)
class RewardConfig:
    """Reward weights. Nobody has tuned them yet. See the module docstring for
    the formula."""

    progress_weight: float = 1.0  # reward per radian of signed progress around the loop
    contact_penalty: float = 0.1  # penalty per tick any car geom touches a wall
    rollover_penalty: float = 10.0  # one-time penalty on rollover termination
    stuck_penalty: float = 5.0  # one-time penalty on stuck termination
    lap_bonus: float = 10.0  # one-time bonus on successful lap completion


@dataclass(frozen=True)
class SpawnRange:
    """Bounds of the random spawn perturbation at reset(). Each value is
    sampled uniformly in [-bound, +bound]."""

    lateral_m: float = 0.15  # m, along world Y
    yaw_rad: float = 0.15  # rad, about +Z
    resettle_steps: int = (
        50  # physics steps after the perturbation, to settle the suspension
    )


@dataclass(frozen=True)
class TerminationConfig:
    """Thresholds that end an episode. See _termination."""

    upright_rollover_cos: float = 0.5  # cos(tilt) below this = rolled over
    stuck_window_s: float = 4.0  # s
    stuck_progress_rad: float = 0.05  # rad, min net progress over the window


@dataclass(frozen=True)
class ObservationBounds:
    """Bounds of the observation space, with a margin over the peaks in 160
    random-action episodes (full throttle, hard steering, wall crashes) on
    loop_corridor.xml, with the elliptic friction cone and impratio=10.
    If the track, the spawn range or the car changes, measure the peaks again."""

    lidar_m: float = 50.0  # m
    accel_m_s2: float = 1000.0  # m/s^2; the crash peak is ~517 m/s^2
    gyro_rad_s: float = 100.0  # rad/s; the peak when the car tumbles is ~39 rad/s
    # rad; the crash peak is ~1.2 rad. MuJoCo's 0.45 rad joint limit is soft,
    # so a crash can push past it. Without wall contact the peak is ~0.47 rad.
    steer_rad: float = 2.0
    # m/s; the same limit as test_top_speed_bounded in
    # validation/test_conformance.py. The highest observed speed is 27.6 m/s.
    speed_m_s: float = 40.0


@dataclass(frozen=True)
class NeoRacerEnvConfig:
    """Environment configuration. The track is always loop_corridor.xml."""

    car_path: str = assets.DEFAULT_CAR
    control_hz: float = 30.0  # Hz
    max_episode_steps: int = 1800  # 60 s at 30 Hz
    obs_bounds: ObservationBounds = ObservationBounds()
    spawn: SpawnRange = SpawnRange()
    termination: TerminationConfig = TerminationConfig()
    reward: RewardConfig = RewardConfig()


def _observation_space(b: ObservationBounds) -> spaces.Box:
    low = np.concatenate(
        [
            np.full(8, -1.0),  # lidar: -1 is the sensor's own "no hit" value
            np.full(3, -b.accel_m_s2),  # imu_accel
            np.full(3, -b.gyro_rad_s),  # imu_gyro
            np.full(4, -1.0),  # imu_quat (unit quaternion component range)
            np.array([-b.steer_rad]),  # steer_cmd_pos
            np.array([-b.speed_m_s]),  # estimated_speed
        ]
    ).astype(np.float32)
    high = np.concatenate(
        [
            np.full(8, b.lidar_m),
            np.full(3, b.accel_m_s2),
            np.full(3, b.gyro_rad_s),
            np.full(4, 1.0),
            np.array([b.steer_rad]),
            np.array([b.speed_m_s]),
        ]
    ).astype(np.float32)
    return spaces.Box(low=low, high=high, shape=(OBS_DIM,), dtype=np.float32)


@dataclass(frozen=True)
class Progress:
    """Signed progress of the car around the loop center (the world origin;
    the track is symmetric about it). + is the forward-travel direction."""

    angle: float  # rad; angle of the car around the center, at the last step
    total: float  # rad; cumulative progress in this episode
    delta: float  # rad; progress in the last step
    window: tuple[float, ...]  # total at each of the last steps, oldest first


def _start_progress(angle: float) -> Progress:
    return Progress(angle=angle, total=0.0, delta=0.0, window=(0.0,))


def _advance(p: Progress, angle: float, window_len: int) -> Progress:
    """Progress after the car moved to angle (rad). Keeps the last
    window_len totals."""
    delta = angle - p.angle
    if delta > np.pi:
        delta -= 2.0 * np.pi
    elif delta < -np.pi:
        delta += 2.0 * np.pi
    total = p.total + delta
    return Progress(
        angle=angle, total=total, delta=delta, window=(*p.window, total)[-window_len:]
    )


def _is_stuck(p: Progress, window_len: int, min_progress: float) -> bool:
    if len(p.window) < window_len:
        return False
    return abs(p.total - p.window[0]) < min_progress


def _termination(
    cfg: TerminationConfig, upright_cos: float, stuck: bool, total_progress: float
) -> TerminationReason | None:
    """The reason the episode ends at this step, or None if it continues."""
    if upright_cos < cfg.upright_rollover_cos:
        return "rollover"
    if stuck:
        return "stuck"
    if total_progress >= 2.0 * np.pi:
        return "lap_complete"
    return None


def _reward(
    cfg: RewardConfig, delta: float, contact: bool, reason: TerminationReason | None
) -> float:
    r = cfg.progress_weight * delta
    if contact:
        r -= cfg.contact_penalty
    match reason:
        case None:
            pass
        case "rollover":
            r -= cfg.rollover_penalty
        case "stuck":
            r -= cfg.stuck_penalty
        case "lap_complete":
            r += cfg.lap_bonus
        case _:
            assert_never(reason)
    return float(r)


class NeoRacerEnv(gym.Env):
    """Gymnasium environment driving the NeoRacer car around
    assets/tracks/loop_corridor.xml.

    __init__ compiles one MjModel, and no method changes it. reset() writes
    only MjData (mj_resetData, settle, spawn perturbation), so one episode
    cannot change the model of the next. All random numbers come from
    self.np_random, the seeded generator of this instance.
    """

    # The same type as gym.Env.metadata, so not a ClassVar.
    metadata: dict[str, Any] = {"render_modes": ["human"], "render_fps": 30}  # noqa: RUF012

    def __init__(
        self,
        config: NeoRacerEnvConfig = NeoRacerEnvConfig(),
        render_mode: str | None = None,
    ):
        super().__init__()
        self.config = config
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode: {render_mode!r}")
        self.render_mode = render_mode

        self.model = assets.compose(
            mujoco.MjSpec.from_file(_TRACK), config.car_path, _LOOP_SPAWN_OFFSET
        )
        self.data = mujoco.MjData(self.model)
        self._car_body = self.model.body("car").id
        # qpos address of the car's freejoint: (x, y, z, qw, qx, qy, qz).
        joint = int(self.model.body_jntadr[self._car_body])
        if (
            self.model.body_jntnum[self._car_body] != 1
            or self.model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE
        ):
            raise ValueError(f"the car body in {config.car_path} needs one free joint")
        self._car_qpos = int(self.model.jnt_qposadr[joint])
        self._walls = collision.wall_geom_ids(self.model)
        self._sensor_layout = sensors.sensor_layout(self.model)

        physics_dt = float(self.model.opt.timestep)
        self._substeps_per_action = max(
            1, round((1.0 / config.control_hz) / physics_dt)
        )
        # The true control period, after rounding to a whole number of
        # physics substeps. It can differ slightly from 1/control_hz.
        self.effective_control_dt = self._substeps_per_action * physics_dt

        self._limits = contract.actuator_limits(self.model)

        self.action_space = spaces.Box(
            low=_ACTION_LOW, high=_ACTION_HIGH, dtype=np.float32
        )
        self.observation_space = _observation_space(config.obs_bounds)

        self._stuck_window_ticks = max(
            1, round(config.termination.stuck_window_s * config.control_hz)
        )

        self._viewer = None

        # Episode state. gymnasium requires reset() before step(), and
        # reset() sets these values again.
        self._step_count = 0
        self._progress = _start_progress(0.0)

    # --- internal helpers -----------------------------------------------

    def _angle_around_center(self) -> float:
        """Angle (rad) of the car's position around the world origin."""
        xpos = self.data.xpos[self._car_body]
        return float(np.arctan2(xpos[1], xpos[0]))

    def _build_observation(self) -> tuple[Observation, float]:
        """The observation, and the estimated speed (m/s) in it."""
        readings = sensors.read_layout(self._sensor_layout, self.data)
        lidar = sensors.lidar_ranges(readings).astype(np.float32)
        accel = readings.imu_accel.astype(np.float32)
        gyro = readings.imu_gyro.astype(np.float32)
        quat = readings.imu_quat.astype(np.float32)
        speed = sensors.estimated_linear_speed_ms(sensors.motor_encoder_rad_s(readings))
        scalars = np.array([readings.steer_cmd_pos, speed], dtype=np.float32)
        return np.concatenate([lidar, accel, gyro, quat, scalars]), speed

    def _build_info(
        self,
        speed: float,
        contact: bool,
        upright_cos: float,
        reason: TerminationReason | None,
    ) -> EnvInfo:
        return {
            "progress_rad": self._progress.total,
            "lap_frac": abs(self._progress.total) / (2.0 * np.pi),
            "speed_ms": speed,
            "upright_cos": upright_cos,
            "contact": contact,
            "episode_time_s": self._step_count * self.effective_control_dt,
            "termination_reason": reason,
        }

    # --- Gymnasium API ----------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed, options=options)

        mujoco.mj_resetData(self.model, self.data)
        sim.apply_ctrl_and_step(self.model, self.data, 0.0, sim.SETTLE_STEPS)

        lateral = float(
            self.np_random.uniform(
                -self.config.spawn.lateral_m, self.config.spawn.lateral_m
            )
        )
        yaw = float(
            self.np_random.uniform(
                -self.config.spawn.yaw_rad, self.config.spawn.yaw_rad
            )
        )
        q = self._car_qpos
        self.data.qpos[q + 1] += lateral
        self.data.qpos[q + 3 : q + 7] = _compose_yaw_perturbation(
            self.data.qpos[q + 3 : q + 7], yaw
        )
        mujoco.mj_forward(self.model, self.data)
        for _ in range(self.config.spawn.resettle_steps):
            mujoco.mj_step(self.model, self.data)

        self._step_count = 0
        self._progress = _start_progress(self._angle_around_center())

        obs, speed = self._build_observation()
        upright_cos = sim.car_upright_cos(self.data, self._car_body)
        info = self._build_info(
            speed, contact=False, upright_cos=upright_cos, reason=None
        )
        return obs, info

    def step(self, action):
        action = np.clip(
            np.asarray(action, dtype=np.float32), _ACTION_LOW, _ACTION_HIGH
        )
        sim.apply_command(
            self.data,
            ControlCommand(
                throttle=float(action[0]) * self._limits.torque_nm,
                steer=float(action[1]) * self._limits.steer_rad,
            ),
        )

        contact_this_tick = False
        for _ in range(self._substeps_per_action):
            mujoco.mj_step(self.model, self.data)
            if collision.car_touches_wall(
                self.model, self.data, self._car_body, self._walls
            ):
                contact_this_tick = True

        self._step_count += 1
        window_len = self._stuck_window_ticks + 1
        self._progress = _advance(
            self._progress, self._angle_around_center(), window_len
        )

        upright_cos = sim.car_upright_cos(self.data, self._car_body)
        stuck = _is_stuck(
            self._progress, window_len, self.config.termination.stuck_progress_rad
        )
        reason = _termination(
            self.config.termination, upright_cos, stuck, self._progress.total
        )
        terminated = reason is not None
        truncated = (
            not terminated
        ) and self._step_count >= self.config.max_episode_steps

        reward = _reward(
            self.config.reward, self._progress.delta, contact_this_tick, reason
        )
        obs, speed = self._build_observation()
        info = self._build_info(speed, contact_this_tick, upright_cos, reason)

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return
        if self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self._viewer.sync()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
