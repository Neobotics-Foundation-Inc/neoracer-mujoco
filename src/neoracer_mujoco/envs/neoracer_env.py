"""
Gymnasium environment for the NeoRacer MuJoCo car.

Wraps the neoracer_mujoco toolbox (assets/sim/sensors/collision/contract) in
the standard Gymnasium Env API. No new physics, no new sensors, no new
actuator mapping beyond what contract.py already documents -- this is a thin
RL-facing shell around the same simulator the classical controllers in
control/ drive.

Frame: +X forward, +Y left, +Z up (see assets/neoracer.xml).

Track: assets/tracks/loop_corridor.xml only.

Action space -- Box([-1,-1],[1,1]), float32:
    action[0] = normalized throttle, + = forward
    action[1] = normalized steer,    + = LEFT

    This uses the native NeoRacer contract convention (contract.py:
    ctrl[0:4] +torque=forward, ctrl[4] +angle=left), not racecar_core's
    convention (+angle=right). racecar_core_adapter.MujocoDrive is
    deliberately NOT reused here: it negates steer to cross into
    racecar_core's convention, which would reverse this env's action[1]
    sign relative to ctrl[4]'s own documented meaning. The mapping below
    (clamp, then scale by each actuator's own ctrlrange, read from the
    compiled model rather than hardcoded) mirrors MujocoDrive's pattern
    without its sign flip.

Observation space -- Box(20,), float32, built only from sensors.SensorReadings
(no raw qpos/qvel):
    [0:8]   lidar_ranges   m, -1 = no hit (sensors.lidar_scan)
    [8:11]  imu_accel      m/s^2, body frame, gravity included
    [11:14] imu_gyro       rad/s, body frame
    [14:18] imu_quat       orientation quaternion (w,x,y,z), world frame
    [18]    steer_cmd_pos  rad, virtual steering input
    [19]    estimated_speed  m/s, from motor_encoder_reading (mean of the 4
            wheel encoders, matching the real car's single drive motor --
            NOT ground-truth imu_linvel)

Control rate: 30 Hz, the target LiDAR scan rate of the real hardware.
Physics substeps per action are computed from model.opt.timestep at
construction time, not assumed to be 0.002s -- see __init__.

Reward (RewardConfig, all weights explicit and UNTUNED, see _reward):
    r_t = progress_weight * delta_theta_t  -  contact_penalty * contact_t
    plus one-time terminal adjustments:
        + lap_bonus        on successful lap completion
        - rollover_penalty on rollover termination
        - stuck_penalty    on stuck termination
    delta_theta_t is the signed net change in cumulative angle (radians)
    around the loop's center this tick (see Progress).

Termination (terminated=True) vs. truncation (truncated=True):
    terminated: rollover (upright_cos below threshold), stuck (no net
    angular progress over a trailing window), or lap completion
    (cumulative angle >= 2*pi in the track's intended forward-travel
    direction -- empirically confirmed positive). See _termination.
    truncated: max_episode_steps reached.
    Ordinary wall contact is NOT terminal by itself -- it's recorded in
    info and penalized in the reward; stuck detection is the sole backstop
    for a car wedged against a wall.
"""

import os
from dataclasses import dataclass, field
from typing import ClassVar, Literal, assert_never

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces

from .. import assets, collision, contract, sensors, sim

_SUPPORTED_TRACK_FILENAME = "loop_corridor.xml"
_DEFAULT_TRACK = os.path.join(assets.ASSET_DIR, "tracks", _SUPPORTED_TRACK_FILENAME)

# Spawn offset (m) on loop_corridor.xml: its outer wall is at y=-3 and its
# inner wall at y=-2, so the 1.0 m wide bottom straight is centered at
# y=-2.5. At the origin, the car would spawn inside the inner wall.
_LOOP_SPAWN_OFFSET = (0.0, -2.5, 0.0)

OBS_DIM = 20

_ACTION_LOW = np.array([-1.0, -1.0], dtype=np.float32)
_ACTION_HIGH = np.array([1.0, 1.0], dtype=np.float32)

type TerminationReason = Literal["rollover", "stuck", "lap_complete"]


def _compose_scene(track_path: str, car_path: str) -> mujoco.MjModel:
    """Compose the car onto loop_corridor.xml at _LOOP_SPAWN_OFFSET, with
    the MjSpec.attach_body pattern of examples/track_centering_demo.py.

    Limit: examples/wall_follower_vs/scene_utils.py has a copy of this.
    If a second module in the package needs it, move it to
    neoracer_mujoco.assets.
    """
    scene = mujoco.MjSpec.from_file(track_path)
    car = mujoco.MjSpec.from_file(car_path)
    frame = scene.worldbody.add_frame()
    frame.pos = _LOOP_SPAWN_OFFSET
    frame.attach_body(car.body("car"), "", "")
    return scene.compile()


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
    """Reward weights -- explicit and deliberately UNTUNED. See the module
    docstring for the full formula."""

    progress_weight: float = 1.0  # reward per radian of signed progress around the loop
    contact_penalty: float = 0.1  # penalty per tick any car geom touches a wall
    rollover_penalty: float = 10.0  # one-time penalty on rollover termination
    stuck_penalty: float = 5.0  # one-time penalty on stuck termination
    lap_bonus: float = 10.0  # one-time bonus on successful lap completion


@dataclass(frozen=True)
class NeoRacerEnvConfig:
    """Environment configuration -- track/timing/termination/spawn knobs,
    separate from RewardConfig so reward tuning doesn't touch physics/task
    parameters and vice versa."""

    track_path: str = _DEFAULT_TRACK
    car_path: str = assets.DEFAULT_CAR
    control_hz: float = 30.0  # Hz
    max_episode_steps: int = 1800  # 60 s at 30 Hz
    spawn_lateral_range_m: float = 0.15  # m, +/- bound
    spawn_yaw_range_rad: float = 0.15  # rad, +/- bound
    upright_rollover_cos: float = 0.5  # cos(tilt) below this = rolled over
    stuck_window_s: float = 4.0  # s
    stuck_progress_rad: float = 0.05  # rad, min net progress over the window
    lidar_high_m: float = 50.0  # observation_space upper bound -- see module docstring
    reward: RewardConfig = field(default_factory=RewardConfig)

    def __post_init__(self):
        # The spawn offset, the progress around the world origin, the lap
        # and the reward are specific to the geometry of this one track.
        if not self.track_path.endswith(_SUPPORTED_TRACK_FILENAME):
            raise ValueError(
                "NeoRacerEnv currently only supports "
                f"assets/tracks/{_SUPPORTED_TRACK_FILENAME} "
                f"(got track_path={self.track_path!r}). The spawn offset, "
                "angular-progress/lap-fraction/lap-completion logic, and "
                "reward shaping are all specific to that track's geometry -- "
                "see the module docstring. Other tracks are not supported."
            )


_DEFAULT_CONFIG = NeoRacerEnvConfig()

# Observation bounds, with margin over the peaks seen in 80 adversarial
# episodes (full throttle, hard steering, wall crashes) on loop_corridor.xml:
#   imu_accel: ~511 m/s^2 peak in crashes
#   imu_gyro: ~42 rad/s peak when tumbling
#   steer_cmd_pos: ~0.87 rad peak; the 0.45 rad joint limit is soft in
#     MuJoCo, so crashes can push past it
# Re-check these if the track, spawn range, or car changes.
_ACCEL_BOUND = 1000.0  # m/s^2
_GYRO_BOUND = 100.0  # rad/s
_STEER_BOUND = 1.0  # rad
_SPEED_BOUND = 40.0  # m/s; matches validation/test_conformance.py's own top-speed
# glitch ceiling (test_top_speed_bounded); observed max here was ~27.6 m/s.


def _observation_space(cfg: NeoRacerEnvConfig) -> spaces.Box:
    low = np.concatenate(
        [
            np.full(8, -1.0),  # lidar: -1 is the sensor's own "no hit" value
            np.full(3, -_ACCEL_BOUND),  # imu_accel (m/s^2)
            np.full(3, -_GYRO_BOUND),  # imu_gyro (rad/s)
            np.full(4, -1.0),  # imu_quat (unit quaternion component range)
            np.array([-_STEER_BOUND]),  # steer_cmd_pos (rad)
            np.array([-_SPEED_BOUND]),  # estimated_speed (m/s)
        ]
    ).astype(np.float32)
    high = np.concatenate(
        [
            np.full(8, cfg.lidar_high_m),
            np.full(3, _ACCEL_BOUND),
            np.full(3, _GYRO_BOUND),
            np.full(4, 1.0),
            np.array([_STEER_BOUND]),
            np.array([_SPEED_BOUND]),
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
    cfg: NeoRacerEnvConfig, upright_cos: float, stuck: bool, total_progress: float
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

    One MjModel is compiled once at construction and never mutated;
    reset() only rewrites MjData (mj_resetData + settle + spawn
    perturbation), so episodes cannot leak model-level state into each
    other. All randomness comes from self.np_random (Gymnasium's per-
    instance seeded generator), never global numpy random state.
    """

    metadata: ClassVar[dict] = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        config: NeoRacerEnvConfig = _DEFAULT_CONFIG,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.config = config
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode: {render_mode!r}")
        self.render_mode = render_mode

        self.model = _compose_scene(config.track_path, config.car_path)
        self.data = mujoco.MjData(self.model)
        self._car_body = self.model.body("car").id
        self._walls = collision.wall_geom_ids(self.model)
        self._sensor_layout = sensors.sensor_layout(self.model)

        physics_dt = float(self.model.opt.timestep)
        self._substeps_per_action = max(
            1, round((1.0 / config.control_hz) / physics_dt)
        )
        # The true control period, after rounding to a whole number of
        # physics substeps -- may differ slightly from 1/control_hz.
        self.effective_control_dt = self._substeps_per_action * physics_dt

        self._torque_limit = float(self.model.actuator("fl_motor").ctrlrange[1])
        self._steer_limit = float(self.model.actuator("steer_servo").ctrlrange[1])

        self.action_space = spaces.Box(
            low=_ACTION_LOW, high=_ACTION_HIGH, dtype=np.float32
        )
        self.observation_space = _observation_space(config)

        self._stuck_window_ticks = max(
            1, round(config.stuck_window_s * config.control_hz)
        )

        self._viewer = None

        # Episode state -- placeholder values until the first reset().
        self._step_count = 0
        self._episode_time_s = 0.0
        self._progress = _start_progress(0.0)

    # --- internal helpers -----------------------------------------------

    def _apply_action(self, action: np.ndarray) -> None:
        throttle, steer = float(action[0]), float(action[1])
        self.data.ctrl[contract.DRIVE] = throttle * self._torque_limit
        self.data.ctrl[contract.STEER] = steer * self._steer_limit

    def _angle_around_center(self) -> float:
        """Angle (rad) of the car's position around the world origin."""
        xpos = self.data.xpos[self._car_body]
        return float(np.arctan2(xpos[1], xpos[0]))

    def _upright_cos(self) -> float:
        """cos(tilt) of the car body, as in sim.car_upright_cos."""
        return float(self.data.xmat[self._car_body][8])

    def _build_observation(self) -> np.ndarray:
        readings = sensors.read_layout(self._sensor_layout, self.data)
        lidar = sensors.lidar_scan(readings).ranges.astype(np.float32)
        accel = readings.imu_accel.astype(np.float32)
        gyro = readings.imu_gyro.astype(np.float32)
        quat = readings.imu_quat.astype(np.float32)
        steer_cmd = readings.steer_cmd_pos.astype(np.float32)
        encoder = sensors.motor_encoder_reading(readings)
        speed = np.array([sensors.estimated_linear_speed_ms(encoder)], dtype=np.float32)
        return np.concatenate([lidar, accel, gyro, quat, steer_cmd, speed])

    def _build_info(
        self,
        obs: np.ndarray,
        contact: bool,
        upright_cos: float,
        reason: TerminationReason | None,
    ) -> dict:
        return {
            "progress_rad": self._progress.total,
            "lap_frac": abs(self._progress.total) / (2.0 * np.pi),
            "speed_ms": float(obs[19]),
            "upright_cos": upright_cos,
            "contact": contact,
            "episode_time_s": self._episode_time_s,
            "termination_reason": reason,
        }

    # --- Gymnasium API ----------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed, options=options)

        mujoco.mj_resetData(self.model, self.data)
        sim.run(self.model, self.data, 0.0, sim.SETTLE_STEPS)

        lateral = float(
            self.np_random.uniform(
                -self.config.spawn_lateral_range_m, self.config.spawn_lateral_range_m
            )
        )
        yaw = float(
            self.np_random.uniform(
                -self.config.spawn_yaw_range_rad, self.config.spawn_yaw_range_rad
            )
        )
        self.data.qpos[1] += lateral
        self.data.qpos[3:7] = _compose_yaw_perturbation(self.data.qpos[3:7], yaw)
        mujoco.mj_forward(self.model, self.data)
        for _ in range(50):  # re-settle suspension after the perturbation
            mujoco.mj_step(self.model, self.data)

        self._step_count = 0
        self._episode_time_s = 0.0
        self._progress = _start_progress(self._angle_around_center())

        obs = self._build_observation()
        info = self._build_info(
            obs, contact=False, upright_cos=self._upright_cos(), reason=None
        )
        return obs, info

    def step(self, action):
        action = np.clip(
            np.asarray(action, dtype=np.float32), _ACTION_LOW, _ACTION_HIGH
        )
        self._apply_action(action)

        contact_this_tick = False
        for _ in range(self._substeps_per_action):
            mujoco.mj_step(self.model, self.data)
            if collision.car_touches_wall(
                self.model, self.data, self._car_body, self._walls
            ):
                contact_this_tick = True

        self._step_count += 1
        self._episode_time_s = self._step_count * self.effective_control_dt
        window_len = self._stuck_window_ticks + 1
        self._progress = _advance(
            self._progress, self._angle_around_center(), window_len
        )

        upright_cos = self._upright_cos()
        stuck = _is_stuck(self._progress, window_len, self.config.stuck_progress_rad)
        reason = _termination(self.config, upright_cos, stuck, self._progress.total)
        terminated = reason is not None
        truncated = (
            not terminated
        ) and self._step_count >= self.config.max_episode_steps

        reward = _reward(
            self.config.reward, self._progress.delta, contact_this_tick, reason
        )
        obs = self._build_observation()
        info = self._build_info(obs, contact_this_tick, upright_cos, reason)

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
