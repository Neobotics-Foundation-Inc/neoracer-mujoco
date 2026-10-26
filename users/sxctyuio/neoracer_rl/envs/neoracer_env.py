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
    [0:8]   lidar_ranges   m, -1 = no hit (sensors.lidar_ranges)
    [8:11]  imu_accel      m/s^2, body frame, gravity included
    [11:14] imu_gyro       rad/s, body frame
    [14:18] imu_quat       orientation quaternion (w,x,y,z), world frame
    [18]    steer_cmd_pos  rad, virtual steering input
    [19]    estimated_speed  m/s, from motor_encoder_rad_s (mean of the 4
            wheel encoders, matching the real car's single drive motor --
            NOT ground-truth imu_linvel)

Control rate: 30 Hz, the target LiDAR scan rate of the real hardware.
Physics substeps per action are computed from model.opt.timestep at
construction time, not assumed to be 0.002s -- see __init__.

Reward (RewardConfig, all weights explicit and UNTUNED, see _reward):
    r_t = progress_weight * delta_theta_t
          + corner_weight * corner_closeness_t * max(delta_theta_t, 0)
          - contact_penalty * contact_t
    plus one-time terminal adjustments:
        + lap_bonus        on successful lap completion
        - rollover_penalty on rollover termination
        - stuck_penalty    on stuck termination
        - wall_penalty     on wall termination
    delta_theta_t is the signed net change in cumulative angle (radians)
    around the loop's center this tick (see Progress).
    corner_closeness_t is in [0, 1]: exp(-(d / corner_radius_m)^2), with d
    the distance from the car to the nearest inner-corner apex (_APEXES).
    It multiplies forward progress, so the bonus is earned by driving
    through a corner tight, not by parking next to the apex.

Termination (terminated=True) vs. truncation (truncated=True):
    terminated: rollover (upright_cos below threshold), wall contact (if
    TerminationConfig.terminate_on_wall), stuck (no net angular progress
    over a trailing window), or lap completion (cumulative angle >= 2*pi
    in the track's intended forward-travel direction -- empirically
    confirmed positive). See _termination.
    truncated: max_episode_steps reached.
    With terminate_on_wall=False, wall contact is recorded in info and
    penalized per tick instead, and stuck detection is the backstop for a
    car wedged against a wall.
"""

import os
from dataclasses import dataclass
from typing import Any, Literal, assert_never
from typing_extensions import TypedDict

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces

from neoracer_mujoco import assets, collision, contract, sim
from .. import sensors
from .car_model import apply_car_options, car_info
from .drive_model import DriveModel, DriveModelConfig, apply_surface_friction, clearance_cost, mask_rear, wall_clearance

# The spawn offset, the progress around the world origin, the lap and the
# reward are specific to the geometry of this one track.
_TRACK = os.path.join(assets.ASSET_DIR, "tracks", "loop_corridor.xml")

# Spawn offset (m) on loop_corridor.xml: its outer wall is at y=-3 and its
# inner wall at y=-2, so the 1.0 m wide bottom straight is centered at
# y=-2.5. At the origin, the car would spawn inside the inner wall.
_LOOP_SPAWN_OFFSET = (0.0, -2.5, 0.0)

# Inner-corner apexes (m) of loop_corridor.xml: the midpoints of the inner
# wall's 1 m 45-degree chamfers. The inner wall is x:[-3,3] y:[-2,2] with its
# corners cut along |x| + |y| = 4, i.e. from (3, 1) to (2, 2) top-right.
# The car's center can't get much closer than ~0.16 m (half its width plus
# the wall's half thickness), so closeness tops out around 0.9.
_APEXES = np.array([[2.5, 1.5], [-2.5, 1.5], [-2.5, -1.5], [2.5, -1.5]])

OBS_DIM = 20

_ACTION_LOW = np.array([-1.0, -1.0], dtype=np.float32)
_ACTION_HIGH = np.array([1.0, 1.0], dtype=np.float32)

type TerminationReason = Literal["rollover", "wall", "stuck", "lap_complete"]


class EnvInfo(TypedDict, closed=True):
    """The info dict of reset() and step(). gymnasium requires a dict."""

    progress_rad: float  # rad; cumulative signed progress in this episode
    lap_frac: float  # laps; abs(progress_rad) / 2 pi
    speed_ms: float  # m/s; estimated_speed, observation [19]
    upright_cos: float  # cos(tilt) of the car body
    contact: bool  # a car geom touched a wall in this step
    corner_closeness: float  # 0..1; 1 = at an inner-corner apex. See _corner_closeness
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
    """Reward weights -- explicit and deliberately UNTUNED. See the module
    docstring for the full formula."""

    progress_weight: float = 1.0  # reward per radian of signed progress around the loop
    contact_penalty: float = 0.1  # penalty per tick any car geom touches a wall
    rollover_penalty: float = 10.0  # one-time penalty on rollover termination
    stuck_penalty: float = 5.0  # one-time penalty on stuck termination
    lap_bonus: float = 10.0  # one-time bonus on successful lap completion
    # One-time penalty on wall termination. Keep it <= stuck_penalty: if a
    # crash after real progress costs more than sitting still until the stuck
    # check fires, early policies learn to stall instead of drive.
    wall_penalty: float = 5.0
    # Extra reward per radian of progress at an apex (scaled down by
    # corner_closeness away from it). 0.5 = up to ~45% more for a tight line
    # through each corner. 0 disables.
    corner_weight: float = 0.5
    corner_radius_m: float = 0.5  # m; closeness falls to 1/e at this distance
    # Wall clearance: a per-step cost for running closer than wall_margin_m to
    # a wall (LiDAR range minus the car's own extent along each beam, rear beam
    # excluded): wall_clearance_weight at contact, 0 at the margin. Teaches the
    # car to keep space instead of learning about walls only by crashing.
    # 0 disables (train_ppo.py turns it on with --wall-clearance-weight).
    wall_margin_m: float = 0.08
    wall_clearance_weight: float = 0.0


@dataclass(frozen=True)
class SpawnRange:
    """Bounds of the random spawn perturbation at reset(). Each value is
    sampled uniformly in [-bound, +bound]."""

    lateral_m: float = 0.15  # m, along world Y
    yaw_rad: float = 0.15  # rad, about +Z


@dataclass(frozen=True)
class TerminationConfig:
    """Thresholds that end an episode. See _termination."""

    upright_rollover_cos: float = 0.5  # cos(tilt) below this = rolled over
    stuck_window_s: float = 4.0  # s
    stuck_progress_rad: float = 0.05  # rad, min net progress over the window
    terminate_on_wall: bool = True  # any wall contact ends the episode


@dataclass(frozen=True)
class NeoRacerEnvConfig:
    """Environment configuration. The track is always loop_corridor.xml."""

    car_path: str = assets.DEFAULT_CAR
    control_hz: float = 30.0  # Hz
    max_episode_steps: int = 1800  # 60 s at 30 Hz
    lidar_high_m: float = 50.0  # observation_space upper bound -- see module docstring
    spawn: SpawnRange = SpawnRange()
    termination: TerminationConfig = TerminationConfig()
    reward: RewardConfig = RewardConfig()
    # How actions drive the car (drive_model.py): a DC-motor throttle that
    # settles at a speed, steering capped at the real 18.75 deg, LiDAR blind
    # straight back -- matched to the real car. None: the original model
    # (throttle = raw +-1 N.m per wheel, +-0.4 rad steering, 360 deg LiDAR).
    drive: DriveModelConfig | None = DriveModelConfig()


# Observation bounds, with margin over the peaks seen in 160 random-action
# episodes (full throttle, hard steering, wall crashes) on loop_corridor.xml,
# with the elliptic friction cone and impratio=10:
#   imu_accel: ~517 m/s^2 peak in crashes
#   imu_gyro: ~39 rad/s peak when tumbling
#   steer_cmd_pos: ~1.2 rad peak in a wall crash. The 0.45 rad joint limit is
#     soft in MuJoCo, so a crash can push past it; without wall contact the
#     peak is ~0.47 rad.
# Re-check these if the track, spawn range, or car changes.
_ACCEL_BOUND = 1000.0  # m/s^2
_GYRO_BOUND = 100.0  # rad/s
_STEER_BOUND = 2.0  # rad
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
    cfg: TerminationConfig,
    upright_cos: float,
    contact: bool,
    stuck: bool,
    total_progress: float,
) -> TerminationReason | None:
    """The reason the episode ends at this step, or None if it continues."""
    if upright_cos < cfg.upright_rollover_cos:
        return "rollover"
    if cfg.terminate_on_wall and contact:
        return "wall"
    if stuck:
        return "stuck"
    if total_progress >= 2.0 * np.pi:
        return "lap_complete"
    return None


def _corner_closeness(xy: np.ndarray, radius_m: float) -> float:
    """1 at an inner-corner apex, falling off as a Gaussian with distance."""
    d = float(np.min(np.linalg.norm(_APEXES - xy, axis=1)))
    return float(np.exp(-((d / radius_m) ** 2)))


def _reward(
    cfg: RewardConfig,
    delta: float,
    contact: bool,
    closeness: float,
    reason: TerminationReason | None,
) -> float:
    r = cfg.progress_weight * delta
    r += cfg.corner_weight * closeness * max(delta, 0.0)
    if contact:
        r -= cfg.contact_penalty
    match reason:
        case None:
            pass
        case "rollover":
            r -= cfg.rollover_penalty
        case "wall":
            r -= cfg.wall_penalty
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
        apply_car_options(self.model, config.car_path)  # GitHub car: its timestep/integrator, before use
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

        self._car = car_info(self.model)  # GitHub car (DC motor) or legacy car (4 torque motors)
        self._torque_limit = self._car.throttle_limit
        self._steer_limit = self._car.steer_limit
        self._drive = DriveModel(self.model, config.drive) if config.drive is not None else None
        apply_surface_friction(self.model, config.drive)

        self.action_space = spaces.Box(
            low=_ACTION_LOW, high=_ACTION_HIGH, dtype=np.float32
        )
        self.observation_space = _observation_space(config)

        self._stuck_window_ticks = max(
            1, round(config.termination.stuck_window_s * config.control_hz)
        )

        self._viewer = None

        # Episode state -- placeholder values until the first reset().
        self._step_count = 0
        self._episode_time_s = 0.0
        self._progress = _start_progress(0.0)

    # --- internal helpers -----------------------------------------------

    def _apply_action(self, action: np.ndarray) -> None:
        throttle, steer = float(action[0]), float(action[1])
        if self._drive is not None:  # realistic drive: torques are set every substep
            self._drive.command(throttle, steer, self.effective_control_dt)
            return
        self.data.ctrl[self._car.drive_ids] = throttle * self._torque_limit
        self.data.ctrl[self._car.steer_id] = steer * self._steer_limit

    def _wall_clearance(self) -> float:
        """Gap (m) to the nearest wall from the clean LiDAR (see RewardConfig)."""
        ranges = sensors.lidar_ranges(sensors.read_layout(self._sensor_layout, self.data))
        return wall_clearance(ranges, extent=self._car.beam_extent)

    def _physics_step(self) -> None:
        """One physics substep, with the drive model's torques for the current wheel spin."""
        if self._drive is not None:
            self._drive.apply(self.data)
        mujoco.mj_step(self.model, self.data)

    def _angle_around_center(self) -> float:
        """Angle (rad) of the car's position around the world origin."""
        xpos = self.data.xpos[self._car_body]
        return float(np.arctan2(xpos[1], xpos[0]))

    def _upright_cos(self) -> float:
        """cos(tilt) of the car body, as in sim.car_upright_cos."""
        return float(self.data.xmat[self._car_body][8])

    def _build_observation(self) -> np.ndarray:
        readings = sensors.read_layout(self._sensor_layout, self.data)
        lidar = sensors.lidar_ranges(readings).astype(np.float32)
        if self._drive is not None and self._drive.cfg.rear_blind:
            lidar = mask_rear(lidar)  # the real LiDAR does not see straight back
        accel = readings.imu_accel.astype(np.float32)
        gyro = readings.imu_gyro.astype(np.float32)
        quat = readings.imu_quat.astype(np.float32)
        steer_cmd = readings.steer_cmd_pos.astype(np.float32)
        motor_rad_s = sensors.motor_encoder_rad_s(readings)
        speed = np.array(
            [motor_rad_s * self._car.wheel_radius], dtype=np.float32  # this car's wheels
        )
        return np.concatenate([lidar, accel, gyro, quat, steer_cmd, speed])

    def _build_info(
        self,
        obs: np.ndarray,
        contact: bool,
        upright_cos: float,
        closeness: float,
        reason: TerminationReason | None,
    ) -> EnvInfo:
        return {
            "progress_rad": self._progress.total,
            "lap_frac": abs(self._progress.total) / (2.0 * np.pi),
            "speed_ms": float(obs[19]),
            "upright_cos": upright_cos,
            "contact": contact,
            "corner_closeness": closeness,
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
                -self.config.spawn.lateral_m, self.config.spawn.lateral_m
            )
        )
        yaw = float(
            self.np_random.uniform(
                -self.config.spawn.yaw_rad, self.config.spawn.yaw_rad
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
        if self._drive is not None:
            self._drive.reset()

        obs = self._build_observation()
        info = self._build_info(
            obs,
            contact=False,
            upright_cos=self._upright_cos(),
            closeness=_corner_closeness(
                self.data.xpos[self._car_body][:2], self.config.reward.corner_radius_m
            ),
            reason=None,
        )
        return obs, info

    def step(self, action):
        action = np.clip(
            np.asarray(action, dtype=np.float32), _ACTION_LOW, _ACTION_HIGH
        )
        self._apply_action(action)

        contact_this_tick = False
        for _ in range(self._substeps_per_action):
            self._physics_step()
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
        stuck = _is_stuck(
            self._progress, window_len, self.config.termination.stuck_progress_rad
        )
        reason = _termination(
            self.config.termination,
            upright_cos,
            contact_this_tick,
            stuck,
            self._progress.total,
        )
        terminated = reason is not None
        truncated = (
            not terminated
        ) and self._step_count >= self.config.max_episode_steps

        closeness = _corner_closeness(
            self.data.xpos[self._car_body][:2], self.config.reward.corner_radius_m
        )
        reward = _reward(
            self.config.reward,
            self._progress.delta,
            contact_this_tick,
            closeness,
            reason,
        )
        clearance = self._wall_clearance()
        reward -= clearance_cost(clearance, self.config.reward.wall_margin_m, self.config.reward.wall_clearance_weight)
        obs = self._build_observation()
        info = self._build_info(obs, contact_this_tick, upright_cos, closeness, reason)
        info["wall_clearance_m"] = clearance

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
