"""
Gymnasium environment for the NeoRacer MuJoCo car (issue #7).

Wraps the existing neoracer_mujoco toolbox (assets/sim/sensors/collision/
contract) in the standard Gymnasium Env API. No new physics, no new sensors,
no new actuator mapping beyond what contract.py already documents -- this is
a thin RL-facing shell around the same simulator every classical controller
in control/ already drives.

Track: assets/tracks/loop_corridor.xml (issue #7's first-pass scope). This
module does not import from, depend on, or modify examples/wall_follower_vs/
or vehicle_space_controller.py in any way -- see _compose_scene() and
_compose_yaw_perturbation() below for independently-reimplemented equivalents
of the small pieces of logic this env happens to need in common with that
experimental code.

Action space -- Box([-1,-1],[1,1]), float32:
    action[0] = normalized throttle, + = forward
    action[1] = normalized steer,    + = LEFT

    This uses the native NeoRacer contract convention (contract.py:
    ctrl[0:4] +torque=forward, ctrl[4] +angle=left), not racecar_core's
    convention (+angle=right). racecar_core_adapter.MujocoDrive is
    deliberately NOT reused here: it negates steer to cross into
    racecar_core's convention, which would silently reverse this env's
    action[1] sign relative to ctrl[4]'s own documented meaning. The mapping
    below (clamp, then scale by each actuator's own ctrlrange, read from the
    compiled model rather than hardcoded) mirrors MujocoDrive's pattern
    without importing its sign flip.

Observation space -- Box(20,), float32, built only from sensors.SensorReadings
(no raw qpos/qvel):
    [0:8]   lidar_ranges   m, -1 = no hit (sensors.lidar_scan)
    [8:11]  imu_accel      m/s^2, body frame, gravity included
    [11:14] imu_gyro       rad/s, body frame
    [14:18] imu_quat       orientation quaternion (w,x,y,z), world frame
    [18]    steer_cmd_pos  rad, virtual steering input
    [19]    estimated_speed  m/s, from motor_encoder_reading (mean of the 4
            wheel encoders, matching the real car's single drive motor --
            NOT ground-truth imu_linvel, per issue #7 design decision)

Control rate: 30 Hz (matches the repo's existing CONTROL_HZ precedent in
examples/wall_follower_vs/vehicle_space_controller.py, itself grounded in
the real hardware's target LiDAR scan rate). Physics substeps per action are
computed from model.opt.timestep at construction time, not assumed to be
0.002s forever -- see __init__.

Reward (RewardConfig, all weights explicit and UNTUNED -- issue #7 first
pass only):
    r_t = progress_weight * delta_theta_t  -  contact_penalty * contact_t
    plus one-time terminal adjustments:
        + lap_bonus        on successful lap completion
        - rollover_penalty on rollover termination
        - stuck_penalty    on stuck termination
    delta_theta_t is the signed net change in cumulative angle (radians)
    around the loop's center this tick -- the same progress metric PR #29's
    Stage D validation used for this exact track, reimplemented
    independently here (see _angle_around_center/_update_progress).

Termination (terminated=True) vs. truncation (truncated=True):
    terminated: rollover (upright_cos below threshold), stuck (no net
    angular progress over a trailing window), or lap completion
    (|cumulative angle| >= 2*pi).
    truncated: max_episode_steps reached.
    Ordinary wall contact is NOT terminal by itself -- it's recorded in
    info and penalized in the reward; stuck detection is the sole backstop
    for a car wedged against a wall, mirroring the already-validated
    loop_trials.py harness convention (its 100-trial Stage D run had zero
    contacts and only ever ended on rollover/stuck/success).
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass, field
from typing import ClassVar

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from .. import assets as _assets
from .. import collision, contract
from .. import sensors as _sensors
from .. import sim as _sim

_DEFAULT_TRACK = os.path.join(_assets.ASSET_DIR, "tracks", "loop_corridor.xml")

# Ring-corridor spawn offset: derived directly from loop_corridor.xml's own
# documented geometry (its docstring: outer wall y=-3, inner wall y=-2, so
# the 1.0m-wide bottom-straight corridor is centered at y=-2.5), NOT copied
# from examples/wall_follower_vs/ or the experimental adapter branch.
# Composing the car onto loop_corridor.xml with no offset would spawn it at
# the scene's raw origin, inside the inner wall's hollow interior -- off the
# drivable ring entirely.
_LOOP_SPAWN_OFFSET = (0.0, -2.5, 0.0)

OBS_DIM = 20


def _compose_scene(track_path: str, car_path: str) -> mujoco.MjModel:
    """Compose the car onto a track scene at runtime.

    Mirrors the MjSpec.attach_body composition pattern used throughout this
    repo (examples/track_centering_demo.py, validation/test_collision_
    reset.py) -- kept private and reimplemented here rather than imported
    from examples/wall_follower_vs/scene_utils.py, which is explicitly not a
    stable API (see README.md/CLAUDE.md) and which this env must not depend
    on. Promote to neoracer_mujoco.assets instead of duplicating a third
    time if a second production (non-example, non-test) consumer needs it.

    Applies _LOOP_SPAWN_OFFSET when composing loop_corridor.xml specifically
    (matched by filename), since that offset is a property of that track's
    own corridor geometry, not something generic to every track.
    """
    scene = mujoco.MjSpec.from_file(track_path)
    car = mujoco.MjSpec.from_file(car_path)
    frame = scene.worldbody.add_frame()
    if track_path.endswith("loop_corridor.xml"):
        frame.pos = _LOOP_SPAWN_OFFSET
    frame.attach_body(car.body("car"), "", "")
    return scene.compile()


def _compose_yaw_perturbation(q: np.ndarray, yaw: float) -> np.ndarray:
    """Compose an extra yaw-about-+Z rotation (radians) onto quaternion q
    via the Hamilton product q * dq.

    Bounded spawn randomization following the same approach validated in
    PR #29's Stage D harness, but reimplemented independently here rather
    than imported from examples/wall_follower_vs/math_utils.py -- this env
    must not depend on the experimental wall-follower code.
    """
    half = yaw / 2.0
    dq = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])
    w0, x0, y0, z0 = q
    w1, x1, y1, z1 = dq
    return np.array(
        [
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        ]
    )


@dataclass(frozen=True)
class RewardConfig:
    """Reward weights -- explicit and deliberately UNTUNED (issue #7 first
    pass). See NeoRacerEnv's module docstring for the full formula."""

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
    car_path: str = _assets.DEFAULT_CAR
    control_hz: float = 30.0  # matches the repo's existing CONTROL_HZ precedent
    max_episode_steps: int = (
        1800  # 60s at 30Hz, matches loop_trials.py's MAX_SIM_TIME_S
    )
    spawn_lateral_range_m: float = 0.15  # +/- bound, matches PR #29's validated range
    spawn_yaw_range_rad: float = 0.15
    upright_rollover_cos: float = 0.5  # cos(tilt) below this = rolled over
    stuck_window_s: float = 4.0
    stuck_progress_rad: float = (
        0.05  # min net angular progress required over the window
    )
    lidar_high_m: float = 50.0  # observation_space upper bound -- see module docstring
    reward: RewardConfig = field(default_factory=RewardConfig)


# Empirically-derived bounds (issue #7), NOT the nominal/steady-state
# ranges -- verified against 80 adversarial episodes (full-throttle,
# oscillating hard steer and random actions, deliberately wall-slam-seeking)
# on loop_corridor.xml, picked with real margin over the observed worst case
# rather than clipping physically valid readings to fit a guessed bound:
#   imu_accel: nominal is ~9.81 m/s^2 at rest, but a hard wall-collision
#     impulse can spike it far higher -- observed max ~511 m/s^2 during
#     rollover-inducing collisions. Bound: +/-1000.
#   imu_gyro: observed max ~42.4 rad/s during a violent tumble. Bound: +/-100.
#   steer_cmd_pos: steer_input's own mechanical joint range is +/-0.45 rad
#     (limited="true" in neoracer.xml), tighter than ctrl[4]'s +/-0.4 command
#     range -- but MuJoCo enforces joint limits as a soft constraint, not a
#     hard wall, so a violent enough collision can transiently push the
#     reading past the nominal limit. Observed max ~0.87 rad. Bound: +/-1.0.
# Re-verify these if the track, spawn range, or car model changes.
_ACCEL_BOUND = 1000.0
_GYRO_BOUND = 100.0
_STEER_BOUND = 1.0
_SPEED_BOUND = 40.0  # matches validation/test_conformance.py's own top-speed
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
        config: NeoRacerEnvConfig | None = None,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.config = config or NeoRacerEnvConfig()
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode: {render_mode!r}")
        self.render_mode = render_mode

        self.model = _compose_scene(self.config.track_path, self.config.car_path)
        self.data = mujoco.MjData(self.model)

        physics_dt = float(self.model.opt.timestep)
        self._substeps_per_action = max(
            1, round((1.0 / self.config.control_hz) / physics_dt)
        )
        # The true control period, after rounding to a whole number of
        # physics substeps -- may differ slightly from 1/control_hz.
        self.effective_control_dt = self._substeps_per_action * physics_dt

        self._torque_limit = float(self.model.actuator("fl_motor").ctrlrange[1])
        self._steer_limit = float(self.model.actuator("steer_servo").ctrlrange[1])

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = _observation_space(self.config)

        self._stuck_window_ticks = max(
            1, round(self.config.stuck_window_s * self.config.control_hz)
        )
        self._angle_history: deque[float] = deque(maxlen=self._stuck_window_ticks + 1)

        self._viewer = None

        # Episode state -- placeholder values until the first reset().
        self._step_count = 0
        self._episode_time_s = 0.0
        self._cum_angle = 0.0
        self._prev_angle = 0.0
        self._last_delta_angle = 0.0

    # --- internal helpers -----------------------------------------------

    def _apply_action(self, action: np.ndarray) -> None:
        throttle, steer = float(action[0]), float(action[1])
        self.data.ctrl[contract.DRIVE] = throttle * self._torque_limit
        self.data.ctrl[contract.STEER] = steer * self._steer_limit

    def _angle_around_center(self) -> float:
        """Angle (radians) of the car's position around loop_corridor.xml's
        geometric center (world origin -- the track is symmetric about it,
        per its own docstring). Independently reimplemented from the same
        progress concept PR #29's loop_trials.py validated, not imported."""
        xpos = self.data.body("car").xpos
        return float(np.arctan2(xpos[1], xpos[0]))

    def _update_progress(self) -> None:
        angle = self._angle_around_center()
        delta = angle - self._prev_angle
        if delta > np.pi:
            delta -= 2.0 * np.pi
        elif delta < -np.pi:
            delta += 2.0 * np.pi
        self._cum_angle += delta
        self._prev_angle = angle
        self._last_delta_angle = delta
        self._angle_history.append(self._cum_angle)

    def _is_stuck(self) -> bool:
        hist = self._angle_history
        if len(hist) < hist.maxlen:
            return False
        return abs(self._cum_angle - hist[0]) < self.config.stuck_progress_rad

    def _build_observation(self) -> np.ndarray:
        readings = _sensors.read(self.model, self.data)
        lidar = _sensors.lidar_scan(readings).ranges.astype(np.float32)
        accel = readings.imu_accel.astype(np.float32)
        gyro = readings.imu_gyro.astype(np.float32)
        quat = readings.imu_quat.astype(np.float32)
        steer_cmd = readings.steer_cmd_pos.astype(np.float32)
        encoder = _sensors.motor_encoder_reading(readings)
        speed = np.array(
            [_sensors.estimated_linear_speed_ms(encoder)], dtype=np.float32
        )
        return np.concatenate([lidar, accel, gyro, quat, steer_cmd, speed])

    def _compute_reward(
        self, contact: bool, terminated: bool, reason: str | None
    ) -> float:
        cfg = self.config.reward
        r = cfg.progress_weight * self._last_delta_angle
        if contact:
            r -= cfg.contact_penalty
        if terminated:
            if reason == "rollover":
                r -= cfg.rollover_penalty
            elif reason == "stuck":
                r -= cfg.stuck_penalty
            elif reason == "lap_complete":
                r += cfg.lap_bonus
        return float(r)

    def _build_info(
        self, obs: np.ndarray, contact: bool, upright_cos: float, reason: str | None
    ) -> dict:
        return {
            "progress_rad": self._cum_angle,
            "lap_frac": abs(self._cum_angle) / (2.0 * np.pi),
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
        _sim.settle(self.model, self.data)

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
        self._cum_angle = 0.0
        self._last_delta_angle = 0.0
        self._prev_angle = self._angle_around_center()
        self._angle_history.clear()
        self._angle_history.append(self._cum_angle)

        obs = self._build_observation()
        upright_cos = _sim.car_upright_cos(self.data)
        info = self._build_info(
            obs, contact=False, upright_cos=upright_cos, reason=None
        )
        return obs, info

    def step(self, action):
        action = np.clip(
            np.asarray(action, dtype=np.float32),
            self.action_space.low,
            self.action_space.high,
        )
        self._apply_action(action)

        contact_this_tick = False
        for _ in range(self._substeps_per_action):
            mujoco.mj_step(self.model, self.data)
            if collision.car_hit_wall(self.model, self.data):
                contact_this_tick = True

        self._step_count += 1
        self._episode_time_s = self._step_count * self.effective_control_dt
        self._update_progress()

        upright_cos = _sim.car_upright_cos(self.data)
        terminated = False
        reason: str | None = None

        if upright_cos < self.config.upright_rollover_cos:
            terminated = True
            reason = "rollover"
        elif self._is_stuck():
            terminated = True
            reason = "stuck"
        elif abs(self._cum_angle) >= 2.0 * np.pi:
            terminated = True
            reason = "lap_complete"

        truncated = (
            not terminated
        ) and self._step_count >= self.config.max_episode_steps

        reward = self._compute_reward(contact_this_tick, terminated, reason)
        obs = self._build_observation()
        info = self._build_info(obs, contact_this_tick, upright_cos, reason)

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return
        if self._viewer is None:
            import mujoco.viewer

            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self._viewer.sync()
        return

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
