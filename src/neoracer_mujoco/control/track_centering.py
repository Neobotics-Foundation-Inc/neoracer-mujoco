"""
Classical (non-learned) lateral track-centering controller for NeoRacer.

Keeps the car near the middle of a two-sided corridor by reading the 8-beam
LiDAR ring off SensorReadings (see neoracer_mujoco.sensors) and computing a
PD-style steering correction plus throttle shaping. Pure control logic only —
this module never loads a model, steps the simulator, or reads sensors off a
live mujoco.MjData; see examples/track_centering_demo.py for that.

Coordinate + steering convention (matches assets/neoracer.xml / README):
  +X = forward, +Y = left, +Z = up.
  ctrl[0:4] = wheel torque (N·m), + = forward.
  ctrl[4]   = steer_servo target angle (rad), + = LEFT, range [-0.4, 0.4].

LiDAR beam layout (verified empirically against the compiled model, not
assumed from the XML comments): 8 beams at 45° steps, purely horizontal.
  lidar_000 = straight ahead (+X)      lidar_180 = straight behind (-X)
  lidar_045 = front-left                lidar_225 = rear-right
  lidar_090 = left (+Y)                 lidar_270 = right (-Y)
  lidar_135 = rear-left                 lidar_315 = front-right
Each reading is meters to the nearest hit, or -1 if nothing is within range.

Centering-error sign convention:
  lateral_error = (mean of left-side beams) - (mean of right-side beams).
    Positive  -> more clearance on the LEFT (car is closer to the RIGHT
                 wall) -> controller steers left (+) toward the open side.
    Negative  -> more clearance on the RIGHT -> steers right (-).
  heading_error: on a straight corridor with parallel walls, the front and
    rear beam on the same side read equal distances when the car's heading
    is parallel to the walls (both measure the same perpendicular distance
    via symmetric 45° beam geometry). If the car's nose is angled toward a
    wall, that wall's FRONT beam shortens relative to its REAR beam.
        left_skew  = rear_left  - front_left   (positive => angled toward left wall)
        right_skew = rear_right - front_right  (positive => angled toward right wall)
        heading_error = right_skew - left_skew
    Positive heading_error -> angled toward the right wall -> steers left (+)
    to straighten out; negative -> angled toward the left wall -> steers right.
  Both signals are in meters; steer = Kp_lateral*lateral_error +
  Kp_heading*heading_error + Kd*d(lateral_error)/dt, clamped to ±steer_limit.

  IMPORTANT: lateral_error is a CLEARANCE-DIFFERENCE metric, not the car's
  physical lateral distance from the track centerline — do not compare its
  magnitude directly to a physical offset in meters. Two of the three beams
  per side (the 45 degree diagonals) travel 1/sin(45 degrees) ~= 1.41x
  farther than the perpendicular distance to reach a parallel wall, so this
  metric is systematically larger in magnitude than the car's actual
  sideways offset. Example (straight corridor, half-width 0.5 m): a car
  physically 0.25 m off-center produces lateral_error ~= -0.64 m, not -0.5 m
  (let alone -0.25 m) — verified both analytically and against the compiled
  simulator. Track relative change in lateral_error over time to judge
  centering performance, not its raw value against a physical distance.

Assumptions and known limitations:
  - heading_error's symmetry argument assumes straight, parallel walls. On a
    curve or at a corner it will be biased and can fight the lateral term;
    this controller is not designed for curved track validation.
  - No IMU-based heading estimate is used (per the task, PR #12's typed
    IMUReading is not assumed merged); heading is inferred purely from the
    LiDAR skew above.
  - Steering smoothing is a rad/s SLEW-RATE limiter (dt-scaled, matching the
    rate-limiting pattern already used in manual_drive.py), not a low-pass
    filter, so it is only as smooth as the caller's step rate.
  - Gains below are a reasonable starting point tuned against the demo
    script's corridor run, not a plant-derived/optimal controller.

Invalid-sensor handling (see _beam_distance): a beam is "corrupt" if its
field is missing, an empty array, NaN, or infinite. A legitimate -1 (no hit
within range) is NOT corrupt — it's real data meaning "clear", and is
substituted with `max_beam_range` (an open/clear reading), same as a corrupt
beam's fallback value. If MORE than `max_corrupt_beams` of the 8 beams are
corrupt in a single reading, compute() skips the normal control law entirely
and returns a safe fallback command: steering slewed toward 0 and throttle
cut to `safe_fallback_throttle_fraction` of target — never an extreme
steering command on bad data.
"""

from dataclasses import dataclass

import numpy as np

from ..sensors import SensorReadings

LEFT_BEAMS = ("lidar_045", "lidar_090", "lidar_135")
RIGHT_BEAMS = ("lidar_315", "lidar_270", "lidar_225")
# Deliberately lidar_000 ONLY, not a wider cone: lidar_045/lidar_315 already
# do double duty in LEFT_BEAMS/RIGHT_BEAMS. Including them here too would
# couple the forward-obstacle slowdown to side-wall proximity during a
# lateral correction (a wall picked up obliquely by a diagonal beam would
# look like a forward obstacle and cut throttle when nothing is ahead).
FRONT_CONE_BEAMS = ("lidar_000",)
FRONT_LEFT_BEAM = "lidar_045"
REAR_LEFT_BEAM = "lidar_135"
FRONT_RIGHT_BEAM = "lidar_315"
REAR_RIGHT_BEAM = "lidar_225"
ALL_BEAMS = (
    "lidar_000",
    "lidar_045",
    "lidar_090",
    "lidar_135",
    "lidar_180",
    "lidar_225",
    "lidar_270",
    "lidar_315",
)


@dataclass(frozen=True)
class TrackCenteringConfig:
    """Named, documented tuning parameters — nothing in compute() is a bare
    unexplained constant; every knob used by the control law lives here."""

    target_throttle: float = 0.10  # N·m per wheel, base forward drive command
    kp_lateral: float = 1.2  # rad of steer per meter of lateral_error
    kp_heading: float = 3.0  # rad of steer per meter of heading_error
    kd: float = 0.05  # rad of steer per (m/s) of lateral_error rate
    steer_limit: float = 0.4  # rad; matches ctrl[4] ctrlrange in neoracer.xml
    steer_slew_limit: float = 3.0  # rad/s max steering rate of change (smoothing)
    min_safe_forward_distance: float = (
        0.25  # m; front clearance at/below which throttle -> 0
    )
    slowdown_threshold: float = (
        0.8  # m; front clearance below which throttle ramps down
    )
    steer_slowdown_gain: float = 0.6  # throttle cut fraction at full steer deflection
    max_beam_range: float = 3.0  # m; substituted for no-hit(-1) or corrupt beams
    max_corrupt_beams: int = 4  # of 8; more triggers the safe fallback command
    safe_fallback_throttle_fraction: float = 0.3  # of target_throttle, during fallback


@dataclass(frozen=True)
class ControlCommand:
    """throttle applies to ctrl[0:4] (N·m, + = forward); steer applies to
    ctrl[4] (rad, + = left)."""

    throttle: float
    steer: float


@dataclass(frozen=True)
class CenteringSignals:
    """Intermediate error signals, exposed separately from ControlCommand so
    the demo/tests can inspect and log them without recomputing."""

    lateral_error: float  # m; + = more clearance on the left
    heading_error: float  # m; + = angled toward the right wall
    front_distance: (
        float  # m; straight-ahead clearance (lidar_000 only — see FRONT_CONE_BEAMS)
    )
    n_corrupt: int  # count of the 8 beams that were NaN/Inf/missing


def _beam_distance(
    sensors: SensorReadings, name: str, cfg: TrackCenteringConfig
) -> tuple[float, bool]:
    """
    Read one LiDAR beam safely.

    Returns (distance, corrupt). `corrupt` is True only for NaN/Inf/missing/
    empty data — NOT for a legitimate -1 no-hit reading, which is real data
    meaning "nothing within range". Both corrupt and no-hit readings resolve
    to cfg.max_beam_range (an "open" distance), since a bad sensor reading
    should never be misread as "wall right next to me".
    """
    arr = getattr(sensors, name, None)
    if arr is None or len(arr) == 0:
        return cfg.max_beam_range, True

    value = float(arr[0])
    if not np.isfinite(value):
        return cfg.max_beam_range, True
    if value < 0:
        return cfg.max_beam_range, False  # legitimate no-hit, not corrupt

    return min(value, cfg.max_beam_range), False


def compute_signals(
    sensors: SensorReadings, cfg: TrackCenteringConfig
) -> CenteringSignals:
    """Pure function: SensorReadings -> CenteringSignals. No state, no I/O —
    the piece of the control law that's cheapest to unit test directly."""
    readings = {}
    n_corrupt = 0
    for name in ALL_BEAMS:
        value, corrupt = _beam_distance(sensors, name, cfg)
        readings[name] = value
        n_corrupt += corrupt

    left_mean = float(np.mean([readings[n] for n in LEFT_BEAMS]))
    right_mean = float(np.mean([readings[n] for n in RIGHT_BEAMS]))
    lateral_error = left_mean - right_mean

    left_skew = readings[REAR_LEFT_BEAM] - readings[FRONT_LEFT_BEAM]
    right_skew = readings[REAR_RIGHT_BEAM] - readings[FRONT_RIGHT_BEAM]
    heading_error = right_skew - left_skew

    front_distance = float(min(readings[n] for n in FRONT_CONE_BEAMS))

    return CenteringSignals(
        lateral_error=lateral_error,
        heading_error=heading_error,
        front_distance=front_distance,
        n_corrupt=n_corrupt,
    )


class TrackCenteringController:
    """
    Stateful PD-style lateral centering controller.

    Usage:
        controller = TrackCenteringController(TrackCenteringConfig())
        control = controller.compute(sensor_readings, dt)
        data.ctrl[0:4] = control.throttle
        data.ctrl[4]   = control.steer

    State (previous lateral error, previous steering command) is used for the
    derivative term and the steering slew-rate limiter. Call reset() when
    starting a new episode/rollout so stale history doesn't leak in — e.g. a
    D-term spike from the last episode's final error.
    """

    def __init__(self, config: TrackCenteringConfig = TrackCenteringConfig()):
        self.config = config
        self.reset()

    def reset(self) -> None:
        """Clear derivative and steering-smoothing history."""
        self._prev_lateral_error = 0.0
        self._prev_steer = 0.0

    def _slew(self, target: float, dt: float) -> float:
        cfg = self.config
        max_delta = cfg.steer_slew_limit * max(dt, 0.0)
        delta = max(-max_delta, min(max_delta, target - self._prev_steer))
        steer = self._prev_steer + delta
        return max(-cfg.steer_limit, min(cfg.steer_limit, steer))

    def compute(self, sensors: SensorReadings, dt: float) -> ControlCommand:
        cfg = self.config
        signals = compute_signals(sensors, cfg)

        if signals.n_corrupt > cfg.max_corrupt_beams:
            # Too much of the LiDAR ring is unusable to trust the control
            # law this step. Slew steering toward 0 (never jump straight to
            # it) and crawl forward at a reduced, safe throttle.
            steer = self._slew(0.0, dt)
            self._prev_steer = steer
            self._prev_lateral_error = 0.0  # don't let a garbage step poison the D-term
            return ControlCommand(
                throttle=cfg.target_throttle * cfg.safe_fallback_throttle_fraction,
                steer=steer,
            )

        error_rate = 0.0
        if dt > 0:
            error_rate = (signals.lateral_error - self._prev_lateral_error) / dt
        self._prev_lateral_error = signals.lateral_error

        steer_raw = (
            cfg.kp_lateral * signals.lateral_error
            + cfg.kp_heading * signals.heading_error
            + cfg.kd * error_rate
        )
        steer_raw = max(-cfg.steer_limit, min(cfg.steer_limit, steer_raw))
        steer = self._slew(steer_raw, dt)
        self._prev_steer = steer

        steer_severity = abs(steer) / cfg.steer_limit if cfg.steer_limit > 0 else 0.0
        throttle = cfg.target_throttle * (
            1.0 - cfg.steer_slowdown_gain * steer_severity
        )

        front = signals.front_distance
        if front <= cfg.min_safe_forward_distance:
            obstacle_scale = 0.0
        elif front >= cfg.slowdown_threshold:
            obstacle_scale = 1.0
        else:
            span = cfg.slowdown_threshold - cfg.min_safe_forward_distance
            obstacle_scale = (front - cfg.min_safe_forward_distance) / span
        throttle = max(0.0, throttle * obstacle_scale)

        return ControlCommand(throttle=throttle, steer=steer)
