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
  - No IMU-based heading estimate is used; heading is inferred purely from
    the LiDAR skew above.
  - Steering smoothing is a rad/s SLEW-RATE limiter (dt-scaled, see
    steering.py), not a low-pass filter, so it is only as smooth as the
    caller's step rate.
  - Gains below are a reasonable starting point tuned against the demo
    script's corridor run, not a plant-derived/optimal controller.

Invalid-sensor handling (see sensors.parse_beam): a beam is corrupt if its
value is NaN or infinite. A legitimate -1 (no hit within range) is NOT
corrupt — it's real data meaning "clear". Both read as `max_beam_range` (an
open/clear reading). If MORE than `max_corrupt_beams` of the 8 beams are
corrupt in a single reading, step() skips the normal control law entirely
and returns steering.fallback(): steering slewed toward 0 and throttle cut
to `safe_fallback_throttle_fraction` of target — never an extreme steering
command on bad data.

Usage, one call per control step (a rollout is a fold over step):
    state = PDState()
    state, command = step(TrackCenteringConfig(), state, sensor_readings, dt)
    data.ctrl[0:4] = command.throttle
    data.ctrl[4]   = command.steer
"""

from dataclasses import dataclass

import numpy as np

from ..sensors import (
    FRONT_CONE_BEAMS,
    FRONT_LEFT_BEAM,
    FRONT_RIGHT_BEAM,
    LEFT_BEAMS,
    REAR_LEFT_BEAM,
    REAR_RIGHT_BEAM,
    RIGHT_BEAMS,
    SensorReadings,
    beam_distance,
    is_corrupt,
    lidar_scan,
    parse_beam,
)
from . import steering
from .steering import ControlCommand, PDState, ShapingConfig


@dataclass(frozen=True)
class TrackCenteringConfig:
    """Named, documented tuning parameters — nothing in step() is a bare
    unexplained constant; every knob used by the control law lives here."""

    kp_lateral: float = 1.2  # rad of steer per meter of lateral_error
    kp_heading: float = 3.0  # rad of steer per meter of heading_error
    kd: float = 0.05  # rad of steer per (m/s) of lateral_error rate
    max_beam_range: float = 3.0  # m; substituted for no-hit(-1) or corrupt beams
    max_corrupt_beams: int = 4  # of 8; more triggers the safe fallback command
    shaping: ShapingConfig = ShapingConfig()  # steer and throttle limits


@dataclass(frozen=True)
class CenteringSignals:
    """Intermediate error signals, exposed separately from ControlCommand so
    the demo/tests can inspect and log them without recomputing."""

    lateral_error: float  # m; + = more clearance on the left
    heading_error: float  # m; + = angled toward the right wall
    front_distance: float  # m; straight-ahead clearance (FRONT_CONE_BEAMS)
    n_corrupt: int  # count of the 8 beams that were NaN/Inf


def compute_signals(
    sensors: SensorReadings, cfg: TrackCenteringConfig
) -> CenteringSignals:
    """Pure function: SensorReadings -> CenteringSignals."""
    beams = [parse_beam(float(r)) for r in lidar_scan(sensors).ranges]
    d = [beam_distance(b, cfg.max_beam_range) for b in beams]

    left_mean = float(np.mean([d[i] for i in LEFT_BEAMS]))
    right_mean = float(np.mean([d[i] for i in RIGHT_BEAMS]))
    left_skew = d[REAR_LEFT_BEAM] - d[FRONT_LEFT_BEAM]
    right_skew = d[REAR_RIGHT_BEAM] - d[FRONT_RIGHT_BEAM]

    return CenteringSignals(
        lateral_error=left_mean - right_mean,
        heading_error=right_skew - left_skew,
        front_distance=float(min(d[i] for i in FRONT_CONE_BEAMS)),
        n_corrupt=sum(is_corrupt(b) for b in beams),
    )


def step(
    cfg: TrackCenteringConfig, state: PDState, sensors: SensorReadings, dt: float
) -> tuple[PDState, ControlCommand]:
    """One PD control step: (state, readings) -> (next state, command)."""
    signals = compute_signals(sensors, cfg)
    if signals.n_corrupt > cfg.max_corrupt_beams:
        return steering.fallback(cfg.shaping, state, dt)

    error_rate = steering.error_rate(state, signals.lateral_error, dt)
    steer_raw = (
        cfg.kp_lateral * signals.lateral_error
        + cfg.kp_heading * signals.heading_error
        + cfg.kd * error_rate
    )
    return steering.command(
        cfg.shaping, state, signals.lateral_error, steer_raw, signals.front_distance, dt
    )
