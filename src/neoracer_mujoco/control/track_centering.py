"""
Classical lateral track-centering controller for NeoRacer.

The controller keeps the car near the middle of a corridor with two walls. It
reads the 8-beam LiDAR ring from SensorReadings (see neoracer_mujoco.sensors)
and computes a PD steer command and a speed. This module is pure control
logic: it does not load a model, step the simulator or read MjData. See
examples/track_centering_demo.py for a demo that does.

Frame: +X forward, +Y left, +Z up.
Speed (m/s), + = forward.
steer_servo target angle (rad), + = left, up to full lock either way
(neoracer_spec.derive.steer_lock).

LiDAR beams: 8 horizontal beams at 45° steps.
  lidar_000 = straight ahead (+X)      lidar_180 = straight behind (-X)
  lidar_045 = front-left                lidar_225 = rear-right
  lidar_090 = left (+Y)                 lidar_270 = right (-Y)
  lidar_135 = rear-left                 lidar_315 = front-right
Each reading is the distance (m) to the nearest hit, or -1 if no geom is in
range.

Error signals:
  lateral_error = (mean of the left beams) - (mean of the right beams).
    If it is positive, the left side has more clearance, so the car is closer
    to the right wall. The controller then steers left (+).
  heading_error: in a straight corridor with parallel walls, the front and rear
    diagonal beams on one side read the same distance when the car is parallel
    to the walls. If the nose turns toward a wall, the front beam on that side
    gets shorter than the rear beam.
        left_skew  = rear_left  - front_left   (+ = nose toward the left wall)
        right_skew = rear_right - front_right  (+ = nose toward the right wall)
        heading_error = right_skew - left_skew
    If heading_error is positive, the nose points toward the right wall, so
    the controller steers left (+).
  steer = kp_lateral * lateral_error + kp_heading * heading_error
          + kd * d(lateral_error)/dt, clamped to ±steer_limit.

lateral_error is a difference of clearances, not the sideways offset of the
car. Four of the six side beams are diagonal. A diagonal beam travels about
1.41 times the perpendicular distance to a parallel wall. So lateral_error is
larger than the real offset. For example, in a corridor with a half-width of
0.5 m, a car 0.25 m off center gives lateral_error ~= -0.64 m. To judge the
centering, compare lateral_error over time, not with a distance in meters.

Limit: heading_error assumes straight, parallel walls. On a curve it has a
bias and can work against the lateral term. If you use this controller on a
curved track, set kp_heading to 0 or use wall_following instead.

Limit: the controller estimates the heading only from the LiDAR skew, not
from the IMU. If the walls are not parallel, add an IMU heading estimate.

Limit: the steer smoothing is a rate limit (rad/s, see steering.py), not a
low-pass filter. So the smoothness depends on the control step rate.

Limit: the default gains come from the demo corridor run. They are not
derived from a model of the car. If the car oscillates, tune them again.

Bad sensor data (see sensors.parse_beam): a beam is corrupt if its value is
NaN or infinite. A -1 (no hit) is valid data and means "clear". Both read as
max_beam_range. If more than max_corrupt_beams of the 8 beams are corrupt,
step() does not use the control law. It returns steering.fallback(): the
steer moves toward 0 and the speed drops to
safe_fallback_speed_fraction of target_speed.

Usage: call step in a loop, and pass each new state to the next call.
    state = PDState()
    state, command = step(TrackCenteringConfig(), state, sensor_readings, dt)
    car = racecar.step(scene, data, racecar.ask(car, command))
"""

from dataclasses import dataclass

import numpy as np

from ..sensors import (
    FRONT_BEAM,
    FRONT_LEFT_BEAM,
    FRONT_RIGHT_BEAM,
    LEFT_BEAMS,
    REAR_LEFT_BEAM,
    REAR_RIGHT_BEAM,
    RIGHT_BEAMS,
    SensorReadings,
    beam_distance,
    is_corrupt,
    lidar_ranges,
    parse_beam,
)
from . import steering
from .steering import PDState, ShapingConfig, SpeedAngle


@dataclass(frozen=True)
class TrackCenteringConfig:
    """Tuning parameters for step()."""

    kp_lateral: float = 0.5
    """Radians of steer per meter of lateral_error."""
    kp_heading: float = 0.5
    """Radians of steer per meter of heading_error. With the servo's delay, a
    stronger heading term swings the car from wall to wall, and the shaping slows
    it for the large steer."""
    kd: float = 0.05
    """Radians of steer per m/s of lateral_error's rate."""
    max_beam_range: float = 3.0
    """The distance (m) substituted for a beam with no hit (-1) or a corrupt one."""
    max_corrupt_beams: int = 4
    """How many of the 8 beams may be corrupt. More triggers the safe fallback
    command."""
    shaping: ShapingConfig = ShapingConfig()


@dataclass(frozen=True)
class CenteringSignals:
    """The error signals that step() uses. The demo and the tests read them
    for logs and checks."""

    lateral_error: float
    """The left clearance less the right (m), + = more clearance on the left."""
    heading_error: float
    """The right skew less the left (m), + = angled toward the right wall."""
    front_distance: float
    """The clearance straight ahead (m), FRONT_BEAM's reading."""
    n_corrupt: int
    """How many of the 8 beams were NaN or infinite."""


def compute_signals(
    sensors: SensorReadings, cfg: TrackCenteringConfig
) -> CenteringSignals:
    """Compute the error signals from the LiDAR readings."""
    beams = [parse_beam(float(r)) for r in lidar_ranges(sensors)]
    d = [beam_distance(b, cfg.max_beam_range) for b in beams]

    left_mean = float(np.mean([d[i] for i in LEFT_BEAMS]))
    right_mean = float(np.mean([d[i] for i in RIGHT_BEAMS]))
    left_skew = d[REAR_LEFT_BEAM] - d[FRONT_LEFT_BEAM]
    right_skew = d[REAR_RIGHT_BEAM] - d[FRONT_RIGHT_BEAM]

    return CenteringSignals(
        lateral_error=left_mean - right_mean,
        heading_error=right_skew - left_skew,
        front_distance=d[FRONT_BEAM],
        n_corrupt=sum(is_corrupt(b) for b in beams),
    )


def step(
    cfg: TrackCenteringConfig, state: PDState, sensors: SensorReadings, dt: float
) -> tuple[PDState, SpeedAngle]:
    """Compute one control command and the next controller state."""
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
