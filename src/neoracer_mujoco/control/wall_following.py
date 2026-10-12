"""
Classical wall-following controller for NeoRacer.

The controller keeps the car at a fixed distance from one wall (follow_side).
It reads the 8-beam LiDAR ring from SensorReadings (see
neoracer_mujoco.sensors). track_centering.py balances the clearance between
two walls. This controller uses only one wall, so it also works where there
is no second wall, for example in a bend or on an open course.

Frame: +X forward, +Y left, +Z up.
ctrl[0:4] = wheel torque (N·m), + = forward. ctrl[4] = steer angle (rad), + = left.

Signals (see compute_signals()). They have the same meaning for the left and
the right wall:
  distance_error = side beam distance - target_distance.
    If it is positive, the car is too far from the wall and steers toward it.
  wall_skew = rear side beam - front side beam.
    If it is positive, the nose points toward the wall and the car steers away.
step() multiplies by SideBeams.toward_wall (+1 or -1) to get the steer
command. So the commands for the left and the right wall are exact mirrors.

Bad sensor data is handled as in track_centering.py (see sensors.parse_beam):
NaN or Inf is corrupt, and -1 (no hit) is valid. Both read as max_beam_range.
Only the 4 beams that this controller reads count in n_corrupt. If more than
max_corrupt_beams are corrupt, step() returns steering.fallback(): the steer
moves toward 0 and the throttle drops.

Usage: call step in a loop, and pass each new state to the next call.
    state = PDState()
    state, command = step(WallFollowingConfig(), state, sensor_readings, dt)
"""

from dataclasses import dataclass
from typing import Literal, assert_never

from ..sensors import (
    FRONT_BEAM,
    FRONT_LEFT_BEAM,
    FRONT_RIGHT_BEAM,
    LEFT_BEAM,
    REAR_LEFT_BEAM,
    REAR_RIGHT_BEAM,
    RIGHT_BEAM,
    SensorReadings,
    beam_distance,
    is_corrupt,
    lidar_ranges,
    parse_beam,
)
from . import steering
from .steering import ControlCommand, PDState, ShapingConfig

type Side = Literal["left", "right"]


@dataclass(frozen=True)
class WallFollowingConfig:
    """Tuning parameters for step()."""

    follow_side: Side = "right"  # which wall to track
    target_distance: float = 0.3  # m; desired standoff from the followed wall
    kp_distance: float = 1.5  # rad steer per m of distance_error
    kp_heading: float = 3.0  # rad steer per m of wall_skew
    kd: float = 0.05  # rad steer per (m/s) of distance_error rate
    max_beam_range: float = 3.0  # m; substituted for no-hit(-1) or corrupt beams
    max_corrupt_beams: int = 2  # of the 4 beams used here; more triggers fallback
    shaping: ShapingConfig = ShapingConfig()  # steer and throttle limits


@dataclass(frozen=True)
class WallFollowingSignals:
    distance_error: float  # m; + = farther from the wall than target_distance
    wall_skew: float  # m; + = nose angled toward the wall
    front_distance: float  # m; clearance straight ahead (FRONT_BEAM)
    n_corrupt: int  # of the 4 beams used here that were NaN/Inf


@dataclass(frozen=True)
class SideBeams:
    """The beams that watch one wall, as indices into lidar_ranges()."""

    side: int  # points perpendicular to the wall
    front: int
    rear: int
    toward_wall: Literal[1, -1]  # sign of a steer toward the wall


def _side_beams(side: Side) -> SideBeams:
    match side:
        case "left":
            return SideBeams(LEFT_BEAM, FRONT_LEFT_BEAM, REAR_LEFT_BEAM, 1)
        case "right":
            return SideBeams(RIGHT_BEAM, FRONT_RIGHT_BEAM, REAR_RIGHT_BEAM, -1)
        case _:
            assert_never(side)


def compute_signals(
    sensors: SensorReadings, cfg: WallFollowingConfig
) -> WallFollowingSignals:
    """Compute the error signals from the LiDAR readings."""
    sb = _side_beams(cfg.follow_side)
    ranges = lidar_ranges(sensors)
    beams = {
        i: parse_beam(float(ranges[i]))
        for i in (sb.side, sb.front, sb.rear, FRONT_BEAM)
    }
    d = {i: beam_distance(b, cfg.max_beam_range) for i, b in beams.items()}

    return WallFollowingSignals(
        distance_error=d[sb.side] - cfg.target_distance,
        wall_skew=d[sb.rear] - d[sb.front],
        front_distance=d[FRONT_BEAM],
        n_corrupt=sum(is_corrupt(b) for b in beams.values()),
    )


def step(
    cfg: WallFollowingConfig, state: PDState, sensors: SensorReadings, dt: float
) -> tuple[PDState, ControlCommand]:
    """Compute one control command and the next controller state."""
    signals = compute_signals(sensors, cfg)
    if signals.n_corrupt > cfg.max_corrupt_beams:
        return steering.fallback(cfg.shaping, state, dt)

    error_rate = steering.error_rate(state, signals.distance_error, dt)
    steer_raw = _side_beams(cfg.follow_side).toward_wall * (
        cfg.kp_distance * signals.distance_error
        - cfg.kp_heading * signals.wall_skew
        + cfg.kd * error_rate
    )
    return steering.command(
        cfg.shaping,
        state,
        signals.distance_error,
        steer_raw,
        signals.front_distance,
        dt,
    )
