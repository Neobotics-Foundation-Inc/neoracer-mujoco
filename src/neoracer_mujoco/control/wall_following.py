"""
Classical (non-learned) wall-following controller for NeoRacer.

Maintains a fixed standoff distance from ONE wall (configurable follow_side)
using the 8-beam LiDAR ring off SensorReadings (see neoracer_mujoco.sensors).
Unlike track_centering.py, which balances clearance between TWO walls, this
tracks a single wall so it stays useful where a second wall isn't
guaranteed (e.g. around a bend, or an otherwise open course).

Conventions match assets/neoracer.xml: +X fwd, +Y left, +Z up;
ctrl[0:4] = wheel torque (+ = fwd); ctrl[4] = steer angle (+ = left).

Signals (see compute_signals()), both direction-agnostic w.r.t. follow_side:
  distance_error = side-beam distance - target_distance.
    + => farther than target => steer TOWARD the wall.
  wall_skew = rear_side_beam - front_side_beam.
    + => nose angled TOWARD the wall => steer AWAY from it.
step() applies one +-1 sign flip (toward_wall, see _side_beams) to turn
these into an actual steer command, so "left" and "right" are exact mirrors.

Invalid-beam handling is the same as in track_centering.py (see
sensors.parse_beam): NaN/Inf is corrupt; a legitimate -1 no-hit is not.
Both read as max_beam_range, an "open" reading. Only the 4 beams this
controller reads count toward n_corrupt. Past max_corrupt_beams corrupt,
step() returns steering.fallback(): steer slewed toward 0, reduced throttle.

Usage, one call per control step (a rollout is a fold over step):
    state = PDState()
    state, command = step(WallFollowingConfig(), state, sensor_readings, dt)
"""

from dataclasses import dataclass
from typing import Literal, assert_never

from ..sensors import (
    FRONT_CONE_BEAMS,
    FRONT_LEFT_BEAM,
    FRONT_RIGHT_BEAM,
    LEFT_BEAM,
    REAR_LEFT_BEAM,
    REAR_RIGHT_BEAM,
    RIGHT_BEAM,
    Corrupt,
    SensorReadings,
    beam_distance,
    lidar_scan,
    parse_beam,
)
from . import steering
from .steering import ControlCommand, PDState

type Side = Literal["left", "right"]


@dataclass(frozen=True)
class WallFollowingConfig:
    """Named, documented tuning parameters -- nothing in step() is a bare
    unexplained constant."""

    follow_side: Side = "right"  # which wall to track
    target_distance: float = 0.3  # m; desired standoff from the followed wall
    target_throttle: float = 0.10  # N·m per wheel, base forward drive command
    kp_distance: float = 1.5  # rad steer per m of distance_error
    kp_heading: float = 3.0  # rad steer per m of wall_skew
    kd: float = 0.05  # rad steer per (m/s) of distance_error rate
    steer_limit: float = 0.4  # rad; matches ctrl[4] ctrlrange in neoracer.xml
    steer_slew_limit: float = 3.0  # rad/s max steering rate of change
    min_safe_forward_distance: float = 0.25  # m; front clearance -> throttle 0
    slowdown_threshold: float = (
        0.8  # m; front clearance below which throttle ramps down
    )
    steer_slowdown_gain: float = 0.6  # throttle cut fraction at full steer deflection
    max_beam_range: float = 3.0  # m; substituted for no-hit(-1) or corrupt beams
    max_corrupt_beams: int = 2  # of the 4 beams used here; more triggers fallback
    safe_fallback_throttle_fraction: float = 0.3  # of target_throttle, during fallback


@dataclass(frozen=True)
class WallFollowingSignals:
    distance_error: float  # m; + = farther from the wall than target_distance
    wall_skew: float  # m; + = nose angled toward the wall
    front_distance: float  # m; straight-ahead clearance (FRONT_CONE_BEAMS)
    n_corrupt: int  # of the 4 beams used here that were NaN/Inf


def _side_beams(side: Side) -> tuple[int, int, int, float]:
    """(side beam, front beam, rear beam, toward_wall) for one side. The side
    beam points perpendicular to the wall. toward_wall is the sign of a steer
    command toward the wall: + for the left wall, - for the right wall."""
    match side:
        case "left":
            return LEFT_BEAM, FRONT_LEFT_BEAM, REAR_LEFT_BEAM, 1.0
        case "right":
            return RIGHT_BEAM, FRONT_RIGHT_BEAM, REAR_RIGHT_BEAM, -1.0
        case _:
            assert_never(side)


def compute_signals(
    sensors: SensorReadings, cfg: WallFollowingConfig
) -> WallFollowingSignals:
    """Pure function: SensorReadings -> WallFollowingSignals."""
    side, front, rear, _ = _side_beams(cfg.follow_side)
    ranges = lidar_scan(sensors).ranges
    beams = {
        i: parse_beam(float(ranges[i])) for i in (side, front, rear, *FRONT_CONE_BEAMS)
    }
    d = {i: beam_distance(b, cfg.max_beam_range) for i, b in beams.items()}

    return WallFollowingSignals(
        distance_error=d[side] - cfg.target_distance,
        wall_skew=d[rear] - d[front],
        front_distance=min(d[i] for i in FRONT_CONE_BEAMS),
        n_corrupt=sum(isinstance(b, Corrupt) for b in beams.values()),
    )


def step(
    cfg: WallFollowingConfig, state: PDState, sensors: SensorReadings, dt: float
) -> tuple[PDState, ControlCommand]:
    """One PD control step: (state, readings) -> (next state, command)."""
    signals = compute_signals(sensors, cfg)
    if signals.n_corrupt > cfg.max_corrupt_beams:
        return steering.fallback(cfg, state, dt)

    _, _, _, toward_wall = _side_beams(cfg.follow_side)
    error_rate = steering.error_rate(state, signals.distance_error, dt)
    steer_raw = toward_wall * (
        cfg.kp_distance * signals.distance_error
        - cfg.kp_heading * signals.wall_skew
        + cfg.kd * error_rate
    )
    return steering.command(
        cfg, state, signals.distance_error, steer_raw, signals.front_distance, dt
    )
