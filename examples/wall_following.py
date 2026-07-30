"""
Classical (non-learned) wall-following controller for NeoRacer (issue #6).

Maintains a fixed standoff distance from ONE wall (configurable follow_side)
using the 8-beam LiDAR ring off SensorReadings (see sensor_logger.py).
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
compute() applies one +-1 sign flip (_toward_wall_sign) to turn these into
an actual steer command, so "left" and "right" are exact mirrors.

Invalid-beam handling mirrors track_centering.py's _beam_distance: NaN/
Inf/missing is "corrupt" (-> max_beam_range, an "open" reading); a
legitimate -1 no-hit is NOT corrupt and resolves the same way. Only the 4
beams this controller reads count toward n_corrupt. Past max_corrupt_beams
corrupt, compute() returns a safe fallback: steer slewed toward 0, reduced
throttle.
"""

from dataclasses import dataclass

import numpy as np
from sensor_logger import (
    FRONT_CONE_BEAMS,
    FRONT_LEFT_BEAM,
    FRONT_RIGHT_BEAM,
    REAR_LEFT_BEAM,
    REAR_RIGHT_BEAM,
    SensorReadings,
)

# The beam pointed roughly perpendicular to each side wall -- the most
# direct standoff-distance measurement for that side.
SIDE_BEAM = {"left": "lidar_090", "right": "lidar_270"}
FRONT_SIDE_BEAM = {"left": FRONT_LEFT_BEAM, "right": FRONT_RIGHT_BEAM}
REAR_SIDE_BEAM = {"left": REAR_LEFT_BEAM, "right": REAR_RIGHT_BEAM}


@dataclass(frozen=True)
class WallFollowingConfig:
    """Named, documented tuning parameters -- nothing in compute() is a bare
    unexplained constant."""

    follow_side: str = "right"  # "left" or "right" -- which wall to track
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

    def __post_init__(self):
        if self.follow_side not in SIDE_BEAM:
            raise ValueError(
                f"follow_side must be 'left' or 'right', got {self.follow_side!r}"
            )


@dataclass(frozen=True)
class ControlCommand:
    """throttle -> ctrl[0:4] (N·m, + = fwd); steer -> ctrl[4] (rad, + = left)."""

    throttle: float
    steer: float


@dataclass(frozen=True)
class WallFollowingSignals:
    distance_error: float  # m; + = farther from the wall than target_distance
    wall_skew: float  # m; + = nose angled toward the wall
    front_distance: float  # m; straight-ahead clearance (FRONT_CONE_BEAMS)
    n_corrupt: int  # of the 4 beams used here that were NaN/Inf/missing


def _beam_distance(
    sensors: SensorReadings, name: str, max_beam_range: float
) -> tuple[float, bool]:
    """Read one LiDAR beam safely: (distance, corrupt). A legitimate -1
    no-hit is NOT corrupt; both resolve to max_beam_range."""
    arr = getattr(sensors, name, None)
    if arr is None or len(arr) == 0:
        return max_beam_range, True
    value = float(arr[0])
    if not np.isfinite(value):
        return max_beam_range, True
    if value < 0:
        return max_beam_range, False
    return min(value, max_beam_range), False


def compute_signals(
    sensors: SensorReadings, cfg: WallFollowingConfig
) -> WallFollowingSignals:
    """Pure function: SensorReadings -> WallFollowingSignals."""
    side, front, rear = (
        SIDE_BEAM[cfg.follow_side],
        FRONT_SIDE_BEAM[cfg.follow_side],
        REAR_SIDE_BEAM[cfg.follow_side],
    )
    readings, n_corrupt = {}, 0
    for name in (side, front, rear, *FRONT_CONE_BEAMS):
        value, corrupt = _beam_distance(sensors, name, cfg.max_beam_range)
        readings[name] = value
        n_corrupt += corrupt

    return WallFollowingSignals(
        distance_error=readings[side] - cfg.target_distance,
        wall_skew=readings[rear] - readings[front],
        front_distance=min(readings[n] for n in FRONT_CONE_BEAMS),
        n_corrupt=n_corrupt,
    )


class WallFollowingController:
    """
    Stateful PD-style single-wall standoff controller.

        controller = WallFollowingController(WallFollowingConfig())
        control = controller.compute(sensor_readings, dt)
        data.ctrl[0:4] = control.throttle
        data.ctrl[4]   = control.steer

    Call reset() when starting a new episode/rollout so stale derivative/
    slew history doesn't leak in.
    """

    def __init__(self, config: WallFollowingConfig | None = None):
        self.config = config if config is not None else WallFollowingConfig()
        # +1 for follow_side="left", -1 for "right" -- see module docstring.
        self._toward_wall_sign = 1.0 if self.config.follow_side == "left" else -1.0
        self.reset()

    def reset(self) -> None:
        self._prev_distance_error = 0.0
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
            steer = self._slew(0.0, dt)
            self._prev_steer = steer
            self._prev_distance_error = (
                0.0  # don't let a garbage step poison the D-term
            )
            return ControlCommand(
                throttle=cfg.target_throttle * cfg.safe_fallback_throttle_fraction,
                steer=steer,
            )

        error_rate = 0.0
        if dt > 0:
            error_rate = (signals.distance_error - self._prev_distance_error) / dt
        self._prev_distance_error = signals.distance_error

        steer_raw = self._toward_wall_sign * (
            cfg.kp_distance * signals.distance_error
            - cfg.kp_heading * signals.wall_skew
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
