"""
Steering and throttle shaping shared by the LiDAR controllers
(track_centering.py, wall_following.py).

Frame: +X forward, +Y left, +Z up.
ctrl[0:4] = wheel torque (N·m), + = forward. ctrl[4] = steer angle (rad), + = left.

Each controller computes one error signal (m) and a raw steer command (rad).
command() then clamps the steer, limits its rate of change, and reduces the
throttle for a large steer or a close obstacle ahead. fallback() is the
command for a step with too many corrupt beams.
"""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ControlCommand:
    """throttle goes to ctrl[0:4] (N·m, + = forward); steer goes to ctrl[4]
    (rad, + = left)."""

    throttle: float
    steer: float


@dataclass(frozen=True)
class PDState:
    """Controller state from one step to the next. Start an episode with
    PDState()."""

    prev_error: float = 0.0  # m; error signal of the last step, for the D term
    prev_steer: float = 0.0  # rad; steer command of the last step, for the slew limit


class ShapingParams(Protocol):
    """The tuning parameters that command() and fallback() read."""

    @property
    def target_throttle(self) -> float: ...  # N·m per wheel
    @property
    def steer_limit(self) -> float: ...  # rad
    @property
    def steer_slew_limit(self) -> float: ...  # rad/s
    @property
    def steer_slowdown_gain(self) -> float: ...  # throttle cut at full steer
    @property
    def min_safe_forward_distance(self) -> float: ...  # m; throttle 0 at or below
    @property
    def slowdown_threshold(self) -> float: ...  # m; throttle ramps down below
    @property
    def safe_fallback_throttle_fraction(self) -> float: ...  # of target_throttle


def _slew(p: ShapingParams, prev_steer: float, target: float, dt: float) -> float:
    max_delta = p.steer_slew_limit * max(dt, 0.0)
    delta = max(-max_delta, min(max_delta, target - prev_steer))
    steer = prev_steer + delta
    return max(-p.steer_limit, min(p.steer_limit, steer))


def error_rate(state: PDState, error: float, dt: float) -> float:
    """Rate of change of the error signal (m/s). Zero if dt is not positive."""
    if dt > 0:
        return (error - state.prev_error) / dt
    return 0.0


def command(
    p: ShapingParams,
    state: PDState,
    error: float,
    steer_raw: float,
    front_distance: float,
    dt: float,
) -> tuple[PDState, ControlCommand]:
    """Shape a raw steer command (rad) into the next state and command.
    error (m) is kept for the next D term. front_distance (m) is the
    clearance straight ahead."""
    steer_raw = max(-p.steer_limit, min(p.steer_limit, steer_raw))
    steer = _slew(p, state.prev_steer, steer_raw, dt)

    steer_severity = abs(steer) / p.steer_limit if p.steer_limit > 0 else 0.0
    throttle = p.target_throttle * (1.0 - p.steer_slowdown_gain * steer_severity)

    if front_distance <= p.min_safe_forward_distance:
        obstacle_scale = 0.0
    elif front_distance >= p.slowdown_threshold:
        obstacle_scale = 1.0
    else:
        span = p.slowdown_threshold - p.min_safe_forward_distance
        obstacle_scale = (front_distance - p.min_safe_forward_distance) / span
    throttle = max(0.0, throttle * obstacle_scale)

    return PDState(prev_error=error, prev_steer=steer), ControlCommand(
        throttle=throttle, steer=steer
    )


def fallback(
    p: ShapingParams, state: PDState, dt: float
) -> tuple[PDState, ControlCommand]:
    """Command for a step with bad LiDAR data: slew the steer toward 0 and
    drive at a reduced throttle. The error history is cleared, so that the
    bad step does not cause a D-term spike in the next step."""
    steer = _slew(p, state.prev_steer, 0.0, dt)
    return PDState(prev_error=0.0, prev_steer=steer), ControlCommand(
        throttle=p.target_throttle * p.safe_fallback_throttle_fraction, steer=steer
    )
