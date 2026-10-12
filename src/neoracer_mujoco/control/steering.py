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


@dataclass(frozen=True)
class ShapingConfig:
    """Steer and throttle limits that command() and fallback() apply."""

    target_throttle: float = 0.10  # N·m per wheel, base forward drive command
    steer_limit: float = 0.4  # rad; matches ctrl[4] ctrlrange in neoracer.xml
    steer_slew_limit: float = 3.0  # rad/s max steering rate of change
    steer_slowdown_gain: float = 0.6  # throttle cut fraction at full steer deflection
    min_safe_forward_distance: float = 0.25  # m; front clearance -> throttle 0
    slowdown_threshold: float = (
        0.8  # m; front clearance below which throttle ramps down
    )
    safe_fallback_throttle_fraction: float = 0.3  # of target_throttle, during fallback


def _slew(p: ShapingConfig, prev_steer: float, target: float, dt: float) -> float:
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
    p: ShapingConfig,
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
    p: ShapingConfig, state: PDState, dt: float
) -> tuple[PDState, ControlCommand]:
    """Command for a step with bad LiDAR data: slew the steer toward 0 and
    drive at a reduced throttle. The error history is cleared, so that the
    bad step does not cause a D-term spike in the next step."""
    steer = _slew(p, state.prev_steer, 0.0, dt)
    return PDState(prev_error=0.0, prev_steer=steer), ControlCommand(
        throttle=p.target_throttle * p.safe_fallback_throttle_fraction, steer=steer
    )
