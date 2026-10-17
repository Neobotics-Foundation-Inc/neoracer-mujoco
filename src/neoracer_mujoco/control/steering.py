"""
Steering and throttle shaping shared by the LiDAR controllers
(track_centering.py, wall_following.py).

Frame: +X forward, +Y left, +Z up.
Throttle in [-1, 1], + = forward. Steer angle (rad), + = left.

Each controller computes one error signal (m) and a raw steer command (rad).
command() then clamps the steer, limits its rate of change, and reduces the
throttle for a large steer or a close obstacle ahead. fallback() is the
command for a step with too many corrupt beams.

Every dt in this package is the control step (s), and it must be positive.
The caller makes it once, from the model timestep or the env config.
"""

from dataclasses import dataclass

from neoracer_spec import derive
from neoracer_spec.neoracer import NEORACER


@dataclass(frozen=True)
class ControlCommand:
    """throttle goes to the drive motor (in [-1, 1], + = forward); steer goes
    to the steering servo (rad, + = left). sim.apply_command writes both."""

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

    # base forward drive command, in [-1, 1]. From rest it pushes as 0.1 N·m on
    # each wheel did before the motor model, and cruises at about 2 m/s.
    target_throttle: float = 0.5
    # rad, full lock: the steer_servo's ctrlrange
    steer_limit: float = derive.steer_lock(NEORACER)
    steer_slew_limit: float = 3.0  # rad/s max steering rate of change
    steer_slowdown_gain: float = 0.6  # throttle cut fraction at full steer deflection
    # m; at this front clearance or less, the throttle is 0
    min_safe_forward_distance: float = 0.25
    # m; below this front clearance, the throttle ramps down
    slowdown_threshold: float = 0.8
    safe_fallback_throttle_fraction: float = 0.3  # of target_throttle, during fallback

    def __post_init__(self) -> None:
        # command() divides by steer_limit and by the slowdown span, so these
        # invariants are checked once here (AGENTS.md rule 2).
        if not self.steer_limit > 0:
            raise ValueError(f"steer_limit must be > 0, got {self.steer_limit}")
        if not self.steer_slew_limit >= 0:
            raise ValueError(
                f"steer_slew_limit must be >= 0, got {self.steer_slew_limit}"
            )
        if not self.slowdown_threshold > self.min_safe_forward_distance:
            raise ValueError(
                "slowdown_threshold must be > min_safe_forward_distance, got "
                f"{self.slowdown_threshold} <= {self.min_safe_forward_distance}"
            )


def _slew(p: ShapingConfig, prev_steer: float, target: float, dt: float) -> float:
    max_delta = p.steer_slew_limit * dt
    delta = max(-max_delta, min(max_delta, target - prev_steer))
    steer = prev_steer + delta
    return max(-p.steer_limit, min(p.steer_limit, steer))


def error_rate(state: PDState, error: float, dt: float) -> float:
    """Rate of change of the error signal (m/s)."""
    return (error - state.prev_error) / dt


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
    steer = _slew(p, state.prev_steer, steer_raw, dt)

    steer_severity = abs(steer) / p.steer_limit
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
