"""
Steering and speed shaping shared by the LiDAR controllers
(track_centering.py, wall_following.py), and the two commands a car takes.

Frame: +X forward, +Y left, +Z up.
Speed (m/s) and throttle in [-1, 1], + = forward. Steer angle (rad), + = left.

Each controller computes one error signal (m) and a raw steer command (rad).
command() then clamps the steer, limits its rate of change, and lowers the
speed for a large steer or a close obstacle ahead. fallback() is the command
for a step with too many corrupt beams. A controller asks for a speed, as a
program on the car does, and the car's speed loop holds it (racecar.ask).

Every dt in this package is the control step (s), and it must be positive.
The caller makes it once, from the model timestep or the env config.
"""

from dataclasses import dataclass

from neoracer_spec import derive
from neoracer_spec.neoracer import NEORACER


@dataclass(frozen=True)
class SpeedAngle:
    """What a program on the car asks for: target, the speed (m/s, + =
    forward) for the firmware's speed loop to hold, and steer, the steering
    servo's angle (rad, + = left). racecar.ask gives it to a car."""

    target: float
    steer: float


@dataclass(frozen=True)
class ControlCommand:
    """throttle goes to the drive motor (in [-1, 1], + = forward); steer goes
    to the steering servo (rad, + = left). sim.apply_command writes both. It
    is the motor's own input, under the speed loop, which a program on the
    car does not set."""

    throttle: float
    steer: float


@dataclass(frozen=True)
class PDState:
    """Controller state from one step to the next. Start an episode with
    PDState()."""

    prev_error: float = 0.0
    """The error signal of the last step (m), for the D term."""
    prev_steer: float = 0.0
    """The steer command of the last step (rad), for the slew limit."""


@dataclass(frozen=True)
class ShapingConfig:
    """Steer and speed limits that command() and fallback() apply."""

    target_speed: float = 1.75
    """The speed (m/s) on a straight with the way ahead clear."""
    steer_limit: float = derive.steer_lock(NEORACER)
    """Full lock (rad): the steer_servo's ctrlrange."""
    steer_slew_limit: float = 3.0
    """The steer's largest rate of change (rad/s)."""
    steer_slowdown_gain: float = 0.6
    """The fraction of the speed cut at full steer deflection."""
    min_safe_forward_distance: float = 0.25
    """The front clearance (m) at or below which the speed is 0."""
    slowdown_threshold: float = 0.8
    """The front clearance (m) below which the speed ramps down."""
    safe_fallback_speed_fraction: float = 0.3
    """The speed during fallback, as a fraction of target_speed."""

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
) -> tuple[PDState, SpeedAngle]:
    """Shape a raw steer command (rad) into the next state and command.
    error (m) is kept for the next D term. front_distance (m) is the
    clearance straight ahead."""
    steer = _slew(p, state.prev_steer, steer_raw, dt)

    steer_severity = abs(steer) / p.steer_limit
    speed = p.target_speed * (1.0 - p.steer_slowdown_gain * steer_severity)

    if front_distance <= p.min_safe_forward_distance:
        obstacle_scale = 0.0
    elif front_distance >= p.slowdown_threshold:
        obstacle_scale = 1.0
    else:
        span = p.slowdown_threshold - p.min_safe_forward_distance
        obstacle_scale = (front_distance - p.min_safe_forward_distance) / span
    speed = max(0.0, speed * obstacle_scale)

    return PDState(prev_error=error, prev_steer=steer), SpeedAngle(
        target=speed, steer=steer
    )


def fallback(p: ShapingConfig, state: PDState, dt: float) -> tuple[PDState, SpeedAngle]:
    """Command for a step with bad LiDAR data: slew the steer toward 0 and
    drive at a reduced speed. The error history is cleared, so that the bad
    step does not cause a D-term spike in the next step."""
    steer = _slew(p, state.prev_steer, 0.0, dt)
    return PDState(prev_error=0.0, prev_steer=steer), SpeedAngle(
        target=p.target_speed * p.safe_fallback_speed_fraction, steer=steer
    )
