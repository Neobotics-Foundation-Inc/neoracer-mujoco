"""
The speed that the car's firmware reports from its motor encoder.

Every window (EncoderParams.window, the spec's MotorEncoder.report_period),
the car's ESP32 (its firmware's task_control) reads the motor encoder's
count over the window and turns it into the tyre's surface speed. It then
low-pass filters it, with the weight EncoderParams.alpha on the new
window's speed. It reports the result as /encoder/speed, which a program
reads as rc.physics.get_encoder_speed().

This module computes the same speed from the drivetrain's angle, the
drive_pos sensor. The physics step does not divide the window. So a window
closes at the first step on or after each tick, and its speed is the angle
it covered over its own length. Windows then run a whole number of steps or
one more, and close as often as on the car.

Limit: the speed is not quantized. One count of the encoder, a pulse
through the drivetrain's reduction, is about a millimetre a second over a
window, small beside any speed a program acts on.
"""

from dataclasses import dataclass

from neoracer_spec.derive import EncoderParams

from .sensors import WHEEL_RADIUS


@dataclass(frozen=True)
class EncoderState:
    """The encoder between two physics steps. Start it with start()."""

    angle: float  # rad, the drivetrain's angle when the window opened
    span: float  # s since the window opened
    due: float  # s until the firmware's next tick
    speed: float  # m/s, + = forward: the last speed the firmware reported


def start(params: EncoderParams, angle: float) -> EncoderState:
    """The encoder of a car at rest, its drivetrain at angle (rad)."""
    return EncoderState(angle=angle, span=0.0, due=params.window, speed=0.0)


# A window closes when less than this is left of it (s), so that a step that
# lands on a tick, up to rounding, closes the window.
_EPS = 1e-9


def step(
    params: EncoderParams, state: EncoderState, angle: float, dt: float
) -> EncoderState:
    """The encoder after a physics step of dt (s) that left the drivetrain at
    angle (rad)."""
    span, due = state.span + dt, state.due - dt
    if due >= _EPS:
        return EncoderState(state.angle, span, due, state.speed)
    raw = (angle - state.angle) * WHEEL_RADIUS / span  # m/s over the window
    speed = params.alpha * raw + (1.0 - params.alpha) * state.speed
    return EncoderState(angle, 0.0, due + params.window, speed)
