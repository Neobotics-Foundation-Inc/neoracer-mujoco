"""
How the car's firmware holds the speed that a program commands.

On the car, rc.drive.set_speed_angle(speed, angle) asks for a speed: speed
times the max speed that the program set, times SpeedLoopParams.max_speed.
Every period the car's ESP32 (its firmware's task_control) measures the
speed as it does for the encoder (encoder.py), with a filter of its own,
and sets the ESC's pulse. Under stop_speed it sends neutral and clears the
loop. Otherwise it sends a feed-forward on the commanded speed, plus a PID
on the error between the commanded speed and the measured one (its
pid_update): an error under error_deadband reads as none, the integral is
held within integral_limit, and the derivative is the error's change over
the period.

This module computes the same pulse, as the throttle that sim.apply_command
writes: the pulse over the ESC's full_throttle_pulse, the share of the
battery's voltage that the motor gets, held within [-1, 1]. A period ends at
the first physics step on or after each tick, as the encoder's window does,
and the throttle holds until the next one. Its functions work on one car's
values and on a batch's, one a row (encoder.Value); the rows of a batch tick
together, and batch.drive runs them.

Limit: the ESC's output is taken as proportional to the pulse past neutral
(spec.Esc.full_throttle_pulse). Nobody has measured it.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from neoracer_spec.derive import SpeedLoopParams

from . import encoder
from .encoder import Value


@dataclass(frozen=True)
class SpeedLoopState:
    """The loop between two physics steps. Start it with start()."""

    measure: encoder.EncoderState
    """The speed it measures, as the encoder does."""
    integral: Value
    """The error's integral (m)."""
    error: Value
    """The error at the last tick (m/s)."""
    throttle: Value
    """The share of the battery's voltage that the motor gets, in [-1, 1]."""


def start(params: SpeedLoopParams, angle: Value) -> SpeedLoopState:
    """The loop of a car at rest, or a batch's, its drivetrain at angle (rad):
    neutral."""
    zero = np.zeros_like(np.asarray(angle, dtype=np.float64))[()]
    return SpeedLoopState(encoder.start(params.measure, angle), zero, zero, zero)


def restart(
    state: SpeedLoopState, rows: npt.NDArray[np.bool_], angle: Value
) -> SpeedLoopState:
    """state with the loops of rows, a mask, started over at rest, their
    drivetrains at angle (rad). The firmware's ticks go on as they were."""
    m = state.measure
    return SpeedLoopState(
        encoder.EncoderState(
            np.where(rows, angle, m.angle), m.span, m.due, np.where(rows, 0.0, m.speed)
        ),
        np.where(rows, 0.0, state.integral),
        np.where(rows, 0.0, state.error),
        np.where(rows, 0.0, state.throttle),
    )


def steps_to_tick(state: SpeedLoopState, dt: float) -> int:
    """The physics steps of dt (s) up to and with the loop's next tick."""
    due, steps = state.measure.due, 0
    while True:
        due, steps = due - dt, steps + 1
        if due < encoder.WINDOW_EPS:
            return steps


def target(params: SpeedLoopParams, speed: float, max_speed: float = 1.0) -> float:
    """The speed (m/s) that racecar_core's set_speed_angle(speed, angle)
    commands, speed in [-1, 1] and max_speed (set_max_speed) in [0, 1], each
    clamped there."""
    return _clamp(speed, -1.0, 1.0) * _clamp(max_speed, 0.0, 1.0) * params.max_speed


def step(
    params: SpeedLoopParams,
    state: SpeedLoopState,
    commanded: Value,
    angle: Value,
    dt: float,
) -> SpeedLoopState:
    """The loop after a physics step of dt (s) that left the drivetrain at
    angle (rad), with commanded (m/s) the speed asked for."""
    measure = encoder.step(params.measure, state.measure, angle, dt)
    if measure.span != 0.0:  # the window is open: no tick
        return SpeedLoopState(measure, state.integral, state.error, state.throttle)
    integral, error, throttle = tick(
        params, commanded, measure.speed, state.integral, state.error
    )
    return SpeedLoopState(measure, integral, error, throttle)


def tick(
    params: SpeedLoopParams,
    commanded: Value,
    measured: Value,
    integral: Value,
    error: Value,
) -> tuple[Value, Value, Value]:
    """The firmware's loop at a tick: the error's integral, the error and the
    throttle after it, from the speed commanded and the speed measured (m/s)
    and the integral and the error before it. Under stop_speed it sends
    neutral and clears the integral and the error."""
    commanded = np.asarray(commanded, dtype=np.float64)
    period = params.measure.window
    new_error = commanded - measured
    new_error = np.where(np.abs(new_error) < params.error_deadband, 0.0, new_error)
    limit = params.integral_limit
    new_integral = np.clip(integral + new_error * period, -limit, limit)
    derivative = (new_error - error) / period
    pid = params.kp * new_error + params.ki * new_integral + params.kd * derivative
    pulse = np.clip(
        feed_forward(params, commanded) + np.trunc(pid),
        -params.pulse_limit,
        params.pulse_limit,
    )
    throttle = np.clip(pulse / params.full_throttle_pulse, -1.0, 1.0)
    stopped = np.abs(commanded) < params.stop_speed
    return (
        np.where(stopped, 0.0, new_integral)[()],
        np.where(stopped, 0.0, new_error)[()],
        np.where(stopped, 0.0, throttle)[()],
    )


def feed_forward(params: SpeedLoopParams, commanded: Value) -> Value:
    """The feed-forward's pulse (us) for commanded (m/s), at or over
    stop_speed: deadband and feed_forward per m/s, rounded to a microsecond
    and held within [deadband, feed_forward_limit], toward commanded."""
    commanded = np.asarray(commanded, dtype=np.float64)
    pulse = np.clip(
        np.floor(params.deadband + np.abs(commanded) * params.feed_forward + 0.5),
        params.deadband,
        params.feed_forward_limit,
    )
    return np.where(commanded > 0.0, pulse, -pulse)[()]


def _clamp(x: float, low: float, high: float) -> float:
    return min(max(x, low), high)
