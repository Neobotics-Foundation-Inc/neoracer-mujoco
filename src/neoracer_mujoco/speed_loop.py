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
and the throttle holds until the next one.

Limit: the ESC's output is taken as proportional to the pulse past neutral
(spec.Esc.full_throttle_pulse). Nobody has measured it.
"""

import math
from dataclasses import dataclass

from neoracer_spec.derive import SpeedLoopParams

from . import encoder


@dataclass(frozen=True)
class SpeedLoopState:
    """The loop between two physics steps. Start it with start()."""

    measure: encoder.EncoderState
    """The speed it measures, as the encoder does."""
    integral: float
    """The error's integral (m)."""
    error: float
    """The error at the last tick (m/s)."""
    throttle: float
    """The share of the battery's voltage that the motor gets, in [-1, 1]."""


def start(params: SpeedLoopParams, angle: float) -> SpeedLoopState:
    """The loop of a car at rest, its drivetrain at angle (rad): neutral."""
    return SpeedLoopState(encoder.start(params.measure, angle), 0.0, 0.0, 0.0)


def target(params: SpeedLoopParams, speed: float, max_speed: float = 1.0) -> float:
    """The speed (m/s) that racecar_core's set_speed_angle(speed, angle)
    commands, speed in [-1, 1] and max_speed (set_max_speed) in [0, 1], each
    clamped there."""
    return _clamp(speed, -1.0, 1.0) * _clamp(max_speed, 0.0, 1.0) * params.max_speed


def step(
    params: SpeedLoopParams,
    state: SpeedLoopState,
    commanded: float,
    angle: float,
    dt: float,
) -> SpeedLoopState:
    """The loop after a physics step of dt (s) that left the drivetrain at
    angle (rad), with commanded (m/s) the speed asked for."""
    measure = encoder.step(params.measure, state.measure, angle, dt)
    if measure.span != 0.0:  # the window is open: no tick
        return SpeedLoopState(measure, state.integral, state.error, state.throttle)
    if abs(commanded) < params.stop_speed:
        return SpeedLoopState(measure, 0.0, 0.0, 0.0)
    period = params.measure.window
    error = commanded - measure.speed
    if abs(error) < params.error_deadband:
        error = 0.0
    integral = _clamp(
        state.integral + error * period, -params.integral_limit, params.integral_limit
    )
    derivative = (error - state.error) / period
    pid = params.kp * error + params.ki * integral + params.kd * derivative
    pulse = _clamp(
        feed_forward(params, commanded) + math.trunc(pid),
        -params.pulse_limit,
        params.pulse_limit,
    )
    throttle = _clamp(pulse / params.full_throttle_pulse, -1.0, 1.0)
    return SpeedLoopState(measure, integral, error, throttle)


def feed_forward(params: SpeedLoopParams, commanded: float) -> float:
    """The feed-forward's pulse (us) for commanded (m/s), at or over
    stop_speed: deadband and feed_forward per m/s, rounded to a microsecond
    and held within [deadband, feed_forward_limit], toward commanded."""
    pulse = math.floor(params.deadband + abs(commanded) * params.feed_forward + 0.5)
    pulse = _clamp(pulse, params.deadband, params.feed_forward_limit)
    return pulse if commanded > 0.0 else -pulse


def _clamp(x: float, low: float, high: float) -> float:
    return min(max(x, low), high)
