"""
Laws for neoracer_mujoco.speed_loop, the firmware's loop that holds the speed
a program commands, and for the racecar_core adapter's drive, which runs it.
"""

import copy
import itertools
import math
from typing import Any

import numpy as np
import pytest
from cases import case

from neoracer_mujoco import contract, load, sim, speed_loop
from neoracer_mujoco.racecar_core_adapter import MujocoRacecar
from neoracer_spec.derive import (
    SPEED_LOOP_NONNEGATIVE,
    SPEED_LOOP_POSITIVE,
    SpeedLoopParams,
    derive,
)
from neoracer_spec.neoracer import NEORACER
from neoracer_spec.parts import SOLVER

PARAMS = derive(NEORACER).speed_loop

DT = SOLVER.timestep
"""The car's physics step (s)."""


def _run(
    params: SpeedLoopParams, commanded: list[float], angles: list[float]
) -> list[speed_loop.SpeedLoopState]:
    """The loop after each step of a drivetrain that passes through angles
    (rad), the first one at the start, with commanded (m/s) at each step."""
    state, states = speed_loop.start(params, angles[0]), []
    for target, angle in zip(commanded, angles[1:], strict=True):
        state = speed_loop.step(params, state, target, angle, DT)
        states.append(state)
    return states


def _turning(omega: float, seconds: float) -> list[float]:
    """The angles (rad) of a drivetrain that turns at omega (rad/s) from 0,
    a step apart."""
    return [omega * i * DT for i in range(round(seconds / DT) + 1)]


ROW_VALUES = ("angle", "speed", "integral", "error", "throttle")
"""What a loop holds for each row."""


def _rows(state: speed_loop.SpeedLoopState) -> dict[str, np.ndarray]:
    """state's row values, by name."""
    m = state.measure
    values = (m.angle, m.speed, state.integral, state.error, state.throttle)
    return {n: np.asarray(v) for n, v in zip(ROW_VALUES, values, strict=True)}


def test_the_throttle_changes_only_at_the_firmwares_ticks():
    """The loop sets the throttle when its window closes, on the firmware's
    20 ms ticks, and holds it between them."""
    angles = _turning(20.0, 1.0)
    states = _run(PARAMS, [1.5] * (len(angles) - 1), angles)
    ticks = [i for i, s in enumerate(states) if s.measure.span == 0.0]
    assert len(ticks) == 50
    for i, (last, state) in enumerate(itertools.pairwise(states), start=1):
        if i not in ticks:
            assert state.throttle == last.throttle


def test_a_tick_from_rest_is_the_firmwares_pid_from_a_cleared_loop():
    """At the first tick after a command, a still drivetrain measures 0, so
    the error is the commanded speed v, its integral v T and its change v / T
    over the period T. The pulse is the feed-forward plus the PID's output
    truncated to a microsecond, held within pulse_limit, and the throttle is
    the pulse over full_throttle_pulse."""
    period = PARAMS.measure.window
    for v in (0.3, 1.0, -0.6, 2.0, -6.0):
        with case(v):
            states = _run(PARAMS, [v] * 8, [0.0] * 9)
            first = next(s for s in states if s.measure.span == 0.0)
            pid = PARAMS.kp * v + PARAMS.ki * v * period + PARAMS.kd * v / period
            pulse = speed_loop.feed_forward(PARAMS, v) + math.trunc(pid)
            pulse = min(max(pulse, -PARAMS.pulse_limit), PARAMS.pulse_limit)
            assert first.throttle == pulse / PARAMS.full_throttle_pulse
            assert -1.0 <= first.throttle <= 1.0


def test_an_esc_at_full_voltage_before_the_pulse_limit_holds_full_throttle():
    """An ESC that gives the motor the battery's whole voltage at a pulse
    under the firmware's pulse limit holds the throttle at 1 for any pulse
    past it, either way."""
    early = copy.replace(PARAMS, full_throttle_pulse=PARAMS.pulse_limit / 2)
    for v in (6.0, -6.0):
        with case(v):
            first = next(
                s for s in _run(early, [v] * 8, [0.0] * 9) if s.measure.span == 0.0
            )
            assert first.throttle == math.copysign(1.0, v)


def test_a_loop_that_the_firmware_cannot_be_is_refused():
    """A loop must send neutral for a commanded speed of 0, so its stop speed
    must be positive, as must its pulse limits and its max speed. A negative
    gain, bound or feed-forward would turn its part of the loop around; 0
    turns it off."""

    def loop(name: str, value: float) -> SpeedLoopParams:
        fields: dict[str, Any] = {**vars(PARAMS), name: value}
        return SpeedLoopParams(**fields)

    for name in SPEED_LOOP_POSITIVE:
        for bad in (0.0, -1.0):
            with case((name, bad)), pytest.raises(ValueError, match=name):
                loop(name, bad)
    for name in SPEED_LOOP_NONNEGATIVE:
        with case(name), pytest.raises(ValueError, match=name):
            loop(name, -1.0)
        loop(name, 0.0)


def test_a_batchs_rows_tick_as_cars_of_their_own():
    """tick on a batch's rows, each with its own commanded and measured
    speeds, integral and last error, gives each row what tick gives that row
    alone, stopped rows and saturated ones among them."""
    commanded = np.array([0.0, 0.3, -1.2, 6.0, 0.04, 2.0])
    measured = np.array([0.5, 0.0, -0.4, 3.4, 0.2, 2.03])
    integral = np.array([0.1, 0.0, -0.3, 900.0, 0.2, 0.0])
    error = np.array([0.2, 0.0, 0.1, 2.6, -0.1, 0.0])
    rows = speed_loop.tick(PARAMS, commanded, measured, integral, error)
    for k in range(len(commanded)):
        with case(k):
            alone = speed_loop.tick(
                PARAMS, commanded[k], measured[k], integral[k], error[k]
            )
            assert tuple(float(np.asarray(v)[k]) for v in rows) == alone


def test_the_target_is_the_held_share_of_the_max_speed():
    """target is speed, held within [-1, 1], times max_speed, held within
    [0, 1], times the loop's max speed, for one car and for each row of a
    batch."""
    speeds = np.array([-2.0, -0.5, 0.0, 0.3, 1.5])
    for max_speed in (-0.5, 0.0, 0.4, 1.0, 2.0):
        with case(max_speed):
            rows = np.asarray(speed_loop.target(PARAMS, speeds, max_speed))
            for k, speed in enumerate(speeds):
                held = min(max(speed, -1.0), 1.0) * min(max(max_speed, 0.0), 1.0)
                one = speed_loop.target(PARAMS, float(speed), max_speed)
                assert rows[k] == one == held * PARAMS.max_speed


def test_rows_that_start_over_start_at_rest_and_keep_the_ticks():
    """restart clears the loops of the rows it is given, at their new angles,
    and leaves the others and the shared ticks as they were, a window open."""
    angles = _turning(10.0, 0.51)
    loop = speed_loop.start(PARAMS, np.full(3, angles[0]))
    for a in angles[1:]:
        loop = speed_loop.step(
            PARAMS, loop, np.array([1.0, 2.0, -1.0]), np.full(3, a), DT
        )
    assert loop.measure.span > 0.0
    rows = np.array([False, True, False])
    again = speed_loop.restart(loop, rows, np.full(3, 7.0))
    assert (again.measure.span, again.measure.due) == (
        loop.measure.span,
        loop.measure.due,
    )
    after, before = _rows(again), _rows(loop)
    assert after["angle"][1] == 7.0
    assert all(after[name][1] == 0.0 for name in ROW_VALUES if name != "angle")
    for k in (0, 2):
        assert all(after[name][k] == before[name][k] for name in ROW_VALUES)


def test_a_row_that_starts_over_starts_every_car_of_it_over():
    """On a loop of two cars a row, as batch.drive_cars runs it, restart with
    a mask of rows starts over every car of those rows, as the mask of each
    car of those rows does."""
    angles = _turning(10.0, 0.51)
    loop = speed_loop.start(PARAMS, np.full((3, 2), angles[0]))
    commanded = np.array([[1.0, -0.5], [2.0, 0.5], [-1.0, 1.5]])
    for a in angles[1:]:
        loop = speed_loop.step(PARAMS, loop, commanded, np.full((3, 2), a), DT)
    rows = np.array([False, True, False])
    angle = np.full((3, 2), 7.0)
    by_row = _rows(speed_loop.restart(loop, rows, angle))
    by_car = _rows(speed_loop.restart(loop, np.repeat(rows[:, None], 2, 1), angle))
    for name in ROW_VALUES:
        np.testing.assert_array_equal(by_row[name], by_car[name], err_msg=name)
    assert by_row["angle"][1].tolist() == [7.0, 7.0]


def test_the_feed_forward_rounds_and_holds_within_its_range():
    """The feed-forward is deadband plus feed_forward per m/s, rounded to a
    microsecond, toward the commanded speed, and held within [deadband,
    feed_forward_limit]."""
    for v, pulse in (
        (0.05, 93.0),
        (1.0, 158.0),
        (-2.5, -260.0),
        (6.0, 498.0),
        (9.0, 500.0),
    ):
        with case(v):
            assert speed_loop.feed_forward(PARAMS, v) == pulse


def test_under_the_stop_speed_the_loop_sends_neutral_and_starts_over():
    """A commanded speed under stop_speed, either way, sends neutral at the
    next tick and clears the integral and the last error, so a new command
    starts as from rest."""
    for stop in (0.0, 0.04, -0.04):
        with case(stop):
            angles = _turning(5.0, 1.0)
            n = len(angles) - 1
            commanded = [2.0] * (n // 2) + [stop] * (n - n // 2)
            states = _run(PARAMS, commanded, angles)
            assert states[n // 2 - 1].integral != 0.0
            after = [s for s in states[n // 2 :] if s.measure.span == 0.0]
            assert all(s.throttle == 0.0 for s in after)
            assert all(s.integral == 0.0 and s.error == 0.0 for s in after)


def test_an_error_under_the_deadband_reads_as_none():
    """A drivetrain at the commanded speed, measured within error_deadband of
    it, adds nothing to the integral, and its pulse is the feed-forward."""
    v = 1.0
    omega = v / PARAMS.measure.radius  # rad/s
    loose = copy.replace(PARAMS, measure=copy.replace(PARAMS.measure, alpha=1.0))
    states = _run(loose, [v] * round(0.2 / DT), _turning(omega, 0.2))
    ticks = [s for s in states if s.measure.span == 0.0]
    assert all(s.integral == 0.0 for s in ticks)
    assert ticks[-1].throttle == speed_loop.feed_forward(loose, v) / (
        loose.full_throttle_pulse
    )


def test_the_loop_is_odd():
    """Negating the commanded speed and the drivetrain's turn negates every
    throttle the loop sets."""
    angles = [0.3 * math.sin(0.05 * i) + 0.4 * i for i in range(400)]
    commanded = [2.0 if i < 200 else 0.7 for i in range(399)]
    forward = _run(PARAMS, commanded, angles)
    backward = _run(PARAMS, [-c for c in commanded], [-a for a in angles])
    for f, b in zip(forward, backward, strict=True):
        assert b.throttle == -f.throttle


def test_the_car_holds_the_speed_that_a_program_asks_for():
    """A program's speed on the NeoRacer asks for that share of max_speed,
    and the loop holds the car on the flat within 0.15 m/s of it after 3 s,
    from 0.5 to 2 m/s. Full speed, beyond what the motor reaches, holds full
    throttle."""
    for speed in (1 / 12, 1 / 6, 1 / 3, 1.0):
        with case(speed):
            scene = load()
            data = sim.settle(scene.model)
            steps = round(3.0 / scene.model.opt.timestep)
            rc = MujocoRacecar(scene, data, max_steps=steps)
            drive = rc.drive
            rc.set_start_update(
                lambda s=speed, d=drive: d.set_speed_angle(s, 0.0), lambda: None
            )
            rc.go()
            ground = float(data.bind_scalar(scene.car.sensors.imu_linvel).data[0])
            target = speed * PARAMS.max_speed
            if speed < 1.0:
                assert ground == pytest.approx(target, abs=0.15)
            else:
                full = contract.actuator_limits(scene.model, scene.car).voltage
                assert ground < target
                assert float(data.bind_scalar(scene.car.drive).ctrl[0]) == full
