"""
Laws for neoracer_mujoco.encoder, the speed that the car's firmware reports
from its motor encoder, and for rc.physics.get_encoder_speed(), which reads
it in the racecar_core adapter.
"""

import itertools

import pytest

from neoracer_mujoco import encoder, sim
from neoracer_mujoco.control.steering import ControlCommand
from neoracer_mujoco.racecar_core_adapter import MujocoRacecar
from neoracer_mujoco.sensors import WHEEL_RADIUS

DT = 1 / 360  # s, the car's physics step


def _spin(omega: float, seconds: float) -> list[encoder.EncoderState]:
    """The encoder after each step of a drivetrain that turns at omega
    (rad/s) from rest, for seconds."""
    state, states = encoder.start(0.0), []
    for i in range(1, round(seconds / DT) + 1):
        state = encoder.step(state, omega * i * DT, DT)
        states.append(state)
    return states


def test_windows_close_fifty_times_a_second():
    """The step does not divide 20 ms, but the windows close on the firmware's
    20 ms ticks: 50 in a second, of seven or eight steps each."""
    states = _spin(10.0, 1.0)
    closes = [i for i, s in enumerate(states) if s.span == 0.0]
    assert len(closes) == 50
    assert {b - a for a, b in itertools.pairwise(closes)} == {7, 8}


def test_the_speed_changes_only_when_a_window_closes():
    states = _spin(10.0, 0.5)
    for last, state in itertools.pairwise(states):
        if state.span != 0.0:
            assert state.speed == last.speed


def test_the_first_window_reports_most_of_a_speed_step():
    """From rest, the first window at a steady speed reports 0.95 of it: the
    filter keeps 0.05 of the last value, 0."""
    states = _spin(10.0, 0.03)
    first = next(s for s in states if s.span == 0.0)
    assert first.speed == pytest.approx(0.95 * 10.0 * WHEEL_RADIUS)


def test_a_steady_speed_reads_the_tyres_surface_speed():
    """Each window measures its own length, so at a steady speed every window
    reads omega * r, whether it ran seven steps or eight, and the filter
    converges to it."""
    omega = -12.0  # rad/s, backward
    assert _spin(omega, 0.5)[-1].speed == pytest.approx(omega * WHEEL_RADIUS)


def test_the_adapter_reads_the_car_speed(scene):
    """rc.physics.get_encoder_speed() of a car that drives straight reads its
    wheels' surface speed, and so about its speed over the ground."""
    model, car = scene.model, scene.car
    data = sim.settle(model)
    rc = MujocoRacecar(scene, data, max_steps=round(1.5 / model.opt.timestep))
    command = ControlCommand(throttle=0.5, steer=0.0)
    rc.set_start_update(lambda: sim.apply_command(car, data, command), lambda: None)
    rc.go()
    ground = float(data.bind_scalar(car.sensors.imu_linvel).data[0])  # m/s
    assert ground > 1.0
    assert rc.physics.get_encoder_speed() == pytest.approx(ground, rel=0.05)
