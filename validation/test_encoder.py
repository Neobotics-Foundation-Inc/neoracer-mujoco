"""
Laws for neoracer_mujoco.encoder, the speed that the car's firmware reports
from its motor encoder, and for rc.physics.get_encoder_speed(), which reads
it in the racecar_core adapter.
"""

import copy
import itertools

import mujoco
import numpy as np
import pytest
from cases import case

from neoracer_mujoco import Scene, assets, batch, encoder, load, sim
from neoracer_mujoco.control.steering import ControlCommand
from neoracer_mujoco.racecar_core_adapter import MujocoRacecar, neoracer_params
from neoracer_spec.derive import EncoderParams, derive
from neoracer_spec.neoracer import NEORACER
from neoracer_spec.parts import SOLVER

PARAMS = derive(NEORACER).encoder

DT = SOLVER.timestep
"""The car's physics step (s)."""

HELD_THROTTLE = copy.replace(neoracer_params(), speed_loop=None)
"""A NeoRacer without its firmware's speed loop, so that a test can hold the
throttle at a value of its own."""


def _spin(
    params: EncoderParams, omega: float, seconds: float
) -> list[encoder.EncoderState]:
    """The encoder after each step of a drivetrain that turns at omega
    (rad/s) from rest, for seconds."""
    state, states = encoder.start(params, 0.0), []
    for i in range(1, round(seconds / DT) + 1):
        state = encoder.step(params, state, omega * i * DT, DT)
        states.append(state)
    return states


def test_windows_close_on_the_firmwares_ticks():
    """The step does not divide 20 ms, but the windows close on the firmware's
    20 ms ticks: 50 in a second, of seven or eight steps each, and the speed
    changes only when a window closes."""
    states = _spin(PARAMS, 10.0, 1.0)
    closes = [i for i, s in enumerate(states) if s.span == 0.0]
    assert len(closes) == 50
    assert {b - a for a, b in itertools.pairwise(closes)} == {7, 8}
    for last, state in itertools.pairwise(states):
        if state.span != 0.0:
            assert state.speed == last.speed


def test_the_filter_converges_to_the_tires_surface_speed():
    """Each window measures its own length, so at a steady speed every window
    reads omega * r, r the params' radius, whether it ran seven steps or
    eight. From rest, the filter, which keeps 1 - alpha of its last value,
    then reports (1 - (1 - alpha)^k) omega r at the k-th window."""
    omega = -12.0  # rad/s, backward
    for params in (PARAMS, copy.replace(PARAMS, radius=2 * PARAMS.radius)):
        with case(params.radius):
            windows = [s.speed for s in _spin(params, omega, 0.5) if s.span == 0.0]
            for k, speed in enumerate(windows, start=1):
                expected = (1 - (1 - params.alpha) ** k) * omega * params.radius
                assert speed == pytest.approx(expected), f"window {k}"


def test_the_adapter_reads_the_car_speed():
    """rc.physics.get_encoder_speed() of a car that drives straight reads its
    wheels' surface speed, and so about its speed over the ground. The
    adapter's encoder is the NeoRacer's, so these tests drive the NeoRacer."""
    scene = load()
    model, car = scene.model, scene.car
    data = sim.settle(model)
    rc = MujocoRacecar(scene, data, max_steps=round(1.5 / model.opt.timestep))
    rc.set_start_update(lambda: rc.drive.set_speed_angle(0.5, 0.0), lambda: None)
    rc.go()
    ground = float(data.bind_scalar(car.sensors.imu_linvel).data[0])  # m/s
    assert ground > 1.0
    assert rc.physics.get_encoder_speed() == pytest.approx(ground, rel=0.05)


def _wheel_angle(scene: Scene, data: mujoco.MjData) -> float:
    """The mean of the four wheels' spin angles (rad), read from qpos: the
    hinge about the axle (+Y) of each wheel's body."""
    spins = [
        j
        for wheel in scene.car.wheels
        for j in wheel.parent.joints
        if j.type == mujoco.mjtJoint.mjJNT_HINGE and np.allclose(j.axis, [0, 1, 0])
    ]
    return float(np.mean([data.bind_scalar(j).qpos[0] for j in spins]))


def test_the_first_window_measures_the_turn_of_the_wheels_in_it():
    """The car speeds up from rest, on data that mj_forward last updated.
    The first speed the encoder reports is alpha times the wheels' turn
    between the first update() and the update() that closes the window,
    over that time: the encoder reads the angle the car has at each
    update(), not the one of a step before."""
    scene = load()
    model, car = scene.model, scene.car
    data = sim.settle(model)
    mujoco.mj_forward(model, data)
    rc = MujocoRacecar(
        scene, data, max_steps=round(0.1 / model.opt.timestep), car=HELD_THROTTLE
    )
    seen = []  # (s, rad, m/s) at each update()
    command = ControlCommand(throttle=1.0, steer=0.0)
    rc.set_start_update(
        lambda: sim.apply_command(car, data, command),
        lambda: seen.append(
            (data.time, _wheel_angle(scene, data), rc.physics.get_encoder_speed())
        ),
    )
    rc.go()
    t0, a0, _ = seen[0]
    t1, a1, first = next(x for x in seen if x[2] != 0.0)
    assert first == pytest.approx(PARAMS.alpha * (a1 - a0) * PARAMS.radius / (t1 - t0))


def test_a_bounded_run_counts_its_last_step():
    """go() with max_steps returns with every step counted: from rest at
    full throttle, 8 steps at 360 Hz (22 ms) pass the first 20 ms tick, so
    the encoder reports a speed."""
    scene = load()
    model, car = scene.model, scene.car
    data = sim.settle(model)
    rc = MujocoRacecar(scene, data, max_steps=8, car=HELD_THROTTLE)
    command = ControlCommand(throttle=1.0, steer=0.0)
    rc.set_start_update(lambda: sim.apply_command(car, data, command), lambda: None)
    rc.go()
    assert model.opt.timestep == pytest.approx(1 / 360)
    assert rc.physics.get_encoder_speed() > 0.0


def test_drive_angle_is_drive_pos_at_the_steps_end():
    """After mj_step the drive_pos sensor still holds the drivetrain's angle
    at the step's start. batch.drive_angle gives it at the step's end, the
    reading that mj_forward refreshes, on both cars and on rows of qpos."""
    for path in (assets.DEFAULT_CAR, assets.ASSET_DIR + "/racecar_neo_v2.xml"):
        with case(path):
            scene = load(path)
            model, car = scene.model, scene.car
            data = sim.settle(model)
            sim.apply_command(car, data, ControlCommand(0.5, 0.0))
            for _ in range(10):
                mujoco.mj_step(model, data)
            sensor = data.bind_scalar(car.sensors.drive_pos)
            start = float(sensor.data[0])
            end = batch.drive_angle(model, car, data.qpos)
            rows = batch.drive_angle(model, car, np.stack([data.qpos] * 2))
            mujoco.mj_forward(model, data)
            assert end == rows[0] == rows[1] == float(sensor.data[0]) != start


def test_a_drive_pos_that_reads_no_fixed_tendons_length_is_refused():
    """drive_angle sums a fixed tendon's length, so it refuses a drive_pos
    sensor that reads anything else: one wheel's joint, or the tendon's
    speed."""
    joint = (mujoco.mjtSensor.mjSENS_JOINTPOS, mujoco.mjtObj.mjOBJ_JOINT, "rr_drive")
    nominal = load()
    tendon = nominal.spec.sensor(nominal.car.sensors.drive_pos.name).objname
    speed = (mujoco.mjtSensor.mjSENS_TENDONVEL, mujoco.mjtObj.mjOBJ_TENDON, tendon)
    for kind, objtype, objname in (joint, speed):
        with case(kind):
            scene = load()
            sensor = scene.spec.sensor(scene.car.sensors.drive_pos.name)
            sensor.type, sensor.objtype, sensor.objname = kind, objtype, objname
            qpos = np.zeros(scene.model.nq)
            with pytest.raises(ValueError, match="fixed tendon's length"):
                batch.drive_angle(scene.spec.compile(), scene.car, qpos)
