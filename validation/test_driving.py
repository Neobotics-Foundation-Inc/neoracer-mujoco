"""
Laws for what the driving demos share (examples/driving.py): how the command
follows the driver's input, and the car that it drives. No window: neither
glfw nor pygame is imported.
"""

import itertools
import os

import mujoco
import numpy as np
import pytest
from cases import case

from examples.driving import Drive, Rates, Shaping, apply_and_step, next_drive
from neoracer_mujoco import assets, contract

# Rates of their own toward an input and back to 0, so the laws tell them apart.
SHAPING = Shaping(
    throttle=Rates(toward=2.0, back=1.0), steer=Rates(toward=3.0, back=5.0)
)
DT = 0.02  # s, one 50 Hz frame


def _hold(drive: Drive, inputs: Drive, frames: int) -> list[Drive]:
    """The commands of frames frames with inputs held, from drive."""
    drives = [drive]
    for _ in range(frames):
        drives.append(next_drive(SHAPING, drives[-1], inputs, DT))
    return drives


def _mirror(d: Drive) -> Drive:
    return Drive(throttle=-d.throttle, steer=-d.steer)


def test_a_held_input_is_reached_without_overshoot_and_held():
    """The command moves toward a held input, clipped to [-1, 1], at the
    rate toward it, never past it, and stays there."""
    for value in (-3.0, -1.0, 0.3, 1.0, 3.0):
        with case(value):
            drives = _hold(Drive(), Drive(throttle=value, steer=-value), 100)
            target = max(-1.0, min(1.0, value))
            assert drives[1].throttle == pytest.approx(
                np.sign(target) * SHAPING.throttle.toward * DT
            )
            assert drives[-1] == Drive(throttle=target, steer=-target)
            assert all(abs(d.throttle) <= abs(target) for d in drives)


def test_a_released_input_returns_the_command_to_exactly_zero():
    """With no input, the command falls to 0 at the rate back, and stays."""
    drives = _hold(Drive(throttle=1.0, steer=-1.0), Drive(), 100)
    assert drives[1].throttle == pytest.approx(1.0 - SHAPING.throttle.back * DT)
    assert drives[1].steer == pytest.approx(-1.0 + SHAPING.steer.back * DT)
    throttles = [d.throttle for d in drives]
    assert all(b <= a for a, b in itertools.pairwise(throttles))
    assert drives[-1] == Drive()


def test_mirrored_inputs_give_a_mirrored_command():
    rng = np.random.default_rng(0)
    for i in range(5):
        with case(i):
            drive = Drive(*rng.uniform(-1.0, 1.0, 2))
            inputs = Drive(*rng.choice([-1.0, 0.0, 1.0], 2))
            mirrored = next_drive(SHAPING, _mirror(drive), _mirror(inputs), DT)
            assert mirrored == _mirror(next_drive(SHAPING, drive, inputs, DT))


def test_a_long_frame_is_cut_and_a_negative_one_changes_nothing():
    long = next_drive(SHAPING, Drive(), Drive(throttle=1.0), 10.0)
    assert long.throttle == pytest.approx(SHAPING.throttle.toward * SHAPING.max_dt)
    drive = Drive(throttle=0.5, steer=-0.5)
    assert next_drive(SHAPING, drive, Drive(1.0, 1.0), -0.1) == drive


def _drive(scene, data, inputs: Drive, seconds: float) -> None:
    drive = Drive()
    for _ in range(round(seconds / DT)):
        drive = next_drive(SHAPING, drive, inputs, DT)
        apply_and_step(scene, data, drive, DT)


def _speed(scene, data) -> float:
    return float(np.linalg.norm(data.bind_scalar(scene.car.joint).qvel[0:2]))


def test_a_steer_of_one_is_full_lock():
    """A held steer input of s in [-1, 1] sends s times the car's steer
    limit (rad) to its servo."""
    scene = assets.load()
    lock = contract.actuator_limits(scene.model, scene.car).steer_rad
    for steer in (-1.0, -0.5, 0.5, 1.0):
        with case(steer):
            data = mujoco.MjData(scene.model)
            _drive(scene, data, Drive(steer=steer), 1.0)
            ctrl = data.bind_scalar(scene.car.steer).ctrl[0]
            assert ctrl == pytest.approx(steer * lock)


def test_the_car_tops_out_then_brakes_when_the_throttle_is_released():
    scene = assets.load()  # the car alone on its own plane
    data = mujoco.MjData(scene.model)
    x0 = float(data.bind_scalar(scene.car.body).xpos[0])
    _drive(scene, data, Drive(throttle=1.0), 3.0)
    assert float(data.bind_scalar(scene.car.body).xpos[0]) - x0 > 1.0
    # The motor's torque falls to 0 at its free speed V / K, so the mean wheel
    # spin (the drivetrain tendon's velocity) stays below it.
    motor = scene.model.bind_scalar(scene.car.drive)
    free = motor.ctrlrange[1] / motor.gainprm[1]  # rad/s
    assert data.bind_scalar(scene.car.drive).velocity[0] < free
    v_release = _speed(scene, data)
    _drive(scene, data, Drive(), 4.0)
    # At zero throttle the motor brakes in proportion to its speed: from full
    # speed the car loses half of it in about 0.35 s.
    assert _speed(scene, data) < 0.5 * v_release


def test_the_car_drives_forward_on_every_track():
    for track in assets.tracks():
        with case(os.path.basename(track)):
            scene = assets.compose(mujoco.MjSpec.from_file(track))
            data = mujoco.MjData(scene.model)
            x0 = float(data.bind_scalar(scene.car.joint).qpos[0])
            # Distance, not the final speed: in 0.5 s at full throttle the car
            # can reach the scenery ahead of it, such as ramp_course's entry
            # lip, 0.45 m from the start.
            _drive(scene, data, Drive(throttle=1.0), 0.5)
            assert float(data.bind_scalar(scene.car.joint).qpos[0]) - x0 > 0.1
