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
from neoracer_mujoco import assets, collision, contract, racecar, speed_loop
from neoracer_mujoco.control import SpeedAngle

LOOP = os.path.join(assets.TRACK_DIR, "loop_corridor.xml")

SHAPING = Shaping(speed=Rates(toward=2.0, back=1.0), steer=Rates(toward=3.0, back=5.0))
"""Rates of their own toward an input and back to 0, so the laws tell them apart."""
DT = 0.02
"""One 50 Hz frame (s)."""


def _hold(drive: Drive, inputs: Drive, frames: int) -> list[Drive]:
    """The commands of frames frames with inputs held, from drive."""
    drives = [drive]
    for _ in range(frames):
        drives.append(next_drive(SHAPING, drives[-1], inputs, DT))
    return drives


def _mirror(d: Drive) -> Drive:
    return Drive(speed=-d.speed, steer=-d.steer)


def test_a_held_input_is_reached_without_overshoot_and_held():
    """The command moves toward a held input, clipped to [-1, 1], at the
    rate toward it, never past it, and stays there."""
    for value in (-3.0, -1.0, 0.3, 1.0, 3.0):
        with case(value):
            drives = _hold(Drive(), Drive(speed=value, steer=-value), 100)
            target = max(-1.0, min(1.0, value))
            assert drives[1].speed == pytest.approx(
                np.sign(target) * SHAPING.speed.toward * DT
            )
            assert drives[-1] == Drive(speed=target, steer=-target)
            assert all(abs(d.speed) <= abs(target) for d in drives)


def test_a_released_input_returns_the_command_to_exactly_zero():
    """With no input, the command falls to 0 at the rate back, and stays."""
    drives = _hold(Drive(speed=1.0, steer=-1.0), Drive(), 100)
    assert drives[1].speed == pytest.approx(1.0 - SHAPING.speed.back * DT)
    assert drives[1].steer == pytest.approx(-1.0 + SHAPING.steer.back * DT)
    speeds = [d.speed for d in drives]
    assert all(b <= a for a, b in itertools.pairwise(speeds))
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
    long = next_drive(SHAPING, Drive(), Drive(speed=1.0), 10.0)
    assert long.speed == pytest.approx(SHAPING.speed.toward * SHAPING.max_dt)
    drive = Drive(speed=0.5, steer=-0.5)
    assert next_drive(SHAPING, drive, Drive(1.0, 1.0), -0.1) == drive


CAR = racecar.neoracer_params()
STOPPED = Drive()


def _drive(
    scene, data, car: racecar.CarState, inputs: Drive, seconds: float, drive=STOPPED
) -> tuple[racecar.CarState, Drive]:
    """The car and the command after seconds with inputs held, from drive."""
    for _ in range(round(seconds / DT)):
        drive = next_drive(SHAPING, drive, inputs, DT)
        car, _ = apply_and_step(scene, data, car, drive, DT)
    return car, drive


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
            car = racecar.start(CAR, scene, data)
            _drive(scene, data, car, Drive(steer=steer), 1.0)
            ctrl = data.bind_scalar(scene.car.steer).ctrl[0]
            assert ctrl == pytest.approx(steer * lock)


def test_the_car_tops_out_then_brakes_when_the_speed_is_released():
    scene = assets.load()  # the car alone on its own plane
    data = mujoco.MjData(scene.model)
    car = racecar.start(CAR, scene, data)
    x0 = float(data.bind_scalar(scene.car.body).xpos[0])
    car, full = _drive(scene, data, car, Drive(speed=1.0), 3.0)
    assert float(data.bind_scalar(scene.car.body).xpos[0]) - x0 > 1.0
    # The motor's torque falls to 0 at its free speed V / K, so the mean wheel
    # spin (the drivetrain tendon's velocity) stays below it.
    motor = scene.model.bind_scalar(scene.car.drive)
    free = motor.ctrlrange[1] / motor.gainprm[1]  # rad/s
    assert data.bind_scalar(scene.car.drive).velocity[0] < free
    _drive(scene, data, car, Drive(), 4.0, full)  # the speed falls at its rate back
    # Asked for 0, the speed loop brakes the car to a stop.
    assert _speed(scene, data) < 0.01


def test_a_frame_says_whether_the_car_touched_a_wall_in_any_step():
    """apply_and_step steps the whole frame and says whether the car touched
    a wall in any of its steps, as the same steps taken one at a time show.
    On the loop, full speed and full lock reach the inner wall."""
    scene = assets.compose(mujoco.MjSpec.from_file(LOOP), visuals=False)
    model, car = scene.model, scene.car
    framed, stepped = mujoco.MjData(model), mujoco.MjData(model)
    framed_car = racecar.start(CAR, scene, framed)
    stepped_car = racecar.start(CAR, scene, stepped)
    hit = Drive(speed=1.0, steer=1.0)
    assert CAR.speed_loop is not None
    full = float(speed_loop.target(CAR.speed_loop, 1.0))
    lock = contract.actuator_limits(model, car).steer_rad
    said, touched = [], []
    for _ in range(round(3.0 / DT)):
        framed_car, said_frame = apply_and_step(scene, framed, framed_car, hit, DT)
        said.append(said_frame)
        stepped_car = racecar.ask(stepped_car, SpeedAngle(full, lock))
        steps = []
        while stepped.time < framed.time:
            stepped_car = racecar.step(scene, stepped, stepped_car)
            steps.append(collision.car_touches_wall(stepped, car))
        touched.append(any(steps))
    assert said == touched
    assert any(said)


def test_the_car_drives_forward_on_every_track():
    for track in assets.tracks():
        with case(os.path.basename(track)):
            scene = assets.compose(mujoco.MjSpec.from_file(track))
            data = mujoco.MjData(scene.model)
            car = racecar.start(CAR, scene, data)
            x0 = float(data.bind_scalar(scene.car.joint).qpos[0])
            # Distance, not the final speed: in 0.5 s at full speed the car
            # can reach the scenery ahead of it, such as ramp_course's entry
            # lip, 0.45 m from the start.
            _drive(scene, data, car, Drive(speed=1.0), 0.5)
            assert float(data.bind_scalar(scene.car.joint).qpos[0]) - x0 > 0.1
