"""
Laws for the physics of the pygame driving demo (examples/pygame_drive/drive_sim.py).
No window: pygame is not imported.
"""

import os

import mujoco
import numpy as np
import pytest

from examples.pygame_drive.drive_sim import (
    DriveParams,
    DriveState,
    apply_and_step,
    next_state,
)
from neoracer_mujoco import assets

P = DriveParams()
DT = 0.02  # s, one 50 Hz frame


def _drive(scene, data, state, throttle, steer, frames):
    for _ in range(frames):
        state = next_state(P, state, throttle, steer, DT)
        apply_and_step(scene, data, P, state, DT)
    return state


def _speed(scene, data) -> float:
    return float(np.linalg.norm(data.bind_scalar(scene.car.joint).qvel[0:2]))  # vx, vy


@pytest.mark.parametrize("throttle", [-3.0, -1.0, 0.3, 1.0, 3.0])
def test_next_state_reaches_the_clipped_input_without_overshoot(throttle):
    state, seen = DriveState(), []
    for _ in range(100):
        state = next_state(P, state, throttle, -throttle, DT)
        seen.append(state.throttle)
    target = max(-1.0, min(1.0, throttle))
    assert state.throttle == target and state.steer == -target
    assert all(abs(t) <= abs(target) for t in seen)


def test_next_state_cuts_a_long_frame():
    assert next_state(P, DriveState(), 1.0, 0.0, 10.0).throttle == pytest.approx(
        P.throttle_slew * P.max_dt
    )


def test_a_negative_frame_time_changes_nothing():
    state = DriveState(throttle=0.5, steer=-0.5)
    assert next_state(P, state, 1.0, 1.0, -0.1) == state
    scene = assets.load()
    data = mujoco.MjData(scene.model)
    apply_and_step(scene, data, P, state, -0.1)
    assert data.time == 0.0


def test_car_tops_out_then_brakes_when_the_throttle_is_released():
    scene = assets.load()  # the car alone on its own plane
    data = mujoco.MjData(scene.model)
    body = scene.car.body
    x0 = float(data.bind_scalar(body).xpos[0])
    state = _drive(scene, data, DriveState(), 1.0, 0.0, 150)  # 3 s full throttle
    assert float(data.bind_scalar(body).xpos[0]) - x0 > 1.0
    # The motor's torque falls to 0 at its free speed V / K, so the mean wheel
    # spin (the drivetrain tendon's velocity) stays below it.
    motor = scene.model.bind_scalar(scene.car.drive)
    free = motor.ctrlrange[1] / motor.gainprm[1]  # rad/s
    assert data.bind_scalar(scene.car.drive).velocity[0] < free
    v_release = _speed(scene, data)
    _drive(scene, data, state, 0.0, 0.0, 200)  # 4 s hands off
    # At zero throttle the motor brakes in proportion to its speed. The time
    # constant is about 4.5 s, mostly the armature of the wheels.
    assert _speed(scene, data) < 0.5 * v_release


@pytest.mark.parametrize("steer", [-1.0, -0.5, 0.5, 1.0])
def test_held_steer_input_sets_the_steer_ctrl(steer):
    """A steer input in [-1, 1], held for 1 s, writes steer * max_steer (rad)
    to the steer servo."""
    scene = assets.load()
    data = mujoco.MjData(scene.model)
    _drive(scene, data, DriveState(), 0.0, steer, 50)  # 1 s
    assert data.bind_scalar(scene.car.steer).ctrl[0] == pytest.approx(
        steer * P.max_steer
    )


@pytest.mark.parametrize("track", assets.tracks(), ids=os.path.basename)
def test_car_drives_forward_on_every_track(track):
    scene = assets.compose(mujoco.MjSpec.from_file(track))
    data = mujoco.MjData(scene.model)
    _drive(scene, data, DriveState(), 1.0, 0.0, 25)  # 0.5 s
    assert float(data.bind_scalar(scene.car.joint).qvel[0]) > 0.1
