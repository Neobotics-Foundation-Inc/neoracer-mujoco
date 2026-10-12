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
    drive_wheel_dofs,
    next_state,
    wheel_torque,
)
from neoracer_mujoco import assets, contract

P = DriveParams()
DT = 0.02  # s, one 50 Hz frame


def _drive(model, data, state, throttle, steer, frames):
    wheels = drive_wheel_dofs(model)
    for _ in range(frames):
        state = next_state(P, state, throttle, steer, DT)
        apply_and_step(model, data, P, wheels, state, DT)
    return state


def _speed(data) -> float:
    return float(np.linalg.norm(data.qvel[0:2]))  # freejoint vx, vy


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
    model = assets.load()
    data = mujoco.MjData(model)
    apply_and_step(model, data, P, drive_wheel_dofs(model), state, -0.1)
    assert data.time == 0.0


def test_wheel_torque_stays_in_limits_and_brakes_a_spinning_wheel():
    spin = np.array([-1000.0, -10.0, 10.0, 1000.0])
    for throttle in (-1.0, 0.0, 1.0):
        assert np.all(np.abs(wheel_torque(P, throttle, spin)) <= P.max_torque)
    assert np.all(np.sign(wheel_torque(P, 0.0, spin)) == -np.sign(spin))


def test_drive_wheel_dofs_are_four_hinge_joints():
    model = assets.load()
    dofs = drive_wheel_dofs(model)
    assert len(set(dofs.tolist())) == 4
    hinge = mujoco.mjtJoint.mjJNT_HINGE
    assert all(model.jnt_type[model.dof_jntid[d]] == hinge for d in dofs)


def test_car_tops_out_then_brakes_when_the_throttle_is_released():
    model = assets.load()  # the car alone on its own plane
    data = mujoco.MjData(model)
    x0 = float(data.body("car").xpos[0])
    state = _drive(model, data, DriveState(), 1.0, 0.0, 150)  # 3 s full throttle
    assert float(data.body("car").xpos[0]) - x0 > 1.0
    spin = float(np.mean(data.qvel[drive_wheel_dofs(model)]))
    assert spin < 1.1 * P.max_torque / P.motor_damping  # the back-EMF balance point
    v_release = _speed(data)
    _drive(model, data, state, 0.0, 0.0, 200)  # 4 s hands off
    assert _speed(data) < 0.1 * v_release


@pytest.mark.parametrize("steer", [-1.0, -0.5, 0.5, 1.0])
def test_held_steer_input_sets_the_steer_ctrl(steer):
    """A steer input in [-1, 1], held for 1 s, writes steer * max_steer (rad)
    to the steer servo."""
    model = assets.load()
    data = mujoco.MjData(model)
    _drive(model, data, DriveState(), 0.0, steer, 50)  # 1 s
    assert data.ctrl[contract.STEER] == pytest.approx(steer * P.max_steer)


@pytest.mark.parametrize("track", assets.tracks(), ids=os.path.basename)
def test_car_drives_forward_on_every_track(track):
    model = assets.compose(mujoco.MjSpec.from_file(track))
    data = mujoco.MjData(model)
    _drive(model, data, DriveState(), 1.0, 0.0, 25)  # 0.5 s
    assert float(data.qvel[0]) > 0.1
