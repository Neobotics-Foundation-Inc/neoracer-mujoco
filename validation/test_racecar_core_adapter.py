"""
Laws for neoracer_mujoco.racecar_core_adapter, the MuJoCo backend for the
racecar_core controller interface.

The ctrl and LiDAR mapping tests use a new MjData and do not step the
physics. The physics tests check the throttle sign and the steer sign on the
moving car.

racecar_core signs: + speed = forward, + angle = right. contract.py signs:
+ torque = forward, + steer = left.
"""

import mujoco
import numpy as np
import pytest

from neoracer_mujoco import contract, sim
from neoracer_mujoco.contract import DRIVE, STEER
from neoracer_mujoco.racecar_core_adapter import (
    RACECAR_CORE_BEAM_ORDER,
    MujocoDrive,
    MujocoLidar,
    MujocoRacecar,
)


def _set_beam(car, data, name, meters):
    adr = car.sensor(name).adr[0]
    data.sensordata[adr] = meters


# --- Drive: the ctrl mapping, without physics ----------------------------------


def test_drive_writes_clipped_and_scaled_commands(car):
    """For each speed, angle and max_speed, set_speed_angle writes
    clip(speed, -1, 1) * clip(max_speed, 0, 1) * torque_nm to each drive
    motor, and -clip(angle, -1, 1) * steer_rad to the steer servo. The
    minus sign changes the racecar_core steer sign to the contract.py sign."""
    limits = contract.actuator_limits(car)
    data = mujoco.MjData(car)
    drive = MujocoDrive(car, data)
    grid = np.linspace(-2.0, 2.0, 17)  # values inside and outside [-1, 1]
    for max_speed in grid:
        drive.set_max_speed(max_speed)
        scale = np.clip(max_speed, 0.0, 1.0)
        for speed in grid:
            for angle in grid:
                drive.set_speed_angle(speed, angle)
                torque = np.clip(speed, -1.0, 1.0) * scale * limits.torque_nm
                steer = -np.clip(angle, -1.0, 1.0) * limits.steer_rad
                np.testing.assert_allclose(data.ctrl[DRIVE], torque, atol=1e-12)
                assert data.ctrl[STEER] == pytest.approx(steer, abs=1e-12)


def test_new_drive_uses_the_full_torque(car):
    """A new MujocoDrive has max_speed 1.0."""
    data = mujoco.MjData(car)
    MujocoDrive(car, data).set_speed_angle(1.0, 0.0)
    assert np.all(data.ctrl[DRIVE] == contract.actuator_limits(car).torque_nm)


def test_stop_zeros_drive_and_steer(car):
    data = mujoco.MjData(car)
    drive = MujocoDrive(car, data)
    drive.set_speed_angle(1.0, 1.0)
    drive.stop()
    assert np.all(data.ctrl[DRIVE] == 0.0)
    assert data.ctrl[STEER] == 0.0


# --- Drive: physics -----------------------------------------------------------


def _drive_for(car, speed: float, angle: float) -> mujoco.MjData:
    """Settle the car, then drive with (speed, angle) for 0.8 s (400 steps
    at the 2 ms timestep)."""
    data = sim.settle(car)
    MujocoDrive(car, data).set_speed_angle(speed, angle)
    for _ in range(400):
        mujoco.mj_step(car, data)
    return data


def test_full_forward_drives_forward_physically(car):
    x0 = float(sim.settle(car).body("car").xpos[0])
    assert float(_drive_for(car, 1.0, 0.0).body("car").xpos[0]) - x0 > 0.1  # m


def test_right_and_left_angles_yaw_as_mirror_images(car):
    """racecar_core angle +1 (right) yaws the car to the right: the yaw rate
    is negative. Angle -1 gives the mirrored yaw rate. The car model is not
    exactly symmetric, so the two rates agree to 10 %."""
    # qvel[5] is the yaw rate of the car freejoint, rad/s, + = left.
    right = float(_drive_for(car, 0.5, 1.0).qvel[5])
    left = float(_drive_for(car, 0.5, -1.0).qvel[5])
    assert right < -0.1, f"right steer gave yaw rate {right:.3f} rad/s"
    assert left == pytest.approx(-right, rel=0.1)


# --- LiDAR: order, units and no return ----------------------------------------


def test_racecar_core_order_is_clockwise_from_the_front():
    """Index i of a racecar_core scan points 45 * i degrees clockwise from
    the front. A lidar_NNN beam points NNN degrees counterclockwise."""
    for i, name in enumerate(RACECAR_CORE_BEAM_ORDER):
        assert int(name.removeprefix("lidar_")) == (-45 * i) % 360


def test_each_beam_lands_at_its_scan_index_in_cm(car):
    """Set one beam to a distance d (m) and the others to 0. Then the scan
    has 100 * d (cm) at the index of that beam in RACECAR_CORE_BEAM_ORDER,
    and 0 at each other index."""
    rng = np.random.default_rng(0)
    for i, name in enumerate(RACECAR_CORE_BEAM_ORDER):
        for meters in rng.uniform(0.01, 10.0, 5):
            data = mujoco.MjData(car)
            _set_beam(car, data, name, meters)
            lidar = MujocoLidar(car, data)
            scan = lidar.get_samples()
            assert len(scan) == lidar.get_num_samples() == 8
            assert scan[i] == pytest.approx(100.0 * meters, rel=1e-6)
            assert np.all(np.delete(scan, i) == 0.0)


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf"), float("-inf")])
def test_no_return_reads_zero(car, value):
    """MuJoCo gives -1 for no hit. NaN and Inf are bad values. racecar_core
    uses 0.0 for "no return", so each of these values reads as 0.0."""
    data = mujoco.MjData(car)
    _set_beam(car, data, "lidar_000", value)
    assert MujocoLidar(car, data).get_samples()[0] == 0.0


def test_get_samples_async_matches_get_samples(car):
    """This backend has no separate "go" mode, so both reads agree."""
    data = mujoco.MjData(car)
    _set_beam(car, data, "lidar_000", 1.0)
    lidar = MujocoLidar(car, data)
    assert np.array_equal(lidar.get_samples(), lidar.get_samples_async())


# --- Racecar: event loop ------------------------------------------------------


def test_go_before_set_start_update_raises(car):
    """go() without callbacks fails with a clear error and steps nothing."""
    data = mujoco.MjData(car)
    rc = MujocoRacecar(car, data, max_steps=5)
    with pytest.raises(RuntimeError, match="set_start_update"):
        rc.go()
    assert data.time == 0.0
