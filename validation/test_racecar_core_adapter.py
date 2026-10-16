"""
Laws for neoracer_mujoco.racecar_core_adapter, the MuJoCo backend for the
racecar_core controller interface.

The ctrl and LiDAR mapping tests use a new MjData and do not step the
physics. The physics tests check the throttle sign and the steer sign on the
moving car.

racecar_core signs: + speed = forward, + angle = right. contract.py signs:
+ throttle = forward, + steer = left.
"""

import mujoco
import numpy as np
import pytest

from neoracer_mujoco import Scene, contract, sim
from neoracer_mujoco.racecar_core_adapter import (
    RACECAR_CORE_BEAM_ORDER,
    MujocoDrive,
    MujocoLidar,
    MujocoRacecar,
)


def _set_beam(scene: Scene, data: mujoco.MjData, name: str, meters: float) -> None:
    data.bind_scalar(scene.spec.sensor(name)).data[0] = meters


def _drive_ctrl(scene: Scene, data: mujoco.MjData) -> float:
    """The ctrl of the drive motor: its voltage (V)."""
    return float(data.bind_scalar(scene.car.drive).ctrl[0])


def _steer_ctrl(scene: Scene, data: mujoco.MjData) -> float:
    return float(data.bind_scalar(scene.car.steer).ctrl[0])


# --- Drive: the ctrl mapping, without physics ----------------------------------


def test_drive_writes_clipped_and_scaled_commands(scene):
    """For each speed, angle and max_speed, set_speed_angle writes
    clip(speed, -1, 1) * clip(max_speed, 0, 1) * voltage to the drive
    motor, and -clip(angle, -1, 1) * steer_rad to the steer servo. The
    minus sign changes the racecar_core steer sign to the contract.py sign."""
    limits = contract.actuator_limits(scene.model, scene.car)
    data = mujoco.MjData(scene.model)
    drive = MujocoDrive(scene, data)
    grid = np.linspace(-2.0, 2.0, 17)  # values inside and outside [-1, 1]
    for max_speed in grid:
        drive.set_max_speed(max_speed)
        scale = np.clip(max_speed, 0.0, 1.0)
        for speed in grid:
            for angle in grid:
                drive.set_speed_angle(speed, angle)
                volts = np.clip(speed, -1.0, 1.0) * scale * limits.voltage
                steer = -np.clip(angle, -1.0, 1.0) * limits.steer_rad
                assert _drive_ctrl(scene, data) == pytest.approx(volts, abs=1e-12)
                assert _steer_ctrl(scene, data) == pytest.approx(steer, abs=1e-12)


def test_new_drive_uses_the_full_throttle(scene):
    """A new MujocoDrive has max_speed 1.0."""
    data = mujoco.MjData(scene.model)
    MujocoDrive(scene, data).set_speed_angle(1.0, 0.0)
    voltage = contract.actuator_limits(scene.model, scene.car).voltage
    assert _drive_ctrl(scene, data) == voltage


def test_stop_zeros_drive_and_steer(scene):
    data = mujoco.MjData(scene.model)
    drive = MujocoDrive(scene, data)
    drive.set_speed_angle(1.0, 1.0)
    drive.stop()
    assert _drive_ctrl(scene, data) == 0.0
    assert _steer_ctrl(scene, data) == 0.0


# --- Drive: physics -----------------------------------------------------------


def _drive_for(scene, speed: float, angle: float) -> mujoco.MjData:
    """Settle the car, then drive with (speed, angle) for 0.8 s (400 steps
    at the 2 ms timestep)."""
    data = sim.settle(scene.model)
    MujocoDrive(scene, data).set_speed_angle(speed, angle)
    for _ in range(400):
        mujoco.mj_step(scene.model, data)
    return data


def test_full_forward_drives_forward_physically(scene):
    body = scene.car.body
    x0 = float(sim.settle(scene.model).bind_scalar(body).xpos[0])
    assert float(_drive_for(scene, 1.0, 0.0).bind_scalar(body).xpos[0]) - x0 > 0.1  # m


def test_right_and_left_angles_yaw_as_mirror_images(scene):
    """racecar_core angle +1 (right) yaws the car to the right: the yaw rate
    is negative. Angle -1 gives the mirrored yaw rate. The car model is not
    exactly symmetric, so the two rates agree to 10 %."""
    # qvel[5] of the car's free joint is its yaw rate, rad/s, + = left.
    right = float(_drive_for(scene, 0.5, 1.0).bind_scalar(scene.car.joint).qvel[5])
    left = float(_drive_for(scene, 0.5, -1.0).bind_scalar(scene.car.joint).qvel[5])
    assert right < -0.1, f"right steer gave yaw rate {right:.3f} rad/s"
    assert left == pytest.approx(-right, rel=0.1)


# --- LiDAR: order, units and no return ----------------------------------------


def test_racecar_core_order_is_clockwise_from_the_front():
    """Index i of a racecar_core scan points 45 * i degrees clockwise from
    the front. A lidar_NNN beam points NNN degrees counterclockwise."""
    for i, name in enumerate(RACECAR_CORE_BEAM_ORDER):
        assert int(name.removeprefix("lidar_")) == (-45 * i) % 360


def test_each_beam_lands_at_its_scan_index_in_cm(scene):
    """Set one beam to a distance d (m) and the others to 0. Then the scan
    has 100 * d (cm) at the index of that beam in RACECAR_CORE_BEAM_ORDER,
    and 0 at each other index."""
    rng = np.random.default_rng(0)
    for i, name in enumerate(RACECAR_CORE_BEAM_ORDER):
        for meters in rng.uniform(0.01, 10.0, 5):
            data = mujoco.MjData(scene.model)
            _set_beam(scene, data, name, meters)
            lidar = MujocoLidar(scene, data)
            scan = lidar.get_samples()
            assert len(scan) == lidar.get_num_samples() == 8
            assert scan[i] == pytest.approx(100.0 * meters, rel=1e-6)
            assert np.all(np.delete(scan, i) == 0.0)


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf"), float("-inf")])
def test_no_return_reads_zero(scene, value):
    """MuJoCo gives -1 for no hit. NaN and Inf are bad values. racecar_core
    uses 0.0 for "no return", so each of these values reads as 0.0."""
    data = mujoco.MjData(scene.model)
    _set_beam(scene, data, "lidar_000", value)
    assert MujocoLidar(scene, data).get_samples()[0] == 0.0


def test_get_samples_async_matches_get_samples(scene):
    """This backend has no separate "go" mode, so both reads agree."""
    data = mujoco.MjData(scene.model)
    _set_beam(scene, data, "lidar_000", 1.0)
    lidar = MujocoLidar(scene, data)
    assert np.array_equal(lidar.get_samples(), lidar.get_samples_async())


# --- Racecar: event loop ------------------------------------------------------


def test_go_before_set_start_update_raises(scene):
    """go() without callbacks fails with a clear error and steps nothing."""
    data = mujoco.MjData(scene.model)
    rc = MujocoRacecar(scene, data, max_steps=5)
    with pytest.raises(RuntimeError, match="set_start_update"):
        rc.go()
    assert data.time == 0.0
