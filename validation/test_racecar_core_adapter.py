"""
Laws for neoracer_mujoco.racecar_core_adapter, the MuJoCo backend for the
racecar_core controller interface.

The ctrl and LiDAR mapping tests use a new MjData and do not step the
physics. The physics tests check the throttle sign and the steer sign on the
moving car.

racecar_core signs: + speed = forward, + angle = right. contract.py signs:
+ throttle = forward, + steer = left.
"""

import os

import mujoco
import numpy as np
import pytest

from neoracer_mujoco import Scene, assets, contract, lidar, sim
from neoracer_mujoco.racecar_core_adapter import MujocoDrive, MujocoLidar, MujocoRacecar


@pytest.fixture(scope="module")
def corridor() -> Scene:
    """The car in loop_corridor.xml, where every ray it does not block meets
    a wall within lidar.MAX_RANGE."""
    track = mujoco.MjSpec.from_file(os.path.join(assets.TRACK_DIR, "loop_corridor.xml"))
    return assets.compose(track, visuals=False)


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
    """Settle the car, then drive with (speed, angle) for 0.8 s."""
    data = sim.settle(scene.model)
    MujocoDrive(scene, data).set_speed_angle(speed, angle)
    for _ in range(round(0.8 / scene.model.opt.timestep)):
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


@pytest.mark.parametrize(
    ("bearing", "beam"),
    [(0, "lidar_000"), (45, "lidar_315"), (90, "lidar_270"), (270, "lidar_090")],
)
def test_a_scan_is_clockwise_from_the_front_in_cm(corridor, bearing, beam):
    """Index i of a scan points i * 360 / len(scan) degrees clockwise from the
    front: it reads, in cm, what the rangefinder at that bearing reads in m.
    A lidar_NNN beam points NNN degrees counterclockwise."""
    data = sim.settle(corridor.model)
    scan = MujocoLidar(corridor, data).get_samples()
    i = bearing * len(scan) // 360
    meters = data.bind_scalar(corridor.spec.sensor(beam)).data[0]
    assert meters > 0.0
    assert scan[i] == pytest.approx(100.0 * meters, rel=1e-6)


def test_no_return_and_the_blocked_rear_read_zero(scene, corridor):
    """racecar_core uses 0.0 for "no return". The car alone on its floor sees
    nothing, and in the corridor the rear quarter, which the car blocks,
    reads 0.0 too."""
    alone = MujocoLidar(scene, sim.settle(scene.model))
    assert len(alone.get_samples()) == alone.get_num_samples() == lidar.SLOTS
    assert np.all(alone.get_samples() == 0.0)
    scan = MujocoLidar(corridor, sim.settle(corridor.model)).get_samples()
    assert np.all(scan[~lidar.VISIBLE] == 0.0)
    assert np.all(scan[lidar.VISIBLE] > 0.0)


def test_get_samples_async_matches_get_samples(corridor):
    """This backend has no separate "go" mode, so both reads agree."""
    rc_lidar = MujocoLidar(corridor, sim.settle(corridor.model))
    assert np.array_equal(rc_lidar.get_samples(), rc_lidar.get_samples_async())


# --- Racecar: event loop ------------------------------------------------------


def test_go_before_set_start_update_raises(scene):
    """go() without callbacks fails with a clear error and steps nothing."""
    data = mujoco.MjData(scene.model)
    rc = MujocoRacecar(scene, data, max_steps=5)
    with pytest.raises(RuntimeError, match="set_start_update"):
        rc.go()
    assert data.time == 0.0
