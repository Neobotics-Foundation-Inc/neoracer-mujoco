"""
Laws for neoracer_mujoco.racecar_core_adapter, the MuJoCo backend for the
racecar_core controller interface.

The ctrl and LiDAR mapping tests use a new MjData and do not step the
physics: test_conformance.py shows that the contract.py signs move the car
the right way.

racecar_core signs: + speed = forward, + angle = right. contract.py signs:
+ throttle = forward, + steer = left.
"""

import os

import mujoco
import numpy as np
import pytest

from neoracer_mujoco import Scene, assets, contract, lidar, sim
from neoracer_mujoco.racecar_core_adapter import (
    CarParams,
    MujocoDrive,
    MujocoLidar,
    MujocoRacecar,
)

LIDAR = lidar.LidarParams()


@pytest.fixture(scope="module")
def corridor() -> Scene:
    """The car in loop_corridor.xml, where every ray it does not block meets
    a wall within the LiDAR's range."""
    track = mujoco.MjSpec.from_file(os.path.join(assets.TRACK_DIR, "loop_corridor.xml"))
    return assets.compose(track, visuals=False)


def _drive_ctrl(scene: Scene, data: mujoco.MjData) -> float:
    """The ctrl of the drive motor: its voltage (V) on the NeoRacer."""
    return float(data.bind_scalar(scene.car.drive).ctrl[0])


def _steer_ctrl(scene: Scene, data: mujoco.MjData) -> float:
    return float(data.bind_scalar(scene.car.steer).ctrl[0])


# --- Drive: the ctrl mapping, without physics ----------------------------------


def test_drive_writes_clipped_and_scaled_commands(scene):
    """For each speed, angle and max_speed, set_speed_angle drives the
    motor with the throttle clip(speed, -1, 1) * clip(max_speed, 0, 1),
    through its throttle map (on the NeoRacer, that share of the battery's
    voltage), and writes -clip(angle, -1, 1) * steer_rad to the steer servo.
    The minus sign changes the racecar_core steer sign to the contract.py
    sign. A new drive has max_speed 1, and stop() writes 0 to both."""
    limits = contract.actuator_limits(scene.model, scene.car)
    a, b, c = contract.throttle_map(scene.model, scene.car.drive)
    data = mujoco.MjData(scene.model)
    drive = MujocoDrive(scene, data)
    drive.set_speed_angle(1.0, 0.0)  # a new drive has max_speed 1
    assert _drive_ctrl(scene, data) == pytest.approx(limits.voltage)
    grid = np.linspace(-2.0, 2.0, 17)  # values inside and outside [-1, 1]
    for max_speed in grid:
        drive.set_max_speed(max_speed)
        scale = np.clip(max_speed, 0.0, 1.0)
        for speed in grid:
            for angle in grid:
                drive.set_speed_angle(speed, angle)
                t = np.clip(speed, -1.0, 1.0) * scale
                volts = a * t + b * np.sign(t) + c * abs(np.sign(t))
                steer = -np.clip(angle, -1.0, 1.0) * limits.steer_rad
                assert _drive_ctrl(scene, data) == pytest.approx(volts, abs=1e-12)
                assert _steer_ctrl(scene, data) == pytest.approx(steer, abs=1e-12)
    drive.stop()
    assert _drive_ctrl(scene, data) == 0.0
    assert _steer_ctrl(scene, data) == 0.0


# --- LiDAR: order, units and no return ----------------------------------------


def test_set_max_speed_without_an_argument_restores_the_start(scene):
    """set_max_speed() without an argument goes back to the max speed the
    drive started with, as racecar_core's default does: 1.0 on the NeoRacer,
    0.25 on the V2. Speed 1 drives that share of full throttle at the start
    and after. A drive clamps its start into [0, 1], as set_max_speed clamps
    its argument, and CarParams refuses a start outside it."""
    a, b, c = contract.throttle_map(scene.model, scene.car.drive)
    data = mujoco.MjData(scene.model)
    for start, share in ((1.0, 1.0), (0.25, 0.25), (2.0, 1.0), (-0.25, 0.0)):
        expected = a * share + (b + c if share > 0 else 0.0)
        drive = MujocoDrive(scene, data, start)
        drive.set_speed_angle(1.0, 0.0)
        assert _drive_ctrl(scene, data) == pytest.approx(expected), start
        drive.set_max_speed(0.5)
        drive.set_max_speed()
        drive.set_speed_angle(1.0, 0.0)
        assert _drive_ctrl(scene, data) == pytest.approx(expected), start
    for start in (2.0, -0.25):
        with pytest.raises(ValueError, match="start_max_speed"):
            CarParams(start_max_speed=start)


def test_a_scan_is_clockwise_from_the_front_in_cm(corridor):
    """Index i of a scan points i * 360 / len(scan) degrees clockwise from the
    front: it reads, in cm, what the rangefinder at that bearing reads in m.
    A lidar_NNN beam points NNN degrees counterclockwise."""
    data = sim.settle(corridor.model)
    scan = MujocoLidar(corridor, data).get_samples()
    for bearing, beam in [
        (0, "lidar_000"),
        (45, "lidar_315"),
        (90, "lidar_270"),
        (270, "lidar_090"),
    ]:
        i = bearing * len(scan) // 360
        meters = data.bind_scalar(corridor.spec.sensor(beam)).data[0]
        assert meters > 0.0, beam
        assert scan[i] == pytest.approx(100.0 * meters, rel=1e-6), beam


def test_no_return_and_the_blocked_rear_read_zero(scene, corridor):
    """racecar_core uses 0.0 for "no return". The car alone on its floor sees
    nothing, and in the corridor the rear quarter, which the car blocks,
    reads 0.0 too."""
    alone = MujocoLidar(scene, sim.settle(scene.model))
    assert len(alone.get_samples()) == alone.get_num_samples() == LIDAR.slots
    assert np.all(alone.get_samples() == 0.0)
    scan = MujocoLidar(corridor, sim.settle(corridor.model)).get_samples()
    assert np.all(scan[~lidar.visible(LIDAR)] == 0.0)
    assert np.all(scan[lidar.visible(LIDAR)] > 0.0)


# --- Racecar: event loop ------------------------------------------------------


def test_go_before_set_start_update_raises(scene):
    """go() without callbacks fails with a clear error and steps nothing."""
    data = mujoco.MjData(scene.model)
    rc = MujocoRacecar(scene, data, max_steps=5)
    with pytest.raises(RuntimeError, match="set_start_update"):
        rc.go()
    assert data.time == 0.0
