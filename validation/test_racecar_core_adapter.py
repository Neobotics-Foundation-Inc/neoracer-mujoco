"""
neoracer_mujoco.racecar_core_adapter: the MuJoCo backend for the
racecar_core controller interface (issue #8).

Mirrors test_wall_following.py's split: fast unit checks against a freshly
compiled MjModel/MjData (no settle/step needed for pure ctrl-mapping or
sensordata-mapping checks), plus physics-integration checks for the two
claims that only mean something once the sim actually steps -- throttle
sign and steer sign.
"""

import mujoco
import numpy as np
import pytest

from neoracer_mujoco import sim as _sim
from neoracer_mujoco.contract import DRIVE, STEER
from neoracer_mujoco.racecar_core_adapter import (
    RACECAR_CORE_BEAM_ORDER,
    MujocoDrive,
    MujocoLidar,
)


def _set_beam(car, data, name, meters):
    adr = car.sensor(name).adr[0]
    data.sensordata[adr] = meters


# --- Drive: pure ctrl-mapping (no physics step needed) ------------------


def test_full_forward_maps_to_positive_torque(car):
    data = mujoco.MjData(car)
    MujocoDrive(car, data).set_speed_angle(1.0, 0.0)
    torque_limit = car.actuator("fl_motor").ctrlrange[1]
    assert np.all(data.ctrl[DRIVE] == pytest.approx(torque_limit))


def test_full_backward_maps_to_negative_torque(car):
    data = mujoco.MjData(car)
    MujocoDrive(car, data).set_speed_angle(-1.0, 0.0)
    torque_limit = car.actuator("fl_motor").ctrlrange[1]
    assert np.all(data.ctrl[DRIVE] == pytest.approx(-torque_limit))


def test_positive_angle_maps_to_negative_steer_ctrl(car):
    """racecar_core angle +1 = right; ctrl[4] + = left (contract.py) -- so
    a right command must write a NEGATIVE ctrl[4]. Getting this backwards
    silently inverts every controller that uses the adapter."""
    data = mujoco.MjData(car)
    steer_limit = car.actuator("steer_servo").ctrlrange[1]
    MujocoDrive(car, data).set_speed_angle(0.0, 1.0)
    assert data.ctrl[STEER] == pytest.approx(-steer_limit)


def test_negative_angle_maps_to_positive_steer_ctrl(car):
    data = mujoco.MjData(car)
    steer_limit = car.actuator("steer_servo").ctrlrange[1]
    MujocoDrive(car, data).set_speed_angle(0.0, -1.0)
    assert data.ctrl[STEER] == pytest.approx(steer_limit)


def test_commands_are_clamped(car):
    data = mujoco.MjData(car)
    torque_limit = car.actuator("fl_motor").ctrlrange[1]
    steer_limit = car.actuator("steer_servo").ctrlrange[1]
    MujocoDrive(car, data).set_speed_angle(5.0, -5.0)
    assert np.all(data.ctrl[DRIVE] == pytest.approx(torque_limit))
    assert data.ctrl[STEER] == pytest.approx(steer_limit)


def test_stop_zeros_drive_and_steer(car):
    data = mujoco.MjData(car)
    drive = MujocoDrive(car, data)
    drive.set_speed_angle(1.0, 1.0)
    drive.stop()
    assert np.all(data.ctrl[DRIVE] == 0.0)
    assert data.ctrl[STEER] == 0.0


def test_max_speed_scales_torque(car):
    data = mujoco.MjData(car)
    torque_limit = car.actuator("fl_motor").ctrlrange[1]
    drive = MujocoDrive(car, data)
    drive.set_max_speed(0.5)
    drive.set_speed_angle(1.0, 0.0)
    assert np.all(data.ctrl[DRIVE] == pytest.approx(0.5 * torque_limit))


def test_max_speed_is_clamped(car):
    data = mujoco.MjData(car)
    torque_limit = car.actuator("fl_motor").ctrlrange[1]
    drive = MujocoDrive(car, data)
    drive.set_max_speed(3.0)
    drive.set_speed_angle(1.0, 0.0)
    assert np.all(data.ctrl[DRIVE] == pytest.approx(torque_limit))


# --- Drive: physics integration (only these need a real step loop) ------


def test_full_forward_drives_forward_physically(car):
    data = _sim.settle(car)
    x0 = data.body("car").xpos[0].copy()
    drive = MujocoDrive(car, data)
    drive.set_speed_angle(1.0, 0.0)
    for _ in range(400):
        mujoco.mj_step(car, data)
    assert data.body("car").xpos[0] - x0 > 0.1


def test_full_right_yaws_right_physically(car):
    """racecar_core angle=+1 (right) must yaw the car -z (negative qvel[5]),
    the mirror of test_conformance.py's left_steer_yaws_left."""
    data = _sim.settle(car)
    drive = MujocoDrive(car, data)
    drive.set_speed_angle(0.5, 1.0)
    for _ in range(400):
        mujoco.mj_step(car, data)
    assert data.qvel[5] < -0.1, f"right steer gave yaw rate {data.qvel[5]:.3f}"


def test_full_left_yaws_left_physically(car):
    data = _sim.settle(car)
    drive = MujocoDrive(car, data)
    drive.set_speed_angle(0.5, -1.0)
    for _ in range(400):
        mujoco.mj_step(car, data)
    assert data.qvel[5] > 0.1, f"left steer gave yaw rate {data.qvel[5]:.3f}"


# --- LiDAR: unit conversion, no-return, ordering, scan length ------------


def test_get_num_samples_matches_scan_length(car):
    data = mujoco.MjData(car)
    lidar = MujocoLidar(car, data)
    assert lidar.get_num_samples() == len(lidar.get_samples()) == 8


def test_forward_beam_is_index_zero(car):
    """Both conventions start their scan directly ahead -- index 0 must be
    lidar_000 in both, unaffected by the clockwise/counterclockwise
    re-sequencing applied to every other beam."""
    data = mujoco.MjData(car)
    _set_beam(car, data, "lidar_000", 1.23)
    scan = MujocoLidar(car, data).get_samples()
    assert scan[0] == pytest.approx(123.0)


def test_right_side_beam_lands_clockwise_quarter_turn(car):
    """lidar_270 is this sim's right-side beam (see wall_following.SIDE_BEAM).
    racecar_core's clockwise scan must place the right side one quarter-turn
    (index 2 of 8) after front, matching its own angle convention where
    right = +90 deg clockwise."""
    data = mujoco.MjData(car)
    _set_beam(car, data, "lidar_270", 2.0)
    scan = MujocoLidar(car, data).get_samples()
    assert scan[2] == pytest.approx(200.0)
    assert RACECAR_CORE_BEAM_ORDER[2] == "lidar_270"


def test_left_side_beam_lands_counterclockwise_quarter_turn(car):
    """Mirror of the right-side check: lidar_090 (this sim's left beam) must
    land at index 6 (three-quarters clockwise = one quarter counterclockwise)."""
    data = mujoco.MjData(car)
    _set_beam(car, data, "lidar_090", 2.0)
    scan = MujocoLidar(car, data).get_samples()
    assert scan[6] == pytest.approx(200.0)
    assert RACECAR_CORE_BEAM_ORDER[6] == "lidar_090"


def test_meters_convert_to_centimeters(car):
    data = mujoco.MjData(car)
    _set_beam(car, data, "lidar_000", 0.5)
    scan = MujocoLidar(car, data).get_samples()
    assert scan[0] == pytest.approx(50.0)


def test_no_hit_converts_to_zero(car):
    """This sim's no-hit sentinel is -1 (see contract.py); racecar_core's is
    0.0 -- these must not be confused with each other."""
    data = mujoco.MjData(car)
    _set_beam(car, data, "lidar_000", -1.0)
    scan = MujocoLidar(car, data).get_samples()
    assert scan[0] == 0.0


def test_non_finite_reading_converts_to_zero(car):
    data = mujoco.MjData(car)
    _set_beam(car, data, "lidar_000", float("nan"))
    scan = MujocoLidar(car, data).get_samples()
    assert scan[0] == 0.0


def test_get_samples_async_matches_get_samples(car):
    """No separate 'go' mode in this backend -- both reads must agree."""
    data = mujoco.MjData(car)
    _set_beam(car, data, "lidar_000", 1.0)
    lidar = MujocoLidar(car, data)
    assert np.array_equal(lidar.get_samples(), lidar.get_samples_async())


def test_get_samples_is_deterministic(car):
    data = mujoco.MjData(car)
    _set_beam(car, data, "lidar_045", 0.75)
    lidar = MujocoLidar(car, data)
    assert np.array_equal(lidar.get_samples(), lidar.get_samples())
