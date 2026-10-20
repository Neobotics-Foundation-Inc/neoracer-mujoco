"""
The toolbox names that other repos use. neoracer-rl builds its Gymnasium
environment on these names and installs a released neoracer-mujoco wheel, so
a rename here breaks it only at its next upgrade. This test fails first, in
this repo. If you change one of these names on purpose, change neoracer-rl in
the same release.
"""

import dataclasses
import importlib

from neoracer_mujoco.contract import ActuatorLimits
from neoracer_mujoco.control.steering import ControlCommand
from neoracer_mujoco.scene import Car, Scene
from neoracer_mujoco.sensors import SensorReadings

USED_BY_NEORACER_RL = (
    "assets.ASSET_DIR",
    "assets.DEFAULT_CAR",
    "assets.TRACK_DIR",
    "assets.compose",
    "collision.car_touches_wall",
    "contract.actuator_limits",
    "scene.Car",
    "scene.Scene",
    "sensors.SensorReadings",
    "sensors.estimated_linear_speed_ms",
    "sensors.lidar_ranges",
    "sensors.motor_encoder_rad_s",
    "sensors.read",
    "sim.SETTLE_STEPS",
    "sim.apply_command",
    "sim.apply_ctrl_and_step",
    "sim.car_upright_cos",
)


def test_the_names_used_by_neoracer_rl_exist():
    def exists(name):
        module, attr = name.split(".")
        return hasattr(importlib.import_module(f"neoracer_mujoco.{module}"), attr)

    assert [name for name in USED_BY_NEORACER_RL if not exists(name)] == []


def test_records_used_by_neoracer_rl_keep_their_fields():
    def fields(record):
        return {f.name for f in dataclasses.fields(record)}

    assert {"throttle", "steer"} <= fields(ControlCommand)
    assert {"voltage", "steer_rad"} <= fields(ActuatorLimits)
    assert {"spec", "model", "car"} <= fields(Scene)
    assert {"body", "joint", "drive", "steer", "sensors", "truth"} <= fields(Car)
    assert {"imu_accel", "imu_gyro", "imu_quat", "steer_cmd_pos"} <= fields(
        SensorReadings
    )
