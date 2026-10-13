"""
The toolbox names that other repos use. neoracer-rl builds its Gymnasium
environment on these names and installs a released neoracer-mujoco wheel, so
a rename here breaks it only at its next upgrade. This test fails first, in
this repo. If you change one of these names on purpose, change neoracer-rl in
the same release.
"""

import dataclasses
import importlib

import pytest

from neoracer_mujoco.contract import ActuatorLimits
from neoracer_mujoco.control.steering import ControlCommand

USED_BY_NEORACER_RL = (
    "assets.DEFAULT_CAR",
    "assets.TRACK_DIR",
    "assets.compose",
    "collision.car_touches_wall",
    "collision.wall_geom_ids",
    "contract.actuator_limits",
    "sensors.estimated_linear_speed_ms",
    "sensors.lidar_ranges",
    "sensors.motor_encoder_rad_s",
    "sensors.read_layout",
    "sensors.sensor_layout",
    "sim.SETTLE_STEPS",
    "sim.apply_command",
    "sim.apply_ctrl_and_step",
    "sim.car_upright_cos",
)


@pytest.mark.parametrize("name", USED_BY_NEORACER_RL)
def test_name_used_by_neoracer_rl_exists(name):
    module, attr = name.split(".")
    assert hasattr(importlib.import_module(f"neoracer_mujoco.{module}"), attr)


def test_records_used_by_neoracer_rl_keep_their_fields():
    def fields(record):
        return {f.name for f in dataclasses.fields(record)}

    assert {"throttle", "steer"} <= fields(ControlCommand)
    assert {"torque_nm", "steer_rad"} <= fields(ActuatorLimits)
