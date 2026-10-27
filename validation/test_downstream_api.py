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

from neoracer_mujoco.assets import Entry
from neoracer_mujoco.contract import ActuatorLimits
from neoracer_mujoco.control.steering import ControlCommand
from neoracer_mujoco.physics import PhysicsRanges
from neoracer_mujoco.scene import Car, Scene
from neoracer_mujoco.sensors import CarSensors
from neoracer_mujoco.sim import SpawnRange
from neoracer_mujoco.track_generation import Track
from neoracer_mujoco.truth import TruthSensors
from neoracer_spec.racecar_neo_v2 import Drive

USED_BY_NEORACER_RL = (
    "neoracer_mujoco.assets.ASSET_DIR",
    "neoracer_mujoco.assets.DEFAULT_CAR",
    "neoracer_mujoco.assets.Entry",
    "neoracer_mujoco.assets.TRACK_DIR",
    "neoracer_mujoco.assets.compose",
    "neoracer_mujoco.assets.compose_cars",
    "neoracer_mujoco.batch.Batch",
    "neoracer_mujoco.batch.CpuBatch",
    "neoracer_mujoco.batch.DataBatch",
    "neoracer_mujoco.batch.command",
    "neoracer_mujoco.batch.command_cars",
    "neoracer_mujoco.batch.drive",
    "neoracer_mujoco.batch.drive_angle",
    "neoracer_mujoco.batch.drive_cars",
    "neoracer_mujoco.batch.read",
    "neoracer_mujoco.batch.readings",
    "neoracer_mujoco.collision.car_touches_wall",
    "neoracer_mujoco.contract.actuator_limits",
    "neoracer_mujoco.contract.wheel_radius",
    "neoracer_mujoco.control.PDState",
    "neoracer_mujoco.control.SpeedAngle",
    "neoracer_mujoco.control.steering.ControlCommand",
    "neoracer_mujoco.control.track_centering.TrackCenteringConfig",
    "neoracer_mujoco.control.track_centering.step",
    "neoracer_mujoco.control.wall_following.WallFollowingConfig",
    "neoracer_mujoco.control.wall_following.step",
    "neoracer_mujoco.export.car_xml",
    "neoracer_mujoco.physics.Physics",
    "neoracer_mujoco.physics.PhysicsRanges",
    "neoracer_mujoco.physics.Range",
    "neoracer_mujoco.physics.apply_physics",
    "neoracer_mujoco.physics.sample",
    "neoracer_mujoco.racecar.CarParams",
    "neoracer_mujoco.racecar.neoracer_params",
    "neoracer_mujoco.racecar.racecar_neo_v2_params",
    "neoracer_mujoco.racecar_core_adapter.MujocoRacecar",
    "neoracer_mujoco.scene.Car",
    "neoracer_mujoco.scene.Scene",
    "neoracer_mujoco.sensors.NO_RETURN",
    "neoracer_mujoco.sim.SpawnRange",
    "neoracer_mujoco.sim.apply_command",
    "neoracer_mujoco.sim.car_upright_cos",
    "neoracer_mujoco.sim.perturb_spawn",
    "neoracer_mujoco.sim.settle",
    "neoracer_mujoco.sim.spawn_clears_walls",
    "neoracer_mujoco.sim.steps_in",
    "neoracer_mujoco.speed_loop.SpeedLoopState",
    "neoracer_mujoco.speed_loop.restart",
    "neoracer_mujoco.speed_loop.start",
    "neoracer_mujoco.speed_loop.step",
    "neoracer_mujoco.speed_loop.target",
    "neoracer_mujoco.track_generation.Difficulty",
    "neoracer_mujoco.track_generation.NoValidTrack",
    "neoracer_mujoco.track_generation.Track",
    "neoracer_mujoco.track_generation.WallParams",
    "neoracer_mujoco.track_generation.generate_track",
    "neoracer_mujoco.track_generation.lap_distance",
    "neoracer_mujoco.track_generation.mjcf.to_start_frame",
    "neoracer_mujoco.track_generation.pose_along",
    "neoracer_mujoco.track_generation.settings_for_difficulty",
    "neoracer_mujoco.track_generation.shortest_loop_distance",
    "neoracer_mujoco.track_generation.to_spec",
    "neoracer_spec.neoracer.NEORACER",
    "neoracer_spec.racecar_neo_v2.RACECAR_NEO_V2",
)


USED_BY_NEORACER_RL_WITH_JAX = ("neoracer_mujoco.batch_warp.WarpBatch",)
"""Names that need the jax extra, which neoracer-rl installs."""


def _missing(names: tuple[str, ...]) -> list[str]:
    def exists(name):
        module, attr = name.rsplit(".", 1)
        return hasattr(importlib.import_module(module), attr)

    return [name for name in names if not exists(name)]


def test_the_names_used_by_neoracer_rl_exist():
    assert _missing(USED_BY_NEORACER_RL) == []


def test_the_names_used_by_neoracer_rl_with_the_jax_extra_exist():
    for module in ("jax", "mujoco.mjx", "mujoco_warp"):
        pytest.importorskip(module)
    assert _missing(USED_BY_NEORACER_RL_WITH_JAX) == []


def test_records_used_by_neoracer_rl_keep_their_fields():
    def fields(record):
        return {f.name for f in dataclasses.fields(record)}

    assert {"throttle", "steer"} <= fields(ControlCommand)
    assert {"steer_rad"} <= fields(ActuatorLimits)
    assert {"spec", "model", "cars"} <= fields(Scene)
    assert isinstance(Scene.car, property)
    assert {"joint", "drive", "wheels", "sensors", "truth"} <= fields(Car)
    # The observation reads these of car.sensors, and the progress, the crash
    # and the rollover these of car.truth.
    beams = {f"lidar_{degrees:03d}" for degrees in range(0, 360, 45)}
    wheels = {f"{corner}_wheel_vel" for corner in ("fl", "fr", "rl", "rr")}
    imu = {"imu_accel", "imu_gyro", "imu_quat", "steer_cmd_pos"}
    assert beams | wheels | imu <= fields(CarSensors)
    assert {"pos", "wall_contact", "zaxis"} <= fields(TruthSensors)
    assert {"lateral", "yaw", "resettle"} <= fields(SpawnRange)
    assert {
        "wheel_friction",
        "chassis_mass",
        "motor_torque",
        "steer_stiffness",
        "lidar_rate",
    } <= fields(PhysicsRanges)
    assert {"steady_speed"} <= fields(Drive)
    assert {"prefix", "offset", "heading", "car"} <= fields(Entry)
    assert {"half_width", "curvature", "total_length"} <= fields(Track)
