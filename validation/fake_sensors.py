"""Build SensorReadings for the pure-logic tests, without the simulator."""

from dataclasses import replace

import numpy as np

from neoracer_mujoco.sensors import SensorReadings

_ZERO = SensorReadings(
    imu_accel=np.zeros(3),
    imu_gyro=np.zeros(3),
    imu_quat=np.zeros(4),
    imu_linvel=np.zeros(3),
    steer_cmd_pos=0.0,
    fl_steer_pos=0.0,
    fr_steer_pos=0.0,
    fl_wheel_vel=0.0,
    fr_wheel_vel=0.0,
    rl_wheel_vel=0.0,
    rr_wheel_vel=0.0,
    fl_susp_pos=0.0,
    fr_susp_pos=0.0,
    rl_susp_pos=0.0,
    rr_susp_pos=0.0,
    lidar_000=0.0,
    lidar_045=0.0,
    lidar_090=0.0,
    lidar_135=0.0,
    lidar_180=0.0,
    lidar_225=0.0,
    lidar_270=0.0,
    lidar_315=0.0,
)


def fake_readings(**overrides: float) -> SensorReadings:
    """Return a SensorReadings with every field zero, except the scalar
    fields named in overrides, which get the given value."""
    return replace(_ZERO, **overrides)
