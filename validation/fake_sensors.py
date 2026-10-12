"""Build SensorReadings for the pure-logic tests, without the simulator."""

from dataclasses import fields

import numpy as np

from neoracer_mujoco.sensors import SensorReadings

_WIDTH = frozendict(imu_accel=3, imu_gyro=3, imu_quat=4, imu_linvel=3)


def fake_readings(**overrides: float) -> SensorReadings:
    """A complete reading: every field is zero, except the scalar fields
    named in overrides, which get the given value."""
    values = {
        f.name: np.full(_WIDTH.get(f.name, 1), overrides.pop(f.name, 0.0))
        for f in fields(SensorReadings)
    }
    if overrides:
        raise TypeError(f"not a SensorReadings field: {sorted(overrides)}")
    return SensorReadings(**values)
