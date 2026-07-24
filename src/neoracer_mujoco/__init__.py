"""
NeoRacer MuJoCo toolbox.

Reusable pieces for loading, stepping, and reading the NeoRacer car model.
The car XML in assets/ is the product; this package is the Python around it.

Common entry points::

    from neoracer_mujoco import load, read
    model = load()                 # compile assets/neoracer.xml
    data = model_step_loop(...)    # your own step loop, see examples/
    readings = read(model, data)   # typed SensorReadings

The `contract` module is the single source of truth for what every NeoRacer
car XML must satisfy (actuator/sensor names, ctrl layout, mass band).
"""

from . import contract
from .assets import cars, load
from .sensors import (
    IMUReading,
    LidarScan,
    SensorReadings,
    calibrate_imu,
    lidar_scan,
    read,
    wheel_speed_ms,
)

__all__ = [
    "IMUReading",
    "LidarScan",
    "SensorReadings",
    "calibrate_imu",
    "cars",
    "contract",
    "lidar_scan",
    "load",
    "read",
    "wheel_speed_ms",
]
