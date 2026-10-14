"""
NeoRacer MuJoCo toolbox.

This package loads, steps and reads the NeoRacer car model in
assets/neoracer.xml.

Common entry points::

    import mujoco
    from neoracer_mujoco import load, read
    scene = load()                  # compile assets/neoracer.xml
    data = mujoco.MjData(scene.model)
    mujoco.mj_step(scene.model, data)  # your own step loop, see examples/
    readings = read(scene.car.sensors, data.sensordata)  # typed SensorReadings

A Scene keeps the MjSpec next to the compiled MjModel, and scene.car holds
the car's indices in the model's arrays (see scene.py).

The `contract` module is the one source of truth for the requirements of a
NeoRacer car XML: actuator and sensor names and mass band.
"""

from . import contract
from .assets import cars, load
from .scene import CarIndex, Scene
from .sensors import SensorReadings, lidar_ranges, read

__all__ = [
    "CarIndex",
    "Scene",
    "SensorReadings",
    "cars",
    "contract",
    "lidar_ranges",
    "load",
    "read",
]
