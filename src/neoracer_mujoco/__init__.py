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
    readings = read(scene.car.sensors, data)  # typed SensorReadings

A Scene keeps the MjSpec next to the compiled MjModel, and scene.car holds
the parts of the car as spec elements, to bind to the model or the data
(see scene.py).

The `contract` module is the one source of truth for the requirements of a
NeoRacer car XML, such as its actuator names and mass band; the fields of
sensors.SensorReadings are the sensors it must have.
"""

from . import contract
from .assets import cars, load
from .scene import Car, Scene
from .sensors import SensorReadings, lidar_ranges, read

__all__ = [
    "Car",
    "Scene",
    "SensorReadings",
    "cars",
    "contract",
    "lidar_ranges",
    "load",
    "read",
]
