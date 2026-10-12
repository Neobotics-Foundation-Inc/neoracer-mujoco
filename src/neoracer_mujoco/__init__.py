"""
NeoRacer MuJoCo toolbox.

This package loads, steps and reads the NeoRacer car model in
assets/neoracer.xml.

Common entry points::

    import mujoco
    from neoracer_mujoco import load, read
    model = load()                  # compile assets/neoracer.xml
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)     # your own step loop, see examples/
    readings = read(model, data)    # typed SensorReadings

The `contract` module is the one source of truth for the requirements of a
NeoRacer car XML: actuator and sensor names, ctrl layout and mass band.
"""

from . import contract
from .assets import cars, load
from .sensors import SensorReadings, lidar_ranges, read

__all__ = [
    "SensorReadings",
    "cars",
    "contract",
    "lidar_ranges",
    "load",
    "read",
]
