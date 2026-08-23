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

The `track_generation` subpackage generates closed-loop racetracks, and
`compose()` turns one into scenery the car can drive on::

    from neoracer_mujoco import compose, generate_track
    model = compose(generate_track(seed=0, difficulty=1))
    model = compose("straight_corridor")   # or a hand-written assets/tracks/ XML

Note `track_generation` (the subpackage, which makes tracks) versus `tracks()`
(the assets/tracks/ discovery function, which lists hand-written ones).
"""

from . import contract
from .assets import cars, compose, load, tracks
from .sensors import (
    IMUReading,
    LidarScan,
    SensorReadings,
    calibrate_imu,
    lidar_scan,
    read,
    wheel_speed_ms,
)
from .track_generation import Track, generate_track, settings_for_difficulty

__all__ = [
    "IMUReading",
    "LidarScan",
    "SensorReadings",
    "Track",
    "calibrate_imu",
    "cars",
    "compose",
    "contract",
    "generate_track",
    "lidar_scan",
    "load",
    "read",
    "settings_for_difficulty",
    "tracks",
    "wheel_speed_ms",
]
