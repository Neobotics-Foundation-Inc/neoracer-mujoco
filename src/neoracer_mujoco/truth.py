"""
Ground truth that only the simulator has: where the car is, which way its
top points, and whether it touches a wall. The real car has no such sensors,
so neoracer.xml does not have them. compile_scene adds them to every scene.

They are sensors so that a step can read everything it needs from
data.sensordata. mujoco.rollout and MuJoCo Warp (MJX with impl="warp") return
sensordata for every step of a batch, so code that reads these sensors works
on one MjData and on a batch. (MJX with impl="jax" cannot run these scenes:
it has no collision between the cylinder wheels and the box walls.) Read
them like any other part of the car:

    data.bind_scalar(scene.car.truth.pos).data             # x, y, z (m), world frame
    data.bind_scalar(scene.car.truth.zaxis).data           # the car's +Z, world frame
    data.bind_scalar(scene.car.truth.wall_contact).data[0] # contacts with a wall

The walls are the geoms of the body named "walls" (WALLS_BODY). A track puts
every wall in that body. A scene without one, such as the car alone, gets an
empty walls body, so every scene has the same sensors.

MuJoCo computes sensors in mj_forward, which mj_step calls before it
integrates. After mj_step, these sensors describe the same state as data.xpos
and data.contact: the state at the start of the step.

Frame: +X forward, +Y left, +Z up.
"""

from dataclasses import dataclass
from typing import Literal, assert_never

import mujoco
import numpy as np

WALLS_BODY = "walls"

type Name = Literal["truth_pos", "truth_zaxis", "truth_wall_contact"]
POS: Name = "truth_pos"
ZAXIS: Name = "truth_zaxis"
WALL_CONTACT: Name = "truth_wall_contact"


@dataclass(frozen=True)
class TruthSensors:
    """The truth sensors, as elements of the scene's MjSpec."""

    pos: mujoco.MjsSensor  # framepos of the car body: x, y, z (m), world frame
    zaxis: mujoco.MjsSensor  # framezaxis of the car body, world frame
    # The number of contacts between a geom of the car's body tree and a geom
    # of the walls body. Zero when the car touches no wall.
    wall_contact: mujoco.MjsSensor


def _add_sensor(spec: mujoco.MjSpec, name: Name, car_body: str) -> mujoco.MjsSensor:
    """Add the truth sensor called name, of the body named car_body, to spec."""
    # The body frame, not its inertial frame, in the world frame. For a
    # contact sensor, mjOBJ_XBODY means the body's whole tree (subtree1 in
    # MJCF), and mjOBJ_BODY the body alone (body2).
    xbody = mujoco.mjtObj.mjOBJ_XBODY
    match name:
        case "truth_pos":
            sensor_type = mujoco.mjtSensor.mjSENS_FRAMEPOS
        case "truth_zaxis":
            sensor_type = mujoco.mjtSensor.mjSENS_FRAMEZAXIS
        case "truth_wall_contact":
            return spec.add_sensor(
                name=name,
                type=mujoco.mjtSensor.mjSENS_CONTACT,
                objtype=xbody,
                objname=car_body,
                reftype=mujoco.mjtObj.mjOBJ_BODY,
                refname=WALLS_BODY,
                # data="found": one value, the number of matching contacts.
                # reduce="none", num=1.
                intprm=[1 << mujoco.mjtConDataField.mjCONDATA_FOUND.value, 0, 1],
            )
        case _:
            assert_never(name)
    return spec.add_sensor(name=name, type=sensor_type, objtype=xbody, objname=car_body)


# The fields that decide what a sensor measures and how it reports it. A
# reused sensor must match the truth sensor in each one, or a noise, cutoff
# or delay setting would change what code reads as ground truth.
_SETTINGS = (
    "type",
    "objtype",
    "objname",
    "reftype",
    "refname",
    "intprm",
    "dim",
    "datatype",
    "needstage",
    "cutoff",
    "noise",
    "delay",
    "interval",
    "nsample",
    "interp",
    "userdata",
)


def _setting(sensor: mujoco.MjsSensor, field: str) -> np.ndarray:
    value = getattr(sensor, field)
    return np.asarray(list(value) if field == "userdata" else value)


def _same_sensor(a: mujoco.MjsSensor, b: mujoco.MjsSensor) -> bool:
    return all(np.array_equal(_setting(a, f), _setting(b, f)) for f in _SETTINGS)


def add_sensors(spec: mujoco.MjSpec, car_body: str) -> None:
    """Add the truth sensors of the body named car_body to spec, and an empty
    walls body if spec has none. A truth sensor that spec already has stays,
    so the spec of a Scene can be compiled again. Raise ValueError if spec
    has a sensor with the name of a truth sensor but other settings."""
    if spec.body(WALLS_BODY) is None:
        spec.worldbody.add_body(name=WALLS_BODY)
    reference = mujoco.MjSpec()  # holds the truth sensors to compare against
    for name in (POS, ZAXIS, WALL_CONTACT):
        sensor = spec.sensor(name)
        if sensor is None:
            _add_sensor(spec, name, car_body)
        elif not _same_sensor(sensor, _add_sensor(reference, name, car_body)):
            raise ValueError(f"the spec has a sensor {name} that is not a truth sensor")


def find_sensors(spec: mujoco.MjSpec) -> TruthSensors:
    """Find the sensors that add_sensors added to spec."""
    pos, zaxis, wall_contact = (spec.sensor(n) for n in (POS, ZAXIS, WALL_CONTACT))
    if pos is None or zaxis is None or wall_contact is None:
        raise ValueError("the spec has no truth sensors: call truth.add_sensors")
    return TruthSensors(pos=pos, zaxis=zaxis, wall_contact=wall_contact)
