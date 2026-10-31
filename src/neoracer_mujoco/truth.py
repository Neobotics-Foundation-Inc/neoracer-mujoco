"""
Ground truth that only the simulator has: where each car is, which way its
top points, whether it touches a wall, and whether it touches another car.
The real car has no such sensors, so neoracer.xml does not have them.
compile_scene adds them to every scene.

They are sensors so that a step can read everything it needs from
data.sensordata. mujoco.rollout and MuJoCo Warp (MJX with impl="warp") return
sensordata for every step of a batch, so code that reads these sensors works
on one MjData and on a batch. (MJX with impl="jax" cannot run these scenes:
it has no collision between the cylinder wheels and the box walls.) Read
them like any other part of the car:

    data.bind_scalar(scene.car.truth.pos).data             # x, y, z (m), world frame
    data.bind_scalar(scene.car.truth.zaxis).data           # the car's +Z, world frame
    data.bind_scalar(scene.car.truth.wall_contact).data[0] # contacts with a wall
    data.bind_scalar(scene.car_contacts[0, 1]).data[0]     # contacts between two cars

Each car has its own pos, zaxis and wall_contact, named behind its prefix
(scene.py), and each two cars have a car_contact sensor.

The walls are the geoms of the body named "walls" (WALLS_BODY). A track puts
every wall in that body. A scene without one, such as the car alone, gets an
empty walls body, so every scene has the same sensors.

MuJoCo computes sensors in mj_forward, which mj_step calls before it
integrates. After mj_step, these sensors describe the same state as data.xpos
and data.contact: the state at the start of the step.

Frame: +X forward, +Y left, +Z up.
"""

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypedDict

import mujoco
import numpy as np

from . import contract

WALLS_BODY = "walls"
POS = "truth_pos"
ZAXIS = "truth_zaxis"
WALL_CONTACT = "truth_wall_contact"
CAR_CONTACT = "truth_contact_"
"""The car_contact sensor of the cars behind prefixes p and q, in the
scene's order, is named p + CAR_CONTACT + q + contract.CAR_BODY."""


@dataclass(frozen=True)
class TruthSensors:
    """The truth sensors of a car, as elements of the scene's MjSpec."""

    pos: mujoco.MjsSensor
    """The framepos of the car body: x, y, z (m), in the world frame."""
    zaxis: mujoco.MjsSensor
    """The framezaxis of the car body, in the world frame."""
    wall_contact: mujoco.MjsSensor
    """The number of contacts between a geom of the car's body tree and a geom of
    the walls body. Zero when the car touches no wall."""


def add_sensors(spec: mujoco.MjSpec, prefixes: Sequence[str] = ("",)) -> None:
    """Add to spec the truth sensors of the car behind each of prefixes, whose
    root body is prefix + contract.CAR_BODY, a car_contact sensor for each two
    of them, and an empty walls body if spec has none. A truth sensor that
    spec already has stays, so the spec of a Scene can be compiled again.
    Raise ValueError if spec has a sensor with the name of a truth sensor but
    other settings."""
    if spec.body(WALLS_BODY) is None:
        spec.worldbody.add_body(name=WALLS_BODY)
    reference = mujoco.MjSpec()  # holds the truth sensors to compare against
    for settings in _sensors(prefixes):
        name = settings["name"]
        sensor = spec.sensor(name)
        if sensor is None:
            spec.add_sensor(**settings)
        elif not _same_sensor(sensor, reference.add_sensor(**settings)):
            raise ValueError(f"the spec has a sensor {name} that is not a truth sensor")


def find_sensors(spec: mujoco.MjSpec, prefix: str = "") -> TruthSensors:
    """Find the sensors that add_sensors added to spec for the car behind
    prefix."""
    names = (prefix + POS, prefix + ZAXIS, prefix + WALL_CONTACT)
    pos, zaxis, wall_contact = (spec.sensor(n) for n in names)
    if pos is None or zaxis is None or wall_contact is None:
        raise ValueError("the spec has no truth sensors: call truth.add_sensors")
    return TruthSensors(pos=pos, zaxis=zaxis, wall_contact=wall_contact)


def find_car_contacts(
    spec: mujoco.MjSpec, prefixes: Sequence[str]
) -> Mapping[tuple[int, int], mujoco.MjsSensor]:
    """The car_contact sensor that add_sensors added to spec for the cars
    behind prefixes[i] and prefixes[j], by (i, j), for each i < j. It counts
    the contacts between a geom of one car's body tree and a geom of the
    other's."""
    found = {}
    for (i, p), (j, q) in itertools.combinations(enumerate(prefixes), 2):
        sensor = spec.sensor(_car_contact(p, q))
        if sensor is None:
            raise ValueError("the spec has no truth sensors: call truth.add_sensors")
        found[i, j] = sensor
    return MappingProxyType(found)


class _Sensor(TypedDict, total=False):
    """The arguments of MjSpec.add_sensor that make a truth sensor."""

    name: str
    type: mujoco.mjtSensor
    objtype: mujoco.mjtObj
    objname: str
    reftype: mujoco.mjtObj
    refname: str
    intprm: list[int]


def _sensors(prefixes: Sequence[str]) -> list[_Sensor]:
    """The settings of every truth sensor of the cars behind prefixes."""
    # The body frame, not its inertial frame, in the world frame. For a
    # contact sensor, mjOBJ_XBODY means the body's whole tree (subtree1 or
    # subtree2 in MJCF), and mjOBJ_BODY the body alone (body2).
    xbody = mujoco.mjtObj.mjOBJ_XBODY
    contact = mujoco.mjtSensor.mjSENS_CONTACT
    # data="found": one value, the number of matching contacts. reduce="none",
    # num=1.
    found = [1 << mujoco.mjtConDataField.mjCONDATA_FOUND.value, 0, 1]
    sensors: list[_Sensor] = []
    for p in prefixes:
        car = p + contract.CAR_BODY
        sensors += [
            {
                "name": p + POS,
                "type": mujoco.mjtSensor.mjSENS_FRAMEPOS,
                "objtype": xbody,
                "objname": car,
            },
            {
                "name": p + ZAXIS,
                "type": mujoco.mjtSensor.mjSENS_FRAMEZAXIS,
                "objtype": xbody,
                "objname": car,
            },
            {
                "name": p + WALL_CONTACT,
                "type": contact,
                "objtype": xbody,
                "objname": car,
                "reftype": mujoco.mjtObj.mjOBJ_BODY,
                "refname": WALLS_BODY,
                "intprm": found,
            },
        ]
    sensors += [
        {
            "name": _car_contact(p, q),
            "type": contact,
            "objtype": xbody,
            "objname": p + contract.CAR_BODY,
            "reftype": xbody,
            "refname": q + contract.CAR_BODY,
            "intprm": found,
        }
        for p, q in itertools.combinations(prefixes, 2)
    ]
    return sensors


def _car_contact(p: str, q: str) -> str:
    return p + CAR_CONTACT + q + contract.CAR_BODY


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
"""The fields that decide what a sensor measures and how it reports it. A
reused sensor must match the truth sensor in each one, or a noise, cutoff or
delay setting would change what code reads as ground truth."""


def _same_sensor(a: mujoco.MjsSensor, b: mujoco.MjsSensor) -> bool:
    return all(np.array_equal(_setting(a, f), _setting(b, f)) for f in _SETTINGS)


def _setting(sensor: mujoco.MjsSensor, field: str) -> np.ndarray:
    value = getattr(sensor, field)
    return np.asarray(list(value) if field == "userdata" else value)
