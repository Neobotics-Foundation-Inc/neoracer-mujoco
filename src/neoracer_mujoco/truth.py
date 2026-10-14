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

import mujoco

WALLS_BODY = "walls"

POS = "truth_pos"
ZAXIS = "truth_zaxis"
WALL_CONTACT = "truth_wall_contact"


@dataclass(frozen=True)
class TruthSensors:
    """The truth sensors, as elements of the scene's MjSpec."""

    pos: mujoco.MjsSensor  # framepos of the car body: x, y, z (m), world frame
    zaxis: mujoco.MjsSensor  # framezaxis of the car body, world frame
    # The number of contacts between a geom of the car's body tree and a geom
    # of the walls body. Zero when the car touches no wall.
    wall_contact: mujoco.MjsSensor


def _definitions(car_body: str) -> frozendict[str, frozendict[str, object]]:
    """The MjSpec fields of each truth sensor of the body named car_body."""
    # The body frame, not its inertial frame, measured in the world frame (no
    # reference object). For a contact sensor, mjOBJ_XBODY means the body's
    # whole tree (subtree1 in MJCF), and mjOBJ_BODY the body alone (body2).
    car: frozendict[str, object] = frozendict(
        objtype=mujoco.mjtObj.mjOBJ_XBODY,
        objname=car_body,
        reftype=mujoco.mjtObj.mjOBJ_UNKNOWN,
        refname="",
    )
    return frozendict(
        {
            POS: frozendict(car, type=mujoco.mjtSensor.mjSENS_FRAMEPOS),
            ZAXIS: frozendict(car, type=mujoco.mjtSensor.mjSENS_FRAMEZAXIS),
            WALL_CONTACT: frozendict(
                car,
                type=mujoco.mjtSensor.mjSENS_CONTACT,
                reftype=mujoco.mjtObj.mjOBJ_BODY,
                refname=WALLS_BODY,
                # data="found": one value, the number of matching contacts.
                # reduce="none", num=1.
                intprm=(1 << mujoco.mjtConDataField.mjCONDATA_FOUND.value, 0, 1),
            ),
        }
    )


def _field(sensor: mujoco.MjsSensor, name: str) -> object:
    value = getattr(sensor, name)
    return tuple(int(x) for x in value) if name == "intprm" else value


def add_sensors(spec: mujoco.MjSpec, car_body: str) -> None:
    """Add the truth sensors of the body named car_body to spec, and an empty
    walls body if spec has none. A truth sensor that spec already has stays,
    so the spec of a Scene can be compiled again. Raise ValueError if spec
    has a sensor with the name of a truth sensor but another definition."""
    if spec.body(WALLS_BODY) is None:
        spec.worldbody.add_body(name=WALLS_BODY)
    for name, definition in _definitions(car_body).items():
        sensor = spec.sensor(name)
        if sensor is None:
            spec.add_sensor(name=name, **definition)
        elif any(_field(sensor, k) != v for k, v in definition.items()):
            raise ValueError(f"the spec has a sensor {name} that is not a truth sensor")


def find_sensors(spec: mujoco.MjSpec) -> TruthSensors:
    """Find the sensors that add_sensors added to spec."""
    pos, zaxis, wall_contact = (spec.sensor(n) for n in (POS, ZAXIS, WALL_CONTACT))
    if pos is None or zaxis is None or wall_contact is None:
        raise ValueError("the spec has no truth sensors: call truth.add_sensors")
    return TruthSensors(pos=pos, zaxis=zaxis, wall_contact=wall_contact)
