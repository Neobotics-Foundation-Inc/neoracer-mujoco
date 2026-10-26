"""
A compiled NeoRacer scene, and the parts of its cars.

A Scene keeps the MjSpec next to the MjModel that was compiled from it. Car
holds the parts of a car as elements of that spec: the body, its free
joint, the actuators, the sensors of the real car, and the truth sensors that
only the simulator has (truth.py). Code reads and writes a part by binding
it to the model or the data where it uses it:

    data.bind_scalar(scene.car.steer).ctrl = 0.2    # rad, + = left
    data.bind_scalar(scene.car.joint).qpos          # x, y, z (m), then qw, qx, qy, qz
    model.bind_scalar(scene.car.body).subtreemass   # kg

No code looks up names in the compiled model, computes addresses or keeps
indices. MJX has the same bind (dx.bind(mx, element)), so this code has the
same shape on the JAX backend.

A scene can hold several cars (assets.compose_cars). The names of each
car's elements are behind a prefix of its own, as MjSpec's attach puts them:
the motor of the car behind "b/" is "b/motor". scene.cars holds each car,
and a function of one car takes the Car it works on, so it works on any car
of any scene:

    for car in scene.cars:
        sim.apply_command(car, data, ControlCommand(throttle=0.5, steer=0.0))

bind_scalar is what bind runs for one element. Call it instead of bind: the
type stubs that pybind11-stubgen makes for MuJoCo get bind wrong.

Frame: +X forward, +Y left, +Z up.
"""

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import mujoco

from . import contract, sensors, truth


@dataclass(frozen=True)
class Car:
    """The parts of the car, as elements of the scene's MjSpec."""

    body: mujoco.MjsBody
    """The car's root body."""
    joint: mujoco.MjsJoint
    """The root body's free joint: qpos x, y, z, qw, qx, qy, qz."""
    drive: mujoco.MjsActuator
    """The drive motor (contract.py), + = forward."""
    steer: mujoco.MjsActuator
    """The steering servo (rad, + = left)."""
    throttled: tuple[mujoco.MjsActuator, ...]
    """The actuators that the throttle drives (contract.throttle_map): the motor
    first, then any others the car has."""
    wheels: tuple[mujoco.MjsGeom, mujoco.MjsGeom, mujoco.MjsGeom, mujoco.MjsGeom]
    """The fl, fr, rl, rr wheel geoms that touch the ground."""
    lidar: mujoco.MjsSite
    """The center of the spinning LiDAR (lidar.py)."""
    sensors: sensors.CarSensors
    """The sensors of the real car."""
    truth: truth.TruthSensors
    """What only the simulator knows (truth.py)."""


@dataclass(frozen=True)
class Scene:
    """A compiled scene. spec is the MjSpec that model was compiled from:
    its elements are what code binds to model and to an MjData of model."""

    spec: mujoco.MjSpec
    model: mujoco.MjModel
    cars: tuple[Car, *tuple[Car, ...]]
    """Each car of the scene, in the order of the prefixes that compile_scene
    was given."""
    car_contacts: Mapping[tuple[int, int], mujoco.MjsSensor]
    """The truth sensor of each two cars, cars[i] and cars[j], by (i, j) for
    each i < j: the number of contacts between a geom of one car's body tree
    and a geom of the other's (truth.py)."""

    @property
    def car(self) -> Car:
        """The first car: the scene's car when it has one."""
        return self.cars[0]


def find_car(spec: mujoco.MjSpec, model: mujoco.MjModel, prefix: str = "") -> Car:
    """Find the parts of the car whose names are behind prefix in spec, which
    model was compiled from. Its actuators are the ones whose names are
    behind prefix. Raise ValueError if the car body has no free joint, or an
    actuator, a wheel geom, the LiDAR site or a sensor of contract.py is
    missing, or an actuator has no name, or user numbers that are not a
    throttle map."""
    problems: list[str] = []
    body = _car_body(spec, prefix)
    joints = body.joints
    if len(joints) != 1 or joints[0].type != mujoco.mjtJoint.mjJNT_FREE:
        problems.append(f'the "{body.name}" body needs exactly one free joint')
    required = [prefix + name for name in contract.REQUIRED_ACTUATORS]
    actuators = {name: spec.actuator(name) for name in required}
    problems += [f"actuator {n} is missing" for n, a in actuators.items() if a is None]
    if any(not a.name for a in spec.actuators):
        # attach puts no prefix on an element without a name, so no car
        # would own it.
        problems.append("an actuator has no name")
    others = [
        a
        for a in spec.actuators
        if a.name.startswith(prefix) and a.name not in required
    ]
    problems += [
        f"actuator {a.name} needs {contract.THROTTLE_MAP_SIZE} user numbers"
        for a in others
        if len(a.userdata) != contract.THROTTLE_MAP_SIZE
    ]
    motor = actuators[prefix + contract.DRIVE_ACTUATOR]
    if motor is not None and len(motor.userdata) not in (0, contract.THROTTLE_MAP_SIZE):
        problems.append(
            f"actuator {motor.name} needs 0 or {contract.THROTTLE_MAP_SIZE} user numbers"
        )
    wheels = [spec.geom(prefix + name) for name in contract.WHEEL_GEOMS]
    problems += [
        f"wheel geom {prefix + n} is missing"
        for n, g in zip(contract.WHEEL_GEOMS, wheels, strict=True)
        if g is None
    ]
    lidar = spec.site(prefix + contract.LIDAR_SITE)
    if lidar is None:
        problems.append(f"site {prefix + contract.LIDAR_SITE} is missing")
    if problems:
        raise ValueError("; ".join(problems))
    wfl, wfr, wrl, wrr = wheels
    return Car(
        body=body,
        joint=joints[0],
        drive=motor,
        steer=actuators[prefix + contract.STEER_ACTUATOR],
        throttled=(motor, *others),
        wheels=(wfl, wfr, wrl, wrr),
        lidar=lidar,
        sensors=sensors.find_sensors(spec, model, prefix),
        truth=truth.find_sensors(spec, prefix),
    )


def compile_scene(spec: mujoco.MjSpec, prefixes: Sequence[str] = ("",)) -> Scene:
    """Add the truth sensors of the car behind each of prefixes to spec
    (truth.add_sensors) and compile it into a Scene of those cars. Raise
    ValueError if there is no prefix, or a prefix starts another, which would
    make the other car's elements this one's too, or a car is not a NeoRacer
    car (see find_car)."""
    if not prefixes:
        raise ValueError("a scene needs a car")
    for p, q in itertools.permutations(prefixes, 2):
        if q.startswith(p):
            raise ValueError(f'the prefix "{p}" starts the prefix "{q}"')
    for prefix in prefixes:
        _car_body(spec, prefix)
    truth.add_sensors(spec, prefixes)
    model = spec.compile()
    first, *rest = (find_car(spec, model, p) for p in prefixes)
    return Scene(
        spec=spec,
        model=model,
        cars=(first, *rest),
        car_contacts=truth.find_car_contacts(spec, prefixes),
    )


def _car_body(spec: mujoco.MjSpec, prefix: str) -> mujoco.MjsBody:
    """The root body of the car behind prefix in spec. Raise ValueError if
    spec has none."""
    body = spec.body(prefix + contract.CAR_BODY)
    if body is None:
        raise ValueError(f'the model has no "{prefix + contract.CAR_BODY}" body')
    return body
