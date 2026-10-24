"""
A compiled NeoRacer scene, and the parts of its car.

A Scene keeps the MjSpec next to the MjModel that was compiled from it. Car
holds the parts of the car as elements of that spec: the body, its free
joint, the actuators, the sensors of the real car, and the truth sensors that
only the simulator has (truth.py). Code reads and writes a part by binding
it to the model or the data where it uses it:

    data.bind_scalar(scene.car.steer).ctrl = 0.2    # rad, + = left
    data.bind_scalar(scene.car.joint).qpos          # x, y, z (m), then qw, qx, qy, qz
    model.bind_scalar(scene.car.body).subtreemass   # kg

No code looks up names in the compiled model, computes addresses or keeps
indices. MJX has the same bind (dx.bind(mx, element)), so this code has the
same shape on the JAX backend.

bind_scalar is what bind runs for one element. Call it instead of bind: the
type stubs that pybind11-stubgen makes for MuJoCo get bind wrong.

Frame: +X forward, +Y left, +Z up.
"""

from dataclasses import dataclass

import mujoco

from . import contract, sensors, truth


@dataclass(frozen=True)
class Car:
    """The parts of the car, as elements of the scene's MjSpec."""

    body: mujoco.MjsBody  # the car's root body
    joint: mujoco.MjsJoint  # its free joint: qpos x, y, z, qw, qx, qy, qz
    drive: mujoco.MjsActuator  # the drive motor (contract.py), + = forward
    steer: mujoco.MjsActuator  # the steering servo (rad, + = left)
    # The actuators that the throttle drives (contract.throttle_map): the
    # motor first, then any others the car has.
    throttled: tuple[mujoco.MjsActuator, ...]
    # the fl, fr, rl, rr wheel geoms that touch the ground
    wheels: tuple[mujoco.MjsGeom, mujoco.MjsGeom, mujoco.MjsGeom, mujoco.MjsGeom]
    lidar: mujoco.MjsSite  # the center of the spinning LiDAR (lidar.py)
    sensors: sensors.CarSensors  # the sensors of the real car
    truth: truth.TruthSensors  # what only the simulator knows (truth.py)


@dataclass(frozen=True)
class Scene:
    """A compiled scene. spec is the MjSpec that model was compiled from:
    its elements are what code binds to model and to an MjData of model."""

    spec: mujoco.MjSpec
    model: mujoco.MjModel
    car: Car


def find_car(spec: mujoco.MjSpec, model: mujoco.MjModel) -> Car:
    """Find the parts of the car in spec, which model was compiled from.
    Raise ValueError if the car body has no free joint, or an actuator, a
    wheel geom, the LiDAR site or a sensor of contract.py is missing, or an
    actuator's user numbers are not a throttle map."""
    problems: list[str] = []
    body = _car_body(spec)
    joints = body.joints
    if len(joints) != 1 or joints[0].type != mujoco.mjtJoint.mjJNT_FREE:
        problems.append(f'the "{contract.CAR_BODY}" body needs exactly one free joint')
    actuators = {name: spec.actuator(name) for name in contract.REQUIRED_ACTUATORS}
    problems += [f"actuator {n} is missing" for n, a in actuators.items() if a is None]
    others = [a for a in spec.actuators if a.name not in contract.REQUIRED_ACTUATORS]
    problems += [
        f"actuator {a.name} needs {contract.THROTTLE_MAP_SIZE} user numbers"
        for a in others
        if len(a.userdata) != contract.THROTTLE_MAP_SIZE
    ]
    motor = actuators[contract.DRIVE_ACTUATOR]
    if motor is not None and len(motor.userdata) not in (0, contract.THROTTLE_MAP_SIZE):
        problems.append(
            f"actuator {motor.name} needs 0 or {contract.THROTTLE_MAP_SIZE} user numbers"
        )
    wheels = {name: spec.geom(name) for name in contract.WHEEL_GEOMS}
    problems += [f"wheel geom {n} is missing" for n, g in wheels.items() if g is None]
    lidar = spec.site(contract.LIDAR_SITE)
    if lidar is None:
        problems.append(f"site {contract.LIDAR_SITE} is missing")
    if problems:
        raise ValueError("; ".join(problems))
    wfl, wfr, wrl, wrr = contract.WHEEL_GEOMS
    return Car(
        body=body,
        joint=joints[0],
        drive=actuators[contract.DRIVE_ACTUATOR],
        steer=actuators[contract.STEER_ACTUATOR],
        throttled=(actuators[contract.DRIVE_ACTUATOR], *others),
        wheels=(wheels[wfl], wheels[wfr], wheels[wrl], wheels[wrr]),
        lidar=lidar,
        sensors=sensors.find_sensors(spec, model),
        truth=truth.find_sensors(spec),
    )


def compile_scene(spec: mujoco.MjSpec) -> Scene:
    """Add the truth sensors to spec (truth.add_sensors) and compile it into a
    Scene. Raise ValueError if the car is not a NeoRacer car (see find_car)."""
    _car_body(spec)
    truth.add_sensors(spec)
    model = spec.compile()
    return Scene(spec=spec, model=model, car=find_car(spec, model))


def _car_body(spec: mujoco.MjSpec) -> mujoco.MjsBody:
    """The car's root body in spec. Raise ValueError if spec has none."""
    body = spec.body(contract.CAR_BODY)
    if body is None:
        raise ValueError(f'the model has no "{contract.CAR_BODY}" body')
    return body
