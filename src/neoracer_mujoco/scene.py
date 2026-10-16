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

CAR_BODY = "car"  # the car's root body, with the free joint


@dataclass(frozen=True)
class Car:
    """The parts of the car, as elements of the scene's MjSpec."""

    body: mujoco.MjsBody  # the car's root body
    joint: mujoco.MjsJoint  # its free joint: qpos x, y, z, qw, qx, qy, qz
    drive: mujoco.MjsActuator  # the drive motor (terminal voltage in V, + = forward)
    steer: mujoco.MjsActuator  # the steering servo (rad, + = left)
    # the fl, fr, rl, rr wheel geoms that touch the ground
    wheels: tuple[mujoco.MjsGeom, mujoco.MjsGeom, mujoco.MjsGeom, mujoco.MjsGeom]
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
    wheel geom or a sensor of contract.py is missing."""
    problems: list[str] = []
    body = spec.body(CAR_BODY)
    if body is None:
        raise ValueError(f'the model has no "{CAR_BODY}" body')
    joints = body.joints
    if len(joints) != 1 or joints[0].type != mujoco.mjtJoint.mjJNT_FREE:
        problems.append(f'the "{CAR_BODY}" body needs exactly one free joint')
    actuators = {name: spec.actuator(name) for name in contract.EXPECTED_ACTUATORS}
    problems += [f"actuator {n} is missing" for n, a in actuators.items() if a is None]
    wheels = {name: spec.geom(name) for name in contract.WHEEL_GEOMS}
    problems += [f"wheel geom {n} is missing" for n, g in wheels.items() if g is None]
    if problems:
        raise ValueError("; ".join(problems))
    wfl, wfr, wrl, wrr = contract.WHEEL_GEOMS
    return Car(
        body=body,
        joint=joints[0],
        drive=actuators[contract.DRIVE_ACTUATOR],
        steer=actuators[contract.STEER_ACTUATOR],
        wheels=(wheels[wfl], wheels[wfr], wheels[wrl], wheels[wrr]),
        sensors=sensors.find_sensors(spec, model),
        truth=truth.find_sensors(spec),
    )


def compile_scene(spec: mujoco.MjSpec) -> Scene:
    """Add the truth sensors to spec (truth.add_sensors) and compile it into a
    Scene. Raise ValueError if the car is not a NeoRacer car (see find_car)."""
    if spec.body(CAR_BODY) is None:
        raise ValueError(f'the model has no "{CAR_BODY}" body')
    truth.add_sensors(spec, CAR_BODY)
    model = spec.compile()
    return Scene(spec=spec, model=model, car=find_car(spec, model))
