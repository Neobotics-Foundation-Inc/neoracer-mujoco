"""
A compiled NeoRacer scene, and the indices of the car in its arrays.

A Scene keeps the MjSpec next to the MjModel that was compiled from it.
car_index finds every part of the car once, with model.bind on the elements
of that spec: no name lookups in the compiled model and no address
arithmetic. The code calls model.bind_scalar, which is what model.bind runs
for one element, because the generated type stubs get model.bind wrong. A step loop then reads and writes plain arrays (data.ctrl,
data.qpos, data.xpos, data.sensordata) with these indices. Plain indices
work the same on numpy arrays and on the arrays of other MuJoCo backends.

Frame: +X forward, +Y left, +Z up.
"""

from dataclasses import dataclass

import mujoco

from . import contract, sensors

CAR_BODY = "car"  # the car's root body, with the free joint


@dataclass(frozen=True)
class CarIndex:
    """Where the car is in the arrays of one compiled scene."""

    body: int  # body id of the car's root body; xpos[body], xmat[body]
    qpos: slice  # the car's free joint in qpos: x, y, z (m), then qw, qx, qy, qz
    qvel: slice  # the same joint in qvel: linear (m/s, world), angular (rad/s, body)
    drive: tuple[int, int, int, int]  # ctrl index of the fl, fr, rl, rr motors
    steer: int  # ctrl index of the steering servo
    sensors: sensors.SensorIndex


@dataclass(frozen=True)
class Scene:
    """A compiled scene. spec is the MjSpec that model was compiled from:
    model.bind needs its elements to find other parts of the scene."""

    spec: mujoco.MjSpec
    model: mujoco.MjModel
    car: CarIndex


def car_index(spec: mujoco.MjSpec, model: mujoco.MjModel) -> CarIndex:
    """Find the car in model, which was compiled from spec. Raise ValueError
    if the car body has no free joint or an actuator or sensor of
    contract.py is missing."""
    problems: list[str] = []
    body = spec.body(CAR_BODY)
    if body is None:
        raise ValueError(f'the model has no "{CAR_BODY}" body')
    joints = body.joints
    if len(joints) != 1 or joints[0].type != mujoco.mjtJoint.mjJNT_FREE:
        problems.append(f'the "{CAR_BODY}" body needs exactly one free joint')
    ctrl: dict[str, int] = {}
    for name in contract.EXPECTED_ACTUATORS:
        actuator = spec.actuator(name)
        if actuator is None:
            problems.append(f"actuator {name} is missing")
        else:
            ctrl[name] = model.bind_scalar(actuator).id
    if problems:
        raise ValueError("; ".join(problems))
    joint = model.bind_scalar(joints[0])
    qpos_start, qvel_start = int(joint.qposadr[0]), int(joint.dofadr[0])
    fl, fr, rl, rr, steer = (ctrl[name] for name in contract.EXPECTED_ACTUATORS)
    return CarIndex(
        body=model.bind_scalar(body).id,
        qpos=slice(qpos_start, qpos_start + 7),
        qvel=slice(qvel_start, qvel_start + 6),
        drive=(fl, fr, rl, rr),
        steer=steer,
        sensors=sensors.sensor_index(spec, model),
    )


def compile_scene(spec: mujoco.MjSpec) -> Scene:
    """Compile spec into a Scene. Raise ValueError if the car is not a
    NeoRacer car (see car_index)."""
    model = spec.compile()
    return Scene(spec=spec, model=model, car=car_index(spec, model))
