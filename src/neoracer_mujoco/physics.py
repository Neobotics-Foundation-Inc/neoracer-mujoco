"""
Change the car's physical parameters, for domain randomization: training a
controller on many slightly different cars, so that it also works on the
real one, whose parameters nobody knows exactly.

A Physics holds one factor for each parameter that can change. All 1.0 is the
car of the XML. PhysicsRanges gives the interval of each factor, and sample
draws a Physics from it:

    ranges = PhysicsRanges(wheel_friction=Range(0.5, 1.2))
    params = sample(ranges, rng)  # rng: np.random.Generator
    model = copy.deepcopy(scene.model)
    apply_physics(model, scene, params)  # model now has 0.5 to 1.2 x the grip

apply_physics changes only the compiled MjModel, not the MjSpec, so there is
nothing to recompile. It then runs mj_setConst, which updates what MuJoCo
derives from the masses, so the model is the same, bit for bit, as one
compiled with the new values. FIELDS names every MjModel field that it can
change, which the batches use to give each row its own physics.

batch.py takes one Physics per row. A batch on the CPU keeps one MjModel for
each different Physics, so build its scene with visuals=False: the model is
1.6 MB instead of 35 MB.
"""

import copy
import functools
from collections.abc import Sequence
from dataclasses import dataclass

import mujoco
import numpy as np
import numpy.typing as npt

from .scene import Scene


@dataclass(frozen=True)
class Physics:
    """Factors on the car's parameters, relative to the compiled car."""

    # x the sliding friction of each wheel geom, and so of each wheel's
    # contacts (the wheels have contact priority). Below 1 the wheels slip more.
    wheel_friction: float = 1.0
    chassis_mass: float = 1.0  # x the mass and the inertia of the car body
    motor_torque: float = (
        1.0  # x the torque that each drive motor gives per unit of ctrl
    )
    steer_stiffness: float = 1.0  # x the stiffness (kp) of the steering servo


@dataclass(frozen=True)
class Range:
    """The interval [low, high] of a factor. Range() is the factor 1.0."""

    low: float = 1.0
    high: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 < self.low <= self.high:
            raise ValueError(f"a range needs 0 < low <= high, not {self}")


@dataclass(frozen=True)
class PhysicsRanges:
    """The interval of each factor of Physics. The default changes nothing."""

    wheel_friction: Range = Range()
    chassis_mass: Range = Range()
    motor_torque: Range = Range()
    steer_stiffness: Range = Range()


def sample(ranges: PhysicsRanges, rng: np.random.Generator) -> Physics:
    """A Physics with each factor drawn uniformly from its range, in the
    order of the fields."""

    def draw(r: Range) -> float:
        return float(rng.uniform(r.low, r.high))

    return Physics(
        wheel_friction=draw(ranges.wheel_friction),
        chassis_mass=draw(ranges.chassis_mass),
        motor_torque=draw(ranges.motor_torque),
        steer_stiffness=draw(ranges.steer_stiffness),
    )


def apply_physics(model: mujoco.MjModel, scene: Scene, physics: Physics) -> None:
    """Write into model, a copy of scene.model, the parameters of
    scene.model times the factors of physics, then run mj_setConst."""
    nominal, car = scene.model, scene.car
    for wheel in car.wheels:
        friction = nominal.bind_scalar(wheel).friction[0]
        model.bind_scalar(wheel).friction[0] = friction * physics.wheel_friction
    body, body0 = model.bind_scalar(car.body), nominal.bind_scalar(car.body)
    body.mass = body0.mass * physics.chassis_mass
    body.inertia = body0.inertia * physics.chassis_mass
    for motor in car.drive:
        gain = nominal.bind_scalar(motor).gainprm[0]
        model.bind_scalar(motor).gainprm[0] = gain * physics.motor_torque
    # A position servo's force is kp * (ctrl - q): gainprm[0] = kp and
    # biasprm[1] = -kp.
    servo, servo0 = model.bind_scalar(car.steer), nominal.bind_scalar(car.steer)
    servo.gainprm[0] = servo0.gainprm[0] * physics.steer_stiffness
    servo.biasprm[1] = servo0.biasprm[1] * physics.steer_stiffness
    mujoco.mj_setConst(model, mujoco.MjData(model))


# The MjModel fields that apply_physics writes, and those that mj_setConst
# derives from them. A dot names a field of a field.
FIELDS = (
    "geom_friction",
    "body_mass",
    "body_inertia",
    "actuator_gainprm",
    "actuator_biasprm",
    "body_subtreemass",
    "body_invweight0",
    "dof_invweight0",
    "dof_M0",
    "actuator_acc0",
    "stat.meaninertia",  # scales the solver's tolerance
)


def field(model: mujoco.MjModel, name: str) -> object:
    """The field of model that name in FIELDS names."""
    return functools.reduce(getattr, name.split("."), model)


def field_rows(
    scene: Scene, physics: Sequence[Physics]
) -> dict[str, npt.NDArray[np.float64]]:
    """For each name in FIELDS, the value of that field in a model with
    physics[i] applied, stacked over i: shape (len(physics), *field shape)."""
    scratch = copy.deepcopy(scene.model)
    rows: dict[str, list[npt.NDArray[np.float64]]] = {f: [] for f in FIELDS}
    for p in physics:
        apply_physics(scratch, scene, p)
        for f in FIELDS:
            rows[f].append(np.array(field(scratch, f)))
    return {f: np.stack(v) for f, v in rows.items()}
