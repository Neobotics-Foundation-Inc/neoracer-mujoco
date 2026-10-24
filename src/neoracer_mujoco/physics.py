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
import math
from collections.abc import Sequence
from dataclasses import dataclass, fields
from typing import Any

import mujoco
import numpy as np
import numpy.typing as npt

from . import sensors
from .scene import Scene


@dataclass(frozen=True)
class Physics:
    """Factors on the car's parameters, relative to the compiled car."""

    # x the sliding friction of each wheel geom, and so of each wheel's
    # contacts (the wheels have contact priority). Below 1 the wheels slip more.
    wheel_friction: float = 1.0
    chassis_mass: float = 1.0  # x the mass and the inertia of the car body
    # x the motor's torque, at every speed; x the gain of a speed controller
    motor_torque: float = 1.0
    steer_stiffness: float = 1.0  # x the gain kp of the steering servo's loop
    # x the LiDAR's scan rate: each beam samples its range, and shows the
    # sample, in 1 / lidar_rate times the time (mjcf.beam_timing)
    lidar_rate: float = 1.0

    def __post_init__(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"{f.name} must be finite and > 0, not {value}")


@dataclass(frozen=True)
class Range:
    """The interval [low, high] of a factor. Range() is the factor 1.0."""

    low: float = 1.0
    high: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 < self.low <= self.high < math.inf:
            raise ValueError(f"a range needs 0 < low <= high < inf, not {self}")


@dataclass(frozen=True)
class PhysicsRanges:
    """The interval of each factor of Physics. The default changes nothing."""

    wheel_friction: Range = Range()
    chassis_mass: Range = Range()
    motor_torque: Range = Range()
    steer_stiffness: Range = Range()
    lidar_rate: Range = Range()


def sample(ranges: PhysicsRanges, rng: np.random.Generator) -> Physics:
    """A Physics with each factor drawn uniformly from its range, in the
    order of the fields. A range of one value takes no draw from rng, so
    PhysicsRanges() leaves rng as it was."""

    def draw(r: Range) -> float:
        return r.low if r.low == r.high else float(rng.uniform(r.low, r.high))

    return Physics(
        wheel_friction=draw(ranges.wheel_friction),
        chassis_mass=draw(ranges.chassis_mass),
        motor_torque=draw(ranges.motor_torque),
        steer_stiffness=draw(ranges.steer_stiffness),
        lidar_rate=draw(ranges.lidar_rate),
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
    # A DC motor's torque, the NeoRacer's, is K (V - K w) / R, with
    # gainprm[0] = R and gainprm[1] = K. Dividing R scales the torque at
    # every speed and keeps the free speed V / K. A speed controller's force,
    # the RACECAR Neo V2's, is its gains times ctrl and the speed.
    for actuator in car.throttled:
        a, a0 = model.bind_scalar(actuator), nominal.bind_scalar(actuator)
        if a0.gaintype[0] == mujoco.mjtGain.mjGAIN_DCMOTOR:
            a.gainprm[0] = a0.gainprm[0] / physics.motor_torque
        else:
            a.gainprm[:3] = a0.gainprm[:3] * physics.motor_torque
    # The servo's position loop applies kp (ctrl - q) volts, with
    # gainprm[4] = kp, so near its target its stiffness is proportional to kp.
    servo, servo0 = model.bind_scalar(car.steer), nominal.bind_scalar(car.steer)
    servo.gainprm[4] = servo0.gainprm[4] * physics.steer_stiffness
    # A beam's interval is its period and phase. MjModel's sensor views have
    # no interval or delay, so these two go by the sensor's id.
    for name in sensors.LIDAR_BEAM_ORDER:
        i = nominal.bind_scalar(getattr(car.sensors, name)).id
        model.sensor_interval[i] = nominal.sensor_interval[i] / physics.lidar_rate
        model.sensor_delay[i] = nominal.sensor_delay[i] / physics.lidar_rate
    mujoco.mj_setConst(model, mujoco.MjData(model))


# The MjModel fields that apply_physics writes, and those that mj_setConst
# derives from them. A dot names a field of a field.
FIELDS = (
    "geom_friction",
    "body_mass",
    "body_inertia",
    "actuator_gainprm",
    "sensor_interval",
    "sensor_delay",
    "body_subtreemass",
    "body_invweight0",
    "dof_invweight0",
    "dof_M0",
    "tendon_invweight0",
    "actuator_acc0",
    "stat.meaninertia",  # scales the solver's tolerance
)


def field(model: mujoco.MjModel, name: str) -> npt.NDArray[np.float64]:
    """The value of the field of model that name in FIELDS names."""
    value: Any = model
    for part in name.split("."):
        value = getattr(value, part)
    return np.asarray(value)


def field_rows(
    scene: Scene, physics: Sequence[Physics]
) -> dict[str, npt.NDArray[np.float64]]:
    """For each name in FIELDS, the value of that field in a model with
    physics[i] applied, stacked over i: shape (len(physics), *field shape)."""
    scratch = copy.deepcopy(scene.model)
    distinct: dict[Physics, dict[str, npt.NDArray[np.float64]]] = {}
    for p in physics:
        if p not in distinct:
            apply_physics(scratch, scene, p)
            distinct[p] = {f: np.array(field(scratch, f)) for f in FIELDS}
    return {f: np.stack([distinct[p][f] for p in physics]) for f in FIELDS}
