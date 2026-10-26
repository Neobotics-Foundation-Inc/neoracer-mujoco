"""
Domain randomization of the car's physics -- a port of the upstream repo's
neoracer_mujoco/physics.py (Neobotics-Foundation-Inc/neoracer-mujoco) to this
version's car, which has four torque motors and unnamed wheel geoms where the
upstream car has one DC-motor actuator and named wheels.

Same design and the same four factors, each a multiplier on the nominal
(as-compiled) car; all 1.0 is the car of the XML:

    wheel_friction   x the sliding friction of each wheel geom -- and so of
                     its contacts: the wheels get contact priority here, as
                     upstream, so their friction decides the grip (MuJoCo
                     otherwise takes the larger of the two geoms').
    chassis_mass     x the mass and inertia of the car body (the chassis).
    motor_torque     x the motor's torque at every speed, keeping its free
                     speed: with the realistic drive (drive_model.py) that is
                     DriveModel.torque_scale; with the legacy car, the motor
                     gears.
    steer_stiffness  x the steering servo's position gain kp.

    ranges = PhysicsRanges(wheel_friction=Range(0.75, 1.25))
    physics = sample(ranges, rng)

After changing masses, apply_physics runs mj_setConst, as upstream, so what
MuJoCo derives from the masses is the same as for a model compiled with them.

Upstream gives no recommended ranges -- PhysicsRanges() changes nothing; its
docstring example uses wheel friction 0.5-1.2 and its tests other values only
to exercise the code. GENTLE_RANGES are this project's choice (the same
spread as before this port).
"""

import math
from dataclasses import dataclass, fields

import mujoco
import numpy as np

from .. import contract
from .car_model import car_info

WHEELS = ("fl", "fr", "rl", "rr")


@dataclass(frozen=True)
class Physics:
    """Factors on the car's parameters, relative to the compiled car."""

    wheel_friction: float = 1.0
    chassis_mass: float = 1.0
    motor_torque: float = 1.0
    steer_stiffness: float = 1.0

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


# This project's ranges: grip about 0.6-1.0 around the realistic drive's 0.8.
GENTLE_RANGES = PhysicsRanges(
    wheel_friction=Range(0.75, 1.25),
    chassis_mass=Range(0.9, 1.1),
    motor_torque=Range(0.9, 1.1),
    steer_stiffness=Range(0.85, 1.15),
)


def sample(ranges: PhysicsRanges, rng: np.random.Generator) -> Physics:
    """A Physics with each factor drawn uniformly from its range, in the order
    of the fields. A range of one value takes no draw from rng."""

    def draw(r: Range) -> float:
        return r.low if r.low == r.high else float(rng.uniform(r.low, r.high))

    return Physics(
        wheel_friction=draw(ranges.wheel_friction),
        chassis_mass=draw(ranges.chassis_mass),
        motor_torque=draw(ranges.motor_torque),
        steer_stiffness=draw(ranges.steer_stiffness),
    )


@dataclass
class CarParts:
    """Where one car's randomized parameters live in a compiled model, and
    their nominal values (captured once, before any randomization)."""

    wheel_geoms: np.ndarray
    chassis: int
    drive: np.ndarray  # the four motor actuators
    steer: int
    friction: np.ndarray
    mass: float
    inertia: np.ndarray
    gear: np.ndarray
    servo_gain: float
    servo_bias: float
    native: bool = False  # the GitHub car: DC motor + DC servo (dcmotor actuators)
    motor_r: float = 0.0  # its motor's gainprm[0] = resistance R
    servo_kp: float = 0.0  # its servo's gainprm[4] = position-loop kp


def car_parts(model: mujoco.MjModel, prefix: str = "") -> CarParts:
    """Find the car named by prefix ("" for the single car, "c0_" ... in a
    race), give its wheels contact priority, and record nominal values."""
    wheels = []
    for w in WHEELS:
        body = model.body(f"{prefix}{w}_wheel_body").id
        wheels += [g for g in range(model.ngeom)
                   if model.geom_bodyid[g] == body and (model.geom_contype[g] or model.geom_conaffinity[g])]
    wheel_geoms = np.array(wheels)
    model.geom_priority[wheel_geoms] = max(1, int(model.geom_priority.max()) + 1)
    chassis = model.body(f"{prefix}car").id
    info = car_info(model, prefix)
    drive, steer = info.drive_ids, info.steer_id
    return CarParts(
        wheel_geoms=wheel_geoms,
        chassis=chassis,
        drive=drive,
        steer=steer,
        friction=model.geom_friction[wheel_geoms, 0].copy(),
        mass=float(model.body_mass[chassis]),
        inertia=model.body_inertia[chassis].copy(),
        gear=model.actuator_gear[drive, 0].copy(),
        servo_gain=float(model.actuator_gainprm[steer, 0]),
        servo_bias=float(model.actuator_biasprm[steer, 1]),
        native=info.native,
        motor_r=float(model.actuator_gainprm[drive[0], 0]),
        servo_kp=float(model.actuator_gainprm[steer, 4]),
    )


def apply_physics(model: mujoco.MjModel, parts: CarParts, physics: Physics, drive=None) -> None:
    """Write the nominal values of parts times the factors of physics into
    model. drive: the car's DriveModel (realistic drive), or None (legacy
    torque motors). Call set_const afterwards, once for all cars."""
    model.geom_friction[parts.wheel_geoms, 0] = parts.friction * physics.wheel_friction
    model.body_mass[parts.chassis] = parts.mass * physics.chassis_mass
    model.body_inertia[parts.chassis] = parts.inertia * physics.chassis_mass
    if parts.native:
        # Exactly upstream's apply_physics: the DC motor's torque is
        # K (V - K w) / R with gainprm[0] = R, so dividing R scales the torque
        # at every speed and keeps the free speed V / K; the servo's loop
        # applies kp (target - angle) volts with gainprm[4] = kp.
        model.actuator_gainprm[parts.drive[0], 0] = parts.motor_r / physics.motor_torque
        model.actuator_gainprm[parts.steer, 4] = parts.servo_kp * physics.steer_stiffness
        return
    if drive is not None:  # torque and back-EMF both scale: the free speed stays
        drive.torque_scale = physics.motor_torque
        model.actuator_gear[parts.drive, 0] = parts.gear
    else:
        model.actuator_gear[parts.drive, 0] = parts.gear * physics.motor_torque
    # A position servo is gain*ctrl + bias*qpos with bias = -kp: scale both.
    model.actuator_gainprm[parts.steer, 0] = parts.servo_gain * physics.steer_stiffness
    model.actuator_biasprm[parts.steer, 1] = parts.servo_bias * physics.steer_stiffness


def set_const(model: mujoco.MjModel, scratch: mujoco.MjData) -> None:
    """mj_setConst after mass changes (as upstream), on a scratch MjData of
    model -- never the env's live data, which mid-race holds other cars."""
    mujoco.mj_setConst(model, scratch)
