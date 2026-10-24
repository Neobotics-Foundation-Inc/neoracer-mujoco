"""
The NeoRacer car contract: what contract.py requires of every NeoRacer car model.

A new car is a new XML file in assets/. assets.cars() finds it, and the tests
in validation/ run the conformance tests on it with no test edits. The tests
pass only if the XML satisfies the requirements below.

This module is the one source of truth for the interface. Code, examples and
tests all read it. The sensors that a car must have are the fields of
sensors.SensorReadings.

Frame: +X forward, +Y left, +Z up.
Actuators:
  motor       = the drive, + = forward: the voltage at the NeoRacer's motor
                (V); the speed that the RACECAR Neo V2's firmware holds (m/s)
  steer_servo = steering target angle (rad), + = left
A car may have more actuators, which the throttle drives with the motor, as
motor_engage on the RACECAR Neo V2. throttle_map gives each one's ctrl.
Wheels:
  fl/fr/rl/rr_wheel = the geom of each wheel that touches the ground, a
                      cylinder of the car's wheel radius (wheel_radius)
Their order in the XML is not part of the contract: code binds each one by
its element in the scene's MjSpec (scene.Car).
"""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import mujoco

if TYPE_CHECKING:  # scene.py imports this module
    from .scene import Car

CAR_BODY = "car"
"""The car's root body, with the free joint."""

DRIVE_ACTUATOR = "motor"
STEER_ACTUATOR = "steer_servo"
REQUIRED_ACTUATORS = (DRIVE_ACTUATOR, STEER_ACTUATOR)
THROTTLE_MAP_SIZE = 3
"""The count of an actuator's user numbers (a, b, c), if it has any. They give
its ctrl for a throttle t in [-1, 1]: a t + b sign(t) + c |sign(t)|. Every
actuator but the steer servo has them, except a motor that takes a share of
its limit."""

WHEEL_GEOMS = ("fl_wheel", "fr_wheel", "rl_wheel", "rr_wheel")
"""The geoms that touch the ground, in the order fl, fr, rl, rr."""

LIDAR_SITE = "lidar_mount"
"""The site at the center of the spinning LiDAR (lidar.py). Its +X axis is the
scan's bearing 0, and its XY plane is the plane the scan sweeps. Its user
number is the field that the LiDAR sees (lidar_field)."""
IMU_ACCEL_SENSOR = "imu_accel"
"""The specific force at the IMU (m/s²), in the car frame."""
IMU_GYRO_SENSOR = "imu_gyro"
"""The angular velocity at the IMU (rad/s), in the car frame."""
DRIVE_POS_SENSOR = "drive_pos"
"""The drivetrain's angle (rad), which the encoder counts."""

MASS_MIN, MASS_MAX = 1.0, 8.0
"""The band (kg) that a car's total mass must be in. The real NeoRacer's mass
is its spec's Chassis.mass. The test checks the total mass, not the mass of
each geom, so a mesh can carry mass (a dense battery or motor). If a mesh uses
MuJoCo's default density of 1000 kg/m³, the total mass is above MASS_MAX: the
chassis box as solid water is 21.3 kg.

Limit: if a new car is really heavier, widen this band for its car family. For
an exact check, compare with the documented mass of that car instead."""


type ThrottleMap = tuple[float, float, float]


class Array(Protocol):
    """A numpy or a JAX array, or a numpy scalar: what throttle_ctrl and the
    batch's command and read need of one."""

    def __array_namespace__(self, *, api_version: None = None) -> Any: ...

    def __getitem__(self, key: Any, /) -> Any: ...

    def __mul__(self, other: float, /) -> Any: ...


def throttle_ctrl[A: Array](tmap: ThrottleMap, throttle: A) -> A:
    """The ctrl that throttle, in [-1, 1], gives an actuator of throttle map
    tmap (throttle_map): a t + b sign(t) + c |sign(t)|."""
    a, b, c = tmap
    xp = throttle.__array_namespace__()  # numpy or jax.numpy
    sign = xp.sign(throttle)
    return throttle * a + sign * b + xp.abs(sign) * c


def throttle_map(model: mujoco.MjModel, actuator: mujoco.MjsActuator) -> ThrottleMap:
    """(a, b, c): a throttle t in [-1, 1] sets actuator's ctrl to
    a t + b sign(t) + c |sign(t)|. They are its user numbers: the RACECAR Neo
    V2's motor holds a speed of at least b while the throttle is not 0, and
    its motor_engage is 1. A motor without them is (its ctrlrange limit, 0,
    0): the NeoRacer's takes the throttle's share of the battery's voltage."""
    user = actuator.userdata
    if len(user) == 0:
        return (float(model.bind_scalar(actuator).ctrlrange[1]), 0.0, 0.0)
    a, b, c = (float(u) for u in user)
    return (a, b, c)


@dataclass(frozen=True)
class ActuatorLimits:
    """The ctrlrange limits of the drive and steer actuators. Each range is
    symmetric, so a command in [-1, 1] scales by one limit."""

    voltage: float
    """The limit of the motor's ctrlrange, [-voltage, voltage]: the NeoRacer's
    battery voltage (V), or for the RACECAR Neo V2 the speed that it holds at
    full throttle (m/s)."""
    steer_rad: float
    """The limit of the steer servo's ctrlrange, [-steer_rad, steer_rad]."""


def wheel_radius(model: mujoco.MjModel, car: Car) -> float:
    """The radius (m) of the car's wheels, the size of its wheel geoms. The
    wheels' turn rate (rad/s) times it is the car's speed (m/s) while they
    do not slip."""
    return float(model.bind_scalar(car.wheels[0]).size[0])


def lidar_field(model: mujoco.MjModel, lidar: mujoco.MjsSite) -> float:
    """The bearings (rad) that the LiDAR at the site lidar sees, centered on
    its front: the car blocks the rest. 2 pi if it sees all around."""
    return float(model.bind_scalar(lidar).user[0])


FIELD_EPS = 1e-5
"""A field at the model's precision, 6 significant digits, is off by less than
this (rad), so a bearing that close outside its edge is on it."""


def in_field(bearing: float, field: float) -> bool:
    """Whether a LiDAR that sees field (rad), centered on its front, sees
    bearing (rad in [0, 2 pi), from the front either way). The field is
    closed and symmetric: a bearing on either edge is in it."""
    return min(bearing, 2 * math.pi - bearing) <= field / 2 + FIELD_EPS


@dataclass(frozen=True)
class Drivetrain:
    """The drivetrain's angle (DRIVE_POS_SENSOR) as a sum over qpos: its
    fixed tendon, coef[k] times qpos[qpos[k]] for each k."""

    qpos: tuple[int, ...]
    coef: tuple[float, ...]


def drivetrain(model: mujoco.MjModel, car: Car) -> Drivetrain:
    """The tendon of the car's drive_pos sensor in model. Raise ValueError if
    it is not a fixed tendon."""
    sensor = model.bind_scalar(car.sensors.drive_pos)
    refused = ValueError("the drive_pos sensor does not read a fixed tendon")
    if sensor.objtype[0] != mujoco.mjtObj.mjOBJ_TENDON:
        raise refused
    tendon = int(sensor.objid[0])
    adr, num = int(model.tendon_adr[tendon]), int(model.tendon_num[tendon])
    wraps = range(adr, adr + num)
    if any(model.wrap_type[w] != mujoco.mjtWrap.mjWRAP_JOINT for w in wraps):
        raise refused
    return Drivetrain(
        tuple(int(model.jnt_qposadr[model.wrap_objid[w]]) for w in wraps),
        tuple(float(model.wrap_prm[w]) for w in wraps),
    )


def drive_angle[A: Array](drivetrain: Drivetrain, qpos: A) -> A:
    """The drivetrain's angle (rad) in each row of qpos (..., nq), summed as
    MuJoCo sums the tendon, so it is drive_pos's reading at that qpos. After
    mj_step the sensors still hold the state at the step's start; this reads
    it at the step's end."""
    angle: Any = 0.0
    for adr, coef in zip(drivetrain.qpos, drivetrain.coef, strict=True):
        angle = angle + coef * qpos[..., adr]
    return angle


def actuator_limits(model: mujoco.MjModel, car: Car) -> ActuatorLimits:
    """Read the actuator limits of the car in model, once per model. Raise
    ValueError if a range is not symmetric about 0: the env and MujocoDrive
    negate the limit for a negative command."""
    low, high = model.bind_scalar(car.drive).ctrlrange
    steer_low, steer_high = model.bind_scalar(car.steer).ctrlrange
    problems = []
    if not (high > 0 and low == -high):
        problems.append(f"the drive ctrlrange is not [-a, a] with a > 0: {(low, high)}")
    if not (steer_high > 0 and steer_low == -steer_high):
        problems.append(
            f"the steer ctrlrange is not [-a, a] with a > 0: {(steer_low, steer_high)}"
        )
    if problems:
        raise ValueError("; ".join(problems))
    return ActuatorLimits(voltage=float(high), steer_rad=float(steer_high))
