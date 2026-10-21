"""
The NeoRacer car contract: what contract.py requires of every NeoRacer car model.

A new car is a new XML file in assets/. assets.cars() finds it, and the tests
in validation/ run the conformance tests on it with no test edits. The tests
pass only if the XML satisfies the requirements below.

This module is the one source of truth for the interface. Code, examples and
tests all read it.

Frame: +X forward, +Y left, +Z up.
Actuators:
  motor       = the drive, + = forward: the voltage at the NeoRacer's motor
                (V); the speed that the RACECAR Neo V2's firmware holds (m/s)
  steer_servo = steering target angle (rad), + = left
A car may have more actuators, which the throttle drives with the motor, as
motor_engage on the RACECAR Neo V2. throttle_map gives each one's ctrl.
Wheels:
  fl/fr/rl/rr_wheel = the geom of each wheel that touches the ground
Their order in the XML is not part of the contract: code binds each one by
its element in the scene's MjSpec (scene.Car).
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco

if TYPE_CHECKING:  # scene.py imports this module
    from .scene import Car

# --- actuators ---------------------------------------------------------------
DRIVE_ACTUATOR = "motor"
STEER_ACTUATOR = "steer_servo"
REQUIRED_ACTUATORS = (DRIVE_ACTUATOR, STEER_ACTUATOR)
# An actuator's user numbers (a, b, c), if it has any, give its ctrl for a
# throttle t in [-1, 1]: a t + b sign(t) + c |sign(t)|. Every actuator but
# the steer servo has them, except a motor that takes a share of its limit.
THROTTLE_MAP_SIZE = 3

# --- wheels -----------------------------------------------------------------
# The geoms that touch the ground, in the order fl, fr, rl, rr.
WHEEL_GEOMS = ("fl_wheel", "fr_wheel", "rl_wheel", "rr_wheel")

# --- sensors ----------------------------------------------------------------
# The site at the centre of the spinning LiDAR (lidar.py). Its +X axis is the
# scan's bearing 0, and its XY plane is the plane the scan sweeps.
LIDAR_SITE = "lidar_mount"
REQUIRED_SENSORS = (
    "imu_accel",
    "imu_gyro",
    "imu_quat",
    "imu_linvel",
    "steer_cmd_pos",
    "fl_steer_pos",
    "fr_steer_pos",
    "fl_wheel_vel",
    "fr_wheel_vel",
    "rl_wheel_vel",
    "rr_wheel_vel",
    "drive_pos",
    "fl_susp_pos",
    "fr_susp_pos",
    "rl_susp_pos",
    "rr_susp_pos",
)

# --- physical sanity --------------------------------------------------------
# The real NeoRacer weighs 3.0 kg (spec, 2026-09-29). The test checks the total
# mass, not the mass of each geom, so a mesh can carry mass (a dense battery
# or motor). If a mesh uses MuJoCo's default density of 1000 kg/m³, the total
# mass is above MASS_MAX: the chassis box as solid water is 21.3 kg.
# Limit: if a new car is really heavier, widen this band for its car family.
# For an exact check, compare with the documented mass of that car instead.
MASS_MIN, MASS_MAX = 1.0, 8.0  # kg


type ThrottleMap = tuple[float, float, float]


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

    # The motor's ctrlrange is [-voltage, voltage]: V, the NeoRacer's battery;
    # m/s for the RACECAR Neo V2, the speed it holds at full throttle.
    voltage: float
    steer_rad: float  # rad; ctrlrange is [-steer_rad, steer_rad]


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
