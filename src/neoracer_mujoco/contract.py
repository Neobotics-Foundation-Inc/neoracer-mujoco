"""
The NeoRacer car contract: what contract.py requires of every NeoRacer car model.

A new car is a new XML file in assets/. assets.cars() finds it, and the tests
in validation/ run the conformance tests on it with no test edits. The tests
pass only if the XML satisfies the requirements below.

This module is the one source of truth for the interface. Code, examples and
tests all read it.

Frame: +X forward, +Y left, +Z up.
Actuators:
  fl/fr/rl/rr_motor = wheel drive torque (N·m), + = forward
  steer_servo       = steering target angle (rad), + = left
Their order in the XML is not part of the contract: code binds each one by
its element in the scene's MjSpec (scene.Car).
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco

if TYPE_CHECKING:  # scene.py imports this module
    from .scene import Car

# --- actuators ---------------------------------------------------------------
DRIVE_ACTUATORS = ("fl_motor", "fr_motor", "rl_motor", "rr_motor")
STEER_ACTUATOR = "steer_servo"
EXPECTED_ACTUATORS = (*DRIVE_ACTUATORS, STEER_ACTUATOR)

# --- sensors ----------------------------------------------------------------
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
    "fl_susp_pos",
    "fr_susp_pos",
    "rl_susp_pos",
    "rr_susp_pos",
)

# --- physical sanity --------------------------------------------------------
# The real NeoRacer has a mass of approximately 2.08 kg (from CAD, see aeh961
# neoracer_mjcf/README.md) or 2.6 kg (estimated). The test checks the total
# mass, not the mass of each geom, so a mesh can carry mass (a dense battery
# or motor). If a mesh uses MuJoCo's default density of 1000 kg/m³, the total
# mass is above MASS_MAX: the chassis box as solid water is 19.2 kg.
# Limit: if a new car is really heavier, widen this band for its car family.
# For an exact check, compare with the documented mass of that car instead.
MASS_MIN, MASS_MAX = 1.0, 8.0  # kg


@dataclass(frozen=True)
class ActuatorLimits:
    """The ctrlrange limits of the drive and steer actuators. Each range is
    symmetric, so a command in [-1, 1] scales by one limit."""

    torque_nm: float  # N·m per wheel; ctrlrange is [-torque_nm, torque_nm]
    steer_rad: float  # rad; ctrlrange is [-steer_rad, steer_rad]


def actuator_limits(model: mujoco.MjModel, car: Car) -> ActuatorLimits:
    """Read the actuator limits of the car in model, once per model. Raise
    ValueError if the four drive ranges differ, or a range is not symmetric
    about 0: the env and MujocoDrive send one torque to all four wheels and
    negate the limit for a negative command."""
    drive = [tuple(float(x) for x in model.bind_scalar(a).ctrlrange) for a in car.drive]
    low, high = drive[0]
    steer_low, steer_high = model.bind_scalar(car.steer).ctrlrange
    problems = []
    if any(r != drive[0] for r in drive):
        problems.append(f"the drive ctrlranges differ: {drive}")
    if not (high > 0 and low == -high):
        problems.append(f"the drive ctrlrange is not [-a, a] with a > 0: {drive[0]}")
    if not (steer_high > 0 and steer_low == -steer_high):
        problems.append(
            f"the steer ctrlrange is not [-a, a] with a > 0: {(steer_low, steer_high)}"
        )
    if problems:
        raise ValueError("; ".join(problems))
    return ActuatorLimits(torque_nm=float(high), steer_rad=float(steer_high))
