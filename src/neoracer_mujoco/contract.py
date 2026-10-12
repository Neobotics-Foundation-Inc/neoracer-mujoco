"""
The NeoRacer car contract: what contract.py requires of every NeoRacer car model.

A new car is a new XML file in assets/. assets.cars() finds it, and the tests
in validation/ run the conformance tests on it with no test edits. The tests
pass only if the XML satisfies the requirements below.

This module is the one source of truth for the interface. Code, examples and
tests all read it.

Frame: +X forward, +Y left, +Z up.
Control layout (ctrl[0:4] + ctrl[4]):
  ctrl[0..3] = fl/fr/rl/rr wheel drive torque (N·m), + = forward
  ctrl[4]    = steer_servo target angle (rad), + = left
"""

from dataclasses import dataclass

import mujoco

# --- actuators + ctrl layout ------------------------------------------------
EXPECTED_ACTUATORS = ("fl_motor", "fr_motor", "rl_motor", "rr_motor", "steer_servo")
DRIVE = slice(0, 4)  # ctrl indices for the four wheel motors
STEER = 4  # ctrl index for the steering servo

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
    """The upper ctrlrange limits of the drive and steer actuators."""

    torque_nm: float  # N·m per wheel; fl_motor ctrlrange
    steer_rad: float  # rad; steer_servo ctrlrange


def actuator_limits(model: mujoco.MjModel) -> ActuatorLimits:
    """Read the actuator limits from a compiled car model, once per model."""
    return ActuatorLimits(
        torque_nm=float(model.actuator("fl_motor").ctrlrange[1]),
        steer_rad=float(model.actuator("steer_servo").ctrlrange[1]),
    )
