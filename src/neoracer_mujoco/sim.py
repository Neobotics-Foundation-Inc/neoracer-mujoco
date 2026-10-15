"""
Step and probe helpers for the NeoRacer model: step the physics with a fixed
control, write a control command, and two physics probes (car_mass,
car_upright_cos) that the conformance tests use.

Frame: +X forward, +Y left, +Z up.
"""

import mujoco
import numpy as np

from .control.steering import ControlCommand
from .scene import Car

SETTLE_STEPS = 400  # zero-ctrl steps to reach static equilibrium


def apply_ctrl_and_step(
    model: mujoco.MjModel, data: mujoco.MjData, ctrl: float | np.ndarray, steps: int
) -> None:
    """Set data.ctrl to ctrl and step data forward `steps` times."""
    data.ctrl[:] = ctrl
    for _ in range(steps):
        mujoco.mj_step(model, data)


def apply_command(car: Car, data: mujoco.MjData, command: ControlCommand) -> None:
    """Write command to data.ctrl: the throttle (in [-1, 1]) to the drive
    motor and the steer (rad, + = left) to the steer servo."""
    data.bind_scalar(car.drive).ctrl = command.throttle
    data.bind_scalar(car.steer).ctrl = command.steer


def settle(model: mujoco.MjModel) -> mujoco.MjData:
    """New MjData, stepped SETTLE_STEPS with zero control so the car rests
    on the ground."""
    data = mujoco.MjData(model)
    apply_ctrl_and_step(model, data, 0.0, SETTLE_STEPS)
    return data


def car_mass(model: mujoco.MjModel, car: Car) -> float:
    """Total mass (kg) of the car body and its children."""
    return float(model.bind_scalar(car.body).subtreemass[0])


def car_upright_cos(data: mujoco.MjData, car: Car) -> float:
    """The cosine of the angle between the car body's +Z axis and the world
    +Z axis. It is 1.0 when the car is level and below 0 when it is upside
    down. It reads the truth sensor car.truth.zaxis, so it is up to date after
    mj_forward or mj_step (see truth.py)."""
    return float(data.bind_scalar(car.truth.zaxis).data[2])
