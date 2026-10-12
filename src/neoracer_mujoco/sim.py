"""
Load -> step -> read helpers for the NeoRacer model.

Thin wrappers over mujoco's step loop, plus two physics probes (car_mass,
car_upright_cos) used by the conformance gate. This is the seed of a future
higher-level Sim object; keep it small until that need is real.
"""

import mujoco
import numpy as np

SETTLE_STEPS = 400  # zero-ctrl steps to reach static equilibrium


def run(
    model: mujoco.MjModel, data: mujoco.MjData, ctrl: float | np.ndarray, steps: int
) -> None:
    """Set data.ctrl to ctrl and step data forward `steps` times."""
    data.ctrl[:] = ctrl
    for _ in range(steps):
        mujoco.mj_step(model, data)


def settle(model: mujoco.MjModel) -> mujoco.MjData:
    """New MjData, stepped SETTLE_STEPS with zero control so the car rests
    on the ground."""
    data = mujoco.MjData(model)
    run(model, data, 0.0, SETTLE_STEPS)
    return data


def car_mass(model: mujoco.MjModel) -> float:
    return float(model.body_subtreemass[model.body("car").id])


def car_upright_cos(data: mujoco.MjData) -> float:
    """cos(tilt): body z-axis · world z-axis. 1.0 = perfectly level, <0 = flipped."""
    return float(data.body("car").xmat[8])
