"""
Load -> step -> read helpers for the NeoRacer model.

Thin wrappers over mujoco's step loop, plus two physics probes (car_mass,
car_upright_cos) used by the conformance gate. This is the seed of a future
higher-level Sim object; keep it small until that need is real.
"""

import mujoco

from . import sensors as _sensors

SETTLE_STEPS = 400  # zero-ctrl steps to reach static equilibrium


def compile(path):
    return mujoco.MjModel.from_xml_path(path)


def settle(model, data=None, steps=SETTLE_STEPS):
    """Run `steps` with zero control so the car rests on the ground."""
    if data is None:
        data = mujoco.MjData(model)
    data.ctrl[:] = 0
    for _ in range(steps):
        mujoco.mj_step(model, data)
    return data


def run(model, data, ctrl, steps):
    data.ctrl[:] = ctrl
    for _ in range(steps):
        mujoco.mj_step(model, data)
    return data


def sensors(model, data):
    """Flat {name: ndarray} of all sensor readings. For the typed struct form,
    use neoracer_mujoco.read()."""
    return _sensors.read_raw(model, data)


def car_mass(model):
    return float(model.body_subtreemass[model.body("car").id])


def car_upright_cos(data):
    """cos(tilt): body z-axis · world z-axis. 1.0 = perfectly level, <0 = flipped."""
    return float(data.body("car").xmat[8])
