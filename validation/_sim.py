"""
Shared helpers + the car contract that every NeoRacer-family model must satisfy.

A new car = a new XML dropped in assets/. It is picked up automatically by the
glob below, so test_conformance.py runs the full battery against it with no edits.

Control layout contract (ctrl[0:4] + ctrl[4]):
  ctrl[0..3] = fl/fr/rl/rr wheel drive torque (N·m), positive = forward
  ctrl[4]    = steer_servo target angle (rad), positive = left
"""

import glob
import os

import mujoco

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSET_DIR = os.path.join(REPO, "assets")
CARS = sorted(glob.glob(os.path.join(ASSET_DIR, "*.xml")))

# --- the contract -----------------------------------------------------------
EXPECTED_ACTUATORS = ["fl_motor", "fr_motor", "rl_motor", "rr_motor", "steer_servo"]
DRIVE = slice(0, 4)   # ctrl indices for the four wheel motors
STEER = 4             # ctrl index for the steering servo

REQUIRED_SENSORS = [
    "imu_accel", "imu_gyro", "imu_quat", "imu_linvel",
    "steer_cmd_pos", "fl_steer_pos", "fr_steer_pos",
    "fl_wheel_vel", "fr_wheel_vel", "rl_wheel_vel", "rr_wheel_vel",
    "fl_susp_pos", "fr_susp_pos", "rl_susp_pos", "rr_susp_pos",
]

# Total car mass sanity band. Real NeoRacer is ~2.08 kg (CAD/STEP, see aeh961
# neoracer_mjcf/README.md) / 2.6 kg estimated. Band is checked on TOTAL mass, not
# per-geom: a mesh is allowed to carry mass (dense battery/motor), it just has to
# add up to something physical. The upper bound still trips the classic "default
# 1000 kg/m³ density" bug — e.g. the chassis bbox as solid water = 19.2 kg.
# ponytail: widen per car family if a new vehicle is genuinely heavier; for an
# exact check, assert against that car's documented spec mass instead of a band.
MASS_MIN, MASS_MAX = 1.0, 8.0   # kg

SETTLE_STEPS = 400              # zero-ctrl steps to reach static equilibrium


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
    """Flat {name: ndarray} of all sensor readings (mirrors sensor_logger.read)."""
    out = {}
    for i in range(model.nsensor):
        adr, dim = model.sensor_adr[i], model.sensor_dim[i]
        out[model.sensor(i).name] = data.sensordata[adr:adr + dim].copy()
    return out


def car_mass(model):
    return float(model.body_subtreemass[model.body("car").id])


def car_upright_cos(data):
    """cos(tilt): body z-axis · world z-axis. 1.0 = perfectly level, <0 = flipped."""
    return float(data.body("car").xmat[8])
