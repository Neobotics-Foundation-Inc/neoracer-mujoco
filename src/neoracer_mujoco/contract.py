"""
The NeoRacer car contract: what every NeoRacer-family model must satisfy.

A new car = a new XML dropped in assets/. It is picked up automatically by
assets.cars(), and validation/ runs the full conformance battery against it
with no test edits — but only if it honors the promises below.

This is the single source of truth for the interface. It lives in a real
module (not a test helper) precisely because production code, examples, and
tests all depend on it.

Control layout contract (ctrl[0:4] + ctrl[4]):
  ctrl[0..3] = fl/fr/rl/rr wheel drive torque (N·m), positive = forward
  ctrl[4]    = steer_servo target angle (rad), positive = left
"""

# --- actuators + ctrl layout ------------------------------------------------
EXPECTED_ACTUATORS = ["fl_motor", "fr_motor", "rl_motor", "rr_motor", "steer_servo"]
DRIVE = slice(0, 4)  # ctrl indices for the four wheel motors
STEER = 4  # ctrl index for the steering servo

# --- sensors ----------------------------------------------------------------
REQUIRED_SENSORS = [
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
]

# --- physical sanity --------------------------------------------------------
# Total car mass sanity band. Real NeoRacer is ~2.08 kg (CAD/STEP, see aeh961
# neoracer_mjcf/README.md) / 2.6 kg estimated. Band is checked on TOTAL mass, not
# per-geom: a mesh is allowed to carry mass (dense battery/motor), it just has to
# add up to something physical. The upper bound still trips the classic "default
# 1000 kg/m³ density" bug — e.g. the chassis bbox as solid water = 19.2 kg.
# ponytail: widen per car family if a new vehicle is genuinely heavier; for an
# exact check, assert against that car's documented spec mass instead of a band.
MASS_MIN, MASS_MAX = 1.0, 8.0  # kg
