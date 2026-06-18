"""
Run neoracer.xml in the MuJoCo viewer.

Usage:
    mjpython users/aeh961/neoracer_mjcf/scripts/run.py

Controls (demo loop — constant throttle + sinusoidal steering):
  - All four wheels driven at constant torque.
  - Steering oscillates sinusoidally to verify Ackermann behaviour visually:
    the inner front wheel should always turn more than the outer.
  - Sensor readings are printed to stdout at PRINT_HZ rate.

Actuator mapping (from models/neoracer.xml):
  ctrl[0] = fl_motor  — front-left  wheel torque (N·m)
  ctrl[1] = fr_motor  — front-right wheel torque (N·m)
  ctrl[2] = rl_motor  — rear-left   wheel torque (N·m)
  ctrl[3] = rr_motor  — rear-right  wheel torque (N·m)
  ctrl[4] = steer_servo — virtual steering wheel angle (rad)
              Ackermann equality constraints in the XML automatically split this
              into the correct fl_steer / fr_steer angles — no Python maths needed.
"""

import math
import sys
import time
from pathlib import Path

# Resolve paths relative to this file so the script works from any cwd.
_SCRIPTS_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPTS_DIR.parent
sys.path.insert(0, str(_SCRIPTS_DIR))   # makes sensor_logger importable

import mujoco
import mujoco.viewer

import sensor_logger as sl

# ── physics tunables ──────────────────────────────────────────────────────────
THROTTLE   = 0.35   # N·m applied to all four wheels
STEER_AMP  = 0.35   # rad, peak virtual steering angle for the demo sine sweep
STEER_FREQ = 0.4    # Hz, steering oscillation frequency
PRINT_HZ   = 10     # sensor print rate (every N physics steps)

_XML = _PROJECT_DIR / "models" / "neoracer.xml"


def _build():
    model = mujoco.MjModel.from_xml_path(str(_XML))
    data  = mujoco.MjData(model)
    return model, data


def _control(data: mujoco.MjData, t: float) -> None:
    """Demo controller: constant throttle + sinusoidal steering."""
    steer = STEER_AMP * math.sin(2 * math.pi * STEER_FREQ * t)

    data.ctrl[0] = THROTTLE   # fl
    data.ctrl[1] = THROTTLE   # fr
    data.ctrl[2] = THROTTLE   # rl
    data.ctrl[3] = THROTTLE   # rr
    data.ctrl[4] = steer


def main() -> None:
    model, data = _build()
    car_id = model.body("car").id

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type        = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = car_id
        viewer.cam.distance    = 1.2
        viewer.cam.azimuth     = 90
        viewer.cam.elevation   = -30

        step = 0
        while viewer.is_running():
            _control(data, data.time)
            mujoco.mj_step(model, data)
            viewer.sync()

            step += 1
            if step % PRINT_HZ == 0:
                sl.print_sensors(sl.read(model, data))

            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
