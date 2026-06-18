"""
Run neoracer_car.xml in the MuJoCo viewer.

Usage:
    mjpython users/aeh961/neoracer_car/run.py

Controls (via this script's demo loop — for interactive keyboard control, see MuJoCo's
viewer key bindings or swap the control() call for your own controller):
  - Drives forward at constant throttle.
  - Steering oscillates sinusoidally so you can verify Ackermann behaviour visually:
    the inner front wheel should always turn more than the outer.
  - Sensor readings are printed to stdout each step.

Actuator mapping (from neoracer_car.xml):
  ctrl[0] = fl_motor  — front-left  wheel torque (N·m)
  ctrl[1] = fr_motor  — front-right wheel torque (N·m)
  ctrl[2] = rl_motor  — rear-left   wheel torque (N·m)
  ctrl[3] = rr_motor  — rear-right  wheel torque (N·m)
  ctrl[4] = steer_servo — virtual steering wheel angle (rad)
              The Ackermann equality constraints in the XML automatically split this
              into the correct fl_steer / fr_steer angles — no Python maths needed.
"""

import math
import sys
import time
from pathlib import Path

# Ensure sensor_logger is importable regardless of cwd or invocation style.
sys.path.insert(0, str(Path(__file__).parent))

import mujoco
import mujoco.viewer

import sensor_logger as sl

# ── physics tunables ──────────────────────────────────────────────────────────
THROTTLE    = 0.35   # N·m applied to all four wheels (ESTIMATED; >0.5 wheelies at this scale)
STEER_AMP   = 0.35   # rad, peak virtual steering angle for the demo sine sweep
STEER_FREQ  = 0.4    # Hz, how fast the steering oscillates
PRINT_HZ    = 10     # sensor print rate (every N viewer-sync steps)

_XML = Path(__file__).parent / "neoracer_car.xml"


def _build():
    model = mujoco.MjModel.from_xml_path(str(_XML))
    data  = mujoco.MjData(model)
    return model, data


def _control(data: mujoco.MjData, t: float) -> None:
    """Demo controller: constant throttle + sinusoidal steering."""
    steer = STEER_AMP * math.sin(2 * math.pi * STEER_FREQ * t)

    # AWD: same torque to all four wheels.
    data.ctrl[0] = THROTTLE   # fl
    data.ctrl[1] = THROTTLE   # fr
    data.ctrl[2] = THROTTLE   # rl
    data.ctrl[3] = THROTTLE   # rr

    # Single steering command; XML Ackermann constraints handle the inner/outer split.
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
            t = data.time
            _control(data, t)

            mujoco.mj_step(model, data)
            viewer.sync()

            step += 1
            if step % PRINT_HZ == 0:
                sensors = sl.read(model, data)
                sl.print_sensors(sensors)

            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
