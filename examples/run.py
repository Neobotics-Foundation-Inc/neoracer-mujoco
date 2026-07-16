"""
Run neoracer.xml in the MuJoCo viewer.

Usage:
    mjpython scripts/run.py

SAFE_MODE = True   → gentle inputs, speed-capped, rollover warnings printed
SAFE_MODE = False  → original aggressive inputs (will flip at high speed)

Actuator mapping (from models/neoracer.xml):
  ctrl[0..3] = fl/fr/rl/rr wheel torque (N·m), positive = forward
  ctrl[4]    = steer_servo target angle (rad), positive = left
               Ackermann equality constraints split this into correct fl/fr angles.
"""

import math
import sys
import time
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPTS_DIR.parent
sys.path.insert(0, str(_SCRIPTS_DIR))

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402

import sensor_logger as sl  # noqa: E402

# ── mode switch ───────────────────────────────────────────────────────────────
# Set to False to restore original aggressive inputs (expect rollover at speed).
SAFE_MODE = True

# ── SAFE_MODE parameters (gentle, speed-limited) ──────────────────────────────
_SAFE_THROTTLE    = 0.08   # N·m  — low enough that top speed stays ≈ 1.5 m/s
_SAFE_STEER_AMP   = 0.15   # rad  — modest sweep; inner wheel ≈ 0.16 rad
_SAFE_STEER_FREQ  = 0.30   # Hz
_SAFE_SPEED_LIMIT = 1.5    # m/s  — throttle cuts when avg speed exceeds this

# ── FULL parameters (original values — reproduce the flip) ────────────────────
_FULL_THROTTLE    = 0.35
_FULL_STEER_AMP   = 0.35
_FULL_STEER_FREQ  = 0.40
_FULL_SPEED_LIMIT = None   # unlimited → car accelerates forever → flips

# ── select active set ─────────────────────────────────────────────────────────
THROTTLE    = _SAFE_THROTTLE    if SAFE_MODE else _FULL_THROTTLE
STEER_AMP   = _SAFE_STEER_AMP   if SAFE_MODE else _FULL_STEER_AMP
STEER_FREQ  = _SAFE_STEER_FREQ  if SAFE_MODE else _FULL_STEER_FREQ
SPEED_LIMIT = _SAFE_SPEED_LIMIT if SAFE_MODE else _FULL_SPEED_LIMIT

PRINT_HZ = 20   # log every N physics steps

_XML = _PROJECT_DIR / "assets" / "neoracer.xml"

# Rollover warning threshold — print a warning line when |roll| exceeds this.
_ROLL_WARN_DEG = 25.0


def _quat_to_euler(q):
    """MuJoCo quaternion [w,x,y,z] → (roll, pitch, yaw) in degrees."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    roll  = math.degrees(math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y)))
    # clamp asin argument to [-1,1] to guard against floating-point drift
    sp    = max(-1.0, min(1.0, 2*(w*y - z*x)))
    pitch = math.degrees(math.asin(sp))
    yaw   = math.degrees(math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)))
    return roll, pitch, yaw


def _control(data: mujoco.MjData, t: float, avg_speed_ms: float) -> tuple[float, float]:
    """
    Apply throttle and sinusoidal steering.
    When SAFE_MODE speed limit is active, throttle is cut to zero above the limit.
    Returns (throttle_applied, steer_applied).
    """
    steer    = STEER_AMP * math.sin(2 * math.pi * STEER_FREQ * t)
    throttle = THROTTLE

    if SPEED_LIMIT is not None and avg_speed_ms >= SPEED_LIMIT:
        throttle = 0.0   # coast — no braking, just cut drive torque

    data.ctrl[0] = throttle
    data.ctrl[1] = throttle
    data.ctrl[2] = throttle
    data.ctrl[3] = throttle
    data.ctrl[4] = steer
    return throttle, steer


def _log(data: mujoco.MjData, model: mujoco.MjModel,
         car_id: int, throttle: float, steer: float,
         sensors: "sl.SensorReadings", avg_spd: float) -> None:
    """Print one diagnostic line per PRINT_HZ steps."""
    car_z    = data.xpos[car_id][2]
    q        = sensors.imu_quat  # the orientation the RL agent sees
    roll, pitch, yaw = _quat_to_euler(q)
    yaw_rate = math.degrees(sensors.imu_gyro[2])  # deg/s

    fl_s = sensors.fl_steer_pos[0]
    fr_s = sensors.fr_steer_pos[0]
    susp = [getattr(sensors, f"{w}_susp_pos")[0] for w in ("fl", "fr", "rl", "rr")]

    flag = ""
    if abs(roll) > _ROLL_WARN_DEG:
        flag = f"  *** ROLL WARNING: {roll:+.1f}° ***"

    print(
        f"t={data.time:6.2f}s "
        f"thr={throttle:+.2f} steer={steer:+.3f}(fl={fl_s:+.3f}/fr={fr_s:+.3f}) "
        f"spd={avg_spd:+5.2f}m/s "
        f"z={car_z:.3f}m "
        f"roll={roll:+5.1f}° pitch={pitch:+5.1f}° yaw_rate={yaw_rate:+6.1f}°/s "
        f"susp=[{susp[0]:+.3f} {susp[1]:+.3f} {susp[2]:+.3f} {susp[3]:+.3f}]"
        f"{flag}"
    )


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(_XML))
    data  = mujoco.MjData(model)
    car_id = model.body("car").id

    print(f"NeoRacer viewer — SAFE_MODE={SAFE_MODE}")
    print(f"  throttle={THROTTLE} N·m  steer_amp={STEER_AMP} rad  "
          f"speed_limit={SPEED_LIMIT} m/s")
    print()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type        = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = car_id
        viewer.cam.distance    = 1.2
        viewer.cam.azimuth     = 90
        viewer.cam.elevation   = -30

        step = 0
        avg_speed = 0.0

        while viewer.is_running():
            loop_start = time.time()

            sensors  = sl.read(model, data)
            speeds   = sl.wheel_speed_ms(sensors)
            avg_speed = sum(speeds.values()) / max(len(speeds), 1)

            throttle, steer = _control(data, data.time, avg_speed)
            mujoco.mj_step(model, data)
            viewer.sync()

            step += 1
            if step % PRINT_HZ == 0:
                _log(data, model, car_id, throttle, steer, sensors, avg_speed)

            # sleep only the time left in this timestep, so the sim tracks
            # real time instead of running slower by the compute cost.
            remaining = model.opt.timestep - (time.time() - loop_start)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()