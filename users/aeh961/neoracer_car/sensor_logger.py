"""
Reads all named sensors from a compiled NeoRacer model and returns a structured dict.

Sensor map (matches neoracer_car.xml):
  imu_accel     — 3-axis linear acceleration (m/s²)
  imu_gyro      — 3-axis angular velocity (rad/s)
  imu_quat      — orientation quaternion (w, x, y, z)
  imu_linvel    — 3D linear velocity in world frame (m/s)
  steer_cmd_pos — virtual steering input angle (rad)
  fl/fr_steer_pos — actual front wheel steer angles (rad)
  fl/fr/rl/rr_wheel_vel — wheel angular velocities (rad/s)
  fl/fr/rl/rr_susp_pos  — suspension travel (m; positive = compressed)
  lidar_000..315 — rangefinder distances (m; -1 = no hit)
"""

import numpy as np
import mujoco

WHEEL_RADIUS = 0.05  # EXTRACTED from neoracer_car.xml geom size


def read(model: mujoco.MjModel, data: mujoco.MjData) -> dict:
    """Return a flat dict of all sensor readings."""
    out = {}
    for i in range(model.nsensor):
        name = model.sensor(i).name
        adr = model.sensor_adr[i]
        dim = model.sensor_dim[i]
        out[name] = data.sensordata[adr : adr + dim].copy()
    return out


def wheel_speed_ms(sensors: dict) -> dict:
    """Convert wheel angular velocities (rad/s) to surface speed (m/s)."""
    return {
        name.replace("_vel", "_speed_ms"): float(sensors[name][0]) * WHEEL_RADIUS
        for name in ("fl_wheel_vel", "fr_wheel_vel", "rl_wheel_vel", "rr_wheel_vel")
        if name in sensors
    }


def print_sensors(sensors: dict) -> None:
    """Print a one-line summary of the most useful sensor values."""
    def f(key, idx=0):
        v = sensors.get(key)
        return float(v[idx]) if v is not None else float("nan")

    speeds = wheel_speed_ms(sensors)
    avg_speed = sum(speeds.values()) / max(len(speeds), 1)

    steer_cmd = f("steer_cmd_pos")
    fl_steer  = f("fl_steer_pos")
    fr_steer  = f("fr_steer_pos")
    fl_susp   = f("fl_susp_pos")
    fr_susp   = f("fr_susp_pos")
    rl_susp   = f("rl_susp_pos")
    rr_susp   = f("rr_susp_pos")

    accel_z = f("imu_accel", 2)
    gyro_z  = f("imu_gyro",  2)

    lidar_fwd = f("lidar_000")
    lidar_l   = f("lidar_090")
    lidar_r   = f("lidar_270")

    print(
        f"spd={avg_speed:+5.2f}m/s  "
        f"steer={steer_cmd:+.3f}({fl_steer:+.3f}/{fr_steer:+.3f})  "
        f"susp=[{fl_susp:+.3f} {fr_susp:+.3f} {rl_susp:+.3f} {rr_susp:+.3f}]m  "
        f"az={accel_z:+5.1f}  gz={gyro_z:+.3f}  "
        f"lidar[fwd={lidar_fwd:.2f} L={lidar_l:.2f} R={lidar_r:.2f}]"
    )
