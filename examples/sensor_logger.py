"""
Reads all named sensors from a compiled NeoRacer model into a SensorReadings struct.

Sensor map (matches neoracer.xml):
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

from dataclasses import dataclass, field

import numpy as np
import mujoco

# Shadows the wheel geom size in assets/neoracer.xml — keep in sync if the XML changes.
WHEEL_RADIUS = 0.050  # ESTIMATED: 50 mm radius consistent with STL bbox and URDF CoM


def _empty() -> np.ndarray:
    return np.array([])


@dataclass(frozen=True)
class SensorReadings:
    """
    One named field per sensor in neoracer.xml. Each value is the raw sensordata
    slice (a numpy array), so scalars come back as length-1 arrays — read [0].
    Defaults let tests build partial instances; read() always fills them all.
    """

    # IMU at the chassis centre
    imu_accel: np.ndarray = field(default_factory=_empty)  # linear accel (m/s²)
    imu_gyro: np.ndarray = field(default_factory=_empty)  # angular vel (rad/s)
    imu_quat: np.ndarray = field(default_factory=_empty)  # orientation (w,x,y,z)
    imu_linvel: np.ndarray = field(default_factory=_empty)  # world linear vel (m/s)
    # steering angles (rad)
    steer_cmd_pos: np.ndarray = field(default_factory=_empty)  # virtual command
    fl_steer_pos: np.ndarray = field(default_factory=_empty)  # actual front-left
    fr_steer_pos: np.ndarray = field(default_factory=_empty)  # actual front-right
    # wheel angular velocity (rad/s)
    fl_wheel_vel: np.ndarray = field(default_factory=_empty)
    fr_wheel_vel: np.ndarray = field(default_factory=_empty)
    rl_wheel_vel: np.ndarray = field(default_factory=_empty)
    rr_wheel_vel: np.ndarray = field(default_factory=_empty)
    # suspension travel (m; + = compressed)
    fl_susp_pos: np.ndarray = field(default_factory=_empty)
    fr_susp_pos: np.ndarray = field(default_factory=_empty)
    rl_susp_pos: np.ndarray = field(default_factory=_empty)
    rr_susp_pos: np.ndarray = field(default_factory=_empty)
    # 8-beam lidar ring (m; -1 = no hit)
    lidar_000: np.ndarray = field(default_factory=_empty)
    lidar_045: np.ndarray = field(default_factory=_empty)
    lidar_090: np.ndarray = field(default_factory=_empty)
    lidar_135: np.ndarray = field(default_factory=_empty)
    lidar_180: np.ndarray = field(default_factory=_empty)
    lidar_225: np.ndarray = field(default_factory=_empty)
    lidar_270: np.ndarray = field(default_factory=_empty)
    lidar_315: np.ndarray = field(default_factory=_empty)


def read(model: mujoco.MjModel, data: mujoco.MjData) -> SensorReadings:
    """Read every named sensor into a SensorReadings struct."""
    raw = {}
    for i in range(model.nsensor):
        name = model.sensor(i).name
        adr = model.sensor_adr[i]
        dim = model.sensor_dim[i]
        raw[name] = data.sensordata[adr : adr + dim].copy()
    # ponytail: fields are coupled to neoracer.xml's sensor set — a car XML with
    # a different sensor list fails here loudly. Add fields if the contract grows.
    return SensorReadings(**raw)


def wheel_speed_ms(sensors: SensorReadings) -> dict:
    """Convert each wheel's angular velocity (rad/s) to surface speed (m/s)."""
    return {
        "fl_wheel_speed_ms": float(sensors.fl_wheel_vel[0]) * WHEEL_RADIUS,
        "fr_wheel_speed_ms": float(sensors.fr_wheel_vel[0]) * WHEEL_RADIUS,
        "rl_wheel_speed_ms": float(sensors.rl_wheel_vel[0]) * WHEEL_RADIUS,
        "rr_wheel_speed_ms": float(sensors.rr_wheel_vel[0]) * WHEEL_RADIUS,
    }


def print_sensors(sensors: SensorReadings) -> None:
    """Print a one-line summary of the most useful sensor values."""

    def f(name, idx=0):
        v = getattr(sensors, name)
        return float(v[idx]) if len(v) > idx else float("nan")

    speeds = wheel_speed_ms(sensors)
    avg_speed = sum(speeds.values()) / max(len(speeds), 1)

    steer_cmd = f("steer_cmd_pos")
    fl_steer = f("fl_steer_pos")
    fr_steer = f("fr_steer_pos")
    fl_susp = f("fl_susp_pos")
    fr_susp = f("fr_susp_pos")
    rl_susp = f("rl_susp_pos")
    rr_susp = f("rr_susp_pos")

    accel_z = f("imu_accel", 2)
    gyro_z = f("imu_gyro", 2)

    lidar_fwd = f("lidar_000")
    lidar_l = f("lidar_090")
    lidar_r = f("lidar_270")

    print(
        f"spd={avg_speed:+5.2f}m/s  "
        f"steer={steer_cmd:+.3f}({fl_steer:+.3f}/{fr_steer:+.3f})  "
        f"susp=[{fl_susp:+.3f} {fr_susp:+.3f} {rl_susp:+.3f} {rr_susp:+.3f}]m  "
        f"az={accel_z:+5.1f}  gz={gyro_z:+.3f}  "
        f"lidar[fwd={lidar_fwd:.2f} L={lidar_l:.2f} R={lidar_r:.2f}]"
    )
