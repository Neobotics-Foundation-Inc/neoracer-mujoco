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
from typing import Generic, TypeVar

import mujoco
import numpy as np

# Unconstrained so LidarScan can later be retargeted to another array type
# (e.g. a JAX array) without changing the dataclass itself.
ArrayT = TypeVar("ArrayT")

# Shadows the wheel geom size in assets/neoracer.xml — keep in sync if the XML changes.
WHEEL_RADIUS = 0.050  # ESTIMATED: 50 mm radius consistent with STL bbox and URDF CoM


def _empty() -> np.ndarray:
    return np.array([])


def read_raw(model: mujoco.MjModel, data: mujoco.MjData) -> dict:
    """Every named sensor as a flat {name: ndarray} dict. The shared read loop
    behind both read() (typed struct) and the dict form used by validation."""
    out = {}
    for i in range(model.nsensor):
        adr = model.sensor_adr[i]
        dim = model.sensor_dim[i]
        out[model.sensor(i).name] = data.sensordata[adr : adr + dim].copy()
    return out


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
    # ponytail: fields are coupled to neoracer.xml's sensor set — a car XML with
    # a different sensor list fails here loudly. Add fields if the contract grows.
    return SensorReadings(**read_raw(model, data))


@dataclass(frozen=True)
class IMUReading:
    """
    One IMU sample, packed from neoracer.xml's imu_accel/imu_gyro/imu_quat sensors.

    Frame: the imu site has no rotation relative to its parent body (pos-only,
    no euler/quat in the XML), so this IS the car body frame: +X forward,
    +Y left, +Z up (per the coordinate frame documented in neoracer.xml's
    header comment).

    acceleration     — m/s^2, local (body) frame. GRAVITY IS INCLUDED: MuJoCo's
                        accelerometer reports specific/proper force like a real
                        accelerometer, so at rest this is NOT (0,0,0) — measured
                        empirically at rest on this model as ~(0, 0, +9.81), norm
                        equal to the model's configured gravity magnitude.
    angular_velocity — rad/s, local (body) frame. Measured empirically as ~0
                        (norm ~1e-9) at rest on this model.
    orientation      — quaternion (w, x, y, z) of the IMU frame relative to the
                        WORLD frame. Not body-relative and not zeroed to any
                        hardware reference attitude.

    These are MuJoCo's native readings with no axis/sign correction applied —
    construct directly from SensorReadings (raw) and pass through
    calibrate_imu() (calibrated) below.
    """

    acceleration: np.ndarray
    angular_velocity: np.ndarray
    orientation: np.ndarray


# No hardware bench-calibration data (bias, scale-factor, axis misalignment)
# exists yet for the physical NeoRacer IMU. Note this is distinct from the IMU
# *mount position* offset in osracer.urdf's imu_joint, which is an XML-geometry
# concern tracked separately, not a reading-calibration one.
#
# TODO(hardware-calibration): once bench-characterized bias/misalignment values
# exist for the real IMU, apply them here. Until then this returns a copy of
# raw's values unchanged, so downstream code can be written against the
# calibrated-reading interface now instead of waiting on unmeasured correction
# values.
def calibrate_imu(raw: IMUReading) -> IMUReading:
    """Apply hardware IMU calibration to a raw IMUReading, returning a new
    IMUReading. Currently a value-preserving copy — see the TODO above this
    function."""
    return IMUReading(
        acceleration=raw.acceleration.copy(),
        angular_velocity=raw.angular_velocity.copy(),
        orientation=raw.orientation.copy(),
    )


def wheel_speed_ms(sensors: SensorReadings) -> dict:
    """Convert each wheel's angular velocity (rad/s) to surface speed (m/s)."""
    return {
        "fl_wheel_speed_ms": float(sensors.fl_wheel_vel[0]) * WHEEL_RADIUS,
        "fr_wheel_speed_ms": float(sensors.fr_wheel_vel[0]) * WHEEL_RADIUS,
        "rl_wheel_speed_ms": float(sensors.rl_wheel_vel[0]) * WHEEL_RADIUS,
        "rr_wheel_speed_ms": float(sensors.rr_wheel_vel[0]) * WHEEL_RADIUS,
    }


# Beam order matches the rangefinder sensors in neoracer.xml, ascending 0deg..315deg.
LIDAR_BEAM_ORDER = (
    "lidar_000",
    "lidar_045",
    "lidar_090",
    "lidar_135",
    "lidar_180",
    "lidar_225",
    "lidar_270",
    "lidar_315",
)
LIDAR_ANGLES_DEG = (0, 45, 90, 135, 180, 225, 270, 315)
ALL_BEAMS = LIDAR_BEAM_ORDER

# Named beam groupings shared by LiDAR-based controllers (track_centering.py,
# wall_following.py) so they agree on which beam is "front-left" etc. without
# each controller re-deriving it.
LEFT_BEAMS = ("lidar_045", "lidar_090", "lidar_135")
RIGHT_BEAMS = ("lidar_315", "lidar_270", "lidar_225")
# Deliberately lidar_000 ONLY, not a wider cone: lidar_045/lidar_315 already
# do double duty in LEFT_BEAMS/RIGHT_BEAMS. Including them here too would
# couple a forward-obstacle slowdown to side-wall proximity during a lateral
# correction (a wall picked up obliquely by a diagonal beam would look like
# a forward obstacle and cut throttle when nothing is ahead).
FRONT_CONE_BEAMS = ("lidar_000",)
FRONT_LEFT_BEAM = "lidar_045"
REAR_LEFT_BEAM = "lidar_135"
FRONT_RIGHT_BEAM = "lidar_315"
REAR_RIGHT_BEAM = "lidar_225"


@dataclass(frozen=True)
class LidarScan(Generic[ArrayT]):
    """8-beam LiDAR ring as one ordered scan (see LIDAR_BEAM_ORDER).
    angles[i] and ranges[i] refer to the same beam."""

    angles: ArrayT  # radians
    ranges: ArrayT  # meters; -1 = no hit within cutoff


def lidar_scan(sensors: SensorReadings) -> LidarScan[np.ndarray]:
    """Pack the 8 named lidar_* fields into one ordered LidarScan."""
    ranges = np.array([getattr(sensors, name)[0] for name in LIDAR_BEAM_ORDER])
    angles = np.deg2rad(LIDAR_ANGLES_DEG)
    return LidarScan(angles=angles, ranges=ranges)


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
