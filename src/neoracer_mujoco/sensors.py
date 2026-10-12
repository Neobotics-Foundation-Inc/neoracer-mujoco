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

from dataclasses import dataclass
from typing import assert_never

import mujoco
import numpy as np
import numpy.typing as npt

# Shadows the wheel geom size in assets/neoracer.xml — keep in sync if the XML changes.
WHEEL_RADIUS = 0.050  # m, ESTIMATED: consistent with STL bbox and URDF CoM

type Vec1 = npt.NDArray[np.float64]  # shape (1,), a scalar sensor
type Vec3 = npt.NDArray[np.float64]  # shape (3,)
type Quat = npt.NDArray[np.float64]  # shape (4,), (w, x, y, z)
type Vec8 = npt.NDArray[np.float64]  # shape (8,), one value per LiDAR beam


type SensorLayout = tuple[tuple[str, slice], ...]  # (name, slice of sensordata)


def sensor_layout(model: mujoco.MjModel) -> SensorLayout:
    """The name and the sensordata slice of each sensor. Get it once per
    model: a control loop then reads with read_layout, without name lookups."""
    return tuple(
        (
            model.sensor(i).name,
            slice(model.sensor_adr[i], model.sensor_adr[i] + model.sensor_dim[i]),
        )
        for i in range(model.nsensor)
    )


def _raw(layout: SensorLayout, data: mujoco.MjData) -> dict[str, np.ndarray]:
    return {name: data.sensordata[s].copy() for name, s in layout}


def read_raw(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, np.ndarray]:
    """Every named sensor as a flat {name: ndarray} dict, the form used by
    validation."""
    return _raw(sensor_layout(model), data)


@dataclass(frozen=True)
class SensorReadings:
    """
    One named field per sensor in neoracer.xml. Each value is the raw sensordata
    slice (a numpy array), so scalars come back as length-1 arrays — read [0].
    """

    # IMU at the chassis centre
    imu_accel: Vec3  # linear accel (m/s²)
    imu_gyro: Vec3  # angular vel (rad/s)
    imu_quat: Quat  # orientation (w,x,y,z)
    imu_linvel: Vec3  # world linear vel (m/s)
    # steering angles (rad)
    steer_cmd_pos: Vec1  # virtual command
    fl_steer_pos: Vec1  # actual front-left
    fr_steer_pos: Vec1  # actual front-right
    # wheel angular velocity (rad/s)
    fl_wheel_vel: Vec1
    fr_wheel_vel: Vec1
    rl_wheel_vel: Vec1
    rr_wheel_vel: Vec1
    # suspension travel (m; + = compressed)
    fl_susp_pos: Vec1
    fr_susp_pos: Vec1
    rl_susp_pos: Vec1
    rr_susp_pos: Vec1
    # 8-beam lidar ring (m; -1 = no hit)
    lidar_000: Vec1
    lidar_045: Vec1
    lidar_090: Vec1
    lidar_135: Vec1
    lidar_180: Vec1
    lidar_225: Vec1
    lidar_270: Vec1
    lidar_315: Vec1


def read_layout(layout: SensorLayout, data: mujoco.MjData) -> SensorReadings:
    """Read every named sensor into a SensorReadings struct."""
    # Limit: fields are coupled to neoracer.xml's sensor set — a car XML with
    # a different sensor list fails here loudly. Add fields if the contract grows.
    return SensorReadings(**_raw(layout, data))


def read(model: mujoco.MjModel, data: mujoco.MjData) -> SensorReadings:
    """Read every named sensor into a SensorReadings struct. In a control
    loop, use read_layout with a layout from sensor_layout(model)."""
    return read_layout(sensor_layout(model), data)


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

    acceleration: Vec3
    angular_velocity: Vec3
    orientation: Quat


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


@dataclass(frozen=True)
class WheelSpeeds:
    """Surface speed of each wheel (m/s, + = forward)."""

    fl: float
    fr: float
    rl: float
    rr: float

    def mean(self) -> float:
        return (self.fl + self.fr + self.rl + self.rr) / 4


def wheel_speed_ms(sensors: SensorReadings) -> WheelSpeeds:
    """Convert each wheel's angular velocity (rad/s) to surface speed (m/s)."""
    return WheelSpeeds(
        fl=float(sensors.fl_wheel_vel[0]) * WHEEL_RADIUS,
        fr=float(sensors.fr_wheel_vel[0]) * WHEEL_RADIUS,
        rl=float(sensors.rl_wheel_vel[0]) * WHEEL_RADIUS,
        rr=float(sensors.rr_wheel_vel[0]) * WHEEL_RADIUS,
    )


@dataclass(frozen=True)
class MotorEncoderReading:
    """
    Single motor encoder reading: rad/s, + = forward.

    The real NeoRacer has one drive motor, not one per wheel -- neoracer.xml
    approximates that with four identically-driven wheel joints, so the
    mean of the four wheel_vel sensors is the correct single-encoder proxy.
    Angular velocity only, not ticks/position: the drive joints have no
    <jointpos> sensor to derive that from.
    """

    angular_velocity: float


def motor_encoder_reading(sensors: SensorReadings) -> MotorEncoderReading:
    """
    Derive the single motor encoder reading from the four wheel_vel jointvel
    sensors (mean angular velocity across all driven wheels — see
    MotorEncoderReading for why a mean is the correct single-motor proxy).
    Raw simulator values — no calibration, filtering, or quantization applied.
    """
    wheel_velocities = np.array(
        [
            sensors.fl_wheel_vel[0],
            sensors.fr_wheel_vel[0],
            sensors.rl_wheel_vel[0],
            sensors.rr_wheel_vel[0],
        ]
    )
    return MotorEncoderReading(angular_velocity=float(np.mean(wheel_velocities)))


def estimated_linear_speed_ms(encoder: MotorEncoderReading) -> float:
    """
    ESTIMATE of vehicle speed from motor rotation (m/s) — NOT ground truth.
    encoder.angular_velocity * WHEEL_RADIUS, i.e. it assumes zero wheel slip.
    Will diverge from the true chassis speed (e.g. SensorReadings.imu_linvel)
    under wheelspin or lockup, exactly like a real wheel-speed-based
    speedometer would.
    """
    return encoder.angular_velocity * WHEEL_RADIUS


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

# Beam groups shared by the LiDAR controllers (track_centering.py,
# wall_following.py). Each value is an index into LIDAR_BEAM_ORDER and into
# lidar_ranges().
LEFT_BEAMS = (1, 2, 3)  # lidar_045, lidar_090, lidar_135
RIGHT_BEAMS = (7, 6, 5)  # lidar_315, lidar_270, lidar_225
# Only lidar_000, not a wider cone: lidar_045/lidar_315 are already in
# LEFT_BEAMS/RIGHT_BEAMS. In the cone, a side wall seen by a diagonal beam
# would look like a forward obstacle and cut the throttle.
FRONT_CONE_BEAMS = (0,)  # lidar_000
LEFT_BEAM = 2  # lidar_090
RIGHT_BEAM = 6  # lidar_270
FRONT_LEFT_BEAM = 1  # lidar_045
REAR_LEFT_BEAM = 3  # lidar_135
FRONT_RIGHT_BEAM = 7  # lidar_315
REAR_RIGHT_BEAM = 5  # lidar_225


def lidar_ranges(sensors: SensorReadings) -> Vec8:
    """The 8 lidar_* ranges (m; -1 = no hit) in LIDAR_BEAM_ORDER."""
    s = sensors
    return np.array(
        [
            s.lidar_000[0],
            s.lidar_045[0],
            s.lidar_090[0],
            s.lidar_135[0],
            s.lidar_180[0],
            s.lidar_225[0],
            s.lidar_270[0],
            s.lidar_315[0],
        ]
    )


@dataclass(frozen=True)
class Hit:
    distance: float  # m


@dataclass(frozen=True)
class NoHit: ...


@dataclass(frozen=True)
class Corrupt: ...


type Beam = Hit | NoHit | Corrupt


def parse_beam(value: float) -> Beam:
    """Classify one rangefinder value (m). MuJoCo gives -1 when no geom is
    within the cutoff. NaN and Inf are corrupt."""
    if not np.isfinite(value):
        return Corrupt()
    if value < 0:
        return NoHit()
    return Hit(value)


def beam_distance(beam: Beam, max_range: float) -> float:
    """Distance (m) for a controller. A beam with no hit or a corrupt value
    reads as max_range, so that bad data never looks like a close wall."""
    match beam:
        case Hit(distance=d):
            return min(d, max_range)
        case NoHit() | Corrupt():
            return max_range
        case _:
            assert_never(beam)


def is_corrupt(beam: Beam) -> bool:
    """True for a NaN or Inf value. A controller counts these beams to
    decide on its fallback command."""
    match beam:
        case Corrupt():
            return True
        case Hit() | NoHit():
            return False
        case _:
            assert_never(beam)


def print_sensors(sensors: SensorReadings) -> None:
    """Print a one-line summary of the most useful sensor values."""
    s = sensors
    print(
        f"spd={wheel_speed_ms(s).mean():+5.2f}m/s  "
        f"steer={s.steer_cmd_pos[0]:+.3f}"
        f"({s.fl_steer_pos[0]:+.3f}/{s.fr_steer_pos[0]:+.3f})  "
        f"susp=[{s.fl_susp_pos[0]:+.3f} {s.fr_susp_pos[0]:+.3f} "
        f"{s.rl_susp_pos[0]:+.3f} {s.rr_susp_pos[0]:+.3f}]m  "
        f"az={s.imu_accel[2]:+5.1f}  gz={s.imu_gyro[2]:+.3f}  "
        f"lidar[fwd={s.lidar_000[0]:.2f} L={s.lidar_090[0]:.2f} "
        f"R={s.lidar_270[0]:.2f}]"
    )
