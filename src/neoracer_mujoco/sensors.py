"""
Reads all named sensors of a compiled NeoRacer scene into a SensorReadings struct.

find_sensors finds the sensor elements in the scene's MjSpec once. read then
binds each one to the data (bind_scalar, see scene.py) and copies its values.

Frame: +X forward, +Y left, +Z up.

Sensor map (matches neoracer.xml):
  imu_accel     — 3-axis linear acceleration (m/s²)
  imu_gyro      — 3-axis angular velocity (rad/s)
  imu_quat      — orientation quaternion (w, x, y, z)
  imu_linvel    — 3D linear velocity in world frame (m/s)
  steer_cmd_pos — virtual steering input angle (rad)
  fl/fr_steer_pos — actual front wheel steer angles (rad)
  fl/fr/rl/rr_wheel_vel — wheel angular velocities (rad/s)
  drive_pos     — the drivetrain's angle: the mean wheel angle (rad)
  fl/fr/rl/rr_susp_pos  — suspension travel (m; positive = compressed)
  lidar_000..315 — rangefinder distances (m; -1 = no hit)
"""

from dataclasses import dataclass, fields
from typing import assert_never

import mujoco
import numpy as np
import numpy.typing as npt

from neoracer_spec.neoracer import NEORACER

# The tyre's radius, which the car model's wheel geoms also have.
WHEEL_RADIUS = NEORACER.wheel.radius.value  # m

type Vec3 = npt.NDArray[np.float64]  # shape (3,)
type Quat = npt.NDArray[np.float64]  # shape (4,), (w, x, y, z)
type Vec8 = npt.NDArray[np.float64]  # shape (8,), one value per LiDAR beam


@dataclass(frozen=True)
class SensorReadings:
    """One named field per sensor in neoracer.xml. A sensor with one value is a
    float. A sensor with several values is a numpy array."""

    # IMU at the chassis centre. The imu site has no rotation relative to the
    # car body, so the IMU frame is the body frame: +X forward, +Y left, +Z up.
    # These are MuJoCo's raw values, with no calibration.
    imu_accel: Vec3  # m/s², body frame; includes gravity: ~(0, 0, +9.81) at rest
    imu_gyro: Vec3  # rad/s, body frame
    imu_quat: Quat  # (w,x,y,z) of the IMU frame relative to the world frame
    imu_linvel: Vec3  # world linear vel (m/s)
    # steering angles (rad)
    steer_cmd_pos: float  # virtual command
    fl_steer_pos: float  # actual front-left
    fr_steer_pos: float  # actual front-right
    # wheel angular velocity (rad/s)
    fl_wheel_vel: float
    fr_wheel_vel: float
    rl_wheel_vel: float
    rr_wheel_vel: float
    # rad, the mean of the four wheels' angles, which the motor encoder counts
    drive_pos: float
    # suspension travel (m; + = compressed)
    fl_susp_pos: float
    fr_susp_pos: float
    rl_susp_pos: float
    rr_susp_pos: float
    # 8-beam lidar ring (m; -1 = no hit)
    lidar_000: float
    lidar_045: float
    lidar_090: float
    lidar_135: float
    lidar_180: float
    lidar_225: float
    lidar_270: float
    lidar_315: float


# The number of values of each sensor with more than one value. Every other
# SensorReadings field has one value.
VECTOR_SIZE: frozendict[str, int] = frozendict(
    imu_accel=3, imu_gyro=3, imu_quat=4, imu_linvel=3
)


@dataclass(frozen=True)
class CarSensors:
    """The sensor element of each SensorReadings field, in the scene's
    MjSpec. Make it once per scene with find_sensors."""

    imu_accel: mujoco.MjsSensor
    imu_gyro: mujoco.MjsSensor
    imu_quat: mujoco.MjsSensor
    imu_linvel: mujoco.MjsSensor
    steer_cmd_pos: mujoco.MjsSensor
    fl_steer_pos: mujoco.MjsSensor
    fr_steer_pos: mujoco.MjsSensor
    fl_wheel_vel: mujoco.MjsSensor
    fr_wheel_vel: mujoco.MjsSensor
    rl_wheel_vel: mujoco.MjsSensor
    rr_wheel_vel: mujoco.MjsSensor
    drive_pos: mujoco.MjsSensor
    fl_susp_pos: mujoco.MjsSensor
    fr_susp_pos: mujoco.MjsSensor
    rl_susp_pos: mujoco.MjsSensor
    rr_susp_pos: mujoco.MjsSensor
    lidar_000: mujoco.MjsSensor
    lidar_045: mujoco.MjsSensor
    lidar_090: mujoco.MjsSensor
    lidar_135: mujoco.MjsSensor
    lidar_180: mujoco.MjsSensor
    lidar_225: mujoco.MjsSensor
    lidar_270: mujoco.MjsSensor
    lidar_315: mujoco.MjsSensor


def find_sensors(spec: mujoco.MjSpec, model: mujoco.MjModel) -> CarSensors:
    """Find the sensor of each SensorReadings field in spec, which model was
    compiled from. Raise ValueError if a field has no sensor of that name and
    size."""
    found: dict[str, mujoco.MjsSensor] = {}
    problems: list[str] = []
    for f in fields(SensorReadings):
        sensor = spec.sensor(f.name)
        size = VECTOR_SIZE.get(f.name, 1)
        if sensor is None:
            problems.append(f"{f.name} is missing")
            continue
        dim = int(model.bind_scalar(sensor).dim[0])
        if dim != size:
            problems.append(f"{f.name} has {dim} values, not {size}")
        else:
            found[f.name] = sensor
    if problems:
        raise ValueError(
            f"the model does not have the NeoRacer sensors: {'; '.join(problems)}"
        )
    return CarSensors(**found)


def read(sensors: CarSensors, data: mujoco.MjData) -> SensorReadings:
    """Read every named sensor of data."""
    d, s = data.bind_scalar, sensors
    return SensorReadings(
        imu_accel=d(s.imu_accel).data.copy(),
        imu_gyro=d(s.imu_gyro).data.copy(),
        imu_quat=d(s.imu_quat).data.copy(),
        imu_linvel=d(s.imu_linvel).data.copy(),
        steer_cmd_pos=float(d(s.steer_cmd_pos).data[0]),
        fl_steer_pos=float(d(s.fl_steer_pos).data[0]),
        fr_steer_pos=float(d(s.fr_steer_pos).data[0]),
        fl_wheel_vel=float(d(s.fl_wheel_vel).data[0]),
        fr_wheel_vel=float(d(s.fr_wheel_vel).data[0]),
        rl_wheel_vel=float(d(s.rl_wheel_vel).data[0]),
        rr_wheel_vel=float(d(s.rr_wheel_vel).data[0]),
        drive_pos=float(d(s.drive_pos).data[0]),
        fl_susp_pos=float(d(s.fl_susp_pos).data[0]),
        fr_susp_pos=float(d(s.fr_susp_pos).data[0]),
        rl_susp_pos=float(d(s.rl_susp_pos).data[0]),
        rr_susp_pos=float(d(s.rr_susp_pos).data[0]),
        lidar_000=float(d(s.lidar_000).data[0]),
        lidar_045=float(d(s.lidar_045).data[0]),
        lidar_090=float(d(s.lidar_090).data[0]),
        lidar_135=float(d(s.lidar_135).data[0]),
        lidar_180=float(d(s.lidar_180).data[0]),
        lidar_225=float(d(s.lidar_225).data[0]),
        lidar_270=float(d(s.lidar_270).data[0]),
        lidar_315=float(d(s.lidar_315).data[0]),
    )


def motor_encoder_rad_s(sensors: SensorReadings) -> float:
    """The motor encoder reading: angular velocity (rad/s, + = forward).

    The motor turns with the mean of the four wheels through the open
    differentials, so this function returns the mean of the four wheel_vel
    sensors. The value is raw: no window, filter or quantization. For the
    speed that the car's firmware reports, see encoder.py.
    """
    s = sensors
    return (s.fl_wheel_vel + s.fr_wheel_vel + s.rl_wheel_vel + s.rr_wheel_vel) / 4


def estimated_linear_speed_ms(motor_rad_s: float) -> float:
    """Estimate the car speed (m/s) from the motor speed (rad/s), as
    motor_rad_s * WHEEL_RADIUS. This estimate assumes that the wheels do not
    slip. If a wheel slips or locks, the estimate differs from the true speed
    (SensorReadings.imu_linvel). A real speedometer that measures wheel speed
    has the same error."""
    return motor_rad_s * WHEEL_RADIUS


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
# Only lidar_000 measures the clearance ahead. lidar_045 and lidar_315 are in
# LEFT_BEAMS and RIGHT_BEAMS: as a front beam, a diagonal beam that sees a side
# wall would look like an obstacle ahead and cut the throttle.
FRONT_BEAM = 0  # lidar_000
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
            s.lidar_000,
            s.lidar_045,
            s.lidar_090,
            s.lidar_135,
            s.lidar_180,
            s.lidar_225,
            s.lidar_270,
            s.lidar_315,
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
