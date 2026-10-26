"""
Reads all named sensors of a compiled NeoRacer scene into a SensorReadings struct.

find_sensors finds the sensor elements in the scene's MjSpec once. read then
binds each one to the data (bind_scalar, see scene.py) and copies its values.

Frame: +X forward, +Y left, +Z up.

Sensor map (matches neoracer.xml):
  imu_accel     = 3-axis linear acceleration (m/s²)
  imu_gyro      = 3-axis angular velocity (rad/s)
  imu_quat      = orientation quaternion (w, x, y, z)
  imu_linvel    = 3D linear velocity in world frame (m/s)
  steer_cmd_pos = virtual steering input angle (rad)
  fl/fr_steer_pos = actual front wheel steer angles (rad)
  fl/fr/rl/rr_wheel_vel = wheel angular velocities (rad/s)
  drive_pos     = the drivetrain's angle: the mean wheel angle (rad)
  fl/fr/rl/rr_susp_pos  = suspension travel (m; positive = compressed)
  lidar_000..315 = rangefinder distances (m; -1 = no hit, or a bearing that
                   the car blocks), as the LiDAR's last complete turn reports
                   them (mjcf.beam_timing)

The LiDAR controllers (track_centering.py, wall_following.py) share the beam
groups LEFT_BEAMS, RIGHT_BEAMS and the *_BEAM constants. Each value is an
index into LIDAR_BEAM_ORDER and into lidar_ranges().
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Any, assert_never

import mujoco
import numpy as np
import numpy.typing as npt

from . import contract

NO_RETURN = -1.0
"""The distance (m) that a rangefinder reads with nothing in range."""

type Vec3 = npt.NDArray[np.float64]
"""Shape (3,)."""
type Quat = npt.NDArray[np.float64]
"""Shape (4,), in the order (w, x, y, z)."""
type Vec8 = npt.NDArray[np.float64]
"""Shape (8,), one value per LiDAR beam."""


@dataclass(frozen=True)
class SensorReadings:
    """One named field per sensor in neoracer.xml. A sensor with one value is a
    float. A sensor with several values is a numpy array.

    The IMU is at the chassis center. The imu site has no rotation relative to
    the car body, so the IMU frame is the body frame: +X forward, +Y left, +Z
    up. The imu_* fields are MuJoCo's raw values, with no calibration.

    The *_wheel_vel fields are the wheels' angular velocities (rad/s), and the
    *_susp_pos fields the suspension travel (m, + = compressed). The lidar_*
    fields are an 8-beam LiDAR ring (m, -1 = no hit, or a bearing that the car
    blocks)."""

    imu_accel: Vec3
    """The acceleration (m/s²) in the body frame. It includes gravity: about
    (0, 0, +9.81) at rest."""
    imu_gyro: Vec3
    """The angular velocity (rad/s) in the body frame."""
    imu_quat: Quat
    """The orientation (w, x, y, z) of the IMU frame relative to the world frame."""
    imu_linvel: Vec3
    """The linear velocity (m/s) in the world frame."""
    steer_cmd_pos: float
    """The steering angle (rad) of the virtual command."""
    fl_steer_pos: float
    """The actual front-left steering angle (rad)."""
    fr_steer_pos: float
    """The actual front-right steering angle (rad)."""
    fl_wheel_vel: float
    fr_wheel_vel: float
    rl_wheel_vel: float
    rr_wheel_vel: float
    drive_pos: float
    """The mean of the four wheels' angles (rad), which the motor encoder counts."""
    fl_susp_pos: float
    fr_susp_pos: float
    rl_susp_pos: float
    rr_susp_pos: float
    lidar_000: float
    lidar_045: float
    lidar_090: float
    lidar_135: float
    lidar_180: float
    lidar_225: float
    lidar_270: float
    lidar_315: float


VECTOR_SIZE: Mapping[str, int] = MappingProxyType(
    {"imu_accel": 3, "imu_gyro": 3, "imu_quat": 4, "imu_linvel": 3}
)
"""The number of values of each sensor with more than one value. Every other
SensorReadings field has one value."""


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
    blocked: frozenset[str]
    """The beams of LIDAR_BEAM_ORDER outside the field that the LiDAR sees
    (contract.lidar_field). The car blocks them, so they read NO_RETURN."""


def find_sensors(
    spec: mujoco.MjSpec, model: mujoco.MjModel, prefix: str = ""
) -> CarSensors:
    """Find the sensor of each SensorReadings field of the car whose names are
    behind prefix in spec, which model was compiled from, and the LiDAR
    beams that the car blocks. Raise ValueError if a field has no sensor of
    that name and size, or the LiDAR site is missing."""
    found: dict[str, mujoco.MjsSensor] = {}
    problems: list[str] = []
    lidar = spec.site(prefix + contract.LIDAR_SITE)
    if lidar is None:
        problems.append(f"the {prefix + contract.LIDAR_SITE} site is missing")
    for f in fields(SensorReadings):
        sensor = spec.sensor(prefix + f.name)
        size = VECTOR_SIZE.get(f.name, 1)
        if sensor is None:
            problems.append(f"{prefix + f.name} is missing")
            continue
        dim = int(model.bind_scalar(sensor).dim[0])
        if dim != size:
            problems.append(f"{prefix + f.name} has {dim} values, not {size}")
        else:
            found[f.name] = sensor
    if problems or lidar is None:
        raise ValueError(
            f"the model does not have the NeoRacer sensors: {'; '.join(problems)}"
        )
    field = contract.lidar_field(model, lidar)
    blocked = frozenset(
        name for name in LIDAR_BEAM_ORDER if not beam_in_field(name, field)
    )
    return CarSensors(**found, blocked=blocked)


def beam_in_field(name: str, field: float) -> bool:
    """Whether a LiDAR that sees field (rad), centered on its front, sees the
    beam of LIDAR_BEAM_ORDER called name (contract.in_field)."""
    return contract.in_field(math.radians(int(name.removeprefix("lidar_"))), field)


def read(sensors: CarSensors, data: mujoco.MjData) -> SensorReadings:
    """Read every named sensor of data. A LiDAR beam that the car blocks
    reads NO_RETURN."""

    def value(name: str) -> float | npt.NDArray[np.float64]:
        if name in sensors.blocked:
            return NO_RETURN
        values = data.bind_scalar(getattr(sensors, name)).data
        return values.copy() if name in VECTOR_SIZE else float(values[0])

    # Each field's value has the field's type, which no checker can see here.
    values: dict[str, Any] = {f.name: value(f.name) for f in fields(SensorReadings)}
    return SensorReadings(**values)


def motor_encoder_rad_s(sensors: SensorReadings) -> float:
    """The motor encoder reading: angular velocity (rad/s, + = forward).

    The motor turns with the mean of the four wheels through the open
    differentials, so this function returns the mean of the four wheel_vel
    sensors. The value is raw: no window, filter or quantization. For the
    speed that the car's firmware reports, see encoder.py.
    """
    s = sensors
    return (s.fl_wheel_vel + s.fr_wheel_vel + s.rl_wheel_vel + s.rr_wheel_vel) / 4


LIDAR_BEAM_ORDER = tuple(
    f.name for f in fields(SensorReadings) if f.name.startswith("lidar_")
)
"""The LiDAR beams, in the order of their SensorReadings fields: their bearings
counterclockwise from the front, 0 to 315 degrees."""

LEFT_BEAMS = (1, 2, 3)
"""The indices of lidar_045, lidar_090 and lidar_135."""
RIGHT_BEAMS = (7, 6, 5)
"""The indices of lidar_315, lidar_270 and lidar_225."""
FRONT_BEAM = 0
"""The index of lidar_000, the only beam that measures the clearance ahead.
lidar_045 and lidar_315 are in LEFT_BEAMS and RIGHT_BEAMS: as a front beam, a
diagonal beam that sees a side wall would look like an obstacle ahead and cut
the throttle."""
LEFT_BEAM = 2
"""The index of lidar_090."""
RIGHT_BEAM = 6
"""The index of lidar_270."""
FRONT_LEFT_BEAM = 1
"""The index of lidar_045."""
REAR_LEFT_BEAM = 3
"""The index of lidar_135."""
FRONT_RIGHT_BEAM = 7
"""The index of lidar_315."""
REAR_RIGHT_BEAM = 5
"""The index of lidar_225."""


def lidar_ranges(sensors: SensorReadings) -> Vec8:
    """The 8 lidar_* ranges (m; -1 = no hit) in LIDAR_BEAM_ORDER."""
    return np.array([getattr(sensors, name) for name in LIDAR_BEAM_ORDER])


@dataclass(frozen=True)
class Hit:
    distance: float
    """The distance (m) that the beam reads."""


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
