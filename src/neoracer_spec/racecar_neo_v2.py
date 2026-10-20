"""
The RACECAR Neo V2, part by part, from its measurement worksheet and its
calibration reference.

There is no CAD model of this car yet. The values come from two sources:
- the RACECAR Neo V2 Model Measurement Worksheet: two measurements of the
  car (its C.L and M.Z columns), and the values of a model of the car that
  it checks;
- calibration_reference_car_2026-10: the calibration tools' runs on one
  reference car (2026-10-04 and 05), and the parameter files they wrote.

The value is the calibration reference's where it has one, and otherwise
the C.L measurement. Where the M.Z measurement, the model or an earlier
calibration disagree, they are its conflicts. The worksheet also lists
values it computes from these, such as the wheel angles at full lock from
the turning radius, which are left out here.

These sources measure what the car does, not its motor's and servo's
ratings, and the car's firmware holds a speed rather than a voltage. So its
parts are its own types, not those of spec.py, and there is no model of it
yet.

Frame: the car's base_link, on the floor below the middle of the rear axle
(REP-103): +X forward, +Y left, +Z up. Values are in SI units unless a
field's name says otherwise.
"""

from dataclasses import dataclass
from typing import Literal

from .provenance import (
    Alternative,
    Source,
    Sourced,
    assumed,
    calibrated,
    documented,
    estimated,
    measured,
)
from .spec import Vec3

C_L = Source("C.L", "RACECAR Neo V2 Model Measurement Worksheet, the C.L column")
M_Z = Source("M.Z", "RACECAR Neo V2 Model Measurement Worksheet, the M.Z column")
MODEL = Source(
    "model", "RACECAR Neo V2 Model Measurement Worksheet, the Model Value column"
)
WORKSHEET_RUNS = Source(
    "worksheet runs",
    "RACECAR Neo V2 Model Measurement Worksheet, the lidar-referenced"
    " calibration runs of 2026-10-05",
)
FRAMES = Source("frames.yaml", "the car's frames.yaml, as the worksheet quotes it")


def _reference(name: str, tool: str) -> Source:
    """A file of calibration_reference_car_2026-10."""
    return Source(name, f"calibration_reference_car_2026-10/{name}, from {tool}")


REFERENCE_CAR = _reference("README.md", "the reference car's table")
ODOM = _reference("parameters/odom.local.yaml", "calibrate_odom.py, 2026-10-05")
DYNAMICS = _reference(
    "parameters/dynamics.local.yaml", "calibrate_dynamics.py, 2026-10-05"
)
STEERING = _reference(
    "parameters/throttle.local.md", "calibrate_steering.py, 2026-10-05"
)
LIDAR_YAW = _reference("parameters/frames.local.yaml", "calibrate_lidar.py, 2026-10-05")
IMU_CAL = _reference(
    "parameters/lsm9ds1_cal.local.yaml", "calibrate_imu.py, 2026-10-04"
)


@dataclass(frozen=True)
class Sides[T]:
    """A value that differs between the car's left and right."""

    left: T
    right: T


@dataclass(frozen=True)
class Span:
    """The range of a value over repeated runs."""

    low: float
    high: float


@dataclass(frozen=True)
class AtCommand[T]:
    """A value measured with the car held at a command: a share of full, in
    [-1, 1], + forward or left."""

    command: float
    value: T


@dataclass(frozen=True)
class Chassis:
    wheelbase: Sourced[float]  # m, between the front and rear axles
    track: Sourced[float]  # m, between the rear wheels' centres
    mass: Sourced[float]  # kg, with the battery
    # The share of the weight on the front wheels. The centre of mass is
    # this share of the wheelbase ahead of the rear axle, and laterally centred.
    front_weight_share: Sourced[float]
    com_height: Sourced[float]  # m, of the centre of mass above the floor
    yaw_inertia: Sourced[float]  # kg·m², about the centre of mass
    ride_height: Sourced[float]  # m, of the chassis above the floor, moving


@dataclass(frozen=True)
class Footprint:
    """The car's outline."""

    front: Sourced[float]  # m, x of the front-most point
    rear: Sourced[float]  # m, x of the rear-most point
    left: Sourced[float]  # m, y of the left-most point
    right: Sourced[float]  # m, y of the right-most point
    lidar_top: Sourced[float]  # m, z of the top of the LiDAR
    antenna_top: Sourced[float]  # m, z of the antenna tips, the tallest point


@dataclass(frozen=True)
class Wheel:
    diameter: Sourced[float]  # m, the tyre's
    width: Sourced[float]  # m, the tyre's
    axle_height: Sourced[float]  # m, of the rear axle above the floor


@dataclass(frozen=True)
class Steering:
    # At full lock, at the middle of the rear axle. The wheels' angles at
    # full lock follow from it and the wheelbase and track.
    min_turn_radius: Sourced[Sides[float]]  # m
    # From the centre to full lock, timed from the yaw rate: an upper bound,
    # since it includes the tyres' lag in following the wheels.
    center_to_lock_time: Sourced[Sides[float]]  # s
    # The curvature of the path that a steering command gives, 1/m, + left.
    response: Sourced[tuple[AtCommand[Span], ...]]
    trim: Sourced[float]  # of command, that the firmware adds to every command
    deadband: Sourced[float]  # of command: smaller changes do nothing
    straight_drift_degrees_per_m: Sourced[Span]  # heading change at command 0, + left


@dataclass(frozen=True)
class Drive:
    """The firmware holds a speed: a drive command is a share of max_speed."""

    max_speed: Sourced[float]  # m/s, the firmware's ceiling, rc.drive.set_max_speed
    speed_floor: Sourced[float]  # m/s, the least speed the firmware sets
    steady_speed: Sourced[tuple[AtCommand[float], ...]]  # m/s, true
    rise_time: Sourced[tuple[AtCommand[float], ...]]  # s, from rest to 90% of it
    stop_distance: Sourced[tuple[AtCommand[float], ...]]  # m, after command 0
    stop_time: Sourced[tuple[AtCommand[float], ...]]  # s, after command 0
    reverse: Sourced[Literal["immediate", "delayed", "double command"]]


@dataclass(frozen=True)
class Odometry:
    """The speed and the yaw rate that the firmware reports."""

    min_speed: Sourced[float]  # m/s; below it the reported speed is 0
    speed_scale: Sourced[float]  # true speed / reported speed
    yaw_rate_scale: Sourced[float]  # true yaw rate / reported yaw rate
    # How much more than the true speed the encoder reads at full lock.
    turn_overread: Sourced[Sides[float]]
    telemetry_rate: Sourced[float]  # Hz


@dataclass(frozen=True)
class Lidar:
    position: Sourced[Vec3]  # m, the centre of the scan
    yaw_degrees: Sourced[float]  # of the scan's zero bearing, from +X
    tilt_degrees: Sourced[tuple[float, float]]  # roll, pitch
    samples: Sourced[int]  # a turn
    scan_rate: Sourced[float]  # Hz, turns a second
    min_range: Sourced[float]  # m
    max_range: Sourced[float]  # m


@dataclass(frozen=True)
class Imu:
    """The LSM9DS1 on the drive-controller board. The firmware reports
    (raw - bias) * scale."""

    position: Sourced[Vec3]  # m, the breakout's centre
    rotation_degrees: Sourced[Vec3]  # roll, pitch, yaw of the chip's axes
    accel_bias: Sourced[Vec3]  # m/s²
    accel_scale: Sourced[Vec3]
    gyro_bias: Sourced[Vec3]  # rad/s
    rest_accel: Sourced[Vec3]  # m/s², calibrated, at rest
    rest_gyro: Sourced[Vec3]  # rad/s, calibrated, at rest
    gyro_noise: Sourced[float]  # rad/s, 1 sigma a sample, at rest


@dataclass(frozen=True)
class Camera:
    front: Sourced[float]  # m, x of its front face
    y: Sourced[float]  # m, of its centre
    bottom: Sourced[float]  # m, z of its underside
    pitch_degrees: Sourced[float]


@dataclass(frozen=True)
class RacecarNeoV2:
    chassis: Chassis
    footprint: Footprint
    wheel: Wheel
    steering: Steering
    drive: Drive
    odometry: Odometry
    lidar: Lidar
    imu: Imu
    camera: Camera


RACECAR_NEO_V2 = RacecarNeoV2(
    chassis=Chassis(
        wheelbase=measured(
            0.230,
            C_L,
            conflicts=(Alternative(0.229, M_Z), Alternative(0.225, MODEL)),
        ),
        track=measured(0.178, C_L, conflicts=(Alternative(0.174, M_Z),)),
        mass=measured(2.36, C_L),
        front_weight_share=measured(
            0.522,
            C_L,
            note="axle weights on a kitchen scale: front 1160 g, rear 1060 g;"
            " they sum to 140 g short of the total",
        ),
        com_height=measured(0.095, C_L, note="a sideways tip test: 47 degrees a side"),
        yaw_inertia=estimated(0.032, "a uniform box of 2.36 kg, 347 x 209 mm"),
        ride_height=assumed(0.0205, MODEL, "the model's; not yet measured"),
    ),
    footprint=Footprint(
        front=measured(0.2875, C_L, conflicts=(Alternative(0.286, M_Z),)),
        rear=measured(-0.0597, C_L, conflicts=(Alternative(-0.059, M_Z),)),
        left=measured(0.1045, C_L, conflicts=(Alternative(0.104, M_Z),)),
        right=measured(
            -0.1045,
            C_L,
            note="the model's -0.107 includes the battery connector, which the"
            " outline leaves out",
            conflicts=(Alternative(-0.104, M_Z), Alternative(-0.107, MODEL)),
        ),
        lidar_top=measured(
            0.205, C_L, conflicts=(Alternative(0.204, M_Z), Alternative(0.2105, MODEL))
        ),
        antenna_top=measured(0.275, C_L, conflicts=(Alternative(0.267, MODEL),)),
    ),
    wheel=Wheel(
        diameter=measured(
            0.072,
            C_L,
            note="the speed reads 0.904 of the true speed, as from a 65.1 mm"
            " wheel; likely the firmware's encoder constants, not the wheel",
            conflicts=(Alternative(0.071, M_Z),),
        ),
        width=measured(0.029, C_L, conflicts=(Alternative(0.030, M_Z),)),
        axle_height=measured(0.036, C_L),
    ),
    steering=Steering(
        min_turn_radius=measured(
            Sides(left=0.420, right=0.466),
            DYNAMICS,
            note="the full-lock circles, from the LiDAR. Left lock turns about 10%"
            " tighter: servo travel or linkage geometry, which no trim corrects."
            " The wheels' angles are 34.8 / 24.3 degrees (inner / outer) to the"
            " left and 31.4 / 22.5 to the right; the model has 22.4 / 18.0",
            conflicts=(Alternative(Sides(left=0.419, right=0.465), WORKSHEET_RUNS),),
        ),
        center_to_lock_time=measured(
            Sides(left=0.203, right=0.216),
            DYNAMICS,
            note="the model swings lock to lock in 0.067 s",
        ),
        response=measured(
            (
                AtCommand(-1.0, Span(-2.140, -2.135)),
                AtCommand(-0.5, Span(-2.098, -2.042)),
                AtCommand(-0.25, Span(-1.043, -1.006)),
                AtCommand(0.0, Span(-0.047, 0.007)),
                AtCommand(0.25, Span(0.884, 0.919)),
                AtCommand(0.5, Span(1.844, 1.858)),
                AtCommand(1.0, Span(2.366, 2.401)),
            ),
            STEERING,
            note="arcs at speed 0.25 with trim 0, from the LiDAR, 3 to 8 runs a"
            " command. It flattens past 0.5 and differs by side: use the table,"
            " not a linear model",
        ),
        trim=calibrated(
            0.0,
            STEERING,
            "the LiDAR: the drift at command 0 over the curvature near the centre",
            note="a first fit of +0.0188 read the difference between the sides as"
            " a centre offset; it moved the drift by under 0.1 degree/m",
        ),
        deadband=measured(
            0.02,
            WORKSHEET_RUNS,
            note="0.01 to 0.02: a trim of +0.019 (9 us) had no measurable effect",
        ),
        straight_drift_degrees_per_m=measured(
            Span(-2.70, 0.42),
            STEERING,
            note="8 runs at trim 0: servo and linkage play, inside the deadband."
            " The worksheet gives -1.7 (-0.8 to -2.6)",
        ),
    ),
    drive=Drive(
        max_speed=documented(
            1.0,
            REFERENCE_CAR,
            note="the setting of every run; the car reaches 1.07 m/s",
        ),
        speed_floor=documented(0.5, REFERENCE_CAR),
        steady_speed=measured(
            (AtCommand(0.25, 0.721), AtCommand(0.5, 0.833), AtCommand(1.0, 1.07)),
            DYNAMICS,
            note="from the odometry, with its lidar-calibrated scale. The worksheet:"
            " the firmware holds about 0.175 m/s above each set speed, in its"
            " own encoder units",
        ),
        rise_time=measured(
            (AtCommand(0.25, 0.33), AtCommand(0.5, 0.33), AtCommand(1.0, 0.36)),
            DYNAMICS,
            note="the model takes 1.3 s",
        ),
        stop_distance=measured(
            (AtCommand(0.25, 0.184), AtCommand(0.5, 0.246), AtCommand(1.0, 0.372)),
            DYNAMICS,
            conflicts=(
                Alternative(
                    (
                        AtCommand(0.25, 0.184),
                        AtCommand(0.5, 0.246),
                        AtCommand(1.0, 0.370),
                    ),
                    C_L,
                ),
            ),
        ),
        stop_time=measured(
            (AtCommand(0.25, 0.56), AtCommand(0.5, 0.59), AtCommand(1.0, 0.72)),
            DYNAMICS,
        ),
        reverse=measured("immediate", C_L),
    ),
    odometry=Odometry(
        min_speed=measured(0.07, WORKSHEET_RUNS),
        speed_scale=calibrated(
            0.90406,
            ODOM,
            "the LiDAR, driving straight at a wall",
            note="it depends on the tyres' diameter and wear. Earlier runs against"
            " a tape gave 0.903 and 0.865",
        ),
        yaw_rate_scale=calibrated(
            1.00188, ODOM, "the LiDAR, over three full-lock circles"
        ),
        turn_overread=measured(
            Sides(left=0.0415, right=0.0245),
            DYNAMICS,
            note="the centre shaft is locked, so the encoder reads a speed between"
            " the rear and the front axles'",
        ),
        telemetry_rate=measured(
            170.0, WORKSHEET_RUNS, note="about 1 frame in 5 arrives late, after 10.7 ms"
        ),
    ),
    lidar=Lidar(
        position=measured(
            (0.2383, 0.0, 0.195),
            C_L,
            note="x checked against a wall to about 1 mm",
            conflicts=(
                Alternative((0.240, 0.0, 0.195), M_Z),
                Alternative((0.2383, 0.0, 0.2016), MODEL),
            ),
        ),
        yaw_degrees=calibrated(
            179.619,
            LIDAR_YAW,
            "the odometry's direction of travel, by scan matching",
            note="its motor faces the rear: nominally 180. The mean of six"
            " placements, standard error 0.059 degrees",
            conflicts=(Alternative(179.16, WORKSHEET_RUNS),),
        ),
        tilt_degrees=measured((0.0, 0.0), C_L),
        samples=measured(1080, C_L, conflicts=(Alternative(720, MODEL),)),
        scan_rate=measured(7.4, C_L, conflicts=(Alternative(6.0, MODEL),)),
        min_range=measured(0.05, C_L),
        max_range=measured(12.0, C_L),
    ),
    imu=Imu(
        position=measured(
            (0.190, 0.0, 0.105),
            C_L,
            conflicts=(
                Alternative((0.192, 0.0, 0.105), M_Z),
                Alternative((0.190, 0.0, 0.1135), MODEL),
            ),
        ),
        rotation_degrees=documented(
            (-90.0, 0.0, -90.0), FRAMES, note="the chip is mounted face down"
        ),
        accel_bias=calibrated(
            (-0.2646, -0.0453, 0.0734),
            IMU_CAL,
            "gravity, with each axis up and down",
            note="one board's",
        ),
        accel_scale=calibrated(
            (1.00451, 1.01032, 1.00362), IMU_CAL, "gravity, with each axis up and down"
        ),
        gyro_bias=calibrated(
            (0.02844, 0.00084, -0.06329), IMU_CAL, "the car at rest", note="one board's"
        ),
        rest_accel=measured((-0.15, -0.01, 9.82), C_L),
        rest_gyro=measured((0.0012, -0.0005, 0.0009), C_L),
        gyro_noise=measured(0.039, WORKSHEET_RUNS),
    ),
    camera=Camera(
        front=measured(0.2779, C_L, conflicts=(Alternative(0.275, M_Z),)),
        y=measured(0.0, C_L),
        bottom=measured(
            0.100, C_L, conflicts=(Alternative(0.103, M_Z), Alternative(0.107, MODEL))
        ),
        pitch_degrees=measured(0.0, C_L),
    ),
)
