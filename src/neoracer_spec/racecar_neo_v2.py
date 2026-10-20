"""
The RACECAR Neo V2, part by part, from its measurement worksheet.

There is no CAD model of this car yet. Every value comes from the RACECAR
Neo V2 Model Measurement Worksheet: two measurements of the car (its C.L
and M.Z columns), the values of a model of the car that it checks, and runs
calibrated against the car's LiDAR (2026-10-05). The value is the C.L
measurement; where the M.Z measurement or the model disagree, they are its
conflicts. The worksheet also lists values it computes from these, such as
the wheel angles at full lock from the turning radius, which are left out
here.

The worksheet measures what the car does, not its motor's and servo's
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
CALIBRATION = Source(
    "calibration",
    "RACECAR Neo V2 Model Measurement Worksheet, the lidar-referenced"
    " calibration runs of 2026-10-05",
)
FRAMES = Source("frames.yaml", "the car's frames.yaml, as the worksheet quotes it")


@dataclass(frozen=True)
class Sides[T]:
    """A value that differs between the car's left and right."""

    left: T
    right: T


@dataclass(frozen=True)
class AtCommand[T]:
    """A value measured with the car held at a command, in [0, 1] of full."""

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
    degrees_per_us: Sourced[float]  # of wheel angle per us of servo pulse, near centre
    us_per_command: Sourced[float]  # us of servo pulse per unit of command
    # The bicycle-model steering angle that a steering command gives.
    response_degrees: Sourced[tuple[AtCommand[Sides[float]], ...]]
    deadband: Sourced[float]  # of command: smaller changes do nothing
    straight_drift_degrees_per_m: Sourced[float]  # heading change at command 0, + left


@dataclass(frozen=True)
class Drive:
    """The firmware holds a speed: a drive command is a share of max_speed."""

    max_speed: Sourced[float]  # m/s, the firmware's rc.drive.set_max_speed
    steady_speed: Sourced[tuple[AtCommand[float], ...]]  # m/s, true
    rise_time: Sourced[tuple[AtCommand[float], ...]]  # s, from rest to 90% of it
    stop_distance: Sourced[tuple[AtCommand[float], ...]]  # m, after command 0
    reverse: Sourced[Literal["immediate", "delayed", "double command"]]


@dataclass(frozen=True)
class Odometry:
    """The speed and the yaw rate that the firmware reports."""

    distance_per_transition: Sourced[float]  # m of travel per encoder transition
    min_speed: Sourced[float]  # m/s; below it the reported speed is 0
    speed_scale: Sourced[float]  # true speed / reported speed
    yaw_rate_scale: Sourced[float]  # true yaw rate / reported yaw rate
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
    position: Sourced[Vec3]  # m, the breakout's centre
    rotation_degrees: Sourced[Vec3]  # roll, pitch, yaw of the chip's axes
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
            Sides(left=0.419, right=0.465),
            CALIBRATION,
            note="from the LiDAR; the wheels' angles are 34.8 / 24.3 degrees"
            " (inner / outer) to the left and 31.4 / 22.5 to the right. The"
            " difference is a 1.2 degree offset to the left: a simulator should"
            " use the mean, about 33 / 23, until the steering is calibrated."
            " The model has 22.4 / 18.0",
        ),
        center_to_lock_time=measured(
            Sides(left=0.20, right=0.22),
            C_L,
            note="about 0.07 s of it is the tyres' lag. The model swings lock to"
            " lock in 0.067 s",
        ),
        degrees_per_us=measured(0.104, CALIBRATION),
        us_per_command=measured(475.0, CALIBRATION),
        response_degrees=measured(
            (
                AtCommand(0.25, Sides(left=11.5, right=13.1)),
                AtCommand(0.5, Sides(left=22.9, right=25.3)),
                AtCommand(1.0, Sides(left=28.6, right=26.3)),
            ),
            CALIBRATION,
            note="it flattens past 0.5",
        ),
        deadband=measured(
            0.02,
            CALIBRATION,
            note="0.01 to 0.02: a trim of +0.019 (9 us) had no measurable effect",
        ),
        straight_drift_degrees_per_m=measured(
            -1.7,
            CALIBRATION,
            note="-0.8 to -2.6: hardware variation, below the trim's resolution",
        ),
    ),
    drive=Drive(
        max_speed=measured(
            1.0,
            C_L,
            note="the setting of every drive run; at it the car reaches 1.07 m/s",
        ),
        steady_speed=measured(
            (AtCommand(0.25, 0.72), AtCommand(0.5, 0.83), AtCommand(1.0, 1.07)),
            CALIBRATION,
            note="the firmware holds about 0.175 m/s above each set speed, in its"
            " own encoder units",
        ),
        rise_time=measured(
            (AtCommand(0.5, 0.33), AtCommand(1.0, 0.36)),
            C_L,
            note="the model takes 1.3 s",
        ),
        stop_distance=measured(
            (AtCommand(0.25, 0.184), AtCommand(0.5, 0.246), AtCommand(1.0, 0.370)),
            C_L,
            note="from 0.5 it stops in 0.59 s: 1.3 to 1.5 m/s² of braking",
        ),
        reverse=measured("immediate", C_L),
    ),
    odometry=Odometry(
        distance_per_transition=measured(
            0.00203,
            CALIBRATION,
            note="the firmware's nominal 2.25 mm times the speed scale",
        ),
        min_speed=measured(0.07, CALIBRATION),
        speed_scale=measured(0.90406, CALIBRATION),
        yaw_rate_scale=measured(1.00188, CALIBRATION),
        telemetry_rate=measured(
            170.0, CALIBRATION, note="about 1 frame in 5 arrives late, after 10.7 ms"
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
        yaw_degrees=measured(
            179.16,
            CALIBRATION,
            note="its motor faces the rear; the mount error is 0 to 0.8 degrees."
            " One car's value",
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
        rest_accel=measured((-0.15, -0.01, 9.82), C_L, note="the LSM9DS1's"),
        rest_gyro=measured((0.0012, -0.0005, 0.0009), C_L, note="the LSM9DS1's"),
        gyro_noise=measured(0.039, CALIBRATION),
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
