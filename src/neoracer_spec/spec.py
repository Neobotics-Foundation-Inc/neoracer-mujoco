"""
The specification of a car with Ackermann steering, one drive motor and open
differentials: what the car is, part by part. neoracer.py is the NeoRacer's.

Every value is a provenance.Sourced: the value, how we know it, and what
other sources say. Values are in SI units unless a field's name says
otherwise. No field has a default: a car states every value, with its
source. derive.py computes the simulators' parameters from a Spec.

Frame: +X forward, +Y left, +Z up. A position is in the car's body frame:
its origin at the middle of the wheelbase, on the axle line.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from . import ackermann
from .provenance import Sourced

type Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class Spec:
    chassis: Chassis
    wheel: Wheel
    suspension: Suspension
    battery: Battery
    motor: DriveMotor
    esc: Esc
    servo: SteeringServo
    steering: Steering
    speed_controller: SpeedController
    encoder: MotorEncoder
    imu: Imu
    lidar: Lidar
    camera: Camera
    visuals: Visuals

    def __post_init__(self) -> None:
        c, s = self.chassis, self.steering
        lock = ackermann.steer_for_inner(
            c.wheelbase.value,
            c.track.value,
            math.radians(s.inner_lock_degrees.value),
        )
        if not lock < s.fit_range.value:
            raise ValueError(
                f"full lock, {lock} rad of steering input, is not inside"
                f" fit_range, {s.fit_range.value} rad"
            )


@dataclass(frozen=True)
class Chassis:
    wheelbase: Sourced[float]  # m, between the front and rear axles
    track: Sourced[float]  # m, between the left and right wheels' centres
    mass: Sourced[float]  # kg, the whole car, wheels included
    com_height: Sourced[float]  # m, the chassis' centre of mass above the axles
    inertia: Sourced[InertiaReference]  # scaled to the chassis' mass
    collision_box: Sourced[Box]  # the body less the wheels


@dataclass(frozen=True)
class InertiaReference:
    """A body whose inertia is known, to scale to another of the same shape:
    its principal moments about its centre of mass, and its mass."""

    inertia: Vec3  # kg·m^2
    mass: float  # kg


@dataclass(frozen=True)
class Box:
    center: Vec3  # m
    half_size: Vec3  # m


@dataclass(frozen=True)
class Wheel:
    radius: Sourced[float]  # m, the tyre's
    width: Sourced[float]  # m, the tyre's
    mass: Sourced[float]  # kg, each
    friction: Sourced[float]  # the tyre's sliding friction on the floor
    # Losses from the motor to the tyres, as damping on each wheel's spin.
    drive_damping: Sourced[float]  # N·m·s/rad


@dataclass(frozen=True)
class Suspension:
    stiffness: Sourced[float]  # N/m, each wheel's spring
    damping: Sourced[float]  # N·s/m, each wheel's damper
    droop: Sourced[float]  # m, the wheel's travel down from rest
    jounce: Sourced[float]  # m, the wheel's travel up from rest


@dataclass(frozen=True)
class Battery:
    voltage: Sourced[float]  # V


@dataclass(frozen=True)
class DriveMotor:
    rating: Sourced[DcMotorRating]
    gear_ratio: Sourced[float]  # motor turns per wheel turn


@dataclass(frozen=True)
class DcMotorRating:
    """A DC motor's datasheet line: at voltage, its torque from rest and its
    speed with no load."""

    voltage: float  # V
    stall_torque: float  # N·m
    free_speed_rpm: float  # rpm


def rewound(rating: DcMotorRating, free_speed_rpm: float) -> DcMotorRating:
    """The motor of rating rewound in the same frame to free_speed_rpm at the
    same voltage. With the same copper, turns n give a back-EMF constant
    proportional to n and a resistance to n^2, so the free speed and the stall
    torque both scale as 1 / n."""
    k = free_speed_rpm / rating.free_speed_rpm
    return DcMotorRating(rating.voltage, rating.stall_torque * k, free_speed_rpm)


@dataclass(frozen=True)
class Esc:
    # What the motor does at neutral throttle: short its windings (brake) or
    # leave them open (coast).
    neutral: Sourced[Literal["brake", "coast"]]


@dataclass(frozen=True)
class SteeringServo:
    # Two datasheet lines; the rating at supply is interpolated between them.
    low: Sourced[ServoRating]
    high: Sourced[ServoRating]
    supply: Sourced[float]  # V
    # How far the horn turns from the centre over the firmware's pulse range.
    horn_half_range_degrees: Sourced[float]
    # The servo's own position loop, the model's.
    kp: Sourced[float]  # V/rad
    kd: Sourced[float]  # V·s/rad


@dataclass(frozen=True)
class ServoRating:
    """A hobby servo's datasheet line, in the units it is rated in."""

    voltage: float  # V
    stall_torque_kgf_cm: float  # kgf·cm
    seconds_per_60_degrees: float  # s, with no load


@dataclass(frozen=True)
class Steering:
    # The inner front wheel's angle at full lock.
    inner_lock_degrees: Sourced[float]
    # The share of the horn's half range that the car's driver commands at
    # full lock.
    driver_lock: Sourced[float]
    # The steering input's range (rad): the model's Ackermann constraints fit
    # over it, so it must be wider than full lock.
    fit_range: Sourced[float]
    # Inertia added at the steering input and at each front wheel's steer
    # joint: the servo's rotor, seen through its gears, turns with them.
    armature: Sourced[float]  # kg·m²


@dataclass(frozen=True)
class SpeedController:
    """The firmware's speed loop: a PID on the encoder's speed, with a
    feed-forward, driving the ESC's pulse."""

    period: Sourced[float]  # s
    kp: Sourced[float]  # us of pulse per m/s of error
    ki: Sourced[float]
    kd: Sourced[float]
    speed_filter_weight: Sourced[float]  # of the new speed in the loop's low-pass
    feed_forward: Sourced[float]  # us of pulse per m/s commanded
    deadband: Sourced[float]  # us, the smallest pulse past neutral that it sends
    full_pulse: Sourced[float]  # us past neutral, full throttle
    max_speed: Sourced[float]  # m/s


@dataclass(frozen=True)
class MotorEncoder:
    pulses_per_turn: Sourced[int]  # a turn of the motor
    max_speed_rpm: Sourced[float]  # rpm, its rating
    # Every report_period the firmware turns the count over the period into
    # the tyre's surface speed and low-pass filters it with filter_weight on
    # the new period.
    report_period: Sourced[float]  # s
    filter_weight: Sourced[float]


@dataclass(frozen=True)
class Imu:
    position: Sourced[Vec3]  # m
    rate: Sourced[float]  # Hz, of the readings the car publishes
    accel_range_g: Sourced[float]  # g, the accelerometer's full scale
    gyro_range_dps: Sourced[float]  # degrees/s, the gyroscope's full scale
    magnetometer_range_gauss: Sourced[float]  # gauss


@dataclass(frozen=True)
class Lidar:
    position: Sourced[Vec3]  # m, the centre of the scan
    scan_rate: Sourced[float]  # Hz, turns a second
    resolution_degrees: Sourced[float]  # between two samples
    field_degrees: Sourced[float]  # of bearings it sees, centered on the front
    range_bright: Sourced[float]  # m at least, on a target of 90% reflectivity
    range_dark: Sourced[float]  # m at least, on a target of 10% reflectivity
    accuracy: Sourced[float]  # m


@dataclass(frozen=True)
class Camera:
    position: Sourced[Vec3]  # m, the lens
    sensor: Sourced[Resolution]  # the sensor's most
    fov_degrees: Sourced[float]
    focal_length: Sourced[float]  # m
    shutter: Sourced[Literal["global", "rolling"]]
    stream: Sourced[Resolution]  # what the car's driver asks for


@dataclass(frozen=True)
class Resolution:
    width: int  # pixels
    height: int  # pixels
    fps: float  # Hz


@dataclass(frozen=True)
class Visuals:
    # The wheels' radius in the cosmetic wheel meshes, which are scaled to the
    # tyre.
    mesh_wheel_radius: Sourced[float]  # m
