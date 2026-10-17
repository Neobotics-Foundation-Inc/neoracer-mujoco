"""
The parts of a car with Ackermann steering, one drive motor and open
differentials, as datasheets and measurements give them: the types that a
car's specification is made of. neoracer.py is the NeoRacer's.

A part holds what was read or measured, in SI units unless a field's name
says otherwise, not the simulator's parameters, which derive.py computes.
No field has a default: a car states every value, so its file can say where
each one comes from.

Frame: +X forward, +Y left, +Z up. A position is in the car's body frame:
its origin at the middle of the wheelbase, on the axle line.
"""

import math
from dataclasses import dataclass
from typing import Literal

from . import ackermann

type Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class InertiaReference:
    """A body whose inertia is known, to scale to another of the same shape:
    its principal moments (kg·m^2, about its centre of mass) and its mass."""

    inertia: Vec3  # kg·m^2
    mass: float  # kg


@dataclass(frozen=True)
class Chassis:
    wheelbase: float  # m, between the front and rear axles
    track: float  # m, between the left and right wheels' centres
    mass: float  # kg, the whole car, wheels included
    com_height: float  # m, the chassis' centre of mass above the axles
    inertia: InertiaReference  # scaled to the chassis' mass
    # The box that collides: the body less the wheels.
    box_center: Vec3  # m
    box_half_size: Vec3  # m


@dataclass(frozen=True)
class Wheel:
    radius: float  # m, the tyre's
    width: float  # m, the tyre's
    mass: float  # kg, each
    friction: float  # the tyre's sliding friction on the floor
    # Losses from the motor to the tyres, as damping on each wheel's spin.
    drive_damping: float  # N·m·s/rad


@dataclass(frozen=True)
class Suspension:
    stiffness: float  # N/m, each wheel's spring
    damping: float  # N·s/m, each wheel's damper
    droop: float  # m, the wheel's travel down from rest
    jounce: float  # m, the wheel's travel up from rest


@dataclass(frozen=True)
class Battery:
    voltage: float  # V


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
class DriveMotor:
    rating: DcMotorRating
    gear_ratio: float  # motor turns per wheel turn


@dataclass(frozen=True)
class Esc:
    # What the motor does at neutral throttle: short its windings (brake) or
    # leave them open (coast).
    neutral: Literal["brake", "coast"]


@dataclass(frozen=True)
class ServoRating:
    """A hobby servo's datasheet line, in the units it is rated in."""

    voltage: float  # V
    stall_torque_kgf_cm: float  # kgf·cm
    seconds_per_60_degrees: float  # s, with no load


@dataclass(frozen=True)
class SteeringServo:
    # Two datasheet lines; the rating at supply is interpolated between them.
    low: ServoRating
    high: ServoRating
    supply: float  # V
    # How far the horn turns from the centre over the firmware's pulse range.
    horn_half_range_degrees: float
    # The servo's own position loop. It is not on any datasheet: it is the
    # model's, so that the servo settles as a servo does.
    kp: float  # V/rad
    kd: float  # V·s/rad


@dataclass(frozen=True)
class Steering:
    # The inner front wheel's angle at full lock.
    inner_lock_degrees: float
    # The share of the horn's half range that the car's driver commands at
    # full lock.
    driver_lock: float
    # The steering input's range (rad): the model's Ackermann constraints fit
    # over it, so it must be wider than full lock.
    fit_range: float
    # How far (rad) past the inner wheel's angle at fit_range the front
    # wheels' steer joints reach, so their limits never oppose the
    # constraints.
    wheel_range_margin: float
    # The model's virtual body that the steering input turns, between the
    # servo and the Ackermann constraints. Its mass comes off the chassis'.
    input_body_mass: float  # kg


@dataclass(frozen=True)
class SpeedController:
    """The firmware's speed loop: a PID on the encoder's speed, with a
    feed-forward, driving the ESC's pulse. The simulator does not model it
    yet: a throttle is a share of the battery's voltage."""

    period: float  # s
    kp: float  # us of pulse per m/s of error
    ki: float
    kd: float
    speed_filter_weight: float  # of the new speed in the loop's low-pass
    feed_forward: float  # us of pulse per m/s commanded
    deadband: float  # us, the smallest pulse past neutral that it sends
    full_pulse: float  # us past neutral, full throttle
    max_speed: float  # m/s


@dataclass(frozen=True)
class MotorEncoder:
    pulses_per_turn: int  # a turn of the motor
    max_speed_rpm: float  # rpm, its rating
    # Every report_period the firmware turns the count over the period into
    # the tyre's surface speed and low-pass filters it with filter_weight on
    # the new period.
    report_period: float  # s
    filter_weight: float


@dataclass(frozen=True)
class Imu:
    position: Vec3  # m
    rate: float  # Hz, of the readings the car publishes
    accel_range_g: float  # g, the accelerometer's full scale
    gyro_range_dps: float  # degrees/s, the gyroscope's full scale
    magnetometer_range_gauss: float  # gauss


@dataclass(frozen=True)
class Lidar:
    position: Vec3  # m, the centre of the scan
    scan_rate: float  # Hz, turns a second
    resolution_degrees: float  # between two samples
    field_degrees: float  # of bearings it sees, centered on the front
    range_bright: float  # m at least, on a target of 90% reflectivity
    range_dark: float  # m at least, on a target of 10% reflectivity
    accuracy: float  # m


@dataclass(frozen=True)
class Camera:
    position: Vec3  # m, the lens
    width: int  # pixels, the sensor's
    height: int  # pixels
    fps: float  # Hz, the sensor's most
    fov_degrees: float
    focal_length: float  # m
    shutter: Literal["global", "rolling"]
    # The stream the car's driver asks for.
    stream_width: int  # pixels
    stream_height: int  # pixels
    stream_fps: float  # Hz


@dataclass(frozen=True)
class Visuals:
    # The wheels' radius in the cosmetic wheel meshes, which are scaled to the
    # tyre.
    mesh_wheel_radius: float  # m


@dataclass(frozen=True)
class Car:
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
        lock = ackermann.steer_for_inner(
            self.chassis.wheelbase,
            self.chassis.track,
            math.radians(self.steering.inner_lock_degrees),
        )
        if not lock < self.steering.fit_range:
            raise ValueError(
                f"full lock, {lock} rad of steering input, is not inside"
                f" fit_range, {self.steering.fit_range} rad"
            )
