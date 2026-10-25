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

type DriveCommand = Literal["speed", "throttle"]
"""What a car's computer sets with a drive command: a speed, which the car
holds, or a throttle, the share of the battery's voltage that the motor
gets."""


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
    command: Command
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
    wheelbase: Sourced[float]
    """The distance between the front and rear axles."""
    track: Sourced[float]
    """The distance between the left and right wheels' centers."""
    mass: Sourced[float]
    """The whole car's mass, wheels included."""
    com_height: Sourced[float]
    """The height of the chassis' center of mass above the axles."""
    inertia: Sourced[InertiaReference]
    """Scaled to the chassis' mass."""
    outline: Sourced[Prism]
    """The body, without the wheels: what it collides with, and what the
    other cars' LiDARs see of it."""


@dataclass(frozen=True)
class InertiaReference:
    """A body whose inertia is known, to scale to another of the same shape:
    its principal moments (kg·m²) about its center of mass, and its mass."""

    inertia: Vec3
    mass: float


@dataclass(frozen=True)
class Prism:
    """An upright prism: a convex polygon seen from above, its corners (x, y)
    counterclockwise, from the height bottom up to top."""

    corners: tuple[tuple[float, float], ...]
    bottom: float
    top: float

    def __post_init__(self) -> None:
        n = len(self.corners)
        if n < 3:
            raise ValueError(f"a polygon has at least 3 corners, not {n}")
        # Convex and counterclockwise: every other corner strictly left of
        # each edge, which a polygon whose edges cross fails.
        for i in range(n):
            (ax, ay), (bx, by) = self.corners[i], self.corners[(i + 1) % n]
            for k, (px, py) in enumerate(self.corners):
                if k in (i, (i + 1) % n):
                    continue
                if (bx - ax) * (py - ay) - (by - ay) * (px - ax) <= 0:
                    raise ValueError(
                        f"corner {k} is not left of the edge from corner {i}: the"
                        " corners must run counterclockwise round a convex polygon"
                    )
        if not self.bottom < self.top:
            raise ValueError(f"bottom {self.bottom} is not below top {self.top}")


@dataclass(frozen=True)
class Wheel:
    radius: Sourced[float]
    """The tire's radius."""
    width: Sourced[float]
    """The tire's width."""
    mass: Sourced[float]
    """Each wheel's mass."""
    friction: Sourced[float]
    """The tire's sliding friction on the floor."""
    drive_damping: Sourced[float]
    """The losses from the motor to the tires, as damping on each wheel's spin
    (N·m·s/rad)."""


@dataclass(frozen=True)
class Suspension:
    stiffness: Sourced[float]
    """Each wheel's spring (N/m)."""
    damping: Sourced[float]
    """Each wheel's damper (N·s/m)."""
    droop: Sourced[float]
    """The wheel's travel down from rest."""
    jounce: Sourced[float]
    """The wheel's travel up from rest."""


@dataclass(frozen=True)
class Battery:
    voltage: Sourced[float]


@dataclass(frozen=True)
class DriveMotor:
    rating: Sourced[DcMotorRating]
    gear_ratio: Sourced[float]
    """Motor turns per wheel turn."""


@dataclass(frozen=True)
class DcMotorRating:
    """A DC motor's datasheet line: at voltage, its torque from rest and its
    speed with no load."""

    voltage: float
    stall_torque: float
    free_speed_rpm: float


def rewound(rating: DcMotorRating, free_speed_rpm: float) -> DcMotorRating:
    """The motor of rating rewound in the same frame to free_speed_rpm at the
    same voltage. With the same copper, turns n give a back-EMF constant
    proportional to n and a resistance to n^2, so the free speed and the stall
    torque both scale as 1 / n."""
    k = free_speed_rpm / rating.free_speed_rpm
    return DcMotorRating(rating.voltage, rating.stall_torque * k, free_speed_rpm)


@dataclass(frozen=True)
class Esc:
    neutral: Sourced[Literal["brake", "coast"]]
    """What the motor does at neutral throttle: short its windings (brake) or
    leave them open (coast). The model only brakes, so derive refuses coast."""
    full_throttle_pulse: Sourced[float]
    """The pulse past neutral at which the motor gets the battery's whole
    voltage (us). Under it, the model gives the motor that share of the
    voltage."""


@dataclass(frozen=True)
class SteeringServo:
    """The steering servo. Its rating at supply is interpolated between two
    datasheet lines, low and high."""

    low: Sourced[ServoRating]
    high: Sourced[ServoRating]
    supply: Sourced[float]
    horn_half_range_degrees: Sourced[float]
    """How far the horn turns from the center over the firmware's pulse
    range."""
    kp: Sourced[float]
    """The proportional gain of the servo's own position loop, the model's
    (V/rad)."""
    kd: Sourced[float]
    """The derivative gain of that loop (V·s/rad)."""
    latency: Sourced[float]
    """How long a command takes from the car's board to the servo's loop: the
    firmware's control loop and pulse, and the servo's own dead time."""
    steps_at_rest: Sourced[ServoSteps]
    """The car's steering steps at rest, which latency is fitted to."""


@dataclass(frozen=True)
class ServoSteps:
    """Steps between the center and the driver's full lock, the car at rest
    on the floor, each timed from the command written to the car's board:
    when the chassis first moves, and when each swing ends."""

    first_motion: float
    center_to_lock: float
    lock_to_lock: float
    lock_to_center: float


@dataclass(frozen=True)
class ServoRating:
    """A hobby servo's datasheet line, in the units it is rated in."""

    voltage: float
    stall_torque_kgf_cm: float
    seconds_per_60_degrees: float
    """With no load."""


@dataclass(frozen=True)
class Steering:
    inner_lock_degrees: Sourced[float]
    """The inner front wheel's angle at full lock."""
    driver_lock: Sourced[float]
    """The share of the horn's half range that the car's driver commands at
    full lock."""
    fit_range: Sourced[float]
    """The steering input's range. The model's Ackermann constraints fit over
    it, so it must be wider than full lock."""
    armature: Sourced[float]
    """The inertia added at the steering input and at each front wheel's steer
    joint (kg·m²): the servo's rotor, seen through its gears, turns with
    them."""


@dataclass(frozen=True)
class Command:
    """What the car's computer sends the firmware to drive: a drive command
    and a steering angle."""

    drive: Sourced[DriveCommand]
    """What a drive command sets. On the NeoRacer, a speed, which the speed
    loop (SpeedController) holds."""
    timeout: Sourced[float]
    """The time without a command after which the firmware stops the car and
    centers the steering."""


@dataclass(frozen=True)
class SpeedController:
    """The firmware's speed loop: every period, a PID on the encoder's speed
    and a feed-forward on the commanded speed set the ESC's pulse. A pulse is
    in microseconds past neutral."""

    period: Sourced[float]
    kp: Sourced[float]
    """Microseconds of pulse per m/s of error."""
    ki: Sourced[float]
    """Microseconds of pulse per meter of the error's integral."""
    kd: Sourced[float]
    """Microseconds of pulse per m/s² of the error's change."""
    integral_limit: Sourced[float]
    """The bound on the error's integral (m)."""
    error_deadband: Sourced[float]
    """The error under which the loop reads none (m/s)."""
    stop_speed: Sourced[float]
    """The commanded speed under which the loop sends neutral and starts over."""
    speed_filter_weight: Sourced[float]
    """The weight of the new speed in the loop's low-pass filter."""
    feed_forward: Sourced[float]
    """Microseconds of pulse per m/s commanded."""
    deadband: Sourced[float]
    """The feed-forward's pulse at a commanded speed of 0 (us): the least
    that moves the car."""
    feed_forward_limit: Sourced[float]
    """The most pulse that the feed-forward sends (us)."""
    pulse_limit: Sourced[float]
    """The most pulse that the loop sends either way (us)."""
    max_speed: Sourced[float]
    """The speed that a program's full speed commands."""


@dataclass(frozen=True)
class MotorEncoder:
    """Every report_period the firmware turns the count over the period into
    the tire's surface speed, and low-pass filters it with filter_weight on
    the new period."""

    pulses_per_turn: Sourced[int]
    """Pulses in a turn of the motor."""
    max_speed_rpm: Sourced[float]
    """Its rated speed."""
    report_period: Sourced[float]
    filter_weight: Sourced[float]


@dataclass(frozen=True)
class Imu:
    position: Sourced[Vec3]
    rate: Sourced[float]
    """The rate of the readings that the car publishes."""
    accel_range_g: Sourced[float]
    """The accelerometer's full scale."""
    gyro_range_dps: Sourced[float]
    """The gyroscope's full scale."""
    magnetometer_range_gauss: Sourced[float]
    """The magnetometer's full scale."""


@dataclass(frozen=True)
class Lidar:
    position: Sourced[Vec3]
    """The center of the scan."""
    scan_rate: Sourced[float]
    """Turns a second."""
    resolution_degrees: Sourced[float]
    """The angle between two samples."""
    field_degrees: Sourced[float]
    """The bearings it sees, centered on the front."""
    range_bright: Sourced[float]
    """The least range it reaches on a target of 90% reflectivity."""
    range_dark: Sourced[float]
    """The least range it reaches on a target of 10% reflectivity."""
    accuracy: Sourced[float]


@dataclass(frozen=True)
class Camera:
    position: Sourced[Vec3]
    """The lens's position."""
    sensor: Sourced[Resolution]
    """The sensor's highest resolution."""
    fov_degrees: Sourced[float]
    focal_length: Sourced[float]
    shutter: Sourced[Literal["global", "rolling"]]
    stream: Sourced[Resolution]
    """The resolution that the car's driver asks for."""


@dataclass(frozen=True)
class Resolution:
    width: int
    """The width in pixels."""
    height: int
    """The height in pixels."""
    fps: float


@dataclass(frozen=True)
class Visuals:
    mesh_wheel_radius: Sourced[float]
    """The wheels' radius in the cosmetic wheel meshes, which are scaled to
    the tire."""
