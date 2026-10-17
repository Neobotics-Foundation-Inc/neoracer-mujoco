"""
From a car's parts (parts.Car) to what the simulators run: the MuJoCo
model's numbers (ModelParams, model) and the settings beside the model
(Settings, settings). Pure functions, no MuJoCo.

Numbers that the model holds are kept to its precision, 6 significant
digits (precise), so the code and the XML hold the same floats.

Frame: +X forward, +Y left, +Z up; angles in rad.
"""

import math
from dataclasses import dataclass

import numpy as np

from .. import encoder, lidar
from . import ackermann
from .parts import Car, DcMotorRating, ServoRating, SteeringServo, Vec3

G = 9.81  # m/s^2, MuJoCo's gravity
KGF_CM = 9.80665 / 100  # N·m in a kgf·cm, at standard gravity, as servos are rated

type Nominal = tuple[float, float, float]  # V, N·m, rad/s: a dcmotor's nominal
type Polynomial = tuple[float, float, float, float, float]  # a0..a4


def precise(x: float) -> float:
    """x to the model's precision, 6 significant digits."""
    return float(f"{x:.6g}")


def rad_s(rpm: float) -> float:
    return rpm * 2.0 * math.pi / 60.0


@dataclass(frozen=True)
class DcMotorParams:
    nominal: Nominal  # at the shaft
    gear: float  # shaft turns per turn of what it drives
    ctrl_limit: float  # V, |ctrl| at most


@dataclass(frozen=True)
class ServoParams:
    nominal: Nominal  # at the steering input, through the linkage
    kp: float  # V/rad
    kd: float  # V·s/rad
    voltage_limit: float  # V, the loop's clamp
    lock: float  # rad of steering input, |ctrl| at most


@dataclass(frozen=True)
class ModelParams:
    """The car model's numbers. Positions are in the car's body frame."""

    spawn_height: float  # m, of the body frame above the floor
    chassis_mass: float  # kg
    chassis_inertia: Vec3  # kg·m^2
    com_height: float  # m
    imu_position: Vec3  # m
    lidar_position: Vec3  # m
    half_wheelbase: float  # m
    half_track: float  # m
    wheel_radius: float  # m
    tyre_half_width: float  # m
    wheel_mass: float  # kg
    wheel_friction: float
    wheel_damping: float  # N·m·s/rad
    wheel_mesh_scale: float
    input_body_mass: float  # kg
    spring_stiffness: float  # N/m
    spring_damping: float  # N·s/m
    spring_ref: float  # m, the spring's rest position
    droop: float  # m
    jounce: float  # m
    steer_input_range: float  # rad
    wheel_steer_range: float  # rad
    motor: DcMotorParams
    servo: ServoParams
    ackermann_left: Polynomial  # the left wheel's angle from the steering input


@dataclass(frozen=True)
class Settings:
    """What the simulators need beside the model."""

    lidar: lidar.LidarParams
    encoder: encoder.EncoderParams
    wheel_radius: float  # m, that the encoder's speed is measured at
    steer_lock: float  # rad of steering input, full lock


def nominal(rating: DcMotorRating) -> Nominal:
    return (rating.voltage, rating.stall_torque, rad_s(rating.free_speed_rpm))


def servo_rating(servo: SteeringServo) -> ServoRating:
    """The servo's rating at its supply, interpolated between its datasheet's
    two lines."""
    lo, hi = servo.low, servo.high
    w = (servo.supply - lo.voltage) / (hi.voltage - lo.voltage)

    def lerp(a: float, b: float) -> float:
        return a + w * (b - a)

    return ServoRating(
        servo.supply,
        lerp(lo.stall_torque_kgf_cm, hi.stall_torque_kgf_cm),
        lerp(lo.seconds_per_60_degrees, hi.seconds_per_60_degrees),
    )


def steer_lock(car: Car) -> float:
    """Full lock (rad of steering input), where the inner front wheel reaches
    its lock, to the model's precision."""
    c = car.chassis
    inner = math.radians(car.steering.inner_lock_degrees)
    return precise(ackermann.steer_for_inner(c.wheelbase, c.track, inner))


def linkage_ratio(car: Car) -> float:
    """How far the servo's horn turns per radian of steering input: at full
    lock it is at driver_lock of its half range."""
    s = car.steering
    horn = math.radians(s.driver_lock * car.servo.horn_half_range_degrees)
    return horn / steer_lock(car)


def servo_nominal(car: Car) -> Nominal:
    """The servo at the steering input, through the linkage: its supply, its
    stall torque and its free speed."""
    rating, ratio = servo_rating(car.servo), linkage_ratio(car)
    free = math.radians(60.0) / rating.seconds_per_60_degrees  # rad/s at the horn
    return (rating.voltage, rating.stall_torque_kgf_cm * KGF_CM * ratio, free / ratio)


def ackermann_left(car: Car) -> Polynomial:
    """The left wheel's angle as a 4th-order polynomial of the steering input,
    a least-squares fit to ideal Ackermann steering over its range. The right
    wheel's is its mirror image, (-a0, a1, -a2, a3, -a4)."""
    c, r = car.chassis, car.steering.fit_range
    # An even count of points leaves out 0, where left_wheel divides by 0.
    x = np.linspace(-r, r, 400)
    y = [ackermann.left_wheel(c.wheelbase, c.track, s) for s in x]
    a0, a1, a2, a3, a4 = (float(a) for a in np.polynomial.polynomial.polyfit(x, y, 4))
    return (a0, a1, a2, a3, a4)


def wheel_steer_range(car: Car) -> float:
    """How far (rad) each front wheel's steer joint turns: past the inner
    wheel's angle at the edge of the steering input's range."""
    c, s = car.chassis, car.steering
    return (
        ackermann.left_wheel(c.wheelbase, c.track, s.fit_range) + s.wheel_range_margin
    )


def sprung_mass(car: Car) -> float:
    """The mass on the springs (kg): the car less its four wheels."""
    return car.chassis.mass - 4 * car.wheel.mass


def chassis_mass(car: Car) -> float:
    """The chassis body's mass (kg): the sprung mass less the steering input's
    body."""
    return sprung_mass(car) - car.steering.input_body_mass


def chassis_inertia(car: Car) -> Vec3:
    """The chassis' principal inertia (kg·m^2): its reference's, scaled to its
    mass."""
    ref = car.chassis.inertia
    k = chassis_mass(car) / ref.mass
    return (ref.inertia[0] * k, ref.inertia[1] * k, ref.inertia[2] * k)


def spring_ref(car: Car) -> float:
    """The springs' rest position (m): each holds a quarter of the sprung
    weight with its joint at 0."""
    return -sprung_mass(car) * G / 4 / car.suspension.stiffness


def _precise3(v: Vec3) -> Vec3:
    return (precise(v[0]), precise(v[1]), precise(v[2]))


def model(car: Car) -> ModelParams:
    c, w, s = car.chassis, car.wheel, car.suspension
    motor, servo = car.motor, car.servo
    poly = ackermann_left(car)
    v_nominal, stall, free = nominal(motor.rating)
    s_nominal = servo_nominal(car)
    return ModelParams(
        # 1 mm above its resting height, the axles at the tyre radius
        spawn_height=precise(w.radius + 0.001),
        chassis_mass=precise(chassis_mass(car)),
        chassis_inertia=_precise3(chassis_inertia(car)),
        com_height=precise(c.com_height),
        imu_position=_precise3(car.imu.position),
        lidar_position=_precise3(car.lidar.position),
        half_wheelbase=precise(c.wheelbase / 2),
        half_track=precise(c.track / 2),
        wheel_radius=precise(w.radius),
        tyre_half_width=precise(w.width / 2),
        wheel_mass=precise(w.mass),
        wheel_friction=precise(w.friction),
        wheel_damping=precise(w.drive_damping),
        wheel_mesh_scale=precise(w.radius / car.visuals.mesh_wheel_radius),
        input_body_mass=precise(car.steering.input_body_mass),
        spring_stiffness=precise(s.stiffness),
        spring_damping=precise(s.damping),
        spring_ref=precise(spring_ref(car)),
        droop=precise(s.droop),
        jounce=precise(s.jounce),
        steer_input_range=precise(car.steering.fit_range),
        wheel_steer_range=precise(wheel_steer_range(car)),
        motor=DcMotorParams(
            nominal=(precise(v_nominal), precise(stall), precise(free)),
            gear=precise(motor.gear_ratio),
            ctrl_limit=precise(car.battery.voltage),
        ),
        servo=ServoParams(
            nominal=(
                precise(s_nominal[0]),
                precise(s_nominal[1]),
                precise(s_nominal[2]),
            ),
            kp=precise(servo.kp),
            kd=precise(servo.kd),
            voltage_limit=precise(servo.supply),
            lock=steer_lock(car),
        ),
        ackermann_left=(
            precise(poly[0]),
            precise(poly[1]),
            precise(poly[2]),
            precise(poly[3]),
            precise(poly[4]),
        ),
    )


def settings(car: Car) -> Settings:
    return Settings(
        lidar=lidar.of(car.lidar),
        encoder=encoder.of(car.encoder),
        wheel_radius=precise(car.wheel.radius),
        steer_lock=steer_lock(car),
    )
