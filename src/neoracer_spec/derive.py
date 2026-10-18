"""
From a car's specification (spec.Spec) to what the simulators run
(DerivedParams): the MuJoCo model's numbers (ModelParams) and the settings
beside the model, the LiDAR's scan and the encoder's filter. Pure functions
of the values, without their provenance; no MuJoCo.

Numbers that the model holds are kept to its precision, 6 significant
digits (precise), so the code and the XML hold the same floats.

Frame: +X forward, +Y left, +Z up; angles in rad.
"""

import math
from dataclasses import dataclass

import numpy as np

from . import ackermann
from .neoracer import NEORACER
from .spec import (
    DcMotorRating,
    Lidar,
    MotorEncoder,
    ServoRating,
    Spec,
    SteeringServo,
    Vec3,
)

G = 9.81  # m/s^2, MuJoCo's gravity
KGF_CM = 9.80665 / 100  # N·m in a kgf·cm, at standard gravity, as servos are rated

type Nominal = tuple[float, float, float]  # V, N·m, rad/s: a dcmotor's nominal
type Polynomial = tuple[float, float, float, float, float]  # a0..a4


def precise(x: float) -> float:
    """x to the model's precision, 6 significant digits."""
    return float(f"{x:.6g}")


def rad_s(rpm: float) -> float:
    return rpm * 2.0 * math.pi / 60.0


# --- the settings beside the model -------------------------------------------

_LIDAR = NEORACER.lidar
_ENCODER = NEORACER.encoder


@dataclass(frozen=True)
class LidarParams:
    """The LiDAR's scan. The defaults are the NeoRacer's: lidar_params(NEORACER.lidar)."""

    slots: int = round(360 / _LIDAR.resolution_degrees.value)  # slots a turn
    scan_rate: float = _LIDAR.scan_rate.value  # Hz, turns a second
    max_range: float = _LIDAR.range_bright.value  # m, its range on a bright target
    # rad of bearings it sees, centered on the front
    field: float = math.radians(_LIDAR.field_degrees.value)

    def __post_init__(self) -> None:
        if self.slots < 1:
            raise ValueError(f"slots must be >= 1, not {self.slots}")
        if not self.scan_rate > 0.0:
            raise ValueError(f"scan_rate must be > 0 Hz, not {self.scan_rate}")
        if not self.max_range > 0.0:
            raise ValueError(f"max_range must be > 0 m, not {self.max_range}")
        if not 0.0 < self.field <= 2 * math.pi:
            raise ValueError(f"field must be in (0, 2 pi] rad, not {self.field}")


def lidar_params(lidar: Lidar) -> LidarParams:
    """The scan of the LiDAR: a slot every sample, out to its range on a bright
    target."""
    return LidarParams(
        slots=round(360 / lidar.resolution_degrees.value),
        scan_rate=lidar.scan_rate.value,
        max_range=lidar.range_bright.value,
        field=math.radians(lidar.field_degrees.value),
    )


@dataclass(frozen=True)
class EncoderParams:
    """How the firmware measures the speed from the motor encoder. The
    defaults are the NeoRacer's: encoder_params(NEORACER.encoder)."""

    window: float = _ENCODER.report_period.value  # s, the firmware's control period
    alpha: float = (
        _ENCODER.filter_weight.value
    )  # the weight of the new window in the filter

    def __post_init__(self) -> None:
        if not self.window > 0.0:
            raise ValueError(f"window must be > 0 s, not {self.window}")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], not {self.alpha}")


def encoder_params(encoder: MotorEncoder) -> EncoderParams:
    return EncoderParams(
        window=encoder.report_period.value, alpha=encoder.filter_weight.value
    )


# --- the model's numbers -----------------------------------------------------


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
    box_center: Vec3  # m, the chassis' collision box
    box_half_size: Vec3  # m
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
    steer_armature: float  # kg·m²
    motor: DcMotorParams
    servo: ServoParams
    ackermann_left: Polynomial  # the left wheel's angle from the steering input


@dataclass(frozen=True)
class DerivedParams:
    """Everything the simulators need, derived from a Spec."""

    model: ModelParams
    lidar: LidarParams
    encoder: EncoderParams
    wheel_radius: float  # m, that the encoder's speed is measured at
    steer_lock: float  # rad of steering input, full lock


def nominal(rating: DcMotorRating) -> Nominal:
    return (rating.voltage, rating.stall_torque, rad_s(rating.free_speed_rpm))


def servo_rating(servo: SteeringServo) -> ServoRating:
    """The servo's rating at its supply, interpolated between its datasheet's
    two lines."""
    lo, hi, supply = servo.low.value, servo.high.value, servo.supply.value
    w = (supply - lo.voltage) / (hi.voltage - lo.voltage)

    def lerp(a: float, b: float) -> float:
        return a + w * (b - a)

    return ServoRating(
        supply,
        lerp(lo.stall_torque_kgf_cm, hi.stall_torque_kgf_cm),
        lerp(lo.seconds_per_60_degrees, hi.seconds_per_60_degrees),
    )


def steer_lock(spec: Spec) -> float:
    """Full lock (rad of steering input), where the inner front wheel reaches
    its lock, to the model's precision."""
    c = spec.chassis
    inner = math.radians(spec.steering.inner_lock_degrees.value)
    return precise(ackermann.steer_for_inner(c.wheelbase.value, c.track.value, inner))


def linkage_ratio(spec: Spec) -> float:
    """How far the servo's horn turns per radian of steering input: at full
    lock it is at driver_lock of its half range."""
    lock = spec.steering.driver_lock.value * spec.servo.horn_half_range_degrees.value
    return math.radians(lock) / steer_lock(spec)


def servo_nominal(spec: Spec) -> Nominal:
    """The servo at the steering input, through the linkage: its supply, its
    stall torque and its free speed."""
    rating, ratio = servo_rating(spec.servo), linkage_ratio(spec)
    free = math.radians(60.0) / rating.seconds_per_60_degrees  # rad/s at the horn
    return (rating.voltage, rating.stall_torque_kgf_cm * KGF_CM * ratio, free / ratio)


def ackermann_left(spec: Spec) -> Polynomial:
    """The left wheel's angle as a 4th-order polynomial of the steering input,
    a least-squares fit to ideal Ackermann steering over its range. The right
    wheel's is its mirror image, (-a0, a1, -a2, a3, -a4)."""
    l, w = spec.chassis.wheelbase.value, spec.chassis.track.value
    r = spec.steering.fit_range.value
    # An even count of points leaves out 0, where left_wheel divides by 0.
    x = np.linspace(-r, r, 400)
    y = [ackermann.left_wheel(l, w, s) for s in x]
    a0, a1, a2, a3, a4 = (float(a) for a in np.polynomial.polynomial.polyfit(x, y, 4))
    return (a0, a1, a2, a3, a4)


def wheel_steer_range(spec: Spec) -> float:
    """How far (rad) each front wheel's steer joint turns: past the inner
    wheel's angle at the edge of the steering input's range."""
    l, w = spec.chassis.wheelbase.value, spec.chassis.track.value
    s = spec.steering
    return ackermann.left_wheel(l, w, s.fit_range.value) + s.wheel_range_margin.value


def sprung_mass(spec: Spec) -> float:
    """The mass on the springs (kg): the car less its four wheels."""
    return spec.chassis.mass.value - 4 * spec.wheel.mass.value


def chassis_mass(spec: Spec) -> float:
    """The chassis body's mass (kg): the sprung mass less the steering input's
    body."""
    return sprung_mass(spec) - spec.steering.input_body_mass.value


def chassis_inertia(spec: Spec) -> Vec3:
    """The chassis' principal inertia (kg·m^2): its reference's, scaled to its
    mass."""
    ref = spec.chassis.inertia.value
    k = chassis_mass(spec) / ref.mass
    return (ref.inertia[0] * k, ref.inertia[1] * k, ref.inertia[2] * k)


def spring_ref(spec: Spec) -> float:
    """The springs' rest position (m): each holds a quarter of the sprung
    weight with its joint at 0."""
    return -sprung_mass(spec) * G / 4 / spec.suspension.stiffness.value


def _precise3(v: Vec3) -> Vec3:
    return (precise(v[0]), precise(v[1]), precise(v[2]))


def model(spec: Spec) -> ModelParams:
    c, w, s = spec.chassis, spec.wheel, spec.suspension
    motor, servo = spec.motor, spec.servo
    box = c.collision_box.value
    poly = ackermann_left(spec)
    m_nominal, s_nominal = nominal(motor.rating.value), servo_nominal(spec)
    return ModelParams(
        # 1 mm above its resting height, the axles at the tyre radius
        spawn_height=precise(w.radius.value + 0.001),
        chassis_mass=precise(chassis_mass(spec)),
        chassis_inertia=_precise3(chassis_inertia(spec)),
        com_height=precise(c.com_height.value),
        box_center=_precise3(box.center),
        box_half_size=_precise3(box.half_size),
        imu_position=_precise3(spec.imu.position.value),
        lidar_position=_precise3(spec.lidar.position.value),
        half_wheelbase=precise(c.wheelbase.value / 2),
        half_track=precise(c.track.value / 2),
        wheel_radius=precise(w.radius.value),
        tyre_half_width=precise(w.width.value / 2),
        wheel_mass=precise(w.mass.value),
        wheel_friction=precise(w.friction.value),
        wheel_damping=precise(w.drive_damping.value),
        wheel_mesh_scale=precise(w.radius.value / spec.visuals.mesh_wheel_radius.value),
        input_body_mass=precise(spec.steering.input_body_mass.value),
        spring_stiffness=precise(s.stiffness.value),
        spring_damping=precise(s.damping.value),
        spring_ref=precise(spring_ref(spec)),
        droop=precise(s.droop.value),
        jounce=precise(s.jounce.value),
        steer_input_range=precise(spec.steering.fit_range.value),
        wheel_steer_range=precise(wheel_steer_range(spec)),
        steer_armature=precise(spec.steering.armature.value),
        motor=DcMotorParams(
            nominal=_precise3(m_nominal),
            gear=precise(motor.gear_ratio.value),
            ctrl_limit=precise(spec.battery.voltage.value),
        ),
        servo=ServoParams(
            nominal=_precise3(s_nominal),
            kp=precise(servo.kp.value),
            kd=precise(servo.kd.value),
            voltage_limit=precise(servo.supply.value),
            lock=steer_lock(spec),
        ),
        ackermann_left=(
            precise(poly[0]),
            precise(poly[1]),
            precise(poly[2]),
            precise(poly[3]),
            precise(poly[4]),
        ),
    )


def derive(spec: Spec) -> DerivedParams:
    return DerivedParams(
        model=model(spec),
        lidar=lidar_params(spec.lidar),
        encoder=encoder_params(spec.encoder),
        wheel_radius=precise(spec.wheel.radius.value),
        steer_lock=steer_lock(spec),
    )
