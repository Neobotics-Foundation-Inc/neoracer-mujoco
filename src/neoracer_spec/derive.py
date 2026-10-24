"""
From a car's specification (spec.Spec) to what the simulators run
(DerivedParams): the MuJoCo model's numbers (ModelParams) and the settings
beside the model, the LiDAR's scan and the encoder's filter. Pure functions
of the values, without their provenance; no MuJoCo.

Numbers that the model holds are kept to its precision, 6 significant
digits (parts.precise), so the code and the XML hold the same floats.

Frame: +X forward, +Y left, +Z up; angles in rad.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import ackermann, parts
from .parts import Nominal, precise, precise3
from .spec import (
    DcMotorRating,
    ServoRating,
    Spec,
    SteeringServo,
    Vec3,
)


@dataclass(frozen=True)
class DerivedParams:
    """Everything the simulators need, derived from a Spec."""

    model: ModelParams
    lidar: LidarParams
    encoder: EncoderParams


def derive(spec: Spec) -> DerivedParams:
    return DerivedParams(
        model=model(spec), lidar=lidar_params(spec), encoder=encoder_params(spec)
    )


# --- the settings beside the model --------------------------------------------


@dataclass(frozen=True)
class LidarParams:
    """The LiDAR's scan: lidar_params(car) for a NeoRacer,
    derive_v2.lidar_params(car) for a RACECAR Neo V2."""

    slots: int  # slots a turn
    scan_rate: float  # Hz, turns a second
    max_range: float  # m, its range on a bright target
    field: float  # rad of bearings it sees, centered on the front

    def __post_init__(self) -> None:
        if self.slots < 1:
            raise ValueError(f"slots must be >= 1, not {self.slots}")
        if not self.scan_rate > 0.0:
            raise ValueError(f"scan_rate must be > 0 Hz, not {self.scan_rate}")
        if not self.max_range > 0.0:
            raise ValueError(f"max_range must be > 0 m, not {self.max_range}")
        if not 0.0 < self.field <= 2 * math.pi:
            raise ValueError(f"field must be in (0, 2 pi] rad, not {self.field}")


def lidar_params(spec: Spec) -> LidarParams:
    """The scan of the LiDAR: a slot every sample, out to its range on a bright
    target."""
    lidar = spec.lidar
    return LidarParams(
        slots=round(360 / lidar.resolution_degrees.value),
        scan_rate=lidar.scan_rate.value,
        max_range=lidar.range_bright.value,
        field=math.radians(lidar.field_degrees.value),
    )


@dataclass(frozen=True)
class EncoderParams:
    """How the firmware measures the speed from the motor encoder."""

    window: float  # s, the firmware's control period
    radius: float  # m, the tyre's, that it measures the speed at
    alpha: float  # the weight of the new window in the filter

    def __post_init__(self) -> None:
        if not self.window > 0.0:
            raise ValueError(f"window must be > 0 s, not {self.window}")
        if not self.radius > 0.0:
            raise ValueError(f"radius must be > 0 m, not {self.radius}")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], not {self.alpha}")


def encoder_params(spec: Spec) -> EncoderParams:
    """The firmware's encoder, which measures the speed at the car's tyre."""
    return EncoderParams(
        window=spec.encoder.report_period.value,
        radius=spec.wheel.radius.value,
        alpha=spec.encoder.filter_weight.value,
    )


# --- the model's numbers ------------------------------------------------------


@dataclass(frozen=True)
class ModelParams:
    """The NeoRacer's model: the parts that both cars' models have, and its
    own."""

    common: parts.Model
    motor: DcMotorParams
    wheel_mesh_scale: float  # of the cosmetic wheel meshes, to the tyre


@dataclass(frozen=True)
class DcMotorParams:
    nominal: Nominal  # at the shaft
    gear: float  # shaft turns per turn of what it drives
    ctrl_limit: float  # V, |ctrl| at most


def model(spec: Spec) -> ModelParams:
    c, w, s = spec.chassis, spec.wheel, spec.suspension
    box = c.collision_box.value
    fit = spec.steering.fit_range.value
    friction = precise(w.friction.value)
    k, d, ref = precise(s.stiffness.value), precise(s.damping.value), spring_ref(spec)
    return ModelParams(
        common=parts.Model(
            spawn_height=parts.spawn_height(w.radius.value),
            inertial=parts.Inertial(
                mass=precise(chassis_mass(spec)),
                pos=(0.0, 0.0, precise(c.com_height.value)),
                diaginertia=precise3(chassis_inertia(spec)),
            ),
            boxes=(
                parts.CollisionBox(
                    "chassis", precise3(box.center), precise3(box.half_size)
                ),
            ),
            imu_position=precise3(spec.imu.position.value),
            lidar=parts.LidarBeams(
                position=precise3(spec.lidar.position.value),
                yaw=0.0,
                scan_rate=precise(spec.lidar.scan_rate.value),
                field=precise(lidar_params(spec).field),
            ),
            wheels=parts.Wheels(
                half_wheelbase=precise(c.wheelbase.value / 2),
                half_track=precise(c.track.value / 2),
                radius=precise(w.radius.value),
                half_width=precise(w.width.value / 2),
                mass=precise(w.mass.value),
                damping=precise(w.drive_damping.value),
                friction=(friction, friction),
            ),
            springs=parts.Springs(
                range=(-precise(s.droop.value), precise(s.jounce.value)),
                stiffness=(k, k),
                damping=(d, d),
                springref=(precise(ref), precise(ref)),
            ),
            # The steering input is the virtual wheel's angle.
            steering=parts.steering_input(
                c.wheelbase.value,
                c.track.value,
                stops=(-fit, fit),
                virtual=lambda x: x,
                armature=spec.steering.armature.value,
            ),
            servo=parts.Servo(
                nominal=precise3(servo_nominal(spec)),
                kp=precise(spec.servo.kp.value),
                kd=precise(spec.servo.kd.value),
                voltage_limit=precise(spec.servo.supply.value),
                lock=steer_lock(spec),
            ),
        ),
        motor=DcMotorParams(
            nominal=precise3(nominal(spec.motor.rating.value)),
            gear=precise(spec.motor.gear_ratio.value),
            ctrl_limit=precise(spec.battery.voltage.value),
        ),
        wheel_mesh_scale=precise(w.radius.value / spec.visuals.mesh_wheel_radius.value),
    )


def nominal(rating: DcMotorRating) -> Nominal:
    return (rating.voltage, rating.stall_torque, rad_s(rating.free_speed_rpm))


def rad_s(rpm: float) -> float:
    return rpm * 2.0 * math.pi / 60.0


KGF_CM = 9.80665 / 100  # N·m in a kgf·cm, at standard gravity, as servos are rated


def servo_nominal(spec: Spec) -> Nominal:
    """The servo at the steering input, through the linkage: its supply, its
    stall torque and its free speed."""
    rating, ratio = servo_rating(spec.servo), linkage_ratio(spec)
    free = math.radians(60.0) / rating.seconds_per_60_degrees  # rad/s at the horn
    return (rating.voltage, rating.stall_torque_kgf_cm * KGF_CM * ratio, free / ratio)


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


def linkage_ratio(spec: Spec) -> float:
    """How far the servo's horn turns per radian of steering input: at full
    lock it is at driver_lock of its half range."""
    lock = spec.steering.driver_lock.value * spec.servo.horn_half_range_degrees.value
    return math.radians(lock) / steer_lock(spec)


def steer_lock(spec: Spec) -> float:
    """Full lock (rad of steering input), where the inner front wheel reaches
    its lock, to the model's precision."""
    c = spec.chassis
    inner = math.radians(spec.steering.inner_lock_degrees.value)
    return precise(ackermann.steer_for_inner(c.wheelbase.value, c.track.value, inner))


def chassis_mass(spec: Spec) -> float:
    return parts.chassis_mass(spec.chassis.mass.value, spec.wheel.mass.value)


def sprung_mass(spec: Spec) -> float:
    """The mass on the springs (kg): the car less its four wheels."""
    return spec.chassis.mass.value - 4 * spec.wheel.mass.value


def chassis_inertia(spec: Spec) -> Vec3:
    """The chassis' principal inertia (kg·m^2): its reference's, scaled to its
    mass."""
    ref = spec.chassis.inertia.value
    k = chassis_mass(spec) / ref.mass
    return (ref.inertia[0] * k, ref.inertia[1] * k, ref.inertia[2] * k)


def spring_ref(spec: Spec) -> float:
    """The springs' rest position (m): each holds a quarter of the sprung
    mass."""
    return parts.spring_ref(sprung_mass(spec) / 4, spec.suspension.stiffness.value)
