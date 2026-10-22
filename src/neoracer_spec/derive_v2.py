"""
From the RACECAR Neo V2's specification (racecar_neo_v2.RacecarNeoV2) to
the numbers of its MuJoCo model, assets/racecar_neo_v2.xml (ModelParams),
and the settings beside the model (DerivedParams). Pure functions of the
values, without their provenance; no MuJoCo.

The car's firmware holds a speed, so its drive is two actuators on the
drivetrain, not the NeoRacer's DC motor. They push at the tyres' contact
with the floor: their gear is the tyre's radius, so a force is in N and a
speed in m/s.
- motor: a force of gain * ctrl.
- motor_engage: a force of -gain * speed * ctrl, with ctrl 1 while the
  throttle is not 0 and 0 when it is.
While the car drives they hold the speed ctrl. At throttle 0 neither pushes
and the car coasts on the drivetrain's friction, its drag brake, and the
wheels' damping, the drag that grows with the speed. The throttle sets
ctrl through a map in the actuators' user numbers (throttle_map), so that
the car holds its measured steady speeds against those drags.

Numbers that the model holds are kept to its precision, 6 significant
digits (derive.precise), so the code and the XML hold the same floats.

Frame: the car body's, as the NeoRacer's: +X forward, +Y left, +Z up, its
origin at the middle of the wheelbase on the axle line. The spec's positions
are in base_link, on the floor below the rear axle; body() moves them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from .derive import G, LidarParams, Polynomial, precise
from .racecar_neo_v2 import RacecarNeoV2
from .spec import Vec3

type Sides = tuple[float, float]  # left, right
type Axles = tuple[float, float]  # front, rear


@dataclass(frozen=True)
class DerivedParams:
    """Everything the simulators need, derived from a RacecarNeoV2."""

    model: ModelParams
    lidar: LidarParams
    encoder: HallEncoderParams
    wheel_radius: float  # m, that the encoder's speed is measured at
    steer_lock: float  # rad of steering input at a command of 1
    start_max_speed: (
        float  # the share of full throttle that racecar_core's speed 1 sends
    )


def derive(spec: RacecarNeoV2) -> DerivedParams:
    m = model(spec)
    return DerivedParams(
        model=m,
        lidar=lidar_params(spec),
        encoder=encoder_params(spec),
        wheel_radius=m.wheel_radius,
        steer_lock=m.steering.lock,
        start_max_speed=spec.drive.start_max_speed.value,
    )


def lidar_params(spec: RacecarNeoV2) -> LidarParams:
    """A slot every sample of a turn, out to its range; it sees all around:
    the car blocks none of its turn, but its antennas are in it."""
    lidar = spec.lidar
    return LidarParams(
        slots=lidar.samples.value,
        scan_rate=lidar.scan_rate.value,
        max_range=lidar.max_range.value,
        field=2 * math.pi,
    )


@dataclass(frozen=True)
class HallEncoderParams:
    """How the drive board measures the speed: from the time between the
    edges of the motor's hall sensors."""

    edge_angle: float  # rad of the drivetrain from one edge to the next
    radius: float  # m, the tyre's, that it measures the speed at
    alpha: float  # the weight of each new edge's speed in its filter
    stop_timeout: float  # s without an edge, after which it reads 0

    def __post_init__(self) -> None:
        if not self.edge_angle > 0.0:
            raise ValueError(f"edge_angle must be > 0 rad, not {self.edge_angle}")
        if not self.radius > 0.0:
            raise ValueError(f"radius must be > 0 m, not {self.radius}")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], not {self.alpha}")
        if not self.stop_timeout > 0.0:
            raise ValueError(f"stop_timeout must be > 0 s, not {self.stop_timeout}")


def encoder_params(spec: RacecarNeoV2) -> HallEncoderParams:
    """The drive board's encoder: an edge every 1/edges of a turn of the
    motor, which turns the drivetrain through its gears."""
    o = spec.odometry
    edges = o.edges_per_motor_turn.value * o.motor_turns_per_wheel_turn.value
    return HallEncoderParams(
        edge_angle=2 * math.pi / edges,
        radius=wheel_radius(spec),
        alpha=o.filter_alpha.value,
        stop_timeout=o.stop_timeout.value,
    )


# --- the model ----------------------------------------------------------------


@dataclass(frozen=True)
class ModelParams:
    """The car model's numbers. Positions are in the car's body frame."""

    spawn_height: float  # m, of the body frame above the floor
    chassis_mass: float  # kg
    chassis_com: Vec3  # m
    chassis_inertia: Vec3  # kg·m²
    boxes: tuple[Box, ...]
    antenna_center: Vec3  # m, the left antenna's; the right one is its mirror
    antenna_radius: float  # m
    antenna_half_length: float  # m
    imu_position: Vec3  # m
    lidar_position: Vec3  # m
    lidar_yaw: float  # rad, of the scan's bearing 0 from +X
    lidar_scan_rate: float  # Hz, turns of its head a second
    half_wheelbase: float  # m
    half_track: float  # m
    wheel_radius: float  # m
    tyre_half_width: float  # m
    wheel_mass: float  # kg
    grip: Axles
    input_body_mass: float  # kg
    spring_stiffness: Axles  # N/m
    spring_damping: Axles  # N·s/m
    spring_ref: Axles  # m
    half_travel: float  # m
    drive: DriveParams
    servo: ServoParams
    steering: SteeringParams


@dataclass(frozen=True)
class Box:
    name: str
    center: Vec3  # m
    half_size: Vec3  # m


INPUT_BODY_MASS = 0.01  # kg, the steering input's light virtual body


def model(spec: RacecarNeoV2) -> ModelParams:
    c, sh, w, s = spec.chassis, spec.shape, spec.wheel, spec.suspension
    drop = sh.payload_drop.value
    top = spec.footprint.antenna_top.value
    bottom = sh.antenna_bottom.value - drop
    payload = tuple(
        replace(b, center=(b.center[0], b.center[1], b.center[2] - drop))
        for b in sh.payload_boxes.value
    )
    refs = spring_refs(spec)
    return ModelParams(
        # 1 mm above its resting height, the axles at the tyre radius
        spawn_height=precise(wheel_radius(spec) + 0.001),
        chassis_mass=precise(chassis_mass(spec)),
        chassis_com=_precise3(chassis_com(spec)),
        chassis_inertia=_precise3(chassis_inertia(spec)),
        boxes=tuple(
            Box(b.name, _precise3(body(spec, b.center)), _precise3(b.half_size))
            for b in sh.chassis_boxes.value + payload
        ),
        antenna_center=_precise3(
            body(spec, (sh.antenna_x.value, sh.antenna_y.value, (top + bottom) / 2))
        ),
        antenna_radius=precise(sh.antenna_diameter.value / 2),
        antenna_half_length=precise((top - bottom) / 2),
        imu_position=_precise3(body(spec, spec.imu.position.value)),
        lidar_position=_precise3(body(spec, spec.lidar.position.value)),
        lidar_yaw=precise(math.radians(spec.lidar.yaw_degrees.value - 180)),
        lidar_scan_rate=precise(spec.lidar.scan_rate.value),
        half_wheelbase=precise(c.wheelbase.value / 2),
        half_track=precise(c.track.value / 2),
        wheel_radius=precise(wheel_radius(spec)),
        tyre_half_width=precise(w.width.value / 2),
        wheel_mass=precise(w.mass.value),
        grip=(precise(w.grip.value.front), precise(w.grip.value.rear)),
        input_body_mass=precise(INPUT_BODY_MASS),
        spring_stiffness=(
            precise(s.stiffness.value.front),
            precise(s.stiffness.value.rear),
        ),
        spring_damping=(precise(s.damping.value.front), precise(s.damping.value.rear)),
        spring_ref=(precise(refs[0]), precise(refs[1])),
        half_travel=precise(s.travel.value / 2),
        drive=drive(spec),
        servo=servo(spec),
        steering=steering(spec),
    )


def body(spec: RacecarNeoV2, p: Vec3) -> Vec3:
    """A position in base_link (m) in the car body's frame."""
    x, y, z = p
    return (
        x - spec.chassis.wheelbase.value / 2,
        y,
        z - spec.wheel.axle_height.value,
    )


def _precise3(v: Vec3) -> Vec3:
    return (precise(v[0]), precise(v[1]), precise(v[2]))


# --- mass ---------------------------------------------------------------------


def chassis_mass(spec: RacecarNeoV2) -> float:
    """The chassis body's mass (kg): the car's less its wheels and the
    steering input's body."""
    return spec.chassis.mass.value - 4 * spec.wheel.mass.value - INPUT_BODY_MASS


def chassis_com(spec: RacecarNeoV2) -> Vec3:
    """The chassis body's centre of mass: the car's, less the wheels and the
    steering input's body, which sit on the axle line."""
    x, y, z = (float(v) for v in _chassis_com(spec))
    return (x, y, z)


def _chassis_com(spec: RacecarNeoV2) -> np.ndarray:
    m, mw = spec.chassis.mass.value, spec.wheel.mass.value
    moment = m * car_com(spec) - mw * sum(_wheel_centres(spec))
    moment -= INPUT_BODY_MASS * _input_centre(spec)
    return moment / chassis_mass(spec)


def car_com(spec: RacecarNeoV2) -> np.ndarray:
    """The whole car's centre of mass in the body frame: front_weight_share
    of the wheelbase ahead of the rear axle, centred, at com_height."""
    c = spec.chassis
    x = c.front_weight_share.value * c.wheelbase.value
    return np.array(body(spec, (x, 0.0, c.com_height.value)))


def _wheel_centres(spec: RacecarNeoV2) -> list[np.ndarray]:
    x, y = spec.chassis.wheelbase.value / 2, spec.chassis.track.value / 2
    return [np.array([sx * x, sy * y, 0.0]) for sx in (1, -1) for sy in (1, -1)]


def _input_centre(spec: RacecarNeoV2) -> np.ndarray:
    return np.array([spec.chassis.wheelbase.value / 2, 0.0, 0.0])


def chassis_inertia(spec: RacecarNeoV2) -> Vec3:
    """The chassis body's principal inertia (kg·m²) about its own centre of
    mass: the whole car's, less the wheels' and the steering input's body's
    about the car's centre of mass, moved to the chassis' (parallel axes)."""
    com, mw = car_com(spec), spec.wheel.mass.value
    inertia = np.diag(spec.chassis.inertia.value).astype(float)
    for centre in _wheel_centres(spec):
        inertia -= np.diag(wheel_inertia(spec)) + _shift(mw, centre - com)
    inertia -= _shift(INPUT_BODY_MASS, _input_centre(spec) - com)
    inertia -= _shift(chassis_mass(spec), _chassis_com(spec) - com)
    return (float(inertia[0, 0]), float(inertia[1, 1]), float(inertia[2, 2]))


def _shift(mass: float, d: np.ndarray) -> np.ndarray:
    """A point mass's inertia (kg·m²) about a point d from it."""
    return mass * (float(d @ d) * np.eye(3) - np.outer(d, d))


def wheel_inertia(spec: RacecarNeoV2) -> Vec3:
    """A wheel's principal inertia (kg·m²) about its centre, as a solid
    cylinder: about X, its spin axis Y, and Z."""
    m, r, w = spec.wheel.mass.value, wheel_radius(spec), spec.wheel.width.value
    across = m * (3 * r**2 + w**2) / 12
    return (across, m * r**2 / 2, across)


def wheel_radius(spec: RacecarNeoV2) -> float:
    return spec.wheel.diameter.value / 2


def effective_mass(spec: RacecarNeoV2) -> float:
    """The mass that a force at the tyres accelerates (kg): the car's and
    its wheels' spin. A solid wheel adds half its mass for each of four."""
    return (
        spec.chassis.mass.value + 4 * wheel_inertia(spec)[1] / wheel_radius(spec) ** 2
    )


def spring_refs(spec: RacecarNeoV2) -> Axles:
    """Each axle's springs' rest position (m): each spring holds its share of
    the weight on its axle, less the wheels, with its joint at 0."""
    c, k = spec.chassis, spec.suspension.stiffness.value
    m, share, mw = c.mass.value, c.front_weight_share.value, spec.wheel.mass.value
    front = (m * share - 2 * mw) * G / 2
    rear = (m * (1 - share) - 2 * mw) * G / 2
    return (-front / k.front, -rear / k.rear)


# --- drive --------------------------------------------------------------------


@dataclass(frozen=True)
class DriveParams:
    gain: float  # N per m/s
    gear: float  # m, the tyre's radius
    slope: float  # of the throttle map, m/s
    step: float  # of the throttle map, m/s
    ctrl_limit: float  # m/s, the map at throttle 1
    brake: float  # N·m, the drivetrain's frictionloss
    wheel_damping: float  # N·m·s/rad, each wheel's
    latency: float  # s
    nsample: int


def drive(spec: RacecarNeoV2) -> DriveParams:
    r = wheel_radius(spec)
    slope, step = throttle_map(spec)
    brake, drag = coast_drag(spec)
    latency = spec.drive.latency.value
    return DriveParams(
        gain=precise(spec.drive.gain.value),
        gear=precise(r),
        slope=precise(slope),
        step=precise(step),
        ctrl_limit=precise(slope + step),
        # The drivetrain's length is the wheels' mean angle: a force f on it
        # is f / r at the tyres, and so is a damping of each wheel's spin.
        brake=precise(brake * r),
        wheel_damping=precise(drag * r**2 / 4),
        latency=precise(latency),
        nsample=_nsample(latency),
    )


def throttle_map(spec: RacecarNeoV2) -> tuple[float, float]:
    """The motor's ctrl at a throttle t != 0, sign(t) * (step + slope * |t|),
    as (slope, step): the speed it holds is ctrl less the drags over the
    gain, so ctrl is set above the steady speed by that much."""
    floor, slope = steady_speed_line(spec)
    brake, drag = coast_drag(spec)
    gain = spec.drive.gain.value
    return (slope * (1 + drag / gain), floor * (1 + drag / gain) + brake / gain)


def steady_speed_line(spec: RacecarNeoV2) -> tuple[float, float]:
    """The steady speed (m/s) at a throttle t > 0 as floor + slope * t: a
    least-squares line through the measured steady speeds."""
    table = spec.drive.steady_speed.value
    slope, floor = np.polyfit([a.command for a in table], [a.value for a in table], 1)
    return float(floor), float(slope)


def coast_drag(spec: RacecarNeoV2) -> tuple[float, float]:
    """The forces at the tyres that slow the coasting car: its drag brake
    (N) and its drag per speed (N per m/s)."""
    a0, a1 = spec.drive.coast.value
    m = effective_mass(spec)
    return (m * a0, m * a1)


TIMESTEP = 1 / 360  # s, the model's step, as every track's


def _nsample(latency: float) -> int:
    """The history a delay of latency (s) needs: a sample a step, and two
    to spare."""
    return math.ceil(latency / TIMESTEP) + 2


# --- steering -----------------------------------------------------------------


@dataclass(frozen=True)
class SteeringParams:
    lock: float  # rad of steering input at a command of 1, the servo's ctrlrange
    stops: Sides  # rad, the steering input's range: left > 0, right < 0
    wheel_steer_range: float  # rad, of each front wheel's steer joint
    ackermann_left: Polynomial  # the left wheel's angle from the steering input
    ackermann_right: Polynomial


WHEEL_STEER_MARGIN = 0.015  # rad, so the wheels' steer joint limits never bind


def steering(spec: RacecarNeoV2) -> SteeringParams:
    """The steering input is the command times lock, the mean of the
    linkage's two slopes, so that it is about the virtual wheel's angle. It
    stops where the virtual wheel reaches the linkage's stops. Each front
    wheel's polynomial is a least-squares fit to Ackermann geometry from the
    virtual wheel's angle over that range."""
    k = spec.steering.linkage.value
    lock = (k.slope.left + k.slope.right) / 2
    left = k.stop.left / k.slope.left * lock
    right = k.stop.right / k.slope.right * lock
    l, w = spec.chassis.wheelbase.value, spec.chassis.track.value
    x = np.linspace(-right, left, 401)
    virtual = np.where(x >= 0, x * k.slope.left, x * k.slope.right) / lock
    curvature = np.tan(virtual) / l
    fl = np.arctan(l * curvature / (1 - curvature * w / 2))
    fr = np.arctan(l * curvature / (1 + curvature * w / 2))
    widest = max(float(np.abs(fl).max()), float(np.abs(fr).max()))
    return SteeringParams(
        lock=precise(lock),
        stops=(precise(left), precise(-right)),
        wheel_steer_range=precise(widest + WHEEL_STEER_MARGIN),
        ackermann_left=_quartic(x, fl),
        ackermann_right=_quartic(x, fr),
    )


def _quartic(x: np.ndarray, y: np.ndarray) -> Polynomial:
    a0, a1, a2, a3, a4 = (
        precise(float(a)) for a in np.polynomial.polynomial.polyfit(x, y, 4)
    )
    return (a0, a1, a2, a3, a4)


@dataclass(frozen=True)
class ServoParams:
    nominal: tuple[float, float, float]  # V, N·m, rad/s at the steering input
    kp: float  # V/rad
    kd: float  # V·s/rad
    voltage_limit: float  # V
    latency: float  # s
    nsample: int


def servo(spec: RacecarNeoV2) -> ServoParams:
    s = spec.steering
    latency = s.latency.value
    return ServoParams(
        nominal=_precise3(
            (s.servo_supply.value, s.servo_stall_torque.value, s.servo_free_speed.value)
        ),
        kp=precise(s.servo_kp.value),
        kd=precise(s.servo_kd.value),
        voltage_limit=precise(s.servo_supply.value),
        latency=precise(latency),
        nsample=_nsample(latency),
    )
