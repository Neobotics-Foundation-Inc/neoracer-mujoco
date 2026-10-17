"""
The NeoRacer's hardware, from its datasheets and measurements, and the
numbers of the car model (assets/neoracer.xml) that follow from it.

The dataclasses hold what was read off a datasheet or measured on the car.
The functions below derive the model's parameters from them: the drive
motor's and the steering servo's dcmotor settings, the steering lock and the
Ackermann polynomials, the springs' preload, and the chassis' mass and
inertia. car_xml writes them into the car's XML, and the checked-in XML is
what car_xml writes for Hardware(): a test holds it there. To change the car,
change a dataclass here and run

    python -m neoracer_mujoco.write_car_xml assets/neoracer.xml

Frame: +X forward, +Y left, +Z up. Lengths in m, angles in rad unless named.
"""

import math
import re
from dataclasses import dataclass

import numpy as np

G = 9.81  # m/s^2, MuJoCo's gravity
KGF_CM = 9.80665 / 100  # N·m in a kg·cm (kgf at standard gravity), as servos are rated


@dataclass(frozen=True)
class Chassis:
    """The car's geometry and masses."""

    wheelbase: float = 0.28764  # m, the production car's CAD model
    track: float = 0.224  # m, the production car's CAD model
    wheel_radius: float = 0.0425  # m, the 85 mm tyre of the spec (2026-09-29)
    tyre_width: float = 0.04825  # m, the CAD model's tyre
    mass: float = 3.0  # kg, the whole car, the spec (2026-09-29)
    wheel_mass: float = 0.0587  # kg each, the osracer URDF
    steer_body_mass: float = 0.01  # kg, the virtual body of steer_input
    com_height: float = 0.0189  # m above the axles, the osracer URDF
    # The chassis' inertia in the osracer URDF (base_link), for its mass there.
    # Nobody has measured the production car's.
    urdf_inertia: tuple[float, float, float] = (0.002377, 0.0058394, 0.0070057)
    urdf_mass: float = 1.6211  # kg
    # The radius of the wheels of the cosmetic wheel meshes, which come from
    # the osracer CAD model.
    mesh_wheel_radius: float = 0.05  # m


@dataclass(frozen=True)
class Suspension:
    stiffness: float = 600.0  # N/m per wheel, ESTIMATED


@dataclass(frozen=True)
class Battery:
    voltage: float = 11.1  # V, the car's 3S LiPo


@dataclass(frozen=True)
class DriveMotor:
    """The RS-545 that turns the four wheels through open differentials.

    The NeoRacer docs (hardware/drivetrain) give its free speed, about
    11,000 rpm, but not its voltage or its stall torque, and no RS-545
    datasheet lists that speed. So the model takes the speed at rated_voltage
    and scales a datasheet's stall torque to it: rewound in the same frame, a
    motor's stall torque at a given voltage is proportional to its free speed.
    The RS-545SH-4075 (79.3 mN·m and 9,300 rpm at 12 V) is the datasheet; the
    RS-545SH at 24,000 rpm and 214 mN·m gives the same ratio within 5%."""

    free_speed_rpm: float = 11_000.0  # rpm with no load
    rated_voltage: float = 12.0  # V, ASSUMED: the voltage of the free speed
    reference_stall_torque: float = 0.0793  # N·m, the RS-545SH-4075 at 12 V
    reference_free_speed_rpm: float = 9_300.0  # rpm, the RS-545SH-4075 at 12 V
    # Motor turns per wheel turn: the firmware's OSCORE_GEAR_RATIO_MILLI.
    gear_ratio: float = 10.55


@dataclass(frozen=True)
class ServoRating:
    """A servo datasheet's rating at one supply voltage."""

    voltage: float  # V
    stall_torque_kgf_cm: float  # kg·cm
    seconds_per_60_degrees: float  # s, with no load


@dataclass(frozen=True)
class SteeringServo:
    """The SPT5435LV-180W, as a DC motor with a position loop.

    The servo header runs on the OSCORE board's 5 V rail, VCC_5V_IO (the
    NeoRacer docs, hardware/oscore-board). The datasheet rates the servo at
    4.8 V and 6.0 V, so its rating at the supply is interpolated between
    them."""

    low: ServoRating = ServoRating(4.8, 29.0, 0.16)
    high: ServoRating = ServoRating(6.0, 35.0, 0.14)
    supply: float = 5.0  # V
    # The horn turns 90 degrees over the firmware's 1000 to 2000 us.
    horn_half_range_degrees: float = 45.0
    # The loop's gains, picked so it settles without chatter at the 1/360 s
    # time step.
    kp: float = 50.0  # V/rad
    kd: float = 1.0  # V·s/rad


@dataclass(frozen=True)
class Steering:
    # The inner front wheel's angle at full lock, measured on the car
    # (2026-10-16).
    inner_lock_degrees: float = 30.0
    # The share of the horn's half range that the car's driver uses: the ROS
    # throttle_node's max_steering.
    driver_lock: float = 0.625
    # steer_input's range (rad), over which the Ackermann polynomials fit.
    # It is wider than full lock, so the fit covers every command.
    fit_range: float = 0.45


@dataclass(frozen=True)
class Hardware:
    chassis: Chassis = Chassis()
    suspension: Suspension = Suspension()
    battery: Battery = Battery()
    motor: DriveMotor = DriveMotor()
    servo: SteeringServo = SteeringServo()
    steering: Steering = Steering()

    def __post_init__(self) -> None:
        # steer_input's range and the Ackermann fit must cover full lock.
        if not steer_lock(self) < self.steering.fit_range:
            raise ValueError(
                f"full lock, {steer_lock(self)} rad of steer_input, is not inside"
                f" fit_range, {self.steering.fit_range} rad"
            )


type DcMotor = tuple[float, float, float]  # V, N·m, rad/s: MJCF's nominal


def rpm(speed: float) -> float:
    """rad/s in speed (rpm)."""
    return speed * 2.0 * math.pi / 60.0


def motor_nominal(motor: DriveMotor) -> DcMotor:
    """The drive motor at its shaft: rated voltage, stall torque, free speed."""
    stall = (
        motor.reference_stall_torque
        * motor.free_speed_rpm
        / motor.reference_free_speed_rpm
    )
    return (motor.rated_voltage, stall, rpm(motor.free_speed_rpm))


def servo_rating(servo: SteeringServo) -> ServoRating:
    """The servo's rating at its supply, interpolated between its datasheet's."""
    w = (servo.supply - servo.low.voltage) / (servo.high.voltage - servo.low.voltage)

    def lerp(a: float, b: float) -> float:
        return a + w * (b - a)

    return ServoRating(
        servo.supply,
        lerp(servo.low.stall_torque_kgf_cm, servo.high.stall_torque_kgf_cm),
        lerp(servo.low.seconds_per_60_degrees, servo.high.seconds_per_60_degrees),
    )


def left_wheel_angle(chassis: Chassis, steer: float) -> float:
    """The left front wheel's angle (rad) for a steer_input of steer != 0
    (rad, + = left): ideal Ackermann steering, steer_input being the angle of
    a wheel at the middle of the front axle. The left wheel is the inner one
    in a left turn."""
    l, w = chassis.wheelbase, chassis.track
    return math.atan(l / (l / math.tan(steer) - w / 2))


def steer_lock(hw: Hardware) -> float:
    """steer_input (rad) at full lock, to the model's precision: where the
    inner wheel reaches the measured lock, the inverse of left_wheel_angle."""
    l, w = hw.chassis.wheelbase, hw.chassis.track
    inner = math.radians(hw.steering.inner_lock_degrees)
    return precise(math.atan(l / (l / math.tan(inner) + w / 2)))


def linkage_ratio(hw: Hardware) -> float:
    """How far the servo's horn turns per radian of steer_input: at full lock
    the horn is at driver_lock of its half range."""
    horn = math.radians(hw.steering.driver_lock * hw.servo.horn_half_range_degrees)
    return horn / steer_lock(hw)


def servo_nominal(hw: Hardware) -> DcMotor:
    """The servo at steer_input, through the linkage: supply, stall torque,
    free speed."""
    rating, ratio = servo_rating(hw.servo), linkage_ratio(hw)
    free = math.radians(60.0) / rating.seconds_per_60_degrees  # rad/s at the horn
    return (
        rating.voltage,
        rating.stall_torque_kgf_cm * KGF_CM * ratio,
        free / ratio,
    )


def ackermann(hw: Hardware) -> np.ndarray:
    """polycoef a0..a4 of the left wheel's angle as a polynomial of
    steer_input: a least-squares fit to left_wheel_angle over steer_input's
    range. The right wheel's polynomial is its mirror image,
    (-a0, a1, -a2, a3, -a4). Shape (5,)."""
    # An even count of points leaves out 0, where left_wheel_angle divides by 0.
    x = np.linspace(-hw.steering.fit_range, hw.steering.fit_range, 400)
    y = [left_wheel_angle(hw.chassis, s) for s in x]
    return np.polynomial.polynomial.polyfit(x, y, 4)


def sprung_mass(chassis: Chassis) -> float:
    """The mass on the springs (kg): the car less its four wheels."""
    return chassis.mass - 4 * chassis.wheel_mass


def chassis_mass(chassis: Chassis) -> float:
    """The chassis body's mass (kg): the sprung mass less steer_input's body."""
    return sprung_mass(chassis) - chassis.steer_body_mass


def chassis_inertia(chassis: Chassis) -> tuple[float, float, float]:
    """The chassis' principal inertia (kg·m^2): the URDF's, scaled to its mass."""
    k = chassis_mass(chassis) / chassis.urdf_mass
    ixx, iyy, izz = chassis.urdf_inertia
    return (ixx * k, iyy * k, izz * k)


def springref(hw: Hardware) -> float:
    """The springs' rest position (m): each holds a quarter of the sprung
    weight with the joint at 0."""
    return -sprung_mass(hw.chassis) * G / 4 / hw.suspension.stiffness


def precise(x: float) -> float:
    """x to the model's precision, 6 significant digits."""
    return float(f"{x:.6g}")


def number(x: float) -> str:
    """x to the model's precision as the XML writes it, with no leading zero.
    It reads back as precise(x)."""
    text = repr(precise(x)).removesuffix(".0")
    return re.sub(r"^(-?)0\.", r"\1.", text)


def numbers(*xs: float) -> str:
    return " ".join(number(x) for x in xs)


def _set(xml: str, element: str, attribute: str, value: str) -> str:
    """xml with attribute set to value in the one start tag that matches the
    regex element (which matches from its "<")."""
    tags = list(re.finditer(element + r"[^>]*>", xml))
    if len(tags) != 1:
        raise ValueError(f"{len(tags)} elements match {element!r}, not 1")
    tag = tags[0]
    new, n = re.subn(rf'(\s{attribute}=")[^"]*(")', rf"\g<1>{value}\g<2>", tag.group(0))
    if n != 1:
        raise ValueError(f"{element!r} has {n} {attribute} attributes, not 1")
    return xml[: tag.start()] + new + xml[tag.end() :]


def car_xml(hw: Hardware, xml: str) -> str:
    """The car's XML, xml, with the numbers derived from hw."""
    c = hw.chassis
    x, y, r = c.wheelbase / 2, c.track / 2, c.wheel_radius
    lock, poly = steer_lock(hw), ackermann(hw)
    edits = [
        # The car spawns 1 mm above its resting height: the axles at the tyre
        # radius.
        (r'<body name="car"', "pos", numbers(0, 0, r + 0.001)),
        (r"<inertial\s", "mass", number(chassis_mass(c))),
        (r"<inertial\s", "diaginertia", numbers(*chassis_inertia(c))),
        (r"<inertial\s", "pos", numbers(0, 0, c.com_height)),
        (r'<site name="imu"', "pos", numbers(0, 0, c.com_height)),
        (r'<body name="steer_input_body"', "pos", numbers(x, 0, 0)),
        (
            r'<geom type="cylinder" size="[^"]*" contype',
            "mass",
            number(c.steer_body_mass),
        ),
        (
            r'<default class="wheel">\s*<joint[^>]*>\s*<geom',
            "size",
            numbers(r, c.tyre_width / 2),
        ),
        (
            r'<default class="susp">\s*<joint',
            "stiffness",
            number(hw.suspension.stiffness),
        ),
        (r'<default class="susp">\s*<joint', "springref", number(springref(hw))),
        (
            r'<dcmotor name="motor"',
            "nominal",
            numbers(*motor_nominal(hw.motor)),
        ),
        (r'<dcmotor name="motor"', "gear", number(hw.motor.gear_ratio)),
        (
            r'<dcmotor name="motor"',
            "ctrlrange",
            numbers(-hw.battery.voltage, hw.battery.voltage),
        ),
        (r'<dcmotor name="steer_servo"', "nominal", numbers(*servo_nominal(hw))),
        (
            r'<dcmotor name="steer_servo"',
            "controller",
            numbers(hw.servo.kp, 0, hw.servo.kd, 0, 0, hw.servo.supply),
        ),
        (r'<dcmotor name="steer_servo"', "ctrlrange", numbers(-lock, lock)),
        (
            r'<joint name="steer_input"',
            "range",
            numbers(-hw.steering.fit_range, hw.steering.fit_range),
        ),
        (r'<joint name="ackermann_left"', "polycoef", numbers(*poly)),
        (
            r'<joint name="ackermann_right"',
            "polycoef",
            numbers(*(poly * [-1, 1, -1, 1, -1])),
        ),
    ]
    scale = numbers(*[r / c.mesh_wheel_radius] * 3)
    for corner, (sx, sy) in {
        "fl": (x, y),
        "fr": (x, -y),
        "rl": (-x, y),
        "rr": (-x, -y),
    }.items():
        edits += [
            (rf'<body name="{corner}_wheel_body"', "pos", numbers(sx, sy, 0)),
            (rf'<geom name="{corner}_wheel"', "mass", number(c.wheel_mass)),
            (rf'<mesh name="vis_{corner}_wheel"', "scale", scale),
        ]
    for element, attribute, value in edits:
        xml = _set(xml, element, attribute, value)
    return xml
