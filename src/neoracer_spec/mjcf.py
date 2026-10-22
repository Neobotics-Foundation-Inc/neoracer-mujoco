"""
The car models' numbers (derive.ModelParams, derive_v2.ModelParams)
written into the cars' XML.

car_xml and racecar_neo_v2_xml set each number in the start tag that holds
it and leave the rest of the XML, its structure and its comments, as it is.
The checked-in assets/neoracer.xml and assets/racecar_neo_v2.xml are what
they write for neoracer.NEORACER and racecar_neo_v2.RACECAR_NEO_V2; a test
holds them there. After a change to a car, run

    python -m neoracer_mujoco.write_car assets
"""

import math
import re

from . import derive_v2
from .derive import ModelParams, precise


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


def beam_timing(scan_rate: float, bearing: int) -> tuple[float, float, float]:
    """When the rangefinder of the LiDAR beam at bearing (degrees,
    counterclockwise from the scan's 0) reads, as the period and the phase of
    its interval and its delay, all in s.

    The LiDAR's head turns clockwise from bearing 0 scan_rate times a second
    (Hz), and its driver reports a whole turn when the turn ends, as
    racecar_core's get_samples() does. So the beam samples the range once a
    turn, when the head passes it, and shows the sample when that turn ends.
    The first turn begins at time 0. MuJoCo reads a phase of 0 as the whole
    period, a sample at time 0."""
    period = 1.0 / scan_rate
    passes = (-bearing) % 360 / 360  # when in a turn the head passes the beam
    phase = (passes - 1.0) * period if passes > 0.0 else 0.0
    return period, phase, (1.0 - passes) * period


def _beam_edits(scan_rate: float, bearing: int) -> list[tuple[str, str, str]]:
    """The attributes of the beam's rangefinder that time its readings."""
    period, phase, delay = beam_timing(scan_rate, bearing)
    rangefinder = rf'<rangefinder name="lidar_{bearing:03d}"'
    return [
        # Two samples: the one the beam shows, and the next until it shows.
        (rangefinder, "nsample", "2"),
        (rangefinder, "interval", numbers(period, phase)),
        (rangefinder, "delay", number(delay)),
    ]


def car_xml(p: ModelParams, xml: str) -> str:
    """The car's XML, xml, with the numbers of p."""
    x, y = p.half_wheelbase, p.half_track
    poly = p.ackermann_left
    mirror = (-poly[0], poly[1], -poly[2], poly[3], -poly[4])
    m, s = p.motor, p.servo
    friction = numbers(p.wheel_friction, 0.1, 0.1)
    wheel_default = r'<default class="wheel">\s*<joint[^>]*>\s*<geom'
    susp = r'<default class="susp">\s*<joint'
    edits = [
        (r'<geom type="plane"', "friction", friction),
        (r'<body name="car"', "pos", numbers(0, 0, p.spawn_height)),
        (r"<inertial\s", "mass", number(p.chassis_mass)),
        (r"<inertial\s", "diaginertia", numbers(*p.chassis_inertia)),
        (r"<inertial\s", "pos", numbers(0, 0, p.com_height)),
        (r'<geom name="chassis"', "pos", numbers(*p.box_center)),
        (r'<geom name="chassis"', "size", numbers(*p.box_half_size)),
        (r'<site name="imu"', "pos", numbers(*p.imu_position)),
        (r'<site name="lidar_mount"', "pos", numbers(*p.lidar_position)),
        (r'<site name="lidar_mount"', "user", number(p.lidar_field)),
        (r'<geom name="vis_lidar"', "pos", numbers(*p.lidar_position)),
        (r'<body name="steer_input_body"', "pos", numbers(x, 0, 0)),
        (
            r'<geom type="cylinder" size="[^"]*" contype',
            "mass",
            number(p.input_body_mass),
        ),
        (r'<default class="wheel">\s*<joint', "damping", number(p.wheel_damping)),
        (wheel_default, "size", numbers(p.wheel_radius, p.tyre_half_width)),
        (wheel_default, "friction", friction),
        (susp, "range", numbers(-p.droop, p.jounce)),
        (susp, "stiffness", number(p.spring_stiffness)),
        (susp, "damping", number(p.spring_damping)),
        (susp, "springref", number(p.spring_ref)),
        (
            r'<default class="steer">\s*<joint',
            "range",
            numbers(-p.wheel_steer_range, p.wheel_steer_range),
        ),
        (
            r'<default class="steer">\s*<joint',
            "armature",
            number(p.steer_armature),
        ),
        (r'<dcmotor name="motor"', "nominal", numbers(*m.nominal)),
        (r'<dcmotor name="motor"', "gear", number(m.gear)),
        (r'<dcmotor name="motor"', "ctrlrange", numbers(-m.ctrl_limit, m.ctrl_limit)),
        (r'<dcmotor name="steer_servo"', "nominal", numbers(*s.nominal)),
        (
            r'<dcmotor name="steer_servo"',
            "controller",
            numbers(s.kp, 0, s.kd, 0, 0, s.voltage_limit),
        ),
        (r'<dcmotor name="steer_servo"', "ctrlrange", numbers(-s.lock, s.lock)),
        (
            r'<joint name="steer_input"',
            "range",
            numbers(-p.steer_input_range, p.steer_input_range),
        ),
        (r'<joint name="ackermann_left"', "polycoef", numbers(*poly)),
        (r'<joint name="ackermann_right"', "polycoef", numbers(*mirror)),
    ]
    for bearing in range(0, 360, 45):
        edits.append(
            (rf'<site name="lidar_{bearing:03d}"', "pos", numbers(*p.lidar_position))
        )
        edits += _beam_edits(p.lidar_scan_rate, bearing)
    scale = numbers(*[p.wheel_mesh_scale] * 3)
    for corner, (cx, cy) in {
        "fl": (x, y),
        "fr": (x, -y),
        "rl": (-x, y),
        "rr": (-x, -y),
    }.items():
        edits += [
            (rf'<body name="{corner}_wheel_body"', "pos", numbers(cx, cy, 0)),
            (rf'<geom name="{corner}_wheel"', "mass", number(p.wheel_mass)),
            (rf'<mesh name="vis_{corner}_wheel"', "scale", scale),
        ]
    for element, attribute, value in edits:
        xml = _set(xml, element, attribute, value)
    return xml


def racecar_neo_v2_xml(p: derive_v2.ModelParams, xml: str) -> str:
    """The RACECAR Neo V2's XML, xml, with the numbers of p."""
    x, y = p.half_wheelbase, p.half_track
    d, s, st = p.drive, p.servo, p.steering
    ax, ay, az = p.antenna_center
    edits = [
        (r'<body name="car"', "pos", numbers(0, 0, p.spawn_height)),
        (r"<inertial\s", "pos", numbers(*p.chassis_com)),
        (r"<inertial\s", "mass", number(p.chassis_mass)),
        (r"<inertial\s", "diaginertia", numbers(*p.chassis_inertia)),
        (r'<site name="imu"', "pos", numbers(*p.imu_position)),
        (r'<site name="lidar_mount"', "pos", numbers(*p.lidar_position)),
        (r'<site name="lidar_mount"', "euler", numbers(0, 0, p.lidar_yaw)),
        (r'<site name="lidar_mount"', "user", number(p.lidar_field)),
        (r'<geom name="antenna_left"', "pos", numbers(ax, ay, az)),
        (r'<geom name="antenna_right"', "pos", numbers(ax, -ay, az)),
        (r'<body name="steer_input_body"', "pos", numbers(x, 0, 0)),
        (r'<joint name="steer_input"', "range", numbers(st.stops[1], st.stops[0])),
        (
            r'<geom type="cylinder" size="[^"]*" contype',
            "mass",
            number(p.input_body_mass),
        ),
        (r'<default class="wheel">\s*<joint', "damping", number(d.wheel_damping)),
        (
            r'<default class="wheel">\s*<joint[^>]*>\s*<geom',
            "size",
            numbers(p.wheel_radius, p.tyre_half_width),
        ),
        (
            r'<default class="steer">\s*<joint',
            "range",
            numbers(-st.wheel_steer_range, st.wheel_steer_range),
        ),
        (
            r'<default class="susp">\s*<joint',
            "range",
            numbers(-p.half_travel, p.half_travel),
        ),
        (r'<fixed name="drivetrain"', "frictionloss", number(d.brake)),
        (r'<general name="motor"', "gear", number(d.gear)),
        (r'<general name="motor"', "gainprm", number(d.gain)),
        (r'<general name="motor"', "ctrlrange", numbers(-d.ctrl_limit, d.ctrl_limit)),
        (r'<general name="motor"', "user", numbers(d.slope, d.step, 0)),
        (r'<general name="motor"', "delay", number(d.latency)),
        (r'<general name="motor"', "nsample", str(d.nsample)),
        (r'<general name="motor_engage"', "gear", number(d.gear)),
        (r'<general name="motor_engage"', "gainprm", numbers(0, 0, -d.gain)),
        (r'<general name="motor_engage"', "delay", number(d.latency)),
        (r'<general name="motor_engage"', "nsample", str(d.nsample)),
        (r'<dcmotor name="steer_servo"', "nominal", numbers(*s.nominal)),
        (
            r'<dcmotor name="steer_servo"',
            "controller",
            numbers(s.kp, 0, s.kd, 0, 0, s.voltage_limit),
        ),
        (r'<dcmotor name="steer_servo"', "ctrlrange", numbers(-st.lock, st.lock)),
        (r'<dcmotor name="steer_servo"', "delay", number(s.latency)),
        (r'<dcmotor name="steer_servo"', "nsample", str(s.nsample)),
        (r'<joint name="ackermann_left"', "polycoef", numbers(*st.ackermann_left)),
        (r'<joint name="ackermann_right"', "polycoef", numbers(*st.ackermann_right)),
    ]
    for box in p.boxes:
        edits += [
            (rf'<geom name="{box.name}"', "pos", numbers(*box.center)),
            (rf'<geom name="{box.name}"', "size", numbers(*box.half_size)),
        ]
    for name in ("antenna_left", "antenna_right"):
        edits.append(
            (
                rf'<geom name="{name}"',
                "size",
                numbers(p.antenna_radius, p.antenna_half_length),
            )
        )
    for bearing in range(0, 360, 45):
        a = math.radians(bearing) + p.lidar_yaw
        site = rf'<site name="lidar_{bearing:03d}"'
        edits += [
            (site, "pos", numbers(*p.lidar_position)),
            (site, "zaxis", numbers(math.cos(a), math.sin(a), 0)),
            *_beam_edits(p.lidar_scan_rate, bearing),
        ]
    stiffness, damping, ref, grip = (
        p.spring_stiffness,
        p.spring_damping,
        p.spring_ref,
        p.grip,
    )
    for corner, (cx, cy, axle) in {
        "fl": (x, y, 0),
        "fr": (x, -y, 0),
        "rl": (-x, y, 1),
        "rr": (-x, -y, 1),
    }.items():
        susp = rf'<joint name="{corner}_susp"'
        edits += [
            (rf'<body name="{corner}_wheel_body"', "pos", numbers(cx, cy, 0)),
            (susp, "stiffness", number(stiffness[axle])),
            (susp, "damping", number(damping[axle])),
            (susp, "springref", number(ref[axle])),
            (rf'<geom name="{corner}_wheel"', "mass", number(p.wheel_mass)),
            (
                rf'<geom name="{corner}_wheel"',
                "friction",
                numbers(grip[axle], 0.1, 0.1),
            ),
        ]
    for element, attribute, value in edits:
        xml = _set(xml, element, attribute, value)
    return xml
