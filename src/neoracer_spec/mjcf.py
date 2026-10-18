"""
The car model's numbers (derive.ModelParams) written into the car's XML.

car_xml sets each number in the start tag that holds it and leaves the rest
of the XML, its structure and its comments, as it is. The checked-in
assets/neoracer.xml is what car_xml writes for neoracer.NEORACER; a test
holds it there. After a change to the car, run

    python -m neoracer_mujoco.write_car assets
"""

import re

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
