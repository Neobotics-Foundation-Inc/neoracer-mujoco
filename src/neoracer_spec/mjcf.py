"""
The car models' numbers (derive.ModelParams, derive_v2.ModelParams)
written into the cars' XML.

assets/neoracer.xml and assets/racecar_neo_v2.xml are templates: their
structure, their comments and the numbers that the spec does not give.
car_xml and racecar_neo_v2_xml add each number that the spec gives to the
element that holds it, which an Edit finds by its tag and its name or class,
and refuse a template that already has it, so the spec's numbers live in
the spec alone. neoracer_mujoco.assets reads the cars through them.
"""

import math
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass

from . import derive_v2
from .derive import ModelParams, precise


def number(x: float) -> str:
    """x to the model's precision as the XML writes it, with no leading zero.
    It reads back as precise(x)."""
    text = repr(precise(x)).removesuffix(".0")
    return re.sub(r"^(-?)0\.", r"\1.", text)


def numbers(*xs: float) -> str:
    return " ".join(number(x) for x in xs)


@dataclass(frozen=True)
class Edit:
    """An attribute that the spec gives, on the one element that path finds."""

    path: str  # an ElementTree path from the root, such as named("body", "car")
    attribute: str
    value: str


def named(tag: str, name: str) -> str:
    """The path of the element tag named name."""
    return f".//{tag}[@name='{name}']"


def default(cls: str, tag: str) -> str:
    """The path of the element tag in the default class cls."""
    return f".//default[@class='{cls}']/{tag}"


def put(path: str, **attributes: str) -> list[Edit]:
    """The edits that give the element at path these attributes."""
    return [Edit(path, a, v) for a, v in attributes.items()]


def apply(xml: str, edits: Iterable[Edit]) -> str:
    """xml with each edit's attribute added. Raise ValueError if an edit's
    path finds other than one element, or if the element already has the
    attribute: the spec gives it, so the template must not. The comments
    stay; the layout inside a tag does not."""
    builder = ET.TreeBuilder(insert_comments=True)
    root = ET.fromstring(xml, parser=ET.XMLParser(target=builder))
    for edit in edits:
        found = root.findall(edit.path)
        if len(found) != 1:
            raise ValueError(f"{len(found)} elements match {edit.path!r}, not 1")
        (element,) = found
        if edit.attribute in element.attrib:
            raise ValueError(
                f"{edit.path!r} has {edit.attribute}, which the spec gives"
            )
        element.set(edit.attribute, edit.value)
    return ET.tostring(root, encoding="unicode") + "\n"


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


def _beam_edits(scan_rate: float, bearing: int) -> list[Edit]:
    """The attributes of the beam's rangefinder that time its readings."""
    period, phase, delay = beam_timing(scan_rate, bearing)
    return put(
        named("rangefinder", f"lidar_{bearing:03d}"),
        # Two samples: the one the beam shows, and the next until it shows.
        nsample="2",
        interval=numbers(period, phase),
        delay=number(delay),
    )


def car_xml(p: ModelParams, xml: str) -> str:
    """The car's XML from its template, xml, and the numbers of p."""
    return apply(xml, car_edits(p))


def car_edits(p: ModelParams) -> list[Edit]:
    """The numbers of p, by the element and the attribute that hold each."""
    x, y = p.half_wheelbase, p.half_track
    poly = p.ackermann_left
    mirror = (-poly[0], poly[1], -poly[2], poly[3], -poly[4])
    m, s = p.motor, p.servo
    friction = numbers(p.wheel_friction, 0.1, 0.1)
    edits = [
        *put(".//geom[@type='plane']", friction=friction),
        *put(named("body", "car"), pos=numbers(0, 0, p.spawn_height)),
        *put(
            named("body", "car") + "/inertial",
            mass=number(p.chassis_mass),
            diaginertia=numbers(*p.chassis_inertia),
            pos=numbers(0, 0, p.com_height),
        ),
        *put(
            named("geom", "chassis"),
            pos=numbers(*p.box_center),
            size=numbers(*p.box_half_size),
        ),
        *put(named("site", "imu"), pos=numbers(*p.imu_position)),
        *put(
            named("site", "lidar_mount"),
            pos=numbers(*p.lidar_position),
            user=number(p.lidar_field),
        ),
        *put(named("geom", "vis_lidar"), pos=numbers(*p.lidar_position)),
        *put(named("body", "steer_input_body"), pos=numbers(x, 0, 0)),
        *put(
            named("body", "steer_input_body") + "/geom",
            mass=number(p.input_body_mass),
        ),
        *put(default("wheel", "joint"), damping=number(p.wheel_damping)),
        *put(
            default("wheel", "geom"),
            size=numbers(p.wheel_radius, p.tyre_half_width),
            friction=friction,
        ),
        *put(
            default("susp", "joint"),
            range=numbers(-p.droop, p.jounce),
            stiffness=number(p.spring_stiffness),
            damping=number(p.spring_damping),
            springref=number(p.spring_ref),
        ),
        *put(
            default("steer", "joint"),
            range=numbers(-p.wheel_steer_range, p.wheel_steer_range),
            armature=number(p.steer_armature),
        ),
        *put(
            named("dcmotor", "motor"),
            nominal=numbers(*m.nominal),
            gear=number(m.gear),
            ctrlrange=numbers(-m.ctrl_limit, m.ctrl_limit),
        ),
        *put(
            named("dcmotor", "steer_servo"),
            nominal=numbers(*s.nominal),
            controller=numbers(s.kp, 0, s.kd, 0, 0, s.voltage_limit),
            ctrlrange=numbers(-s.lock, s.lock),
        ),
        *put(
            named("joint", "steer_input"),
            range=numbers(-p.steer_input_range, p.steer_input_range),
        ),
        *put(named("joint", "ackermann_left"), polycoef=numbers(*poly)),
        *put(named("joint", "ackermann_right"), polycoef=numbers(*mirror)),
    ]
    for bearing in range(0, 360, 45):
        edits += put(
            named("site", f"lidar_{bearing:03d}"), pos=numbers(*p.lidar_position)
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
            *put(named("body", f"{corner}_wheel_body"), pos=numbers(cx, cy, 0)),
            *put(named("geom", f"{corner}_wheel"), mass=number(p.wheel_mass)),
            *put(named("mesh", f"vis_{corner}_wheel"), scale=scale),
        ]
    return edits


def racecar_neo_v2_xml(p: derive_v2.ModelParams, xml: str) -> str:
    """The RACECAR Neo V2's XML from its template, xml, and the numbers of p."""
    return apply(xml, racecar_neo_v2_edits(p))


def racecar_neo_v2_edits(p: derive_v2.ModelParams) -> list[Edit]:
    """The numbers of p, by the element and the attribute that hold each."""
    x, y = p.half_wheelbase, p.half_track
    d, s, st = p.drive, p.servo, p.steering
    ax, ay, az = p.antenna_center
    edits = [
        *put(named("body", "car"), pos=numbers(0, 0, p.spawn_height)),
        *put(
            named("body", "car") + "/inertial",
            pos=numbers(*p.chassis_com),
            mass=number(p.chassis_mass),
            diaginertia=numbers(*p.chassis_inertia),
        ),
        *put(named("site", "imu"), pos=numbers(*p.imu_position)),
        *put(
            named("site", "lidar_mount"),
            pos=numbers(*p.lidar_position),
            euler=numbers(0, 0, p.lidar_yaw),
            user=number(p.lidar_field),
        ),
        *put(named("geom", "antenna_left"), pos=numbers(ax, ay, az)),
        *put(named("geom", "antenna_right"), pos=numbers(ax, -ay, az)),
        *put(named("body", "steer_input_body"), pos=numbers(x, 0, 0)),
        *put(named("joint", "steer_input"), range=numbers(st.stops[1], st.stops[0])),
        *put(
            named("body", "steer_input_body") + "/geom",
            mass=number(p.input_body_mass),
        ),
        *put(default("wheel", "joint"), damping=number(d.wheel_damping)),
        *put(default("wheel", "geom"), size=numbers(p.wheel_radius, p.tyre_half_width)),
        *put(
            default("steer", "joint"),
            range=numbers(-st.wheel_steer_range, st.wheel_steer_range),
        ),
        *put(default("susp", "joint"), range=numbers(-p.half_travel, p.half_travel)),
        *put(named("fixed", "drivetrain"), frictionloss=number(d.brake)),
        *put(
            named("general", "motor"),
            gear=number(d.gear),
            gainprm=number(d.gain),
            ctrlrange=numbers(-d.ctrl_limit, d.ctrl_limit),
            user=numbers(d.slope, d.step, 0),
            delay=number(d.latency),
            nsample=str(d.nsample),
        ),
        *put(
            named("general", "motor_engage"),
            gear=number(d.gear),
            gainprm=numbers(0, 0, -d.gain),
            delay=number(d.latency),
            nsample=str(d.nsample),
        ),
        *put(
            named("dcmotor", "steer_servo"),
            nominal=numbers(*s.nominal),
            controller=numbers(s.kp, 0, s.kd, 0, 0, s.voltage_limit),
            ctrlrange=numbers(-st.lock, st.lock),
            delay=number(s.latency),
            nsample=str(s.nsample),
        ),
        *put(named("joint", "ackermann_left"), polycoef=numbers(*st.ackermann_left)),
        *put(named("joint", "ackermann_right"), polycoef=numbers(*st.ackermann_right)),
    ]
    for box in p.boxes:
        edits += put(
            named("geom", box.name),
            pos=numbers(*box.center),
            size=numbers(*box.half_size),
        )
    for name in ("antenna_left", "antenna_right"):
        edits += put(
            named("geom", name),
            size=numbers(p.antenna_radius, p.antenna_half_length),
        )
    for bearing in range(0, 360, 45):
        a = math.radians(bearing) + p.lidar_yaw
        edits += put(
            named("site", f"lidar_{bearing:03d}"),
            pos=numbers(*p.lidar_position),
            zaxis=numbers(math.cos(a), math.sin(a), 0),
        )
        edits += _beam_edits(p.lidar_scan_rate, bearing)
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
        edits += [
            *put(named("body", f"{corner}_wheel_body"), pos=numbers(cx, cy, 0)),
            *put(
                named("joint", f"{corner}_susp"),
                stiffness=number(stiffness[axle]),
                damping=number(damping[axle]),
                springref=number(ref[axle]),
            ),
            *put(
                named("geom", f"{corner}_wheel"),
                mass=number(p.wheel_mass),
                friction=numbers(grip[axle], 0.1, 0.1),
            ),
        ]
    return edits
