"""
The car models' numbers (derive.ModelParams, derive_v2.ModelParams)
written into the cars' XML.

assets/neoracer.xml and assets/racecar_neo_v2.xml are templates: their
structure, their comments and the numbers that the spec does not give.
car_xml and racecar_neo_v2_xml add each number that the spec gives to the
element that holds it, which an Edit finds by its tag and its name or class,
and refuse a template that already has it, so the spec's numbers live in
the spec alone. neoracer_mujoco.assets reads the cars through them.

Each part that both cars' models have (parts) has one writer here, such as
lidar_beams, and each car's edits are those writers' and its own.
"""

import math
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from . import derive_v2, parts
from .derive import ModelParams
from .parts import SOLVER, precise


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


def solver(s: parts.Solver) -> list[Edit]:
    """The model's options. The timestep is written whole, not to the
    model's precision, so that it is s.timestep."""
    return put(
        "./option",
        timestep=repr(s.timestep),
        integrator=s.integrator,
        cone=s.cone,
        impratio=number(s.impratio),
    )


# Each wheel's corner: the signs of its x and y, and its axle (0 front).
CORNERS: Mapping[str, tuple[int, int, int]] = MappingProxyType(
    {"fl": (1, 1, 0), "fr": (1, -1, 0), "rl": (-1, 1, 1), "rr": (-1, -1, 1)}
)


def inertial(i: parts.Inertial) -> list[Edit]:
    return put(
        named("body", "car") + "/inertial",
        mass=number(i.mass),
        diaginertia=numbers(*i.diaginertia),
        pos=numbers(*i.pos),
    )


def collision_boxes(bs: tuple[parts.CollisionBox, ...]) -> list[Edit]:
    return [
        e
        for b in bs
        for e in put(
            named("geom", b.name), pos=numbers(*b.center), size=numbers(*b.half_size)
        )
    ]


def lidar_beams(lid: parts.LidarBeams) -> list[Edit]:
    """The mount and the eight beams: each beam's site points its
    rangefinder along its bearing, and its rangefinder reads as the
    LiDAR's driver reports the beam (beam_timing)."""
    edits = put(
        named("site", "lidar_mount"),
        pos=numbers(*lid.position),
        euler=numbers(0, 0, lid.yaw),
        user=number(lid.field),
    )
    for bearing in range(0, 360, 45):
        a = math.radians(bearing) + lid.yaw
        # A multiple of 90 degrees has a cosine or a sine that comes out
        # about 1e-16 away from 0, which the XML writes as 0.
        x, y = (0.0 if abs(v) < 1e-12 else v for v in (math.cos(a), math.sin(a)))
        edits += put(
            named("site", f"lidar_{bearing:03d}"),
            pos=numbers(*lid.position),
            zaxis=numbers(x, y, 0),
        )
        edits += _beam_edits(lid.scan_rate, bearing)
    return edits


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


def wheels(w: parts.Wheels) -> list[Edit]:
    edits = [
        *put(default("wheel", "joint"), damping=number(w.damping)),
        *put(default("wheel", "geom"), size=numbers(w.radius, w.half_width)),
    ]
    for corner, (sx, sy, axle) in CORNERS.items():
        x, y = sx * w.half_wheelbase, sy * w.half_track
        edits += put(named("body", f"{corner}_wheel_body"), pos=numbers(x, y, 0))
        edits += put(
            named("geom", f"{corner}_wheel"),
            mass=number(w.mass),
            friction=numbers(w.friction[axle], 0.1, 0.1),
        )
    return edits


def springs(s: parts.Springs) -> list[Edit]:
    edits = put(default("susp", "joint"), range=numbers(*s.range))
    for corner, (_, _, axle) in CORNERS.items():
        edits += put(
            named("joint", f"{corner}_susp"),
            stiffness=number(s.stiffness[axle]),
            damping=number(s.damping[axle]),
            springref=number(s.springref[axle]),
        )
    return edits


def steering_input(st: parts.SteeringInput, w: parts.Wheels) -> list[Edit]:
    return [
        *put(named("body", "steer_input_body"), pos=numbers(w.half_wheelbase, 0, 0)),
        *put(
            named("body", "steer_input_body") + "/geom",
            mass=number(st.input_body_mass),
        ),
        *put(named("joint", "steer_input"), range=numbers(*st.stops)),
        *put(
            default("steer", "joint"),
            range=numbers(-st.wheel_range, st.wheel_range),
        ),
        *put(named("joint", "ackermann_left"), polycoef=numbers(*st.ackermann_left)),
        *put(named("joint", "ackermann_right"), polycoef=numbers(*st.ackermann_right)),
    ]


def servo(s: parts.Servo) -> list[Edit]:
    return put(
        named("dcmotor", "steer_servo"),
        nominal=numbers(*s.nominal),
        controller=numbers(s.kp, 0, s.kd, 0, 0, s.voltage_limit),
        ctrlrange=numbers(-s.lock, s.lock),
    )


def delay(path: str, d: parts.Delay) -> list[Edit]:
    """The delay of the actuator at path."""
    return put(path, delay=number(d.latency), nsample=str(d.nsample))


def car_xml(p: ModelParams, xml: str) -> str:
    """The car's XML from its template, xml, and the numbers of p."""
    return apply(xml, car_edits(p))


def car_edits(p: ModelParams) -> list[Edit]:
    """The numbers of p, by the element and the attribute that hold each."""
    m = p.motor
    scale = numbers(*[p.wheel_mesh_scale] * 3)
    return [
        *solver(SOLVER),
        *put(named("body", "car"), pos=numbers(0, 0, p.spawn_height)),
        *inertial(p.inertial),
        *collision_boxes(p.boxes),
        *put(named("site", "imu"), pos=numbers(*p.imu_position)),
        *lidar_beams(p.lidar),
        *wheels(p.wheels),
        *springs(p.springs),
        *steering_input(p.steering, p.wheels),
        *servo(p.servo),
        # The NeoRacer's own: its floor, the meshes of its LiDAR and wheels,
        # its steering's armature and its DC drive motor.
        *put(
            ".//geom[@type='plane']",
            friction=numbers(p.wheels.friction[0], 0.1, 0.1),
        ),
        *put(named("geom", "vis_lidar"), pos=numbers(*p.lidar.position)),
        *[
            e
            for corner in CORNERS
            for e in put(named("mesh", f"vis_{corner}_wheel"), scale=scale)
        ],
        *put(default("steer", "joint"), armature=number(p.steer_armature)),
        *put(
            named("dcmotor", "motor"),
            nominal=numbers(*m.nominal),
            gear=number(m.gear),
            ctrlrange=numbers(-m.ctrl_limit, m.ctrl_limit),
        ),
    ]


def racecar_neo_v2_xml(p: derive_v2.ModelParams, xml: str) -> str:
    """The RACECAR Neo V2's XML from its template, xml, and the numbers of p."""
    return apply(xml, racecar_neo_v2_edits(p))


def racecar_neo_v2_edits(p: derive_v2.ModelParams) -> list[Edit]:
    """The numbers of p, by the element and the attribute that hold each."""
    d = p.drive
    ax, ay, az = p.antenna_center
    antenna = numbers(p.antenna_radius, p.antenna_half_length)
    return [
        *solver(SOLVER),
        *put(named("body", "car"), pos=numbers(0, 0, p.spawn_height)),
        *inertial(p.inertial),
        *collision_boxes(p.boxes),
        *put(named("site", "imu"), pos=numbers(*p.imu_position)),
        *lidar_beams(p.lidar),
        *wheels(p.wheels),
        *springs(p.springs),
        *steering_input(p.steering, p.wheels),
        *servo(p.servo),
        *delay(named("dcmotor", "steer_servo"), p.servo_delay),
        # The V2's own: its antennas and its speed controller, two actuators
        # on the drivetrain.
        *put(named("geom", "antenna_left"), pos=numbers(ax, ay, az), size=antenna),
        *put(named("geom", "antenna_right"), pos=numbers(ax, -ay, az), size=antenna),
        *put(named("fixed", "drivetrain"), frictionloss=number(d.brake)),
        *put(
            named("general", "motor"),
            gear=number(d.gear),
            gainprm=number(d.gain),
            ctrlrange=numbers(-d.ctrl_limit, d.ctrl_limit),
            user=numbers(d.slope, d.step, 0),
        ),
        *delay(named("general", "motor"), d.delay),
        *put(
            named("general", "motor_engage"),
            gear=number(d.gear),
            gainprm=numbers(0, 0, -d.gain),
        ),
        *delay(named("general", "motor_engage"), d.delay),
    ]
