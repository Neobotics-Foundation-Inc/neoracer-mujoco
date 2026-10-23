"""
What the cars' models share: records of the numbers of a part that both
models have, which derive.py and derive_v2.py make from each car's spec and
mjcf.py writes into each car's XML. The cars stay separate; their parts
come from here.

Numbers that a model holds are kept to its precision, 6 significant digits
(precise), so the code and the XML hold the same floats.

Frame: the car body's: +X forward, +Y left, +Z up, its origin at the middle
of the wheelbase, on the axle line.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from .spec import Vec3

type Axles = tuple[float, float]  # front, rear
type Nominal = tuple[float, float, float]  # V, N·m, rad/s: a dcmotor's nominal
type Polynomial = tuple[float, float, float, float, float]  # a0..a4


@dataclass(frozen=True)
class Solver:
    """MuJoCo's options for a car's model. A scene with the car takes them
    from it (neoracer_mujoco.assets.compose), so a track sets none."""

    timestep: float  # s
    integrator: Literal["Euler", "RK4", "implicit", "implicitfast"]
    cone: Literal["pyramidal", "elliptic"]
    impratio: float  # of the frictional to the normal contact's impedance


# Both cars' options. The step is 1/360 s, so that a turn of the NeoRacer's
# LiDAR and a frame of its camera each take a whole number of steps.
# implicitfast integrates the actuators' speed damping (the motors' back-EMF
# and the servo's kd), which explicit Euler cannot hold steady at this step:
# the servo then chatters. The elliptic
# friction cone with a high impratio is MuJoCo's setting for wheeled
# vehicles: it stiffens the tyres' sideways contact, so they creep less (on
# the NeoRacer at 0.4 rad of steering, a full-lock radius of 0.74 m against
# 0.77 m with the defaults and 0.68 m ideal).
SOLVER = Solver(
    timestep=1 / 360, integrator="implicitfast", cone="elliptic", impratio=10
)


@dataclass(frozen=True)
class Inertial:
    """The chassis body's mass and its principal inertia about its centre
    of mass."""

    mass: float  # kg
    pos: Vec3  # m, the centre of mass
    diaginertia: Vec3  # kg·m²


@dataclass(frozen=True)
class CollisionBox:
    """A box that the car's body collides with: the geom of that name."""

    name: str
    center: Vec3  # m
    half_size: Vec3  # m


@dataclass(frozen=True)
class LidarBeams:
    """The LiDAR's eight beams, one every 45 degrees counterclockwise from
    the scan's bearing 0: where they start, where they point, and when they
    read."""

    position: Vec3  # m, the centre of the scan
    yaw: float  # rad, of the scan's bearing 0 from +X
    scan_rate: float  # Hz, turns of its head a second
    field: float  # rad of bearings it sees, centred on bearing 0


@dataclass(frozen=True)
class Wheels:
    """The four wheels, at the corners of the wheelbase and the track."""

    half_wheelbase: float  # m
    half_track: float  # m
    radius: float  # m, the tyre's
    half_width: float  # m, the tyre's
    mass: float  # kg, each
    damping: float  # N·m·s/rad, on each wheel's spin
    friction: Axles  # the tyres' sliding friction on the floor


@dataclass(frozen=True)
class Springs:
    """Each wheel's spring and damper, on a slide joint along Z."""

    range: tuple[float, float]  # m, from full droop (< 0) to full jounce
    stiffness: Axles  # N/m
    damping: Axles  # N·s/m
    springref: Axles  # m, where each spring rests with the car on it


@dataclass(frozen=True)
class SteeringInput:
    """The steering input, a virtual joint at the middle of the front axle
    that the servo turns, and the Ackermann constraints that turn the front
    wheels from it."""

    stops: tuple[float, float]  # rad, the input's range: right (< 0), left
    wheel_range: float  # rad, each front wheel's steer joint, either way
    ackermann_left: Polynomial  # the left wheel's angle from the input
    ackermann_right: Polynomial
    input_body_mass: float  # kg, the input's light virtual body


@dataclass(frozen=True)
class Servo:
    """The steering servo, as a DC motor at the steering input with its own
    position loop."""

    nominal: Nominal  # at the steering input, through the linkage
    kp: float  # V/rad
    kd: float  # V·s/rad
    voltage_limit: float  # V, the loop's clamp
    lock: float  # rad of steering input, |ctrl| at most


@dataclass(frozen=True)
class Delay:
    """How long an actuator holds back each command, and the history of
    commands that it keeps for that."""

    latency: float  # s
    nsample: int


def delay(latency: float) -> Delay:
    """A delay of latency (s), with a sample a step of SOLVER and two to
    spare."""
    return Delay(precise(latency), math.ceil(latency / SOLVER.timestep) + 2)


def spawn_height(wheel_radius: float) -> float:
    """Where the car body spawns (m above the floor): 1 mm above its
    resting height, the axles at the tyre radius."""
    return precise(wheel_radius + 0.001)


def precise(x: float) -> float:
    """x to the model's precision, 6 significant digits."""
    return float(f"{x:.6g}")


def precise3(v: Vec3) -> Vec3:
    return (precise(v[0]), precise(v[1]), precise(v[2]))
