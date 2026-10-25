"""
What the cars' models share: records of the numbers of a part that both
models have, which derive.py and derive_v2.py make from each car's spec and
mjcf.py writes into each car's XML. The cars stay separate; their parts
come from here.

Numbers that a model holds are kept to its precision, 6 significant digits
(precise), so the code and the XML hold the same floats.

Frame: the car body's: +X forward, +Y left, +Z up, its origin at the middle
of the wheelbase, on the axle line. Quantities are in SI units unless a name
says otherwise.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from . import ackermann
from .spec import Vec3

type Axles = tuple[float, float]
"""A value at the front axle, then at the rear."""
type Nominal = tuple[float, float, float]
"""A dcmotor's nominal: its voltage (V), stall torque (N·m) and free speed
(rad/s)."""
type Polynomial = tuple[float, float, float, float, float]
"""A 4th-order polynomial's coefficients, a0 to a4."""


@dataclass(frozen=True)
class Solver:
    """MuJoCo's options for a car's model. A scene with the car takes them
    from it (neoracer_mujoco.assets.compose), so a track sets none."""

    timestep: float
    integrator: Literal["Euler", "RK4", "implicit", "implicitfast"]
    cone: Literal["pyramidal", "elliptic"]
    impratio: float
    """The ratio of the frictional to the normal contact's impedance."""


SOLVER = Solver(
    timestep=1 / 360, integrator="implicitfast", cone="elliptic", impratio=10
)
"""Both cars' options. The step is 1/360 s, so that a turn of the NeoRacer's
LiDAR and a frame of its camera each take a whole number of steps.
implicitfast integrates the actuators' speed damping (the motors' back-EMF
and the servo's kd), which explicit Euler cannot hold steady at this step:
the servo then chatters. The elliptic friction cone with a high impratio is
MuJoCo's setting for wheeled vehicles: it stiffens the tires' sideways
contact, so they creep less (on the NeoRacer at 0.4 rad of steering, a
full-lock radius of 0.74 m against 0.77 m with the defaults and 0.68 m
ideal)."""

G = 9.81
"""MuJoCo's gravity (m/s²)."""


@dataclass(frozen=True)
class Inertial:
    """The chassis body's mass and its principal inertia (kg·m²) about its
    center of mass."""

    mass: float
    pos: Vec3
    """The center of mass."""
    diaginertia: Vec3


@dataclass(frozen=True)
class CollisionBox:
    """A box that the car's body collides with: the geom of that name."""

    name: str
    center: Vec3
    half_size: Vec3


@dataclass(frozen=True)
class CollisionMesh:
    """A convex mesh that the car's body collides with: the geom and the
    mesh of that name, the convex hull of vertices, which MuJoCo makes."""

    name: str
    vertices: tuple[Vec3, ...]


@dataclass(frozen=True)
class LidarBeams:
    """The LiDAR's eight beams, one every 45 degrees counterclockwise from
    the scan's bearing 0: where they start, where they point, and when they
    read."""

    position: Vec3
    """The center of the scan."""
    yaw: float
    """The yaw of the scan's bearing 0 from +X."""
    scan_rate: float
    """Turns of its head a second."""
    field: float
    """The bearings it sees, centered on bearing 0."""


@dataclass(frozen=True)
class Wheels:
    """The four wheels, at the corners of the wheelbase and the track."""

    half_wheelbase: float
    half_track: float
    radius: float
    """The tire's radius."""
    half_width: float
    """Half the tire's width."""
    mass: float
    """Each wheel's mass."""
    damping: float
    """The damping on each wheel's spin (N·m·s/rad)."""
    friction: Axles
    """The tires' sliding friction on the floor."""


@dataclass(frozen=True)
class Springs:
    """Each wheel's spring and damper, on a slide joint along Z."""

    range: tuple[float, float]
    """The joint's range, from full droop (< 0) to full jounce."""
    stiffness: Axles
    """Each wheel's spring, by axle (N/m)."""
    damping: Axles
    """Each wheel's damper, by axle (N·s/m)."""
    springref: Axles
    """Where each spring rests with the car on it."""


@dataclass(frozen=True)
class SteeringInput:
    """The steering input, a virtual joint at the middle of the front axle
    that the servo turns, and the Ackermann constraints that turn the front
    wheels from it."""

    stops: tuple[float, float]
    """The input's range: right (< 0), then left."""
    wheel_range: float
    """The range of each front wheel's steer joint, either way."""
    ackermann_left: Polynomial
    """The left wheel's angle from the input."""
    ackermann_right: Polynomial
    input_body_mass: float
    """The mass of the input's light virtual body."""
    armature: float
    """The inertia added at the input and at each front wheel's steer joint
    (kg·m²)."""


@dataclass(frozen=True)
class Servo:
    """The steering servo, as a DC motor at the steering input with its own
    position loop."""

    nominal: Nominal
    """The servo's nominal at the steering input, through the linkage."""
    kp: float
    """The proportional gain of its position loop (V/rad)."""
    kd: float
    """The derivative gain of its position loop (V·s/rad)."""
    voltage_limit: float
    """The loop's clamp."""
    lock: float
    """The steering input that |ctrl| reaches at most."""
    delay: Delay
    """The delay from a command to the servo acting on it."""


@dataclass(frozen=True)
class Delay:
    """How long an actuator holds back each command, and the history of
    commands that it keeps for that."""

    latency: float
    nsample: int


@dataclass(frozen=True)
class Model:
    """The parts that both cars' models have, which each car's ModelParams
    holds beside its own. Positions are in the car's body frame."""

    spawn_height: float
    """The body frame's height above the floor at spawn."""
    inertial: Inertial
    shapes: tuple[CollisionBox | CollisionMesh, ...]
    """What the car's body collides with."""
    imu_position: Vec3
    lidar: LidarBeams
    wheels: Wheels
    springs: Springs
    steering: SteeringInput
    servo: Servo


INPUT_BODY_MASS = 0.01
"""The mass of the steering input's virtual body, light beside the chassis:
the model's own choice, not a car's."""
WHEEL_RANGE_MARGIN = 0.015
"""How far the front wheels' steer joints reach past the widest angle that
the Ackermann constraints ask of them, so that their limits never oppose the
constraints: the model's own choice, not a car's."""


def steering_input(
    wheelbase: float,
    track: float,
    stops: tuple[float, float],
    virtual: Callable[[float], float],
    armature: float,
) -> SteeringInput:
    """The steering input from stops[0] to stops[1] (rad), which turns the
    virtual wheel at the middle of the front axle to virtual(x) (rad) at x.
    Each front wheel's polynomial is a least-squares fit to ideal Ackermann
    steering from the virtual wheel over that range, and the steer joints
    reach WHEEL_RANGE_MARGIN past the widest angle of either wheel."""
    x = np.linspace(*stops, 401)
    angles = [virtual(float(s)) for s in x]
    left = np.array([ackermann.left_wheel(wheelbase, track, a) for a in angles])
    right = np.array([ackermann.right_wheel(wheelbase, track, a) for a in angles])
    widest = float(max(np.abs(left).max(), np.abs(right).max()))
    return SteeringInput(
        stops=(precise(stops[0]), precise(stops[1])),
        wheel_range=precise(widest + WHEEL_RANGE_MARGIN),
        ackermann_left=quartic(x, left),
        ackermann_right=quartic(x, right),
        input_body_mass=INPUT_BODY_MASS,
        armature=precise(armature),
    )


def quartic(x: np.ndarray, y: np.ndarray) -> Polynomial:
    """The least-squares 4th-order polynomial through (x, y), to the model's
    precision."""
    a0, a1, a2, a3, a4 = (
        precise(float(a)) for a in np.polynomial.polynomial.polyfit(x, y, 4)
    )
    return (a0, a1, a2, a3, a4)


def chassis_mass(car: float, wheel: float) -> float:
    """The chassis body's mass (kg): the car's, less its four wheels and the
    steering input's body."""
    return car - 4 * wheel - INPUT_BODY_MASS


def spring_ref(load: float, stiffness: float) -> float:
    """The rest position (m) of a spring of stiffness (N/m) that holds load
    (kg) with its joint at 0."""
    return -load * G / stiffness


def delay(latency: float) -> Delay:
    """A delay of latency (s), with a sample a step of SOLVER and two to
    spare."""
    return Delay(precise(latency), math.ceil(latency / SOLVER.timestep) + 2)


def spawn_height(wheel_radius: float) -> float:
    """Where the car body spawns (m above the floor): 1 mm above its
    resting height, the axles at the tire radius."""
    return precise(wheel_radius + 0.001)


def precise(x: float) -> float:
    """x to the model's precision, 6 significant digits."""
    return float(f"{x:.6g}")


def precise3(v: Vec3) -> Vec3:
    return (precise(v[0]), precise(v[1]), precise(v[2]))
