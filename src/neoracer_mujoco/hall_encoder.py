"""
The speed that the RACECAR Neo V2's drive board reports from its motor's
hall sensors.

The board times the edges of the hall sensors: 12 a turn of the motor, and
so, through the gears, an edge every 2.25 mm that the tyres roll. The edges
that come in a sample give a speed, the distance they cover over the time
they took, and the board filters it: alpha of the new speed plus 1 - alpha
of the last. After stop_timeout without an edge it reads 0, so the speed of
a car that creeps slower than an edge each stop_timeout reads 0. A program
reads the result as rc.physics.get_encoder_speed().

This module computes the same speed from the drivetrain's angle, the
drive_pos sensor, at each physics step, as RacecarSim does from its wheels'
speed. A step that passes an edge measures the edges in it from the last
edge before them, and it splits its time in proportion to the angle: the
time after the step's last edge is the share of the angle that is left.

Limit: the filter runs once a step that passes an edge, as in RacecarSim;
the board may run it on each edge. At 360 Hz a car at 1 m/s passes about
one edge a step.
"""

import math
from dataclasses import dataclass

from neoracer_spec.derive_v2 import HallEncoderParams


@dataclass(frozen=True)
class HallEncoderState:
    """The encoder between two physics steps. Start it with start()."""

    angle: float  # rad, the drivetrain's angle at the last step
    travel: float  # rad that the drivetrain turned since the last edge
    since: float  # s since the last edge
    speed: float  # m/s, + = forward: the last speed the board reported


def start(params: HallEncoderParams, angle: float) -> HallEncoderState:
    """The encoder of a car at rest, its drivetrain at angle (rad)."""
    return HallEncoderState(angle=angle, travel=0.0, since=0.0, speed=0.0)


def step(
    params: HallEncoderParams, state: HallEncoderState, angle: float, dt: float
) -> HallEncoderState:
    """The encoder after a physics step of dt (s) that left the drivetrain at
    angle (rad)."""
    turn = angle - state.angle
    travel, since = state.travel + abs(turn), state.since + dt
    edges = math.floor(travel / params.edge_angle)
    if edges == 0:
        speed = 0.0 if since > params.stop_timeout else state.speed
        return HallEncoderState(angle, travel, since, speed)
    left = travel - edges * params.edge_angle  # rad past the last edge
    after = since * left / travel  # s since the last edge
    measured = edges * params.edge_angle * params.radius / (since - after)  # m/s
    speed = params.alpha * math.copysign(measured, turn)
    speed += (1.0 - params.alpha) * state.speed
    return HallEncoderState(angle, left, after, speed)
