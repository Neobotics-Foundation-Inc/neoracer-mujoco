"""
What the cars' models share: records of the numbers of a part that both
models have, which derive.py and derive_v2.py make from each car's spec and
mjcf.py writes into each car's XML. The cars stay separate; their parts
come from here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Solver:
    """MuJoCo's options for a car's model. A scene with the car takes them
    from it (neoracer_mujoco.assets.compose), so a track sets none."""

    timestep: float  # s
    integrator: Literal["Euler", "RK4", "implicit", "implicitfast"]
    cone: Literal["pyramidal", "elliptic"]
    impratio: float  # of the frictional to the normal contact's impedance


# Both cars' options. The step is 1/360 s: 12 steps to a 30 Hz LiDAR scan
# and 6 to a 60 Hz frame. implicitfast integrates the actuators' speed
# damping (the motors' back-EMF and the servo's kd), which explicit Euler
# cannot hold steady at this step: the servo then chatters. The elliptic
# friction cone with a high impratio is MuJoCo's setting for wheeled
# vehicles: it stiffens the tyres' sideways contact, so they creep less (on
# the NeoRacer at 0.4 rad of steering, a full-lock radius of 0.74 m against
# 0.77 m with the defaults and 0.68 m ideal).
SOLVER = Solver(
    timestep=1 / 360, integrator="implicitfast", cone="elliptic", impratio=10
)


def nsample(latency: float) -> int:
    """The history that a delay of latency (s) needs: a sample a step of
    SOLVER, and two to spare."""
    return math.ceil(latency / SOLVER.timestep) + 2
