"""
Step and probe helpers for the NeoRacer model: step the physics with a fixed
control, write a control command, settle the car and move it off its start
at random, check that no such move can put a car against a wall, and two
physics probes (car_mass, car_upright_cos) that the conformance tests use.

Frame: +X forward, +Y left, +Z up.
"""

import copy
import math
from dataclasses import dataclass

import mujoco
import numpy as np

from . import contract
from .control.steering import ControlCommand
from .scene import Car, Scene

SETTLE_TIME = 1.0
"""The time (s) of zero ctrl that lets the car reach static equilibrium."""


def apply_ctrl_and_step(
    model: mujoco.MjModel, data: mujoco.MjData, ctrl: float | np.ndarray, steps: int
) -> None:
    data.ctrl[:] = ctrl
    for _ in range(steps):
        mujoco.mj_step(model, data)


def apply_command(car: Car, data: mujoco.MjData, command: ControlCommand) -> None:
    """Write command to data.ctrl: the throttle, in [-1, 1], to the actuators
    it drives, through their throttle maps (contract.throttle_map: on the
    NeoRacer, that fraction of the motor's voltage limit, the battery), and
    the steer (rad, + = left) to the steer servo."""
    t = np.float64(command.throttle)
    for actuator in car.throttled:
        tmap = contract.throttle_map(data.model, actuator)
        data.bind_scalar(actuator).ctrl = contract.throttle_ctrl(tmap, t)
    data.bind_scalar(car.steer).ctrl = command.steer


def settle(model: mujoco.MjModel, data: mujoco.MjData | None = None) -> mujoco.MjData:
    """data, reset, or new MjData if it is None, stepped for SETTLE_TIME with
    zero control so the car rests on the ground."""
    if data is None:
        data = mujoco.MjData(model)
    else:
        mujoco.mj_resetData(model, data)
    apply_ctrl_and_step(model, data, 0.0, settle_steps(model.opt.timestep))
    return data


def settle_steps(timestep: float) -> int:
    """The steps of timestep (s) in SETTLE_TIME (steps_in)."""
    return steps_in(SETTLE_TIME, timestep)


def steps_in(seconds: float, timestep: float) -> int:
    """The steps of timestep (s) in seconds, half a step rounding up, as the
    JS package's settleSteps rounds."""
    return math.floor(seconds / timestep + 0.5)


@dataclass(frozen=True)
class SpawnRange:
    """How far perturb_spawn moves the car: each by an amount uniform in
    [-bound, bound]."""

    lateral: float = 0.15
    """The bound (m) of the move along the car's left."""
    yaw: float = 0.15
    """The bound (rad) of the turn about +Z."""
    resettle: float = 0.14
    """The time (s) of steps after the move, so the suspension settles."""

    def __post_init__(self) -> None:
        for name in ("lateral", "yaw", "resettle"):
            value = getattr(self, name)
            if not 0.0 <= value < math.inf:
                raise ValueError(f"{name} must be finite and >= 0, not {value}")


def perturb_spawn(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    car: Car,
    rng: np.random.Generator,
    spawn: SpawnRange = SpawnRange(),
) -> None:
    """Move the car in data sideways, square to its heading, and turn it
    about +Z, by amounts drawn from rng in that order, then step data for
    spawn.resettle."""
    lateral = float(rng.uniform(-spawn.lateral, spawn.lateral))
    yaw = float(rng.uniform(-spawn.yaw, spawn.yaw))
    qpos = data.bind_scalar(car.joint).qpos  # (x, y, z, qw, qx, qy, qz), a view
    w, x, y, z = qpos[3:7]
    heading = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    qpos[0] -= lateral * math.sin(heading)
    qpos[1] += lateral * math.cos(heading)
    turn = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
    quat = np.empty(4)
    mujoco.mju_mulQuat(quat, qpos[3:7], turn)
    qpos[3:7] = quat
    mujoco.mj_forward(model, data)
    for _ in range(steps_in(spawn.resettle, model.opt.timestep)):
        mujoco.mj_step(model, data)


def spawn_reach(model: mujoco.MjModel, car: Car, spawn: SpawnRange) -> float:
    """How far (m) the move of perturb_spawn, before it resettles, can carry
    a point of the car's colliding geoms: spawn.lateral sideways, plus the
    chord of a turn of spawn.yaw about the car's origin at the farthest such
    a point is from it, with the car where the model puts it."""
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    root = model.bind_scalar(car.body).id
    geoms = np.flatnonzero(
        (model.body_rootid[model.geom_bodyid] == root)
        & ((model.geom_contype != 0) | (model.geom_conaffinity != 0))
    )
    centers = np.linalg.norm(data.geom_xpos[geoms, :2] - data.xpos[root, :2], axis=1)
    far = float((centers + model.geom_rbound[geoms]).max())
    # A turn moves a point farthest at half a turn, whatever the bound past it.
    return spawn.lateral + 2.0 * far * math.sin(min(spawn.yaw, math.pi) / 2)


def spawn_clears_walls(scene: Scene, spawn: SpawnRange) -> bool:
    """Whether no move of perturb_spawn within spawn, before it resettles,
    can put a car of the scene, where its model puts it, against a wall:
    every car is farther than its spawn_reach from every wall, beyond the
    margins of the geoms. The resettle after the move only lets the car's
    suspension settle."""
    model = copy.deepcopy(scene.model)
    for car in scene.cars:
        root = model.bind_scalar(car.body).id
        mine = model.body_rootid[model.geom_bodyid] == root
        model.geom_margin[mine] += spawn_reach(model, car, spawn)
    # MuJoCo makes a contact of two geoms nearer than the sum of their
    # margins, which the wall sensors count. Its midphase skips the geoms of two bodies whose boxes do not overlap,
    # whatever their margins.
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_MIDPHASE
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return all(
        data.bind_scalar(car.truth.wall_contact).data[0] == 0 for car in scene.cars
    )


def car_mass(model: mujoco.MjModel, car: Car) -> float:
    """Total mass (kg) of the car body and its children."""
    return float(model.bind_scalar(car.body).subtreemass[0])


def car_upright_cos(data: mujoco.MjData, car: Car) -> float:
    """The cosine of the angle between the car body's +Z axis and the world
    +Z axis. It is 1.0 when the car is level and below 0 when it is upside
    down. It reads the truth sensor car.truth.zaxis, so it is up to date after
    mj_forward or mj_step (see truth.py)."""
    return float(data.bind_scalar(car.truth.zaxis).data[2])
