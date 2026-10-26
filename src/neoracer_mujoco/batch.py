"""
Step a batch of copies of one scene at once, and read and write the batch
arrays by spec element.

A batch holds nbatch environments of one scene. Each row has its own
physics (physics.ScenePhysics), so a batch can train on many slightly
different cars (see physics.py). You choose the backend:

- CpuBatch, optimized for CPU. It keeps one row of mjSTATE_FULLPHYSICS
  state per environment and steps the rows with mujoco.rollout on a pool of
  threads. The threads run in C, so the batch uses every core on any Python
  build.
- batch_warp.WarpBatch, optimized for GPU. It keeps a batched mjx.Data and
  steps it with MJX on MuJoCo Warp, on a CUDA GPU. It needs the jax extra:
  pip install "neoracer-mujoco[jax]".

Both have the same methods (the Batch protocol):

    sim = CpuBatch(scene, nbatch=64, nthread=4)
    initial = sim.initial([data] * 64, [scene] * 64, [Physics()] * 64)  # per row
    ctrl = command(scene.model, scene.car, throttle, steer)  # (64, nu)
    state, sensordata = sim.step(initial, ctrl, nstep=17)  # (64, 17, nsensordata)
    zaxis = read(scene.model, sensordata[:, -1], scene.car.truth.zaxis)  # (64, 3)
    state = sim.reset(state, done, initial)  # rows where done start over

The state of a row holds its scene and its physics, so a row that starts
over takes those of initial too. CpuBatch can give each row its own scene,
such as a generated track of fixed size; WarpBatch steps every row on its
own scene.

step holds ctrl for nstep physics steps and returns the sensor values after
each one. Each call starts the constraint solver cold (qacc_warmstart = 0),
as mujoco.rollout does, so a CPU row is the same, bit for bit, as one MjData
stepped the same way. DataBatch steps one MjData so, as a batch of one row,
and drive takes it as it takes a batch, so a single-car environment that
owns one MjData steps it as a batch row steps.

A row of a scene of several cars (assets.compose_cars) holds them all:
command_cars gives each car its throttle and steer, and drive_cars runs
each car's speed loop, with the cars on the arrays' last axis.

A batch array has no bind, so command, command_cars, read and drive_angle
bind the element to the model and index the array with its address. They
are the only code that uses an address. They work on numpy and JAX arrays
alike.
"""

import copy
import functools
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import mujoco
import numpy as np
import numpy.typing as npt
from mujoco import rollout

from neoracer_spec.derive import SpeedLoopParams

from . import contract, speed_loop
from .contract import Array
from .physics import Physics, ScenePhysics, apply_physics
from .scene import Car, Scene

type States = npt.NDArray[np.float64]
"""Shape (nbatch, nstate), each row an mjSTATE_FULLPHYSICS state."""
type Rows = npt.NDArray[np.float64]
"""Shape (nbatch, n)."""
type SensorData = npt.NDArray[np.float64]
"""Shape (nbatch, nstep, nsensordata)."""
type Done = npt.NDArray[np.bool_]
"""Shape (nbatch,)."""

FULL_PHYSICS = mujoco.mjtState.mjSTATE_FULLPHYSICS


class Stepper[S, A](Protocol):
    """Rows that step, as drive steps them. S is the state, A the array type
    of ctrl and sensordata."""

    def step(self, state: S, ctrl: A, nstep: int) -> tuple[S, A]: ...

    def qpos(self, state: S) -> A:
        """The joint positions (nbatch, nq) of every row of state."""
        ...


class Batch[S, A, M](Stepper[S, A], Protocol):
    """A batch of nbatch copies of one scene. M is the array type of the
    done mask."""

    nbatch: int

    def initial(
        self,
        datas: Sequence[mujoco.MjData],
        scenes: Sequence[Scene],
        physics: Sequence[ScenePhysics],
    ) -> S: ...

    def reset(self, state: S, done: M, initial: S) -> S: ...


@dataclass(frozen=True)
class CpuState:
    """The state of a CpuBatch: row i steps from rows[i] with models[i]."""

    rows: States
    models: tuple[mujoco.MjModel, ...]
    """One model per row. Rows with the same scene and physics share one."""


class CpuBatch:
    """nbatch rows of scene, or of scenes with the same sizes, such as
    generated tracks with a fixed number of wall boxes, stepped by
    mujoco.rollout on nthread threads. Call close() to stop the threads."""

    def __init__(self, scene: Scene, nbatch: int, nthread: int) -> None:
        self.scene = scene
        self.model = scene.model
        self.nbatch = nbatch
        self._pool = rollout.Rollout(nthread=nthread)
        # rollout steps each row on the MjData of the thread that runs it.
        # The models of all rows have the same sizes, so any of them fits.
        self._datas = [mujoco.MjData(scene.model) for _ in range(max(nthread, 1))]

    def initial(
        self,
        datas: Sequence[mujoco.MjData],
        scenes: Sequence[Scene],
        physics: Sequence[ScenePhysics],
    ) -> CpuState:
        """A state whose row i starts from the physics state of datas[i] and
        steps on scenes[i] with physics[i]. Raise ValueError unless there is
        one of each per row and every scene has the sizes of the batch's."""
        check_one_per_row(datas, self.nbatch, "MjData")
        check_one_per_row(scenes, self.nbatch, "Scene")
        check_one_per_row(physics, self.nbatch, "Physics")
        for scene in scenes:
            check_same_sizes(self.model, scene.model)
        rows = np.empty((self.nbatch, mujoco.mj_stateSize(self.model, FULL_PHYSICS)))
        for row, data in zip(rows, datas, strict=True):
            mujoco.mj_getState(self.model, data, row, FULL_PHYSICS)
        models: dict[tuple[int, ScenePhysics], mujoco.MjModel] = {}
        for scene, p in zip(scenes, physics, strict=True):
            if (id(scene), p) not in models:
                model = scene.model
                if p != Physics():
                    model = copy.deepcopy(scene.model)
                    apply_physics(model, scene, p)
                models[id(scene), p] = model
        return CpuState(
            rows,
            tuple(models[id(s), p] for s, p in zip(scenes, physics, strict=True)),
        )

    def step(
        self, state: CpuState, ctrl: Rows, nstep: int
    ) -> tuple[CpuState, SensorData]:
        """Hold ctrl (nbatch, nu) for nstep >= 1 steps. Return the state after
        the last step and the sensordata after each step."""
        if nstep < 1:
            # rollout reads nstep=0 as "not given" and steps once.
            raise ValueError(f"nstep must be at least 1, not {nstep}")
        states, sensordata = self._pool.rollout(
            list(state.models), self._datas, state.rows, ctrl[:, None, :], nstep=nstep
        )
        # A copy, so the states of the other steps can be freed.
        return CpuState(states[:, -1].copy(), state.models), sensordata

    def qpos(self, state: CpuState) -> Rows:
        """The joint positions (nbatch, nq) of every row of state."""
        start = mujoco.mj_stateSize(self.model, mujoco.mjtState.mjSTATE_TIME)
        return state.rows[:, start : start + self.model.nq]

    def reset(self, state: CpuState, done: Done, initial: CpuState) -> CpuState:
        """state with the rows where done replaced by the rows of initial."""
        models = zip(state.models, initial.models, done, strict=True)
        return CpuState(
            np.where(done[:, None], initial.rows, state.rows),
            tuple(i if d else s for s, i, d in models),
        )

    def close(self) -> None:
        self._pool.close()


class DataBatch:
    """One MjData as a batch of one row, stepped as CpuBatch steps a row. Its
    state is the MjData, which step advances in place."""

    def __init__(self, model: mujoco.MjModel) -> None:
        self.model = model

    def step(
        self, state: mujoco.MjData, ctrl: Rows, nstep: int
    ) -> tuple[mujoco.MjData, SensorData]:
        """Hold ctrl (1, nu) for nstep >= 1 steps of mj_step from a cold
        solver, as mujoco.rollout steps a row. Return state and the sensordata
        after each step (1, nstep, nsensordata)."""
        if nstep < 1:
            raise ValueError(f"nstep must be at least 1, not {nstep}")
        state.qacc_warmstart[:] = 0
        state.ctrl[:] = ctrl[0]
        sensordata = np.empty((1, nstep, self.model.nsensordata))
        for k in range(nstep):
            mujoco.mj_step(self.model, state)
            sensordata[0, k] = state.sensordata
        return state, sensordata

    def qpos(self, state: mujoco.MjData) -> Rows:
        """The joint positions of state as one row (1, nq)."""
        return state.qpos[None]


def check_same_sizes(model: mujoco.MjModel, other: mujoco.MjModel) -> None:
    """Raise ValueError unless other has the size signature of model, the
    hash of its sizes that mujoco.rollout needs to be the same for the
    models of one batch. Names may differ."""
    if other.signature != model.signature:
        raise ValueError(
            "the scenes of a batch need the same sizes, as the same number of "
            "wall boxes gives generated tracks"
        )


def check_one_per_row(items: Sequence[object], nbatch: int, what: str) -> None:
    """Raise ValueError unless items has one item, a what, for each of nbatch
    rows."""
    if len(items) != nbatch:
        raise ValueError(
            f"initial needs one {what} per row: {nbatch}, not {len(items)}"
        )


def command[A: Array](model: mujoco.MjModel, car: Car, throttle: A, steer: A) -> A:
    """The ctrl rows (nbatch, nu) that give the actuators that the throttle
    drives throttle, in [-1, 1], through their throttle maps
    (contract.throttle_map: on the NeoRacer, that fraction of the motor's
    voltage limit, the battery), and the steer servo steer (rad, + = left).
    throttle and steer have shape (nbatch,). Raise ValueError if model has
    actuators that are not the car's."""
    return command_cars(model, [car], throttle[..., None], steer[..., None])


def command_cars[A: Array](
    model: mujoco.MjModel, cars: Sequence[Car], throttle: A, steer: A
) -> A:
    """The ctrl rows (nbatch, nu) that give each of cars its throttle and its
    steer, as command gives one car: throttle[:, k] and steer[:, k], of
    shape (nbatch, len(cars)), are cars[k]'s. Raise ValueError unless each of
    model's actuators is one car's."""
    count = sum(len(car.throttled) + 1 for car in cars)
    if model.nu != count:
        raise ValueError(f"the model has {model.nu} actuators, the cars have {count}")
    xp = throttle.__array_namespace__()  # numpy or jax.numpy
    column = {}
    for k, car in enumerate(cars):
        for actuator in car.throttled:
            tmap = contract.throttle_map(model, actuator)
            value = contract.throttle_ctrl(tmap, throttle[..., k])
            column[int(model.bind_scalar(actuator).id)] = value
        column[int(model.bind_scalar(car.steer).id)] = steer[..., k]
    if len(column) != model.nu:
        raise ValueError(f"the cars share actuators: {len(column)} of {model.nu}")
    return xp.stack([column[i] for i in range(model.nu)], axis=-1)


def drive[S, A: Array](
    sim: Stepper[S, A],
    model: mujoco.MjModel,
    car: Car,
    params: SpeedLoopParams,
    state: S,
    loop: speed_loop.SpeedLoopState,
    commanded: npt.NDArray[np.float64],
    steer: A,
    nstep: int,
) -> tuple[S, A, speed_loop.SpeedLoopState]:
    """Step nstep physics steps with the firmware's speed loop of every row
    (speed_loop.py) holding the speed commanded (m/s), on the host, and the
    steer servo holding steer (rad), an array of sim's, as its ctrl is; both
    (nbatch,). The rows' loops tick together, so the steps run in pieces from
    one tick to the next, the throttle held over each. Returns the state, the
    sensor values after each step (nbatch, nstep, nsensordata), arrays of
    sim's, and the loop.

    Only a piece's last step can close the firmware's window, and the encoder
    reads the drivetrain only then, so the loop reads it once a piece: at the
    piece's end, from qpos (drive_angle), as the sensors after a
    step hold its start.

    Limit: the loop runs on the host in numpy, so on Warp each piece sends
    the throttle to the GPU and brings the drivetrain's angles back."""
    return _drive(
        sim,
        model,
        params,
        state,
        loop,
        commanded,
        steer,
        nstep,
        functools.partial(command, model, car),
        functools.partial(drive_angle, model, car),
    )


def drive_cars[S, A: Array](
    sim: Stepper[S, A],
    model: mujoco.MjModel,
    cars: Sequence[Car],
    params: SpeedLoopParams,
    state: S,
    loop: speed_loop.SpeedLoopState,
    commanded: npt.NDArray[np.float64],
    steer: A,
    nstep: int,
) -> tuple[S, A, speed_loop.SpeedLoopState]:
    """drive for each of cars, each with a speed loop of params:
    commanded[:, k] and steer[:, k], of shape (nbatch, len(cars)), are
    cars[k]'s, and so is each array of the loop's."""
    return _drive(
        sim,
        model,
        params,
        state,
        loop,
        commanded,
        steer,
        nstep,
        functools.partial(command_cars, model, cars),
        functools.partial(_drive_angles, model, cars),
    )


def _drive[S, A: Array](
    sim: Stepper[S, A],
    model: mujoco.MjModel,
    params: SpeedLoopParams,
    state: S,
    loop: speed_loop.SpeedLoopState,
    commanded: npt.NDArray[np.float64],
    steer: A,
    nstep: int,
    ctrl: Callable[[A, A], A],
    angle: Callable[[A], A],
) -> tuple[S, A, speed_loop.SpeedLoopState]:
    """drive, with ctrl giving the ctrl rows of a throttle and a steer, and
    angle the drivetrains' angles in the rows of a qpos."""
    if nstep < 1:
        raise ValueError(f"nstep must be at least 1, not {nstep}")
    dt = float(model.opt.timestep)
    xp = steer.__array_namespace__()  # numpy or jax.numpy
    pieces = []
    left = nstep
    while left > 0:
        k = min(left, speed_loop.steps_to_tick(loop, dt))
        throttle = xp.asarray(np.broadcast_to(loop.throttle, commanded.shape))
        state, sensordata = sim.step(state, ctrl(throttle, steer), k)
        drivetrain = np.asarray(angle(sim.qpos(state)))
        for _ in range(k):
            loop = speed_loop.step(params, loop, commanded, drivetrain, dt)
        pieces.append(sensordata)
        left -= k
    xs = pieces[0].__array_namespace__()  # the backend's: numpy or jax.numpy
    return state, xs.concatenate(pieces, axis=1), loop


def drive_angle[A: Array](model: mujoco.MjModel, car: Car, qpos: A) -> A:
    """The drivetrain's angle (rad) in each row of qpos (..., nq): the length
    of the fixed tendon that the car's drive_pos sensor reads, summed as
    MuJoCo sums it, so it is drive_pos's reading at that qpos. After mj_step,
    and after each step of a rollout, the sensors still hold the state at the
    step's start; this reads it at the step's end. Raise ValueError if
    drive_pos reads anything else."""
    sensor = model.bind_scalar(car.sensors.drive_pos)
    refused = ValueError("the drive_pos sensor does not read a fixed tendon's length")
    if sensor.type[0] != mujoco.mjtSensor.mjSENS_TENDONPOS:
        raise refused
    tendon = int(sensor.objid[0])
    adr, num = int(model.tendon_adr[tendon]), int(model.tendon_num[tendon])
    wraps = range(adr, adr + num)
    if any(model.wrap_type[w] != mujoco.mjtWrap.mjWRAP_JOINT for w in wraps):
        raise refused
    angle: Any = 0.0
    for w in wraps:
        joint = int(model.wrap_objid[w])
        angle = angle + float(model.wrap_prm[w]) * qpos[..., model.jnt_qposadr[joint]]
    return angle


def _drive_angles[A: Array](model: mujoco.MjModel, cars: Sequence[Car], qpos: A) -> A:
    """drive_angle of each of cars, on the last axis: shape (..., len(cars))."""
    xp = qpos.__array_namespace__()  # numpy or jax.numpy
    return xp.stack([drive_angle(model, car, qpos) for car in cars], axis=-1)


def read[A: Array](model: mujoco.MjModel, sensordata: A, sensor: mujoco.MjsSensor) -> A:
    """The values of sensor in sensordata, whose last axis is nsensordata:
    shape (..., dim)."""
    bound = model.bind_scalar(sensor)
    adr, dim = int(bound.adr[0]), int(bound.dim[0])
    return sensordata[..., adr : adr + dim]
