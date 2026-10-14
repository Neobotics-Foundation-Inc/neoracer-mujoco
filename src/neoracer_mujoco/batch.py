"""
Step a batch of copies of one scene at once, and read and write the batch
arrays by spec element.

A batch holds nbatch environments of one MjModel. You choose the backend:

- CpuBatch, optimized for CPU. It keeps one row of mjSTATE_FULLPHYSICS
  state per environment and steps the rows with mujoco.rollout on a pool of
  threads. The threads run in C, so the batch uses every core on any Python
  build.
- batch_warp.WarpBatch, optimized for GPU. It keeps a batched mjx.Data and
  steps it with MJX on MuJoCo Warp, on a CUDA GPU. It needs the jax extra:
  pip install "neoracer-mujoco[jax]".

Both have the same methods (the Batch protocol):

    sim = CpuBatch(scene.model, nbatch=64, nthread=4)
    initial = sim.initial(data)  # every row is a copy of data
    ctrl = command(scene.model, scene.car, throttle, steer)  # (64, nu)
    state, sensordata = sim.step(initial, ctrl, nstep=17)  # (64, 17, nsensordata)
    zaxis = read(scene.model, sensordata[:, -1], scene.car.truth.zaxis)  # (64, 3)
    state = sim.reset(state, done, initial)  # rows where done start over

step holds ctrl for nstep physics steps and returns the sensor values after
each one. Each call starts the constraint solver cold (qacc_warmstart = 0),
as mujoco.rollout does, so a CPU row is the same, bit for bit, as one MjData
stepped the same way.

A batch array has no bind, so command and read bind the element to the
model and index the array with its address. They are the only code that
uses an address. They work on numpy and JAX arrays alike.
"""

from typing import Any, Protocol

import mujoco
import numpy as np
import numpy.typing as npt
from mujoco import rollout

from .scene import Car

type States = npt.NDArray[np.float64]  # shape (nbatch, nstate), mjSTATE_FULLPHYSICS
type Rows = npt.NDArray[np.float64]  # shape (nbatch, n)
type SensorData = npt.NDArray[np.float64]  # shape (nbatch, nstep, nsensordata)
type Done = npt.NDArray[np.bool_]  # shape (nbatch,)

FULL_PHYSICS = mujoco.mjtState.mjSTATE_FULLPHYSICS


class Array(Protocol):
    """A numpy or a JAX array: what command and read need of one."""

    def __array_namespace__(self, *, api_version: None = None) -> Any: ...

    def __getitem__(self, key: Any, /) -> Any: ...


class Batch[S, A, M](Protocol):
    """A batch of nbatch copies of one model. S is the batch state, A the
    array type of ctrl and sensordata, M the array type of the done mask."""

    nbatch: int

    def initial(self, data: mujoco.MjData) -> S: ...

    def step(self, state: S, ctrl: A, nstep: int) -> tuple[S, A]: ...

    def reset(self, state: S, done: M, initial: S) -> S: ...


class CpuBatch:
    """nbatch copies of model, stepped by mujoco.rollout on nthread threads.
    The state is an array of mjSTATE_FULLPHYSICS rows. Call close() to stop
    the threads."""

    def __init__(self, model: mujoco.MjModel, nbatch: int, nthread: int) -> None:
        self.model = model
        self.nbatch = nbatch
        self._pool = rollout.Rollout(nthread=nthread)
        # rollout steps each row on the MjData of the thread that runs it.
        self._datas = [mujoco.MjData(model) for _ in range(max(nthread, 1))]

    def initial(self, data: mujoco.MjData) -> States:
        """A state with every row a copy of the physics state of data."""
        row = np.empty(mujoco.mj_stateSize(self.model, FULL_PHYSICS))
        mujoco.mj_getState(self.model, data, row, FULL_PHYSICS)
        return np.tile(row, (self.nbatch, 1))

    def step(self, state: States, ctrl: Rows, nstep: int) -> tuple[States, SensorData]:
        """Hold ctrl (nbatch, nu) for nstep >= 1 steps. Return the state after
        the last step and the sensordata after each step."""
        if nstep < 1:
            # rollout reads nstep=0 as "not given" and steps once.
            raise ValueError(f"nstep must be at least 1, not {nstep}")
        states, sensordata = self._pool.rollout(
            self.model, self._datas, state, ctrl[:, None, :], nstep=nstep
        )
        # A copy, so the states of the other steps can be freed.
        return states[:, -1].copy(), sensordata

    def reset(self, state: States, done: Done, initial: States) -> States:
        """state with the rows where done replaced by the rows of initial."""
        return np.where(done[:, None], initial, state)

    def close(self) -> None:
        self._pool.close()


def command[A: Array](model: mujoco.MjModel, car: Car, throttle: A, steer: A) -> A:
    """The ctrl rows (nbatch, nu) that give every drive motor throttle (N·m)
    and the steer servo steer (rad, + = left). throttle and steer have shape
    (nbatch,). Raise ValueError if model has actuators that are not the car's."""
    elements = (*car.drive, car.steer)
    if model.nu != len(elements):
        raise ValueError(
            f"the model has {model.nu} actuators, the car has {len(elements)}"
        )
    values = (throttle,) * len(car.drive) + (steer,)
    column = {
        int(model.bind_scalar(a).id): v for a, v in zip(elements, values, strict=True)
    }
    xp = throttle.__array_namespace__()  # numpy or jax.numpy
    return xp.stack([column[i] for i in range(model.nu)], axis=-1)


def read[A: Array](model: mujoco.MjModel, sensordata: A, sensor: mujoco.MjsSensor) -> A:
    """The values of sensor in sensordata, whose last axis is nsensordata:
    shape (..., dim)."""
    bound = model.bind_scalar(sensor)
    adr, dim = int(bound.adr[0]), int(bound.dim[0])
    return sensordata[..., adr : adr + dim]
