"""
The Warp backend of batch.py: nbatch copies of one model in a batched
mjx.Data, stepped by MJX on MuJoCo Warp. It runs on a CUDA GPU, or on the
CPU, much more slowly, when JAX sees no GPU. It needs the jax extra:
pip install "neoracer-mujoco[jax]".

MJX's other backend, impl="jax", cannot run a NeoRacer scene: it has no
collision between the cylinder wheels and the box walls.

The physics is MuJoCo's, but Warp computes collisions and solves the
constraints with its own code, so a row differs from the CPU backend by a
small amount, and the difference grows after a crash, as two CPU runs with
slightly different states do.
"""

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from . import batch

# The most contacts and constraints that one environment holds at once. The
# car on a track has about 10 contacts and 40 constraints.
CONTACTS_PER_ENV = 64
CONSTRAINTS_PER_ENV = 256


class WarpBatch:
    """nbatch copies of model in a batched mjx.Data (impl="warp"). ctrl,
    done and sensordata are JAX arrays."""

    def __init__(self, model: mujoco.MjModel, nbatch: int) -> None:
        self.model = model
        self.nbatch = nbatch
        self._mx = mjx.put_model(model, impl="warp")
        self._step = jax.jit(self._step_batch, static_argnames="nstep")
        # Compiled once here, so that each call of initial reuses it.
        self._initial = jax.jit(jax.vmap(self._one_env))

    def initial(self, datas: Sequence[mujoco.MjData]) -> mjx.Data:
        """A batched mjx.Data whose environment i has the physics state of
        datas[i]. Raise ValueError unless there is one MjData per row."""
        batch.check_one_per_row(datas, self.nbatch)
        fields = ("time", "qpos", "qvel", "act", "ctrl", "history")
        # Stacked on the host, so each field goes to the device once.
        rows = {
            f: jnp.asarray(np.stack([getattr(d, f) for d in datas])) for f in fields
        }
        return self._initial(rows)

    def _one_env(self, row: dict[str, jax.Array]) -> mjx.Data:
        # Made inside vmap, as MJX requires: Warp keeps one contact and
        # constraint buffer for all environments, not one per environment.
        d = mjx.make_data(
            self.model,
            impl="warp",
            naconmax=self.nbatch * CONTACTS_PER_ENV,
            njmax=CONSTRAINTS_PER_ENV,
        )
        return d.replace(**row)

    def _step_batch(
        self, state: mjx.Data, ctrl: jax.Array, nstep: int
    ) -> tuple[mjx.Data, jax.Array]:
        def substep(d: mjx.Data, _: None) -> tuple[mjx.Data, jax.Array]:
            d = mjx.step(self._mx, d)
            return d, d.sensordata

        def one_env(d: mjx.Data, c: jax.Array) -> tuple[mjx.Data, jax.Array]:
            # Start the solver cold, as mujoco.rollout does on the CPU.
            d = d.replace(ctrl=c, qacc_warmstart=jnp.zeros_like(d.qacc_warmstart))
            return jax.lax.scan(substep, d, None, length=nstep)

        return jax.vmap(one_env)(state, ctrl)

    def step(
        self, state: mjx.Data, ctrl: jax.Array, nstep: int
    ) -> tuple[mjx.Data, jax.Array]:
        """Hold ctrl (nbatch, nu) for nstep >= 1 steps. Return the state after
        the last step and the sensordata (nbatch, nstep, nsensordata) after
        each step."""
        if nstep < 1:
            raise ValueError(f"nstep must be at least 1, not {nstep}")
        return self._step(state, ctrl, nstep=nstep)

    def reset(self, state: mjx.Data, done: jax.Array, initial: mjx.Data) -> mjx.Data:
        """state with the environments where done replaced by initial."""
        # Data.where merges only the per-environment fields, as Warp requires.
        return jax.vmap(lambda s, d, i: s.where(d, i))(state, done, initial)
