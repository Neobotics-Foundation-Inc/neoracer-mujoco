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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from . import batch
from . import physics as physics_module
from .physics import ScenePhysics
from .scene import Scene

WARP_PATHS: Mapping[str, str] = MappingProxyType(
    {
        "sensor_interval": "_impl.sensor_interval",
        "sensor_delay": "_impl.sensor_delay",
    }
)
"""Where MJX's model for MuJoCo Warp holds each field of physics.FIELDS that its
Model does not: MuJoCo Warp's own model, _impl, holds the sensors' timing."""

CONTACTS_PER_CAR = 64
"""The most contacts that one car of an environment holds at once. The car on
a track has about 10."""
CONSTRAINTS_PER_CAR = 256
"""The most constraints that one car of an environment holds at once. The car
on a track has about 40."""


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class WarpState:
    """The state of a WarpBatch: environment i steps data[i] with the model
    fields fields[name][i] (physics.FIELDS) in place of the scene's."""

    data: mjx.Data
    fields: dict[str, jax.Array]


class WarpBatch:
    """nbatch copies of scene in a batched mjx.Data (impl="warp"). ctrl,
    done and sensordata are JAX arrays."""

    def __init__(self, scene: Scene, nbatch: int) -> None:
        self.scene = scene
        self.model = scene.model
        self.nbatch = nbatch
        self._mx = mjx.put_model(scene.model, impl="warp")
        self._step = jax.jit(self._step_batch, static_argnames="nstep")
        # Compiled once here, so that each call of initial reuses it.
        self._initial = jax.jit(jax.vmap(self._one_env))

    def initial(
        self,
        datas: Sequence[mujoco.MjData],
        scenes: Sequence[Scene],
        physics: Sequence[ScenePhysics],
    ) -> WarpState:
        """A state whose environment i has the physics state of datas[i] and
        steps with physics[i]. Raise ValueError unless there is one of each
        per row and every scene is the batch's own.

        Limit: MuJoCo Warp collides and casts rays against the geometry of the
        model that put_model was given, so every environment steps on the
        batch's scene. For another track, make another WarpBatch."""
        batch.check_one_per_row(datas, self.nbatch, "MjData")
        batch.check_one_per_row(scenes, self.nbatch, "Scene")
        batch.check_one_per_row(physics, self.nbatch, "Physics")
        if any(scene is not self.scene for scene in scenes):
            raise ValueError("every row of a WarpBatch steps on the batch's scene")
        fields = ("time", "qpos", "qvel", "act", "ctrl", "history")
        # Stacked on the host, so each field goes to the device once.
        rows = {
            f: jnp.asarray(np.stack([getattr(d, f) for d in datas])) for f in fields
        }
        model_rows = physics_module.field_rows(self.scene, physics)
        return WarpState(
            data=self._initial(rows),
            fields={f: jnp.asarray(v) for f, v in model_rows.items()},
        )

    def _one_env(self, row: dict[str, jax.Array]) -> mjx.Data:
        # Made inside vmap, as MJX requires: Warp keeps one contact and
        # constraint buffer for all environments, not one per environment.
        ncar = len(self.scene.cars)
        d = mjx.make_data(
            self.model,
            impl="warp",
            naconmax=self.nbatch * ncar * CONTACTS_PER_CAR,
            njmax=ncar * CONSTRAINTS_PER_CAR,
        )
        return d.replace(**row)

    def _step_batch(
        self, state: WarpState, ctrl: jax.Array, nstep: int
    ) -> tuple[WarpState, jax.Array]:
        def one_env(
            fields: dict[str, jax.Array], d: mjx.Data, c: jax.Array
        ) -> tuple[mjx.Data, jax.Array]:
            model = self._mx.tree_replace(
                {WARP_PATHS.get(f, f): v for f, v in fields.items()}
            )

            def substep(d: mjx.Data, _: None) -> tuple[mjx.Data, jax.Array]:
                d = mjx.step(model, d)
                return d, d.sensordata

            # Start the solver cold, as mujoco.rollout does on the CPU.
            d = d.replace(ctrl=c, qacc_warmstart=jnp.zeros_like(d.qacc_warmstart))
            return jax.lax.scan(substep, d, None, length=nstep)

        data, sensordata = jax.vmap(one_env)(state.fields, state.data, ctrl)
        return WarpState(data, state.fields), sensordata

    def step(
        self, state: WarpState, ctrl: jax.Array, nstep: int
    ) -> tuple[WarpState, jax.Array]:
        """Hold ctrl (nbatch, nu) for nstep >= 1 steps. Return the state after
        the last step and the sensordata (nbatch, nstep, nsensordata) after
        each step."""
        if nstep < 1:
            raise ValueError(f"nstep must be at least 1, not {nstep}")
        return self._step(state, ctrl, nstep=nstep)

    def qpos(self, state: WarpState) -> jax.Array:
        """The joint positions (nbatch, nq) of every environment of state."""
        return state.data.qpos

    def reset(self, state: WarpState, done: jax.Array, initial: WarpState) -> WarpState:
        """state with the environments where done replaced by initial."""
        # Data.where merges only the per-environment fields, as Warp requires.
        data = jax.vmap(lambda s, d, i: s.where(d, i))(state.data, done, initial.data)
        fields = {
            f: jnp.where(done.reshape(-1, *[1] * (v.ndim - 1)), initial.fields[f], v)
            for f, v in state.fields.items()
        }
        return WarpState(data, fields)
