"""
Scene loading and spawn perturbation for the trial scripts of the
vehicle-space wall follower.

Frame: +X forward, +Y left, +Z up.
"""

import os
from dataclasses import dataclass
from typing import Literal, assert_never

import mujoco
import numpy as np

from neoracer_mujoco import assets
from neoracer_mujoco.scene import Scene

type TrackName = Literal["straight_corridor.xml", "loop_corridor.xml"]


def _spawn_offset(track: TrackName) -> assets.Position:
    """World position (m) of the car at the start of track."""
    match track:
        case "straight_corridor.xml":
            return (0.0, 0.0, 0.0)
        case "loop_corridor.xml":
            # The bottom straight is at y = -2.5 m, midway between the inner
            # wall (y = -2 m) and the outer wall (y = -3 m).
            return (0.0, -2.5, 0.0)
        case _:
            assert_never(track)


def load_track_scene(track: TrackName) -> Scene:
    """Compile assets/tracks/<track> with the car at the center of the
    corridor, facing +X."""
    return assets.compose(
        mujoco.MjSpec.from_file(os.path.join(assets.TRACK_DIR, track)),
        pos=_spawn_offset(track),
    )


@dataclass(frozen=True)
class SpawnParams:
    """Range of the random spawn perturbation. Each value is uniform in
    [-bound, +bound]."""

    lateral_m: float = 0.15  # m; bound of the shift along world Y
    yaw_rad: float = 0.15  # rad; bound of the rotation about +Z
    resettle_steps: int = 50  # physics steps after the shift, to settle the suspension


def _compose_yaw(q: np.ndarray, yaw: float) -> np.ndarray:
    """q * dq, where dq is a rotation of yaw (rad) about +Z. q and the result
    are quaternions (w, x, y, z).

    This function writes out the product. mujoco.mju_mulQuat gives a result
    that differs in the last bit, and the seeded trial results must not
    change.
    """
    half = yaw / 2.0
    w0, x0, y0, z0 = q
    w1, x1, y1, z1 = np.cos(half), 0.0, 0.0, np.sin(half)
    return np.array(
        [
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        ]
    )


def perturb_spawn(
    scene: Scene,
    data: mujoco.MjData,
    rng: np.random.Generator,
    params: SpawnParams = SpawnParams(),
) -> None:
    """Shift the car along world Y and turn it about +Z by random amounts
    from rng. Then step data resettle_steps times."""
    lateral = float(rng.uniform(-params.lateral_m, params.lateral_m))
    yaw = float(rng.uniform(-params.yaw_rad, params.yaw_rad))
    # qpos of the car freejoint: (x, y, z, qw, qx, qy, qz). A view of data.qpos.
    qpos = data.bind_scalar(scene.car.joint).qpos
    qpos[1] += lateral
    qpos[3:7] = _compose_yaw(qpos[3:7], yaw)
    model = scene.model
    mujoco.mj_forward(model, data)
    for _ in range(params.resettle_steps):
        mujoco.mj_step(model, data)
