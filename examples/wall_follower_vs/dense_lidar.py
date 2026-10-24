"""
A dense LiDAR for the trial scripts. It casts N rays in a live MuJoCo scene
from the lidar_mount site of the car, to emulate a dense sensor such as the
Lakibeam L1 (270 deg field of view, up to 1080 beams). The 8 rangefinders in
neoracer.xml do not change.

Frame: +X forward, +Y left, +Z up. Beam angles are + = left.
"""

import math
from dataclasses import dataclass

import mujoco
import numpy as np
import numpy.typing as npt

from neoracer_mujoco.scene import Scene

type Scan = tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
"""The beam angles (rad) and their ranges (m)."""


@dataclass(frozen=True)
class DenseLidarParams:
    """The beam layout. The default field of view and range are the Lakibeam
    L1 values."""

    n_beams: int = 1080
    fov_deg: float = 270.0
    """The field of view, centered on +X."""
    max_range_m: float = 12.0
    """A beam with no hit in this range has no return."""

    def __post_init__(self) -> None:
        if self.n_beams < 1:
            raise ValueError(f"n_beams must be >= 1, got {self.n_beams}")


def scan_angles(params: DenseLidarParams) -> npt.NDArray[np.float64]:
    """Beam angles (rad), ascending and symmetric about 0 rad. With
    n_beams = 1080 and fov_deg = 270 the step is 0.25 deg. One beam points
    along +X."""
    if params.n_beams == 1:
        return np.zeros(1)
    half = math.radians(params.fov_deg) / 2.0
    return np.linspace(-half, half, params.n_beams)


@dataclass(frozen=True)
class LidarMount:
    """The spec elements that cast_dense_lidar() reads at each tick."""

    site: mujoco.MjsSite
    """The lidar_mount site, the rays' origin."""
    body: mujoco.MjsBody
    """The car body. Its axes orient the rays, and the rays ignore it."""


def lidar_mount(scene: Scene) -> LidarMount:
    """Find the lidar_mount site and the car body once."""
    site = scene.spec.site("lidar_mount")
    if site is None:
        raise ValueError("the scene has no lidar_mount site")
    return LidarMount(site=site, body=scene.car.body)


def cast_dense_lidar(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    mount: LidarMount,
    params: DenseLidarParams,
) -> Scan:
    """Cast the beams from the lidar mount. Returns (angles_rad, ranges_m)
    in the same layout as synthetic_scenes.cast_scan(). A range is -1.0 if
    the beam has no hit within max_range_m."""
    origin = data.bind_scalar(mount.site).xpos
    body_mat = data.bind_scalar(mount.body).xmat.reshape(3, 3)
    angles = scan_angles(params)
    directions = (
        np.cos(angles)[:, None] * body_mat[:, 0]
        + np.sin(angles)[:, None] * body_mat[:, 1]
    )
    dist = np.empty(params.n_beams)
    geom_id = np.empty(params.n_beams, dtype=np.int32)
    mujoco.mj_multiRay(
        model,
        data,
        origin,
        directions.ravel(),
        None,
        True,
        model.bind_scalar(mount.body).id,
        geom_id,
        dist,
        None,
        params.n_beams,
        params.max_range_m,
    )
    ranges = np.where((dist >= 0.0) & (dist <= params.max_range_m), dist, -1.0)
    return angles, ranges
