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

type Scan = tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]  # rad, m


@dataclass(frozen=True)
class DenseLidarParams:
    """The beam layout. The default field of view and range are the Lakibeam
    L1 values."""

    n_beams: int = 1080
    fov_deg: float = 270.0  # deg; centered on +X
    max_range_m: float = 12.0  # m; a beam with no hit in this range has no return


def scan_angles(params: DenseLidarParams) -> npt.NDArray[np.float64]:
    """Beam angles (rad), ascending and symmetric about 0 rad. With
    n_beams = 1080 and fov_deg = 270 the step is 0.25 deg."""
    half = math.radians(params.fov_deg) / 2.0
    return np.linspace(-half, half, params.n_beams)


@dataclass(frozen=True)
class LidarMount:
    """Model ids that cast_dense_lidar() reads at each tick."""

    site_id: int  # the lidar_mount site: the ray origin
    body_id: int  # the car body: its axes give the ray directions; rays ignore it


def lidar_mount(model: mujoco.MjModel) -> LidarMount:
    """Read the ids of the lidar_mount site and the car body once."""
    return LidarMount(
        site_id=model.site("lidar_mount").id, body_id=model.body("car").id
    )


def cast_dense_lidar(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    mount: LidarMount,
    params: DenseLidarParams,
) -> Scan:
    """Cast the beams from the lidar mount. Returns (angles_rad, ranges_m)
    in the same layout as synthetic_scenes.cast_scan(). A range is -1.0 if
    the beam has no hit within max_range_m."""
    origin = data.site_xpos[mount.site_id]
    body_mat = data.xmat[mount.body_id].reshape(3, 3)
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
        mount.body_id,
        geom_id,
        dist,
        None,
        params.n_beams,
        params.max_range_m,
    )
    ranges = np.where((dist >= 0.0) & (dist <= params.max_range_m), dist, -1.0)
    return angles, ranges
