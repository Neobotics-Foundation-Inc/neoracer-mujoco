"""Stage 4: wall capsules bounding the track corridor.

Two design pieces:
- wall_params: pure NumPy. Given a Track and a side, return the pose+size of K
  capsules forming a continuous rail along that boundary. Consumed at reset to
  rewrite geom fields in place (no recompilation -- INV-1).
- build_model: compile the car XML once with a FIXED pool of 2*K parked wall
  capsules. Fixed topology for every episode/seed, exactly what makes the later
  MJX port a mechanical array replace.

Contact design (per plan): walls get contype=2 conaffinity=1 condim=1 priority=1.
priority=1 is REQUIRED -- without it the wheels' condim=3 wins the contact pair
and walls become frictional, breaking the frictionless-wall design.
"""

from pathlib import Path

import mujoco
import numpy as np

from .config import WallConfig

# repo root: .../users/amoghmpanhale/proc_track/walls.py -> parents[3]
CAR_XML_PATH = str(Path(__file__).resolve().parents[3] / "assets" / "neoracer.xml")


def wall_params(track, side, cfg: WallConfig):
    """Pose + half-length of K capsules along one boundary (side +1 left / -1 right).

    Consecutive boundary points joined by a horizontal capsule; overlap keeps the
    rail continuous around curvature. Returns (pos (K,3), quat (K,4), half_len (K,)).
    """
    M = track.C.shape[0]
    idx = np.arange(cfg.K) * (M // cfg.K)  # M % K == 0
    edge = track.C[idx] + side * track.w[idx, None] * track.Nrm[idx]
    a, b = edge, np.roll(edge, -1, axis=0)
    mid = 0.5 * (a + b)
    d = b - a
    seg_len = np.linalg.norm(d, axis=1)
    half_len = cfg.overlap * 0.5 * seg_len
    u = d / seg_len[:, None]  # unit segment directions
    # quat rotating the capsule local z-axis onto the horizontal direction u:
    # rotation of pi/2 about axis (-u_y, u_x, 0).
    c = np.cos(np.pi / 4)
    quat = np.stack(
        [np.full(cfg.K, c), -c * u[:, 1], c * u[:, 0], np.zeros(cfg.K)], axis=1
    )
    pos = np.concatenate([mid, np.full((cfg.K, 1), cfg.center_z)], axis=1)
    return pos, quat, half_len


def build_model(cfg: WallConfig, car_xml_path=CAR_XML_PATH):
    """Compile the car XML once with a fixed pool of 2*K parked wall capsules.

    Returns (model, wall_gid_start). Left block = [start, start+K),
    right block = [start+K, start+2K) -- contiguous by construction.
    """
    spec = mujoco.MjSpec.from_file(car_xml_path)
    for i in range(2 * cfg.K):
        g = spec.worldbody.add_geom()
        g.name = f"wall_{i:03d}"
        g.type = mujoco.mjtGeom.mjGEOM_CAPSULE
        g.size[:2] = [cfg.radius, 0.1]  # placeholder half-length, rewritten at reset
        g.pos = [0.0, 0.0, -10.0]  # parked below floor; worldbody geoms never hit floor
        g.condim = 1
        g.priority = 1
        g.contype = 2
        g.conaffinity = 1
        g.rgba = [0.8, 0.2, 0.2, 1.0]
    model = spec.compile()
    # Walls are static (worldbody) geoms we reposition every reset. MuJoCo's
    # midphase builds a BVH over static geoms at COMPILE time (from the parked
    # poses); moving geom_pos afterward leaves that tree stale, so wall contacts
    # silently vanish once the model has enough geoms to trigger midphase.
    # Disabling midphase forces the dynamic broadphase (geom_xpos + rbound each
    # step), which sees the moved walls. Verified: without this, chassis can
    # penetrate a wall by 9 cm with zero contacts generated.
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_MIDPHASE
    wall_gid_start = model.geom("wall_000").id
    return model, wall_gid_start
