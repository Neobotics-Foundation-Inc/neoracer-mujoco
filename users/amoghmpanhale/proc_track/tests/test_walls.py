"""T4: wall coverage. Every point on each boundary curve lies within `half_thick`
of some box wall's tangent axis segment -- pins overlap=1.2, K=256 (a gap would
let the car clip the corridor edge).

Also checks the build_model contract (fixed 2*K pool, contiguous gids, contact
masks) once via a session-scoped fixture -- compile is the slow bit.
"""

import numpy as np
import pytest
import mujoco

from proc_track.config import WallConfig, gen_config_for_phase
from proc_track.generator import sample_valid_track
from proc_track.walls import build_model, wall_params

WALLS = WallConfig()


@pytest.fixture(scope="session")
def compiled():
    return build_model(WALLS)


def _boundary_curve(track, side, n=10000):
    """Dense march along one boundary: C + side*w*Nrm, resampled to n points."""
    edge = track.C + side * track.w[:, None] * track.Nrm
    M = edge.shape[0]
    # linear resample in index space (edge is already ~uniform arc length)
    src = np.arange(M)
    tgt = np.linspace(0, M, n, endpoint=False)
    x = np.interp(tgt, src, edge[:, 0], period=M)
    y = np.interp(tgt, src, edge[:, 1], period=M)
    return np.stack([x, y], axis=1)


def _seg_axes_xy(pos, quat, half_len):
    """World-frame xy endpoints of each box wall's tangent axis (local x by quat)."""
    ax = np.zeros((len(pos), 3))
    for i in range(len(pos)):
        mujoco.mju_rotVecQuat(ax[i], np.array([1.0, 0.0, 0.0]), quat[i])
    u = ax[:, :2]
    p0 = pos[:, :2] - half_len[:, None] * u
    p1 = pos[:, :2] + half_len[:, None] * u
    return p0, p1


def _min_dist_to_segments(pts, p0, p1):
    """For each pt, min distance to any segment [p0_k, p1_k]. Vectorized."""
    d = p1 - p0  # (K, 2)
    seg2 = np.sum(d * d, axis=1)  # (K,)
    # (N, K, 2)
    rel = pts[:, None, :] - p0[None, :, :]
    t = np.sum(rel * d[None, :, :], axis=2) / np.maximum(seg2[None, :], 1e-12)
    t = np.clip(t, 0.0, 1.0)
    proj = p0[None, :, :] + t[:, :, None] * d[None, :, :]
    dist = np.linalg.norm(pts[:, None, :] - proj, axis=2)  # (N, K)
    return dist.min(axis=1)


@pytest.mark.parametrize("seed", range(25))
def test_T4_wall_coverage(seed):
    track = sample_valid_track(seed, gen_config_for_phase(3))
    for side in (+1, -1):
        pos, quat, half_len = wall_params(track, side, WALLS)
        p0, p1 = _seg_axes_xy(pos, quat, half_len)
        pts = _boundary_curve(track, side)
        worst = _min_dist_to_segments(pts, p0, p1).max()
        assert worst <= WALLS.half_thick, (
            f"seed {seed} side {side}: gap {worst*1000:.1f} mm > half_thick "
            f"{WALLS.half_thick*1000:.0f} mm"
        )


def test_build_model_contract(compiled):
    model, start = compiled
    K = WALLS.K
    # contiguous block of 2*K wall geoms
    assert model.geom("wall_000").id == start
    assert model.geom(f"wall_{2*K-1:03d}").id == start + 2 * K - 1
    gids = start + np.arange(2 * K)
    # contact masks + box type + priority (frictionless-wall contract)
    assert np.all(model.geom_contype[gids] == 2)
    assert np.all(model.geom_conaffinity[gids] == 1)
    assert np.all(model.geom_priority[gids] == 1)
    assert np.all(model.geom_condim[gids] == 1)
    assert np.all(model.geom_type[gids] == mujoco.mjtGeom.mjGEOM_BOX)


def test_wall_orientation_is_live(compiled):
    """Walls must honor per-reset geom_quat. Parked-at-identity geoms compile with
    geom_sameframe set, which makes mj_kinematics ignore geom_quat and render/collide
    every wall axis-aligned (a comb across the track). build_model clears the flag."""
    model, start = compiled
    gids = start + np.arange(2 * WALLS.K)
    assert np.all(model.geom_sameframe[gids] == 0)
