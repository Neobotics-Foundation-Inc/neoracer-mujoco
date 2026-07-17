"""T2 clean spawn + rollout determinism + obs sanity + flag semantics.

Env compile is the slow bit -> a session-scoped env, reset per test.
"""

import numpy as np
import pytest
import mujoco

from proc_track.env import OBS_DIM, ProcTrackEnv
from proc_track.walls import CAR_XML_PATH


@pytest.fixture(scope="session")
def env():
    return ProcTrackEnv(phase=3)


def _bare_car_z_after(substeps):
    """Body z of the car alone (no walls/track) after `substeps` zero-ctrl steps."""
    m = mujoco.MjModel.from_xml_path(CAR_XML_PATH)
    d = mujoco.MjData(m)
    d.ctrl[:] = 0
    for _ in range(substeps):
        mujoco.mj_step(m, d)
    return float(d.xpos[m.body("car").id, 2])


def test_T2_clean_spawn(env):
    """200 seeds: only wheel-floor contacts at t=0; chassis z tracks the car-only
    settle height after 0.2 s (walls/track must not disturb the car)."""
    frame = env.cfg.frame_skip
    ref_z = _bare_car_z_after(int(0.2 / 0.002))  # 100 substeps == 0.2 s

    wheel = set(int(g) for g in env.wheel_gids)
    floor = env.floor_gid

    for seed in range(200):
        env.reset(seed)
        # every contact at spawn is a wheel-floor pair
        for k in range(env.data.ncon):
            c = env.data.contact[k]
            g1, g2 = int(c.geom1), int(c.geom2)
            pair = {g1, g2}
            assert floor in pair and (pair - {floor}) <= wheel, (
                f"seed {seed}: non wheel-floor contact {g1},{g2}"
            )
        # step 0.2 s of zero action, compare settle height
        for _ in range(int(0.2 / (0.002 * frame))):  # 10 control steps
            env.step(np.zeros(2))
        z = float(env.data.xpos[env.car_bid, 2])
        assert abs(z - ref_z) < 0.005, f"seed {seed}: z {z:.4f} vs ref {ref_z:.4f}"


def test_rollout_determinism():
    """Same seed + same actions -> bit-identical qpos trajectory and obs."""
    rng = np.random.default_rng(0)
    actions = rng.uniform(-1, 1, (100, 2))

    def rollout():
        e = ProcTrackEnv(phase=3)
        obs, _ = e.reset(123)
        traj = [e.data.qpos.copy()]
        obses = [obs.copy()]
        for a in actions:
            o, _, term, trunc, _ = e.step(a)
            traj.append(e.data.qpos.copy())
            obses.append(o.copy())
            if term or trunc:
                break
        return traj, obses

    t1, o1 = rollout()
    t2, o2 = rollout()
    assert len(t1) == len(t2)
    for a, b in zip(t1, t2):
        assert np.array_equal(a, b)
    for a, b in zip(o1, o2):
        assert np.array_equal(a, b)


def test_obs_shape_and_finite(env):
    obs, info = env.reset(7)
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
    assert np.all(np.isfinite(obs))
    assert "track_L" in info
    for _ in range(20):
        obs, r, term, trunc, _ = env.step(np.array([0.2, 0.5]))
        assert obs.shape == (OBS_DIM,)
        assert np.all(np.isfinite(obs))
        assert np.isfinite(r)
        if term or trunc:
            break


def test_flags_mutually_exclusive(env):
    """terminated and truncated are never both set in one step."""
    env.reset(3)
    for _ in range(200):
        _, _, term, trunc, _ = env.step(np.array([0.0, 1.0]))
        assert not (term and trunc)
        if term or trunc:
            break
