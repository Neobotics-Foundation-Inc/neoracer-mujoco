"""
Tests for neoracer_mujoco.envs.NeoRacerEnv (issue #7): the Gymnasium
environment interface itself, not RL training (none exists yet).

Gymnasium is an optional `rl` dependency (pyproject.toml's
[project.optional-dependencies]) -- pytest.importorskip below means this
file is skipped, not failed, on a plain `pip install -e .` checkout without
it, matching README.md's "any Python with mujoco + pytest installed" claim
for `pytest validation/`.

Several tests deliberately manipulate the env's internal progress/history
state directly (documented at each call site) rather than physically
driving a full lap or a real rollover through random actions: no trained
policy exists yet to complete a lap, and waiting on random actions to
stumble into a rollover would make the test slow and flaky. These tests
exercise the termination CONDITION logic directly; the physics themselves
(rollover detection, collision detection, settle/step) already have their
own conformance tests elsewhere in this suite.
"""

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env

from neoracer_mujoco.envs import NeoRacerEnv, NeoRacerEnvConfig


@pytest.fixture
def env():
    e = NeoRacerEnv()
    yield e
    e.close()


# --- construction / spaces --------------------------------------------------


def test_env_constructs(env):
    assert isinstance(env, gym.Env)


def test_action_space_is_valid_box(env):
    assert isinstance(env.action_space, spaces.Box)
    assert env.action_space.shape == (2,)
    assert env.action_space.dtype == np.float32
    np.testing.assert_array_equal(env.action_space.low, [-1.0, -1.0])
    np.testing.assert_array_equal(env.action_space.high, [1.0, 1.0])


def test_observation_space_is_valid_box(env):
    assert isinstance(env.observation_space, spaces.Box)
    assert env.observation_space.shape == (20,)
    assert env.observation_space.dtype == np.float32
    assert np.all(np.isfinite(env.observation_space.low))
    assert np.all(np.isfinite(env.observation_space.high))


def test_gymnasium_check_env_passes(env):
    """Gymnasium's own env checker -- API shape, dtypes, spaces, seeding."""
    check_env(env, skip_render_check=True)


def test_unsupported_track_raises_clear_error():
    """This first-pass env's progress/lap/reward semantics are specific to
    loop_corridor.xml (no generic track abstraction exists yet) -- an
    unsupported track_path must fail loudly at construction, not silently
    apply loop-specific semantics to a track they don't fit."""
    import os

    from neoracer_mujoco import assets as _assets

    other_track = os.path.join(_assets.ASSET_DIR, "tracks", "straight_corridor.xml")
    with pytest.raises(ValueError, match="loop_corridor"):
        NeoRacerEnv(config=NeoRacerEnvConfig(track_path=other_track))


# --- reset() / step() basic contract ----------------------------------------


def test_reset_observation_in_space(env):
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert isinstance(info, dict)


def test_step_observation_in_space_and_types(env):
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(
        np.array([0.3, 0.1], dtype=np.float32)
    )
    assert env.observation_space.contains(obs)
    assert np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    assert "termination_reason" in info


def test_repeated_resets_work(env):
    for seed in range(5):
        obs, _info = env.reset(seed=seed)
        assert env.observation_space.contains(obs)


def test_full_bounded_episode_runs_without_crashing(env):
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(env.config.max_episode_steps):
        action = rng.uniform(-1, 1, size=2).astype(np.float32)
        obs, reward, terminated, truncated, _info = env.step(action)
        assert np.all(np.isfinite(obs))
        assert np.isfinite(reward)
        if terminated or truncated:
            break
    else:
        pytest.fail("episode neither terminated nor truncated within max_episode_steps")


# --- determinism / seeding ---------------------------------------------------


def test_deterministic_reset_same_seed():
    """Two independent envs, same seed => bit-identical reset observation
    and identical trajectory under the same action sequence."""
    env1, env2 = NeoRacerEnv(), NeoRacerEnv()
    obs1, _ = env1.reset(seed=42)
    obs2, _ = env2.reset(seed=42)
    np.testing.assert_array_equal(obs1, obs2)

    rng = np.random.default_rng(0)
    for _ in range(20):
        action = rng.uniform(-1, 1, size=2).astype(np.float32)
        r1 = env1.step(action)
        r2 = env2.step(action)
        np.testing.assert_array_equal(r1[0], r2[0])  # obs
        assert r1[1] == r2[1]  # reward
        assert r1[2] == r2[2] and r1[3] == r2[3]  # terminated, truncated
    env1.close()
    env2.close()


def test_different_seeds_produce_different_spawns():
    env = NeoRacerEnv()
    obs_a, _ = env.reset(seed=1)
    obs_b, _ = env.reset(seed=2)
    assert not np.array_equal(obs_a, obs_b)
    env.close()


def test_reset_does_not_use_global_numpy_random(env, monkeypatch):
    """Seeding two envs from Gymnasium's own generator, not global
    np.random, means poisoning global np.random must not change the
    result."""
    obs1, _ = env.reset(seed=7)

    def _boom(*args, **kwargs):
        raise AssertionError("global np.random was used")

    monkeypatch.setattr(np.random, "uniform", _boom)
    monkeypatch.setattr(np.random, "seed", _boom)
    obs2, _ = env.reset(seed=7)
    np.testing.assert_array_equal(obs1, obs2)


# --- action handling ----------------------------------------------------------


def test_out_of_range_actions_are_clamped_not_rejected():
    """action=[5, -5] must behave identically to the clamped action
    [1, -1], not raise, not silently do something else."""
    env_a, env_b = NeoRacerEnv(), NeoRacerEnv()
    env_a.reset(seed=3)
    env_b.reset(seed=3)
    env_a.step(np.array([5.0, -5.0], dtype=np.float32))
    env_b.step(np.array([1.0, -1.0], dtype=np.float32))
    np.testing.assert_array_equal(env_a.data.ctrl, env_b.data.ctrl)
    env_a.close()
    env_b.close()


def test_repeated_random_actions_produce_no_nans(env):
    env.reset(seed=0)
    rng = np.random.default_rng(123)
    for _ in range(300):
        action = rng.uniform(-1, 1, size=2).astype(np.float32)
        obs, reward, terminated, truncated, _info = env.step(action)
        assert np.all(np.isfinite(obs))
        assert np.isfinite(reward)
        if terminated or truncated:
            env.reset(seed=0)


# --- termination / truncation -------------------------------------------------


def test_rollover_terminates(env):
    """Directly tips the car past the upright_cos threshold (see module
    docstring for why this isn't driven via random actions) and confirms
    the very next step reports terminated with reason 'rollover'."""
    import mujoco

    env.reset(seed=0)
    # 90-degree rotation about the body X axis: the car's z-axis rotates
    # to point along world -Y, so upright_cos (z-axis . world z) becomes 0,
    # well under the 0.5 threshold.
    half = np.pi / 4.0
    env.data.qpos[3:7] = [np.cos(half), np.sin(half), 0.0, 0.0]
    mujoco.mj_forward(env.model, env.data)
    from neoracer_mujoco import sim as _sim

    assert _sim.car_upright_cos(env.data) < env.config.upright_rollover_cos

    _obs, reward, terminated, truncated, info = env.step(
        np.array([0.0, 0.0], dtype=np.float32)
    )
    assert terminated
    assert not truncated
    assert info["termination_reason"] == "rollover"
    assert reward < 0


def test_stuck_terminates():
    """Physically drives the sim with zero throttle -- the car should not
    make meaningful progress, so the stuck detector should fire on its
    own within stuck_window_s, with no white-box state injection needed."""
    env = NeoRacerEnv()
    env.reset(seed=0)
    terminated = truncated = False
    reason = None
    max_ticks = env._stuck_window_ticks + 20
    for _ in range(max_ticks):
        _obs, _reward, terminated, truncated, info = env.step(
            np.array([0.0, 0.0], dtype=np.float32)
        )
        if terminated or truncated:
            reason = info["termination_reason"]
            break
    assert terminated, "expected the stuck detector to fire on zero throttle"
    assert reason == "stuck"
    env.close()


def test_lap_completion_terminates_successfully(env):
    """Directly sets cumulative progress past the lap threshold (see
    module docstring: no trained policy exists yet to actually complete a
    lap) and confirms the next step reports terminated/'lap_complete' with
    a positive (bonus-dominated) reward.

    +2*pi is the empirically-confirmed intended forward-travel direction
    (positive throttle from the loop_corridor spawn produces positive
    cumulative angle -- see test_negative_full_lap_progress_does_not_
    complete_lap for the negative-direction counterpart)."""
    env.reset(seed=0)
    env._cum_angle = 2.0 * np.pi + 0.5  # already past the 2*pi threshold
    env._angle_history.clear()
    env._angle_history.append(env._cum_angle)

    _obs, reward, terminated, truncated, info = env.step(
        np.array([1.0, 0.0], dtype=np.float32)
    )
    assert terminated
    assert not truncated
    assert info["termination_reason"] == "lap_complete"
    assert reward > 0


def test_negative_full_lap_progress_does_not_complete_lap(env):
    """A full lap's worth of progress in the WRONG (negative) direction
    must not be treated as a successful lap: only positive cumulative
    angle -- the empirically-confirmed forward-travel direction from the
    loop_corridor spawn -- counts (see module docstring)."""
    env.reset(seed=0)
    env._cum_angle = -(2.0 * np.pi + 0.5)  # full lap of NEGATIVE progress
    env._angle_history.clear()
    env._angle_history.append(env._cum_angle)

    _obs, _reward, terminated, truncated, info = env.step(
        np.array([1.0, 0.0], dtype=np.float32)
    )
    assert info["termination_reason"] != "lap_complete"
    # A single step with a fresh angle_history can't trigger rollover or
    # stuck either -- so a fixed-up condition should simply not terminate.
    assert not terminated
    assert not truncated


def test_timeout_truncates():
    env = NeoRacerEnv(config=NeoRacerEnvConfig(max_episode_steps=5))
    env.reset(seed=0)
    terminated = truncated = False
    for i in range(5):
        _obs, _reward, terminated, truncated, info = env.step(
            np.array([0.0, 0.0], dtype=np.float32)
        )
    assert truncated
    assert not terminated
    assert info["termination_reason"] is None
    env.close()


# --- state isolation -----------------------------------------------------------


def test_model_state_does_not_leak_between_episodes(env):
    """The MjModel is compiled once and never written to by reset()/step()
    -- a regression guard, not just a design claim."""
    body_pos_before = env.model.body_pos.copy()
    geom_size_before = env.model.geom_size.copy()
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(50):
        env.step(rng.uniform(-1, 1, size=2).astype(np.float32))
    env.reset(seed=1)
    np.testing.assert_array_equal(env.model.body_pos, body_pos_before)
    np.testing.assert_array_equal(env.model.geom_size, geom_size_before)
