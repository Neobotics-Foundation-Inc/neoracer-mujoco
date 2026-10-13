"""
Tests for neoracer_mujoco.envs.NeoRacerEnv: the Gymnasium environment
interface. These tests do not train a policy. validation/test_train_ppo.py
tests the PPO baseline.

Random actions rarely complete a lap or roll the car over. So the tests
call the pure progress and termination functions directly, with laws over
many random inputs. Two env tests then check that step() uses them.
"""

import os
import pickle
from pathlib import Path

import mujoco
import numpy as np
import pytest
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env

from neoracer_mujoco import assets, sim
from neoracer_mujoco.envs import (
    NeoRacerEnv,
    NeoRacerEnvConfig,
    RewardConfig,
    TerminationConfig,
    TerminationReason,
    advance,
    is_stuck,
    reward,
    start_progress,
    termination,
)

STEER_CMD_POS = 18  # observation index of steer_cmd_pos (rad)
LAP = 2.0 * np.pi  # rad, one full lap of progress


@pytest.fixture
def env():
    e = NeoRacerEnv()
    yield e
    e.close()


# --- construction / spaces --------------------------------------------------


def test_env_rejects_a_car_without_a_free_joint(tmp_path):
    """__init__ parses the car's free joint once. A car without one fails
    there with a clear error, not later in reset()."""
    xml = Path(assets.DEFAULT_CAR).read_text()
    xml = xml.replace("<freejoint/>", "").replace(
        'file="meshes/', f'file="{os.path.join(assets.ASSET_DIR, "meshes")}/'
    )
    car = tmp_path / "no_freejoint.xml"
    car.write_text(xml)
    with pytest.raises(ValueError, match="needs one free joint"):
        NeoRacerEnv(NeoRacerEnvConfig(car_path=str(car)))


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
    """Gymnasium's env checker checks the API, the dtypes, the spaces and the
    seeding."""
    check_env(env, skip_render_check=True)


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
    """Two envs with the same seed give the same reset observation. With the
    same actions, they give the same step results."""
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


def _global_random_state() -> bytes:
    """The state of the global np.random generator, as bytes that compare
    with ==."""
    return pickle.dumps(np.random.get_state())


def test_reset_and_step_do_not_use_global_numpy_random(env):
    """The env takes its random numbers from env.np_random only. So reset()
    and step() do not change the state of the global np.random generator."""
    before = _global_random_state()
    env.reset(seed=7)
    rng = np.random.default_rng(0)
    for _ in range(10):
        env.step(rng.uniform(-1, 1, size=2).astype(np.float32))
    env.reset()
    assert _global_random_state() == before


# --- action handling ----------------------------------------------------------


def test_out_of_range_actions_are_clamped_not_rejected():
    """step() clamps the action to [-1, 1]. So action [5, -5] writes the
    same ctrl as action [1, -1], and does not raise an error."""
    env_a, env_b = NeoRacerEnv(), NeoRacerEnv()
    env_a.reset(seed=3)
    env_b.reset(seed=3)
    env_a.step(np.array([5.0, -5.0], dtype=np.float32))
    env_b.step(np.array([1.0, -1.0], dtype=np.float32))
    np.testing.assert_array_equal(env_a.data.ctrl, env_b.data.ctrl)
    env_a.close()
    env_b.close()


def test_wall_crash_observations_stay_in_space(env):
    """A recorded wall crash pushes steer_cmd_pos to about 1.2 rad, past its
    soft 0.45 rad joint limit. Every observation must still be inside
    observation_space. The actions are episode 23 of a full-throttle random
    run: default_rng(0), which is PCG64(0), advanced past the 3480 steps
    (2 draws each) before it."""
    bits = np.random.PCG64(0)
    bits.advance(2 * 3480)
    rng = np.random.Generator(bits)
    env.reset(seed=23)
    peak = 0.0
    for _ in range(46):
        action = rng.uniform(-1, 1, size=2).astype(np.float32)
        action[0] = 1.0
        obs, _reward, _terminated, _truncated, _info = env.step(action)
        assert env.observation_space.contains(obs)
        peak = max(peak, abs(float(obs[STEER_CMD_POS])))
    assert peak > 1.0, "the recorded crash no longer happens; record a new one"


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


# --- progress, termination and reward: pure functions -------------------------


def _wrap(angle: float) -> float:
    """angle (rad) in [-pi, pi), the range of the env's angle around the center."""
    return float(np.remainder(angle + np.pi, 2.0 * np.pi) - np.pi)


def test_advance_gives_a_step_in_minus_pi_to_pi():
    """For any two angles, advance gives a delta in [-pi, pi]. The delta
    differs from the plain difference only by a full turn."""
    rng = np.random.default_rng(0)
    for old, new in rng.uniform(-np.pi, np.pi, (2000, 2)):
        p = advance(start_progress(old), new, window_len=5)
        assert -np.pi <= p.delta <= np.pi
        turns = (p.delta - (new - old)) / LAP
        assert turns == pytest.approx(round(turns), abs=1e-9)
        assert p.total == p.delta and p.angle == new


def test_advance_adds_up_the_true_progress():
    """If the car moves less than half a turn per step, the total of
    advance is the true change of angle, although each angle wraps."""
    rng = np.random.default_rng(1)
    window_len = 7
    for _ in range(50):
        steps = rng.uniform(-0.9 * np.pi, 0.9 * np.pi, 40)  # rad per step
        start = float(rng.uniform(-np.pi, np.pi))
        p = start_progress(start)
        for true_angle in start + np.cumsum(steps):
            p = advance(p, _wrap(true_angle), window_len)
            assert len(p.window) <= window_len
            assert p.window[-1] == p.total
        assert p.total == pytest.approx(float(steps.sum()), abs=1e-9)


def test_is_stuck_only_with_a_full_window_and_little_progress():
    """is_stuck is False until the window has window_len totals. Then it is
    True if and only if the progress over the window is below min_progress."""
    rng = np.random.default_rng(2)
    window_len, min_progress = 6, 0.05
    for _ in range(200):
        p = start_progress(0.0)
        scale = rng.choice([0.001, 0.01, 0.1])  # rad per step
        for step, angle in enumerate(np.cumsum(rng.normal(0.0, scale, 15))):
            p = advance(p, _wrap(angle), window_len)
            full = step + 2 >= window_len  # start_progress puts one total in
            progress = abs(p.total - p.window[0])
            expected = full and progress < min_progress
            assert is_stuck(p, window_len, min_progress) == expected


def test_termination_reason_follows_the_priority():
    """Rollover comes first, then stuck, then a lap. A lap is complete if
    and only if the total progress is at least +2 pi. Progress in the
    negative direction never completes a lap."""
    cfg = TerminationConfig()
    rng = np.random.default_rng(3)
    totals = [*rng.uniform(-2.0 * LAP, 2.0 * LAP, 500), LAP, -LAP, 0.0]
    for total in totals:
        for upright_cos in (-1.0, cfg.upright_rollover_cos - 1e-9, 0.9, 1.0):
            for stuck in (False, True):
                reason = termination(cfg, upright_cos, stuck, total)
                if upright_cos < cfg.upright_rollover_cos:
                    assert reason == "rollover"
                elif stuck:
                    assert reason == "stuck"
                elif total >= LAP:
                    assert reason == "lap_complete"
                else:
                    assert reason is None


def test_reward_adds_the_terminal_term():
    """Without a termination, the reward is progress_weight * delta minus
    contact_penalty for a contact. A termination adds lap_bonus or subtracts
    its penalty."""
    cfg = RewardConfig()
    terminal: frozendict[TerminationReason, float] = frozendict(
        {
            "lap_complete": cfg.lap_bonus,
            "rollover": -cfg.rollover_penalty,
            "stuck": -cfg.stuck_penalty,
        }
    )
    rng = np.random.default_rng(4)
    for delta in rng.uniform(-np.pi, np.pi, 200):
        for contact in (False, True):
            base = reward(cfg, delta, contact, None)
            expected = cfg.progress_weight * delta - cfg.contact_penalty * contact
            assert base == pytest.approx(expected)
            for reason, term in terminal.items():
                assert reward(cfg, delta, contact, reason) == pytest.approx(base + term)


# --- termination and truncation in the env ------------------------------------


def test_rollover_terminates(env):
    """Tip the car onto its side. The next step ends the episode with the
    reason "rollover" and a negative reward."""
    env.reset(seed=0)
    # A 90 degree rotation about the body X axis turns the car +Z axis to
    # world -Y. So upright_cos becomes 0, below upright_rollover_cos.
    half = np.pi / 4.0  # rad, half of the rotation angle
    env.data.qpos[3:7] = [np.cos(half), np.sin(half), 0.0, 0.0]
    mujoco.mj_forward(env.model, env.data)
    assert (
        sim.car_upright_cos(env.data, env.model.body("car").id)
        < env.config.termination.upright_rollover_cos
    )

    _obs, reward, terminated, truncated, info = env.step(
        np.array([0.0, 0.0], dtype=np.float32)
    )
    assert terminated
    assert not truncated
    assert info["termination_reason"] == "rollover"
    assert reward < 0


def test_stuck_terminates(env):
    """With zero throttle, the car makes no progress. So the stuck check
    ends the episode after stuck_window_s."""
    env.reset(seed=0)
    cfg = env.config
    window_steps = round(cfg.termination.stuck_window_s * cfg.control_hz)
    reason = None
    for _ in range(window_steps + 20):
        _obs, _reward, terminated, truncated, info = env.step(
            np.array([0.0, 0.0], dtype=np.float32)
        )
        if terminated or truncated:
            reason = info["termination_reason"]
            break
    assert reason == "stuck"


def test_timeout_truncates():
    config = NeoRacerEnvConfig(max_episode_steps=5)
    env = NeoRacerEnv(config=config)
    env.reset(seed=0)
    results = [
        env.step(np.array([0.0, 0.0], dtype=np.float32))
        for _ in range(config.max_episode_steps)
    ]
    _obs, _reward, terminated, truncated, info = results[-1]
    assert truncated
    assert not terminated
    assert info["termination_reason"] is None
    assert not any(r[3] for r in results[:-1]), "truncated before the last step"
    env.close()


# --- state isolation -----------------------------------------------------------


def test_model_state_does_not_leak_between_episodes(env):
    """reset() and step() write only env.data. They do not change env.model."""
    body_pos_before = env.model.body_pos.copy()
    geom_size_before = env.model.geom_size.copy()
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(50):
        env.step(rng.uniform(-1, 1, size=2).astype(np.float32))
    env.reset(seed=1)
    np.testing.assert_array_equal(env.model.body_pos, body_pos_before)
    np.testing.assert_array_equal(env.model.geom_size, geom_size_before)
