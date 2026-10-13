"""
Laws for the PPO baseline (examples/train_ppo.py): GAE, the clipped loss,
the observation statistics, and a short training run on NeoRacerEnv.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from examples.train_ppo import (
    Batch,
    PPOParams,
    RolloutState,
    clipped_surrogate,
    collect_rollout,
    empty_obs_stats,
    env_action,
    gae,
    init_actor_critic,
    init_opt_state,
    load_policy,
    ppo_loss,
    ppo_update,
    save_policy,
    update_obs_stats,
)
from neoracer_mujoco.envs import NeoRacerEnv
from neoracer_mujoco.envs.neoracer_env import OBS_DIM


def _reward_to_go(rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
    """Undiscounted sum of the rewards to the end of each episode, per column."""
    out = np.zeros_like(rewards)
    running = np.zeros(rewards.shape[1])
    for t in reversed(range(rewards.shape[0])):
        running = rewards[t] + running * (1.0 - dones[t])
        out[t] = running
    return out


@pytest.mark.parametrize("seed", range(5))
def test_gae_with_unit_gamma_lambda_and_zero_values_is_reward_to_go(seed):
    rng = np.random.default_rng(seed)
    rewards = rng.normal(size=(12, 3))
    dones = (rng.uniform(size=(12, 3)) < 0.2).astype(np.float64)
    zeros = jnp.zeros((12, 3))
    adv, ret = gae(
        jnp.asarray(rewards), zeros, jnp.asarray(dones), jnp.zeros(3), 1.0, 1.0
    )
    expected = _reward_to_go(rewards, dones)
    np.testing.assert_allclose(adv, expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(ret, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("seed", range(5))
def test_gae_does_not_leak_across_a_done_step(seed):
    rng = np.random.default_rng(seed)
    rewards = jnp.asarray(rng.normal(size=(10, 2)))
    values = jnp.asarray(rng.normal(size=(10, 2)))
    last_value = jnp.asarray(rng.normal(size=2))
    dones = np.zeros((10, 2))
    dones[4] = 1.0  # an episode ends after step 4
    adv, _ = gae(rewards, values, jnp.asarray(dones), last_value, 0.97, 0.9)
    # Changes after the done step do not change the advantages up to it.
    changed_rewards = rewards.at[5:].add(100.0)
    changed_values = values.at[5:].add(-50.0)
    adv2, _ = gae(
        changed_rewards, changed_values, jnp.asarray(dones), last_value + 7.0, 0.97, 0.9
    )
    np.testing.assert_allclose(adv[:5], adv2[:5], rtol=1e-6, atol=1e-6)
    # At the done step, the advantage is the reward minus the value.
    np.testing.assert_allclose(adv[4], rewards[4] - values[4], rtol=1e-6, atol=1e-6)


def test_clipped_surrogate_is_zero_for_zero_advantage():
    ratio = jnp.linspace(0.1, 3.0, 50)
    np.testing.assert_array_equal(clipped_surrogate(ratio, jnp.zeros(50), 0.2), 0.0)


@pytest.mark.parametrize("advantage", [-2.0, -0.5, 0.5, 2.0])
def test_clipped_surrogate_has_no_gradient_past_the_clip_in_the_improving_direction(
    advantage,
):
    eps = 0.2
    # A ratio that already moved past the clip in the direction that the
    # advantage rewards gets no more gradient.
    past = 1.5 if advantage > 0 else 0.5
    grad = jax.grad(lambda r: clipped_surrogate(r, jnp.asarray(advantage), eps))(past)
    assert float(grad) == 0.0
    np.testing.assert_allclose(
        clipped_surrogate(jnp.asarray(past), jnp.asarray(advantage), eps),
        -(1.0 + np.sign(advantage) * eps) * advantage,
        rtol=1e-6,
    )
    # Inside the clip range, the loss is -ratio * advantage.
    inside = jax.grad(lambda r: clipped_surrogate(r, jnp.asarray(advantage), eps))(1.05)
    np.testing.assert_allclose(inside, -advantage, rtol=1e-6)


def test_policy_loss_is_zero_when_all_advantages_are_zero():
    params = PPOParams()
    net = init_actor_critic(jax.random.key(0), params)
    rng = np.random.default_rng(0)
    batch = Batch(
        obs=jnp.asarray(rng.normal(size=(32, OBS_DIM))),
        action=jnp.asarray(rng.normal(size=(32, 2))),
        log_prob=jnp.asarray(rng.normal(size=32)),
        advantage=jnp.zeros(32),
        target=jnp.zeros(32),
    )
    _, info = ppo_loss(net, batch, params)
    assert float(info.policy_loss) == 0.0


@pytest.mark.parametrize(
    "field, bad",
    [
        ("gamma", 0.0),
        ("gae_lambda", 1.5),
        ("max_throttle", 0.0),
        ("rollout_steps", 0),
        ("minibatches", 300),  # more than rollout_steps (256)
        ("clip_eps", 0.0),
        ("learning_rate", -1e-4),
        ("max_grad_norm", 0.0),
        ("obs_clip", -1.0),
        ("value_coef", -0.5),
        ("entropy_coef", -0.01),
    ],
)
def test_ppo_params_reject_values_outside_their_domain(field, bad):
    with pytest.raises(ValueError, match=field):
        PPOParams(**{field: bad})


@pytest.mark.parametrize("max_throttle", [0.1, 0.2, 1.0])
def test_env_action_stays_in_the_scaled_box_and_keeps_in_range_steer(max_throttle):
    rng = np.random.default_rng(0)
    actions = rng.normal(scale=2.0, size=(200, 2))
    out = np.stack([env_action(a, max_throttle) for a in actions])
    assert np.all(np.abs(out[:, 0]) <= max_throttle)
    assert np.all(np.abs(out[:, 1]) <= 1.0)
    inside = np.all(np.abs(actions) <= 1.0, axis=1)
    np.testing.assert_allclose(out[inside, 0], max_throttle * actions[inside, 0])
    np.testing.assert_array_equal(out[inside, 1], actions[inside, 1])


@pytest.mark.parametrize("seed", range(3))
def test_running_obs_stats_match_numpy_over_all_batches(seed):
    rng = np.random.default_rng(seed)
    batches = [
        rng.normal(
            loc=rng.uniform(-50, 50, OBS_DIM),
            scale=rng.uniform(0.01, 100, OBS_DIM),
            size=(int(rng.integers(1, 40)), OBS_DIM),
        )
        for _ in range(6)
    ]
    stats = empty_obs_stats()
    for b in batches:
        stats = update_obs_stats(stats, b)
    data = np.concatenate(batches)
    assert stats.count == data.shape[0]
    np.testing.assert_allclose(stats.mean, data.mean(axis=0), rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(stats.var, data.var(axis=0), rtol=1e-9, atol=1e-9)


def test_short_training_run_changes_the_params_and_keeps_them_finite(tmp_path):
    params = PPOParams(rollout_steps=128, epochs=2, minibatches=2, hidden=(16,))
    envs = [NeoRacerEnv() for _ in range(2)]
    obs = np.stack([env.reset(seed=i)[0] for i, env in enumerate(envs)]).astype(
        np.float64
    )
    state = RolloutState(
        obs, np.zeros(2), np.zeros(2, dtype=np.int64), empty_obs_stats()
    )
    net = init_actor_critic(jax.random.key(1), params)
    opt_state = init_opt_state(net, params)
    start = net
    # One update from 128 * 2 = 256 env steps.
    batch, state, _ = collect_rollout(envs, state, net, jax.random.key(10), params)
    net, opt_state, info = ppo_update(net, opt_state, batch, jax.random.key(20), params)
    assert all(np.isfinite(float(x)) for x in jax.tree.leaves(info))
    leaves, start_leaves = jax.tree.leaves(net), jax.tree.leaves(start)
    assert all(np.all(np.isfinite(x)) for x in leaves)
    assert any(
        not np.array_equal(a, b) for a, b in zip(leaves, start_leaves, strict=True)
    )
    assert state.stats.count == 128 * 2

    # A saved policy loads back to the same arrays.
    save_policy(str(tmp_path / "p.npz"), net, state.stats)
    loaded, stats = load_policy(str(tmp_path / "p.npz"))
    for a, b in zip(jax.tree.leaves(loaded), leaves, strict=True):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(stats.mean, state.stats.mean)
