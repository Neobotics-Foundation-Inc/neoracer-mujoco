"""
Train a PPO policy on NeoRacerEnv, or evaluate a saved policy.

PPO (proximal policy optimization) is a policy-gradient method.
It collects a batch of steps with the current policy, then makes a few
gradient steps on a loss that limits how far the new policy moves from the old one.
This script is a small baseline, not a tuned agent.

The policy is a Gaussian distribution over (throttle, steer). An MLP gives the mean,
and a learned vector gives the log standard deviation. env_action() clips each
sampled action to [-1, 1] and scales the throttle by PPOParams.max_throttle
before env.step(). The log probability uses the unclipped sample.
A second MLP (the critic) estimates the value of an observation.

The script normalizes each observation with a running mean and variance,
because the LiDAR value -1 (no hit) and the large IMU ranges otherwise
dominate the input of the MLP.

The pure core is: the MLP, gae(), clipped_surrogate(), ppo_loss(), ppo_update()
and the observation statistics. main() and the functions below "Effects" step
the environments, print the log and read or write the .npz file.

Usage:
    python3 -m examples.train_ppo --total-steps 1000000 --seed 0 --out ppo.npz
    python3 -m examples.train_ppo --eval ppo.npz --episodes 10
    python3 -m examples.train_ppo --eval-baseline random --episodes 10
"""

import argparse
import functools
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, assert_never, cast

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import optax

from neoracer_mujoco.envs import NeoRacerEnv
from neoracer_mujoco.envs.neoracer_env import OBS_DIM, EnvInfo, TerminationReason

ACTION_DIM = 2  # (throttle, steer), see neoracer_env.py

type ObsBatch = npt.NDArray[np.float64]  # shape (batch, OBS_DIM)


@dataclass(frozen=True)
class PPOParams:
    """Hyperparameters of PPO. One update uses num_envs * rollout_steps env steps."""

    rollout_steps: int = 256  # env steps per environment between two updates
    gamma: float = 0.99  # discount per env step (30 Hz), horizon ~ 1/(1-gamma) steps
    gae_lambda: float = 0.95  # GAE weight between 1-step (0) and Monte Carlo (1)
    clip_eps: float = 0.2  # PPO clip range of the probability ratio
    learning_rate: float = 3e-4  # Adam step size
    epochs: int = 10  # passes over each rollout batch
    minibatches: int = 4  # minibatches per epoch
    value_coef: float = 0.5  # weight of the value loss in the total loss
    entropy_coef: float = 0.0  # weight of the entropy bonus in the total loss
    max_grad_norm: float = 0.5  # gradient clip (global L2 norm)
    hidden: tuple[int, ...] = (64, 64)  # hidden layer widths of both MLPs
    # Initial log standard deviation of each action. With -0.5 (std 0.6), the
    # policy did not find the left turn at the first corner in 300k steps.
    # With 0.0 (std 1), it completed laps after ~100k steps.
    init_log_std: float = 0.0
    obs_clip: float = 10.0  # normalized observation clip, in standard deviations
    # Env throttle = max_throttle * policy throttle. The env maps throttle 1 to
    # the full motor torque. A LiDAR centering controller completes a lap at
    # env throttle 0.1 (~4 m/s) and hits the wall at 0.2 (~5.7 m/s).
    # With 1.0, PPO also learns laps, but 3 of 20 evaluation episodes failed
    # after 300k steps (2 rollovers, 1 stuck). With 0.1, all 20 completed a lap.
    max_throttle: float = 0.1  # env throttle units, in (0, 1]

    def __post_init__(self) -> None:
        if not 0.0 < self.gamma <= 1.0 or not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gamma must be in (0, 1] and gae_lambda in [0, 1]")
        if not 0.0 < self.max_throttle <= 1.0:
            raise ValueError("max_throttle must be in (0, 1]")
        if self.rollout_steps < 1 or self.epochs < 1 or self.minibatches < 1:
            raise ValueError("rollout_steps, epochs and minibatches must be >= 1")
        # A batch has rollout_steps * num_envs samples, so this check gives
        # each minibatch at least one sample for any num_envs >= 1.
        if self.minibatches > self.rollout_steps:
            raise ValueError("minibatches must be <= rollout_steps")
        if (
            min(self.clip_eps, self.learning_rate, self.max_grad_norm, self.obs_clip)
            <= 0
        ):
            raise ValueError(
                "clip_eps, learning_rate, max_grad_norm and obs_clip must be > 0"
            )
        if self.value_coef < 0 or self.entropy_coef < 0:
            raise ValueError("value_coef and entropy_coef must be >= 0")


# --- Network -------------------------------------------------------------


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Dense:
    weight: jax.Array  # shape (inputs, outputs)
    bias: jax.Array  # shape (outputs,)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ActorCritic:
    actor: tuple[Dense, ...]  # obs -> action mean
    critic: tuple[Dense, ...]  # obs -> value
    log_std: jax.Array  # shape (ACTION_DIM,), log standard deviation of each action


def init_mlp(
    key: jax.Array, sizes: Sequence[int], out_scale: float
) -> tuple[Dense, ...]:
    """An MLP with orthogonal weights and zero biases. The last layer uses
    out_scale as the gain, the hidden layers use sqrt(2)."""
    keys = jax.random.split(key, len(sizes) - 1)
    gains = [np.sqrt(2.0)] * (len(sizes) - 2) + [out_scale]
    return tuple(
        Dense(
            weight=jax.nn.initializers.orthogonal(gain)(k, (n_in, n_out)),
            bias=jnp.zeros(n_out),
        )
        for k, n_in, n_out, gain in zip(keys, sizes[:-1], sizes[1:], gains, strict=True)
    )


@functools.partial(jax.jit, static_argnames=("params",))
def init_actor_critic(key: jax.Array, params: PPOParams) -> ActorCritic:
    actor_key, critic_key = jax.random.split(key)
    return ActorCritic(
        # A small gain makes the initial action mean close to 0.
        actor=init_mlp(actor_key, (OBS_DIM, *params.hidden, ACTION_DIM), 0.01),
        critic=init_mlp(critic_key, (OBS_DIM, *params.hidden, 1), 1.0),
        log_std=jnp.full(ACTION_DIM, params.init_log_std),
    )


def mlp(layers: tuple[Dense, ...], x: jax.Array) -> jax.Array:
    """tanh hidden layers, linear output."""
    for layer in layers[:-1]:
        x = jnp.tanh(x @ layer.weight + layer.bias)
    return x @ layers[-1].weight + layers[-1].bias


def value(net: ActorCritic, obs: jax.Array) -> jax.Array:
    """The critic value of each normalized observation. Shape (batch,)."""
    return mlp(net.critic, obs)[..., 0]


def gaussian_log_prob(mean: jax.Array, log_std: jax.Array, x: jax.Array) -> jax.Array:
    """log density of x under a diagonal Gaussian, summed over the last axis."""
    z = (x - mean) / jnp.exp(log_std)
    return jnp.sum(-0.5 * z**2 - log_std - 0.5 * jnp.log(2.0 * jnp.pi), axis=-1)


@jax.jit
def sample_action(
    net: ActorCritic, obs: jax.Array, key: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """(unclipped action, its log probability, value) for a batch of normalized
    observations."""
    mean = mlp(net.actor, obs)
    action = mean + jnp.exp(net.log_std) * jax.random.normal(key, mean.shape)
    return action, gaussian_log_prob(mean, net.log_std, action), value(net, obs)


@jax.jit
def greedy_action(net: ActorCritic, obs: jax.Array) -> jax.Array:
    """The mean action of the policy (deterministic), before env_action()."""
    return mlp(net.actor, obs)


def env_action(action: npt.NDArray, max_throttle: float) -> npt.NDArray[np.float64]:
    """The env action for a policy action: clip each value to [-1, 1], then
    scale the throttle by max_throttle."""
    return np.clip(action, -1.0, 1.0) * np.array([max_throttle, 1.0])


# --- Observation normalization --------------------------------------------


@dataclass(frozen=True)
class ObsStats:
    """Running mean and variance of the raw observations."""

    mean: npt.NDArray[np.float64]  # shape (OBS_DIM,)
    var: npt.NDArray[np.float64]  # shape (OBS_DIM,), population variance
    count: float  # observations seen; 0 = no data yet


def empty_obs_stats() -> ObsStats:
    return ObsStats(mean=np.zeros(OBS_DIM), var=np.ones(OBS_DIM), count=0.0)


def update_obs_stats(stats: ObsStats, batch: ObsBatch) -> ObsStats:
    """Combine the statistics with one batch (parallel variance formula,
    Chan et al. 1979)."""
    n = float(batch.shape[0])
    total = stats.count + n
    delta = batch.mean(axis=0) - stats.mean
    m2 = (
        stats.var * stats.count
        + batch.var(axis=0) * n
        + delta**2 * stats.count * n / total
    )
    return ObsStats(mean=stats.mean + delta * n / total, var=m2 / total, count=total)


def normalize_obs(stats: ObsStats, obs: ObsBatch, clip: float) -> ObsBatch:
    return np.clip((obs - stats.mean) / np.sqrt(stats.var + 1e-8), -clip, clip)


# --- GAE and the PPO loss ---------------------------------------------------


def gae(
    rewards: jax.Array,
    values: jax.Array,
    dones: jax.Array,
    last_value: jax.Array,
    gamma: float,
    lam: float,
) -> tuple[jax.Array, jax.Array]:
    """Generalized advantage estimation over a rollout of shape (T, N).

    dones[t] is 1 if the episode ended after step t (terminated or truncated).
    Then step t does not use the value of the next observation.
    last_value (shape (N,)) is the value of the observation after step T-1.
    Returns (advantages, returns), both of shape (T, N); returns = advantages + values.
    """
    next_values = jnp.concatenate([values[1:], last_value[None]], axis=0)
    not_done = 1.0 - dones
    deltas = rewards + gamma * next_values * not_done - values

    def backward(carry: jax.Array, x: tuple[jax.Array, jax.Array]):
        delta, nd = x
        adv = delta + gamma * lam * nd * carry
        return adv, adv

    _, advantages = jax.lax.scan(
        backward, jnp.zeros_like(last_value), (deltas, not_done), reverse=True
    )
    return advantages, advantages + values


def clipped_surrogate(
    ratio: jax.Array, advantage: jax.Array, clip_eps: float
) -> jax.Array:
    """The PPO policy loss of each sample: -min(r A, clip(r, 1-eps, 1+eps) A)."""
    clipped = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    return -jnp.minimum(ratio * advantage, clipped * advantage)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Batch:
    """Flat training samples. All fields have the same first axis."""

    obs: jax.Array  # normalized observation, shape (B, OBS_DIM)
    action: jax.Array  # unclipped sampled action, shape (B, ACTION_DIM)
    log_prob: jax.Array  # log probability of action under the rollout policy
    advantage: jax.Array
    target: jax.Array  # value target (GAE return)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LossInfo:
    policy_loss: jax.Array
    value_loss: jax.Array
    entropy: jax.Array  # nats, sum over the action dimensions
    approx_kl: jax.Array  # estimate of KL(old || new), nats
    clip_frac: jax.Array  # fraction of samples with |ratio - 1| > clip_eps


def ppo_loss(
    net: ActorCritic, batch: Batch, params: PPOParams
) -> tuple[jax.Array, LossInfo]:
    """Total loss = policy loss + value_coef * value loss - entropy_coef * entropy.
    The function normalizes the advantages over the batch."""
    adv = (batch.advantage - batch.advantage.mean()) / (batch.advantage.std() + 1e-8)
    log_prob = gaussian_log_prob(mlp(net.actor, batch.obs), net.log_std, batch.action)
    log_ratio = log_prob - batch.log_prob
    ratio = jnp.exp(log_ratio)
    policy_loss = clipped_surrogate(ratio, adv, params.clip_eps).mean()
    value_loss = 0.5 * jnp.mean((value(net, batch.obs) - batch.target) ** 2)
    entropy = jnp.sum(net.log_std + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e))
    total = policy_loss + params.value_coef * value_loss - params.entropy_coef * entropy
    return total, LossInfo(
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy=entropy,
        approx_kl=jnp.mean((ratio - 1.0) - log_ratio),
        clip_frac=jnp.mean(jnp.abs(ratio - 1.0) > params.clip_eps),
    )


def make_optimizer(params: PPOParams) -> optax.GradientTransformation:
    return optax.chain(
        optax.clip_by_global_norm(params.max_grad_norm),
        optax.adam(params.learning_rate, eps=1e-5),
    )


def init_opt_state(net: ActorCritic, params: PPOParams) -> optax.OptState:
    # optax's type hint for params lists only arrays and containers, not
    # registered dataclasses. optax accepts any JAX pytree.
    return make_optimizer(params).init(cast(optax.Params, net))


@functools.partial(jax.jit, static_argnames=("params",))
def ppo_update(
    net: ActorCritic,
    opt_state: optax.OptState,
    batch: Batch,
    key: jax.Array,
    params: PPOParams,
) -> tuple[ActorCritic, optax.OptState, LossInfo]:
    """params.epochs passes of minibatch Adam steps over a shuffled batch.
    Returns the new network, the new optimizer state and the mean loss info."""
    optimizer = make_optimizer(params)
    size = batch.obs.shape[0]
    mb_size = size // params.minibatches

    def minibatch_step(carry, idx):
        net, opt_state = carry
        mb = jax.tree.map(lambda x: x[idx], batch)
        grads, info = jax.grad(ppo_loss, has_aux=True)(net, mb, params)
        updates, opt_state = optimizer.update(grads, opt_state, net)
        return (optax.apply_updates(net, updates), opt_state), info

    def epoch(carry, epoch_key):
        perm = jax.random.permutation(epoch_key, size)[: params.minibatches * mb_size]
        return jax.lax.scan(minibatch_step, carry, perm.reshape(params.minibatches, -1))

    (net, opt_state), infos = jax.lax.scan(
        epoch, (net, opt_state), jax.random.split(key, params.epochs)
    )
    return net, opt_state, jax.tree.map(jnp.mean, infos)


# --- Effects: environments, rollouts, logging, files -----------------------

type Outcome = TerminationReason | Literal["truncated"]


@dataclass(frozen=True)
class EpisodeResult:
    episode_return: float
    lap_frac: float  # laps at the last step; negative if the car drove backward
    outcome: Outcome
    steps: int  # env steps (30 Hz)


@dataclass(frozen=True)
class RolloutState:
    """The state that a rollout carries to the next rollout."""

    obs: npt.NDArray[np.float64]  # raw observation of each env, shape (N, OBS_DIM)
    episode_return: npt.NDArray[np.float64]  # return so far of each env, shape (N,)
    episode_steps: npt.NDArray[np.int64]  # steps so far of each env, shape (N,)
    stats: ObsStats


def _outcome(info: EnvInfo) -> Outcome:
    """The outcome of an episode, from the info of its last step. The env sets a
    termination reason exactly when terminated is True."""
    reason = info["termination_reason"]
    return "truncated" if reason is None else reason


def collect_rollout(
    envs: list[NeoRacerEnv],
    state: RolloutState,
    net: ActorCritic,
    key: jax.Array,
    params: PPOParams,
) -> tuple[Batch, RolloutState, list[EpisodeResult]]:
    """Step every env params.rollout_steps times with the sampled policy.
    The function resets an env when its episode ends.

    On truncation, the reward of the last step gets gamma * V(final observation),
    so that the time limit does not look like a terminal state to GAE.
    """
    t_len, n = params.rollout_steps, len(envs)
    obs_buf = np.zeros((t_len, n, OBS_DIM), dtype=np.float32)
    act_buf = np.zeros((t_len, n, ACTION_DIM), dtype=np.float32)
    logp_buf = np.zeros((t_len, n), dtype=np.float32)
    val_buf = np.zeros((t_len, n), dtype=np.float32)
    rew_buf = np.zeros((t_len, n), dtype=np.float32)
    done_buf = np.zeros((t_len, n), dtype=np.float32)
    raw_obs, ep_ret, ep_steps, stats = (
        state.obs.copy(),
        state.episode_return.copy(),
        state.episode_steps.copy(),
        state.stats,
    )
    finished: list[EpisodeResult] = []
    step_keys = jax.random.split(key, t_len)

    for t in range(t_len):
        stats = update_obs_stats(stats, raw_obs)
        obs = normalize_obs(stats, raw_obs, params.obs_clip)
        action, logp, val = sample_action(net, jnp.asarray(obs), step_keys[t])
        action_np = np.asarray(action)
        obs_buf[t], act_buf[t], logp_buf[t], val_buf[t] = obs, action_np, logp, val
        for i, env in enumerate(envs):
            next_obs, reward, terminated, truncated, info = env.step(
                env_action(action_np[i], params.max_throttle)
            )
            ep_ret[i] += reward
            ep_steps[i] += 1
            rew_buf[t, i] = reward
            if terminated or truncated:
                done_buf[t, i] = 1.0
                if truncated:
                    final = normalize_obs(stats, next_obs[None], params.obs_clip)
                    rew_buf[t, i] += params.gamma * float(
                        value(net, jnp.asarray(final))[0]
                    )
                finished.append(
                    EpisodeResult(
                        float(ep_ret[i]),
                        info["lap_frac"],
                        _outcome(info),
                        int(ep_steps[i]),
                    )
                )
                ep_ret[i], ep_steps[i] = 0.0, 0
                next_obs, _ = env.reset()
            raw_obs[i] = next_obs

    last_obs = normalize_obs(stats, raw_obs, params.obs_clip)
    last_value = value(net, jnp.asarray(last_obs))
    advantage, target = gae(
        jnp.asarray(rew_buf),
        jnp.asarray(val_buf),
        jnp.asarray(done_buf),
        last_value,
        params.gamma,
        params.gae_lambda,
    )
    batch = Batch(
        obs=jnp.asarray(obs_buf).reshape(t_len * n, OBS_DIM),
        action=jnp.asarray(act_buf).reshape(t_len * n, ACTION_DIM),
        log_prob=jnp.asarray(logp_buf).reshape(-1),
        advantage=advantage.reshape(-1),
        target=target.reshape(-1),
    )
    return batch, RolloutState(raw_obs, ep_ret, ep_steps, stats), finished


@dataclass(frozen=True)
class UpdateLog:
    update: int
    env_steps: int
    episodes: tuple[EpisodeResult, ...]  # episodes that ended in this rollout
    loss: LossInfo
    action_std: tuple[float, ...]  # standard deviation of each action


def format_log(log: UpdateLog, steps_per_s: float) -> str:
    eps = log.episodes
    ret = np.mean([e.episode_return for e in eps]) if eps else float("nan")
    lap = np.mean([e.lap_frac for e in eps]) if eps else float("nan")
    reasons = Counter(e.outcome for e in eps)
    return (
        f"update {log.update:4d}  steps {log.env_steps:8d}  {steps_per_s:5.0f} steps/s  "
        f"episodes {len(eps):3d}  return {ret:+7.2f}  lap_frac {lap:+.3f}  "
        f"laps {reasons['lap_complete']:2d} stuck {reasons['stuck']:2d} "
        f"rollover {reasons['rollover']:2d} trunc {reasons['truncated']:2d}  "
        f"pi {float(log.loss.policy_loss):+.4f} v {float(log.loss.value_loss):.4f} "
        f"kl {float(log.loss.approx_kl):.4f} clip {float(log.loss.clip_frac):.2f} "
        f"std {log.action_std[0]:.2f},{log.action_std[1]:.2f}"
    )


def train(
    params: PPOParams, total_steps: int, seed: int, num_envs: int
) -> tuple[ActorCritic, ObsStats]:
    """Train for at least total_steps env steps. Print one log line per update.
    Env i starts from seed * 1000 + i."""
    key = jax.random.key(seed)
    key, init_key = jax.random.split(key)
    net = init_actor_critic(init_key, params)
    opt_state = init_opt_state(net, params)
    envs = [NeoRacerEnv() for _ in range(num_envs)]
    first_obs = np.stack(
        [env.reset(seed=seed * 1000 + i)[0] for i, env in enumerate(envs)]
    )
    state = RolloutState(
        obs=first_obs.astype(np.float64),
        episode_return=np.zeros(num_envs),
        episode_steps=np.zeros(num_envs, dtype=np.int64),
        stats=empty_obs_stats(),
    )
    steps_per_update = params.rollout_steps * num_envs
    updates = -(-total_steps // steps_per_update)  # ceiling division
    for update in range(1, updates + 1):
        start = time.perf_counter()
        key, rollout_key, update_key = jax.random.split(key, 3)
        batch, state, finished = collect_rollout(envs, state, net, rollout_key, params)
        net, opt_state, loss = ppo_update(net, opt_state, batch, update_key, params)
        log = UpdateLog(
            update=update,
            env_steps=update * steps_per_update,
            episodes=tuple(finished),
            loss=loss,
            action_std=tuple(float(s) for s in np.exp(np.asarray(net.log_std))),
        )
        print(
            format_log(log, steps_per_update / (time.perf_counter() - start)),
            flush=True,
        )
    for env in envs:
        env.close()
    return net, state.stats


def save_policy(path: str, net: ActorCritic, stats: ObsStats) -> None:
    arrays: dict[str, np.ndarray] = {
        "log_std": np.asarray(net.log_std),
        "obs_mean": stats.mean,
        "obs_var": stats.var,
        "obs_count": np.array(stats.count),
    }
    for name, layers in (("actor", net.actor), ("critic", net.critic)):
        for i, layer in enumerate(layers):
            arrays[f"{name}.{i}.weight"] = np.asarray(layer.weight)
            arrays[f"{name}.{i}.bias"] = np.asarray(layer.bias)
    np.savez(path, allow_pickle=False, **arrays)


def load_policy(path: str) -> tuple[ActorCritic, ObsStats]:
    """Parse a file from save_policy(). Raises KeyError if an array is missing."""
    with np.load(path) as f:

        def layers(name: str) -> tuple[Dense, ...]:
            count = sum(
                1 for k in f.files if k.startswith(f"{name}.") and k.endswith(".weight")
            )
            if count == 0:
                raise KeyError(f"{path} has no {name} layers")
            return tuple(
                Dense(
                    jnp.asarray(f[f"{name}.{i}.weight"]),
                    jnp.asarray(f[f"{name}.{i}.bias"]),
                )
                for i in range(count)
            )

        net = ActorCritic(layers("actor"), layers("critic"), jnp.asarray(f["log_std"]))
        stats = ObsStats(f["obs_mean"], f["obs_var"], float(f["obs_count"]))
    return net, stats


# --- Evaluation -------------------------------------------------------------


@dataclass(frozen=True)
class Trained:
    net: ActorCritic
    stats: ObsStats
    params: PPOParams  # obs_clip and max_throttle must be the training values


@dataclass(frozen=True)
class RandomActions: ...


@dataclass(frozen=True)
class ZeroAction: ...


type EvalPolicy = Trained | RandomActions | ZeroAction
type Baseline = Literal["random", "zero"]


def evaluate(
    policy: EvalPolicy, episodes: int, first_seed: int, rng: np.random.Generator
) -> list[EpisodeResult]:
    """Run episodes with env seeds first_seed, first_seed + 1, ...
    A trained policy acts with its mean action (deterministic).
    RandomActions draws uniform actions from rng."""
    env = NeoRacerEnv()
    results = []
    for k in range(episodes):
        obs, _ = env.reset(seed=first_seed + k)
        ep_ret, steps = 0.0, 0
        while True:
            match policy:
                case Trained(net=net, stats=stats, params=p):
                    norm = normalize_obs(
                        stats, obs[None].astype(np.float64), p.obs_clip
                    )
                    mean = np.asarray(greedy_action(net, jnp.asarray(norm)))[0]
                    action = env_action(mean, p.max_throttle)
                case RandomActions():
                    action = rng.uniform(-1.0, 1.0, ACTION_DIM)
                case ZeroAction():
                    action = np.zeros(ACTION_DIM)
                case _:
                    assert_never(policy)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_ret += reward
            steps += 1
            if terminated or truncated:
                results.append(
                    EpisodeResult(
                        ep_ret,
                        info["lap_frac"],
                        _outcome(info),
                        steps,
                    )
                )
                break
    env.close()
    return results


def print_eval(name: str, results: list[EpisodeResult], first_seed: int) -> None:
    print(f"[eval] {name}: {len(results)} episodes, env seeds {first_seed}..")
    for k, r in enumerate(results):
        print(
            f"  seed {first_seed + k:6d}  return {r.episode_return:+8.3f}  "
            f"lap_frac {r.lap_frac:+.3f}  steps {r.steps:4d}  outcome {r.outcome}"
        )
    print(
        f"  mean return {np.mean([r.episode_return for r in results]):+.3f}  "
        f"mean lap_frac {np.mean([r.lap_frac for r in results]):+.3f}  "
        f"outcomes {dict(Counter(r.outcome for r in results))}"
    )


def parse_baseline(text: str) -> Baseline:
    match text:
        case "random" | "zero":
            return text
        case _:
            raise argparse.ArgumentTypeError(
                f"baseline must be random or zero, not {text!r}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--total-steps", type=int, default=1_000_000, help="env steps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--out", help="save the trained policy to this .npz file")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--eval", metavar="PATH", help="evaluate a saved policy")
    mode.add_argument("--eval-baseline", type=parse_baseline, help="random or zero")
    parser.add_argument("--episodes", type=int, default=10, help="evaluation episodes")
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=10_000,
        help="env seed of the first evaluation episode",
    )
    args = parser.parse_args()
    params = PPOParams()
    rng = np.random.default_rng(args.seed)  # actions of the random baseline

    if args.eval is not None:
        net, stats = load_policy(args.eval)
        policy: EvalPolicy = Trained(net, stats, params)
        results = evaluate(policy, args.episodes, args.eval_seed, rng)
        print_eval(f"policy {args.eval}", results, args.eval_seed)
        return
    if args.eval_baseline is not None:
        baseline: Baseline = args.eval_baseline
        match baseline:
            case "random":
                policy = RandomActions()
            case "zero":
                policy = ZeroAction()
            case _:
                assert_never(baseline)
        results = evaluate(policy, args.episodes, args.eval_seed, rng)
        print_eval(baseline, results, args.eval_seed)
        return

    print(f"[train_ppo] {params}")
    print(
        f"[train_ppo] total_steps={args.total_steps} seed={args.seed} num_envs={args.num_envs}"
    )
    net, stats = train(params, args.total_steps, args.seed, args.num_envs)
    if args.out is not None:
        save_policy(args.out, net, stats)
        print(f"[train_ppo] saved the policy to {args.out}")


if __name__ == "__main__":
    main()
