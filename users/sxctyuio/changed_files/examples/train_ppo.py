"""
Train NeoRacerEnv with PPO (Stable-Baselines3), on CPU.

    pip install -e ".[rl]" "stable-baselines3[extra]"
    python3 examples/train_ppo.py --smoke                    # 1-minute sanity run
    python3 examples/train_ppo.py                            # full run
    python3 examples/train_ppo.py --sim2real --out runs/s2r  # with domain randomization
    python3 examples/train_ppo.py --cars 3 --out runs/race   # multi-car racing, one shared policy
    python3 examples/train_ppo.py --cars 3 --sim2real --out runs/race_s2r   # ... randomized per car
    python3 examples/train_ppo.py --cars 2-4 --out runs/race_mix  # 2-, 3- and 4-car races side by side
    python3 examples/train_ppo.py --track procedural --out runs/proc   # random generated tracks
    python3 examples/train_ppo.py --eval runs/ppo/best       # replay (mjpython on macOS for --render)
    tensorboard --logdir runs/ppo/tb                         # curves

Each env runs in its own process (SubprocVecEnv), so --num-envs should be
about your CPU core count. Observations and rewards are normalized
(VecNormalize): the obs span +-1 (quaternion) to +-1000 (accel), which PPO
handles poorly raw. The normalization stats are saved next to the model and
must be loaded with it.

--cars N (2-4) trains MultiRacerEnv: N cars race in one scene, one shared
policy drives them all (parameter sharing). --cars 2-4 (or 2,3,4) runs races
of every size side by side, cycling through the sizes; the number of races is
about num_envs / mean size, rounded up to a whole cycle, each in its own
process, so keep --num-envs about your core count. See neoracer_mujoco/envs/multi_racer_env.py for the race rules.
"""

import argparse
import glob
import importlib.util
import json
import os
import pickle

import numpy as np
import torch

import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import VecEnvWrapper, DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize

from neoracer_mujoco import assets
from neoracer_mujoco.envs import NeoRacerEnv, NeoRacerEnvConfig, RewardConfig, TerminationConfig
from neoracer_mujoco.envs.procedural_env import ProceduralEnvConfig, ProceduralRacerEnv, bank_file, build_bank
from neoracer_mujoco.envs.multi_racer_env import MultiRacerConfig, MultiRacerVecEnv, RaceConfig
from neoracer_mujoco.envs.car_model import GITHUB_CAR, GITHUB_CAR_HELP
from neoracer_mujoco.envs.drive_model import DriveModelConfig
from neoracer_mujoco.envs.history import HistoryWrapper
from neoracer_mujoco.envs.sim2real import Sim2RealConfig, Sim2RealWrapper


class TimePenalty(gym.Wrapper):
    """Subtract a small constant every step, so a faster lap scores higher.

    Speed reward without trusting the speed sensor: the env's progress reward
    pays the same 2*pi per lap however long it takes, and the wheel-encoder
    speed would reward wheelspin against a wall. A per-step cost only pays
    off by actually finishing sooner. Keep it well below the progress earned
    per step (~0.035-0.06 rad), or standing still / crashing becomes attractive.
    """

    def __init__(self, env: gym.Env, penalty: float):
        super().__init__(env)
        self.penalty = penalty

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward - self.penalty, terminated, truncated, info


def make_env(car: str, rank: int, seed: int, render_mode: str | None = None,
             sim2real: bool = False, time_penalty: float = 0.0, cfg: NeoRacerEnvConfig | None = None,
             proc: ProceduralEnvConfig | None = None, history: int = 0):
    def _init():
        if proc is not None:  # random generated tracks
            env = ProceduralRacerEnv(proc, render_mode=render_mode)
        else:
            env = NeoRacerEnv(cfg or NeoRacerEnvConfig(car_path=car), render_mode=render_mode)
        if sim2real:
            env = Sim2RealWrapper(env)
        if time_penalty > 0:
            env = TimePenalty(env, time_penalty)
        if history > 0:  # memory of the last observations and actions (outermost: sees the final obs)
            env = HistoryWrapper(env, history)
        env.reset(seed=seed + rank)
        return env

    return _init


def env_config(args) -> NeoRacerEnvConfig:
    """Env config from the command-line reward / termination options."""
    return NeoRacerEnvConfig(
        car_path=args.car,
        drive=None if args.vehicle == "legacy" else DriveModelConfig(surface_friction=args.grip),
        termination=TerminationConfig(terminate_on_wall=args.terminate_on_wall),
        reward=RewardConfig(wall_penalty=args.wall_penalty, stuck_penalty=args.stuck_penalty,
                            rollover_penalty=args.rollover_penalty, corner_weight=args.corner_weight,
                            wall_clearance_weight=args.wall_clearance_weight, wall_margin_m=args.wall_margin),
    )


def VEHICLE(args) -> str:
    """Which car the run trains on, recorded for render_policy.py (with "grip")."""
    return {"github": "github-v1", "realistic": "realistic-v2", "legacy": "legacy"}[args.vehicle]


def resolve_car(args) -> None:
    """--vehicle picks the car file unless --car names one."""
    if args.legacy_vehicle:
        args.vehicle = "legacy"
    if args.car is None:
        if args.vehicle == "github":
            if not os.path.exists(GITHUB_CAR):
                raise SystemExit(GITHUB_CAR_HELP)
            args.car = GITHUB_CAR
        else:
            args.car = assets.DEFAULT_CAR


def proc_config(args) -> ProceduralEnvConfig | None:
    """ProceduralRacerEnv config for --track procedural, else None."""
    if args.track != "procedural":
        return None
    bank = None
    if 0 < args.track_bank < args.track_pool:
        raise SystemExit(f"--track-pool {args.track_pool} is larger than --track-bank {args.track_bank}: "
                         "the pool would repeat tracks. Use a bank of at least the pool size.")
    if args.track_bank > 0:  # pre-generated tracks: built once in parallel, reused by later runs
        bank = build_bank(bank_file(args.track_bank_dir, args.difficulty, args.track_bank, args.track_bank_seed),
                          args.track_bank, args.difficulty, args.track_bank_seed)
    return ProceduralEnvConfig(base=env_config(args), difficulties=tuple(args.difficulty), bank_path=bank,
                               pool_size=args.track_pool,
                               new_track_every=args.track_swap or (5 if bank else 20))


def parse_cars(text: str) -> list[int]:
    """--cars: "3", "2-4" or "2,3,4" -> the race sizes to cycle through."""
    if "-" in text:
        lo, hi = (int(x) for x in text.split("-"))
        return list(range(lo, hi + 1))
    return [int(x) for x in text.split(",")]


def race_sizes(sizes: list[int], num_envs: int) -> list[int]:
    """One race per process, cycling through sizes: about num_envs slots in
    total, rounded up to a whole cycle so every size is equally represented."""
    cycles = max(1, round(num_envs / (sum(sizes) / len(sizes)) / len(sizes)))
    return sizes * cycles


def race_config(args) -> MultiRacerConfig:
    """MultiRacerEnv config from the command-line options (--cars > 1)."""
    single = env_config(args)
    opposite = args.start == "opposite"
    laps = args.laps or (3 if opposite else 1)
    crash = args.crash_speed if args.crash_speed is not None else (0.5 if opposite else 0.0)
    return MultiRacerConfig(
        car_path=args.car, n_cars=max(args.cars), termination=single.termination, reward=single.reward,
        drive=single.drive,
        max_episode_steps=max(1800, 600 * laps),  # 60 s, or 20 s per lap
        race=RaceConfig(overtake_reward=args.overtake_reward, collision_penalty=args.collision_penalty,
                        victim_share=args.victim_share, near_miss_weight=args.near_miss_weight,
                        time_penalty=args.env_time_penalty, start=args.start, laps=laps,
                        crash_speed_ms=crash, grid_lane_m=args.grid_lane, grid_row_m=args.grid_row,
                        respawn=args.respawn),
    )


END_REASONS = ("rollover", "stuck", "wall", "lap_complete", "collision", "hit", "won", "lost", "time_limit")


class ScheduledTimePenalty(VecEnvWrapper):
    """Subtract a per-step time penalty that ramps linearly from start to end
    over ramp_steps total env steps, then stays at end -- a reward curriculum:
    learn to drive safely first, then get asked for more speed. Applied here,
    in the main process, so it works the same for every env type."""

    def __init__(self, venv, start: float, end: float, ramp_steps: int):
        super().__init__(venv)
        self.start, self.end, self.ramp_steps = start, end, max(1, ramp_steps)
        self.steps = 0

    @property
    def value(self) -> float:
        frac = min(self.steps / self.ramp_steps, 1.0)
        return self.start + (self.end - self.start) * frac

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        rewards = rewards - self.value
        self.steps += self.num_envs
        return obs, rewards, dones, infos


class ActionSmoothness(VecEnvWrapper):
    """Per-step cost for changing the commands: weight * sum((a_t - a_t-1)^2)
    over throttle and steering, per env slot (per car in races). Smooth
    driving pays almost nothing; flicking the steering back and forth every
    step pays a lot. The history resets when a slot's episode ends (a racing
    respawn included), so a new episode's first action is free. Uses the
    policy's commands -- what it controls -- not the noisy, delayed actions
    sim2real applies."""

    def __init__(self, venv, weight: float):
        super().__init__(venv)
        self.weight = weight
        self._prev = None
        self._have_prev = np.zeros(self.num_envs, dtype=bool)
        self._actions = None
        self.last_mean_cost = 0.0

    def reset(self):
        self._have_prev[:] = False
        return self.venv.reset()

    def step_async(self, actions):
        self._actions = np.asarray(actions, dtype=np.float32).reshape(self.num_envs, -1)
        self.venv.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        a = np.clip(self._actions, -1.0, 1.0)
        cost = np.zeros(self.num_envs, dtype=np.float32)
        if self._prev is not None:
            cost = np.where(self._have_prev, self.weight * np.sum((a - self._prev) ** 2, axis=1), 0.0)
        self._prev = a.copy()
        self._have_prev = ~np.asarray(dones, dtype=bool)
        self.last_mean_cost = float(cost.mean())
        return obs, rewards - cost, dones, infos


class LogSmoothness(BaseCallback):
    def __init__(self, wrapper: ActionSmoothness):
        super().__init__()
        self.wrapper = wrapper

    def _on_step(self) -> bool:
        self.logger.record_mean("train/smoothness_cost", self.wrapper.last_mean_cost)
        return True


class LogTimePenalty(BaseCallback):
    def __init__(self, wrapper: ScheduledTimePenalty):
        super().__init__()
        self.wrapper = wrapper

    def _on_step(self) -> bool:
        self.logger.record("train/time_penalty", self.wrapper.value)
        return True


class EpisodeStats(BaseCallback):
    """Log lap fraction and termination reasons to TensorBoard."""

    def _on_step(self) -> bool:
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done:
                self.logger.record_mean("episode/lap_frac", info["lap_frac"])
                reason = info["termination_reason"] or "time_limit"
                for r in END_REASONS:
                    self.logger.record_mean(f"episode/end_{r}", float(reason == r))
                if "crashes_caused" in info:
                    self.logger.record_mean("episode/crashes_caused", info["crashes_caused"])
                    self.logger.record_mean("episode/crashes_hit", info["crashes_hit"])
                if info.get("lap_times_s"):
                    self.logger.record_mean("episode/best_lap_s", min(info["lap_times_s"]))
                if reason in ("lap_complete", "won"):
                    self.logger.record_mean("episode/lap_time_s", info["episode_time_s"])
        return True


class SaveNormalizerOnBest(BaseCallback):
    """Save the normalization stats next to best_model.zip every time it
    updates, so runs/.../best can be rendered while training continues."""

    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def _on_step(self) -> bool:
        self.model.get_vec_normalize_env().save(self.path)
        return True


def init_from(model: PPO, env: VecNormalize, src_dir: str, history: int = 0) -> None:
    """Warm start: copy a trained policy (and its normalization stats) into
    model. The source observation must be a prefix of the new one -- e.g. the
    19-D single-car sim2real policy into the 20-D racing sim2real policy, or
    the 20-D plain one into the 21-D racing one. The extra inputs (place) get
    zero weights, so the warm-started policy initially drives exactly like
    the source and learns what place is worth from there."""
    zips = sorted(glob.glob(os.path.join(src_dir, "*.zip")))
    stats = os.path.join(src_dir, "vecnormalize.pkl")
    if not zips or not os.path.exists(stats):
        raise SystemExit(f"--init-from {src_dir}: needs a model .zip and vecnormalize.pkl")
    old = PPO.load(zips[0], device="cpu")
    d_old, d_new = old.observation_space.shape[0], model.observation_space.shape[0]
    if d_old > d_new:
        raise SystemExit(f"--init-from: source obs is {d_old}-D, new is {d_new}-D")
    old_sd, new_sd = old.policy.state_dict(), model.policy.state_dict()
    for k, v in new_sd.items():
        o = old_sd.get(k)
        if o is None:
            raise SystemExit(f"--init-from: source policy has no {k} (different net_arch?)")
        if o.shape == v.shape:
            new_sd[k] = o.clone()
        elif o.ndim == 2 and o.shape[0] == v.shape[0] and o.shape[1] == d_old and v.shape[1] == d_new:
            w = torch.zeros_like(v)  # first layer: new inputs start at zero weight
            w[:, :d_old] = o
            new_sd[k] = w
        else:
            raise SystemExit(f"--init-from: {k} is {tuple(o.shape)} in the source, {tuple(v.shape)} here")
    model.policy.load_state_dict(new_sd)

    with open(stats, "rb") as f:
        old_norm = pickle.load(f)  # VecNormalize pickles without its env
    rms = env.obs_rms
    old_mean, old_var = old_norm.obs_rms.mean, old_norm.obs_rms.var
    rms.mean[:d_old], rms.var[:d_old] = old_mean, old_var
    if d_new > d_old and history > 0 and d_new == d_old * history + 2 * history:
        # Memory: past observations are distributed like the current one, so
        # they get the same stats; past actions are spread over [-1, 1].
        for k in range(1, history):
            rms.mean[k * d_old:(k + 1) * d_old], rms.var[k * d_old:(k + 1) * d_old] = old_mean, old_var
        rms.mean[history * d_old:], rms.var[history * d_old:] = 0.0, 0.3
    elif d_new > d_old:
        rms.mean[d_old:], rms.var[d_old:] = 0.5, 1.0 / 12.0  # place: uniform-ish in [0, 1]
    rms.count = old_norm.obs_rms.count
    # Reward normalization too: the copied value function predicts returns on
    # the source run's normalized scale. A fresh reward normalizer starts that
    # scale over, so the first updates see large value errors and a noisy
    # policy gradient -- enough to push a converged policy off its peak.
    if getattr(old_norm, "ret_rms", None) is not None and env.norm_reward:
        env.ret_rms.mean = np.copy(old_norm.ret_rms.mean)
        env.ret_rms.var = np.copy(old_norm.ret_rms.var)
        env.ret_rms.count = old_norm.ret_rms.count
    extra = f", {d_new - d_old} new input(s) at zero weight" if d_new > d_old else ", same inputs"
    print(f"warm start from {zips[0]}: {d_old}-D policy -> {d_new}-D{extra}")


def train(args: argparse.Namespace) -> None:
    os.makedirs(args.out, exist_ok=True)
    # With a ramped time penalty the envs get none; ScheduledTimePenalty applies it.
    args.env_time_penalty = 0.0 if args.time_penalty_end is not None else args.time_penalty
    if max(args.cars) > 1 and args.track == "procedural":
        raise SystemExit("--track procedural is single-car for now (no --cars)")
    if max(args.cars) > 1:
        if min(args.cars) < 2 or max(args.cars) > 4:
            raise SystemExit("--cars: race sizes must be 2-4")
        if args.start == "opposite" and args.cars != [2]:
            raise SystemExit("--start opposite is a 2-car format: use --cars 2")
        cfg = race_config(args)
        s2r = Sim2RealConfig() if args.sim2real else None  # every car randomized independently
        sizes = race_sizes(args.cars, args.num_envs)
        env = VecMonitor(MultiRacerVecEnv(cfg, cars=sizes, seed=args.seed, sim2real=s2r, history=args.history))
        # One race of each size, in this process: evaluation covers every field size.
        eval_env = VecMonitor(MultiRacerVecEnv(cfg, cars=args.cars, seed=args.seed + 10_000, sim2real=s2r,
                                               history=args.history,
                                               processes=False))
        with open(os.path.join(args.out, "race.json"), "w") as f:  # render_policy.py reads this
            json.dump({"cars": args.cars, "sim2real": args.sim2real, "start": args.start,
                       "laps": cfg.race.laps, "crash_speed": cfg.race.crash_speed_ms,
                       "grid_lane": args.grid_lane, "grid_row": args.grid_row, "respawn": args.respawn,
                       "history": args.history, "vehicle": VEHICLE(args), "grip": args.grip}, f)
        print(f"racing: {len(sizes)} races of {sizes} cars = {env.num_envs} policy slots, "
              f"start {args.start}, {cfg.race.laps} lap(s)" + (", sim2real" if args.sim2real else "")
              + (", crashed cars respawn" if args.respawn else ", any crash ends the race"))
    else:
        proc = proc_config(args)
        vec = DummyVecEnv if args.num_envs == 1 else SubprocVecEnv
        env = VecMonitor(vec([make_env(args.car, i, args.seed, sim2real=args.sim2real,
                                       time_penalty=args.env_time_penalty, cfg=env_config(args), proc=proc,
                                       history=args.history)
                              for i in range(args.num_envs)]))
        eval_env = VecMonitor(DummyVecEnv([make_env(args.car, 10_000, args.seed, sim2real=args.sim2real,
                                                    time_penalty=args.env_time_penalty, cfg=env_config(args),
                                                    proc=proc, history=args.history)]))
        with open(os.path.join(args.out, "env.json"), "w") as f:  # render_policy.py reads this
            json.dump({"track": args.track, "difficulty": args.difficulty, "sim2real": args.sim2real,
                       "track_bank": args.track_bank,
                       "history": args.history, "vehicle": VEHICLE(args), "grip": args.grip}, f)
        if proc is not None:
            print(f"procedural tracks, difficulty {args.difficulty}, pool {proc.pool_size} per env, "
                  f"swap one every {proc.new_track_every} episodes, "
                  + (f"from a bank of {args.track_bank} saved tracks" if proc.bank_path else "generated as needed")
                  + (", sim2real" if args.sim2real else ""))
    n_envs = env.num_envs
    schedule = None
    if args.time_penalty_end is not None:  # ramped speed incentive, applied here (envs got 0)
        ramp = int(args.timesteps * args.time_penalty_ramp)
        schedule = env = ScheduledTimePenalty(env, args.time_penalty, args.time_penalty_end, ramp)
        # Evaluation always uses the final value, so evals stay comparable and
        # "best" is not biased toward early checkpoints with a smaller penalty.
        eval_env = ScheduledTimePenalty(eval_env, args.time_penalty_end, args.time_penalty_end, 1)
        print(f"time penalty ramps {args.time_penalty} -> {args.time_penalty_end} "
              f"over the first {args.time_penalty_ramp:.0%} of training ({ramp:,} steps)")
    smooth = None
    if args.smoothness_weight > 0:  # cost for changing commands step to step
        smooth = env = ActionSmoothness(env, args.smoothness_weight)
        eval_env = ActionSmoothness(eval_env, args.smoothness_weight)
        print(f"action smoothness cost: {args.smoothness_weight} x (change in throttle and steering)^2 per step")
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    n_steps = 64 if args.smoke else 1024  # per env; rollout = n_steps * num_envs
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=n_steps,
        batch_size=min(512, n_steps * n_envs),
        n_epochs=10,
        # Decays linearly to 0 over training (SB3 passes progress_remaining,
        # 1 -> 0), so late updates can't wreck a policy that already works.
        learning_rate=lambda progress: args.lr * progress,
        target_kl=args.target_kl,  # stop an update early if the policy moves too far
        gamma=0.99,  # ~3 s horizon at 30 Hz
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
        # TensorBoard logging only if installed (stable-baselines3[extra]).
        tensorboard_log=os.path.join(args.out, "tb") if importlib.util.find_spec("tensorboard") else None,
        seed=args.seed,
        verbose=1,
    )

    if args.init_from:
        init_from(model, env, args.init_from, args.history)

    every = max(1, (2_000 if args.smoke else 200_000) // n_envs)
    callbacks = [
        EpisodeStats(),
        *([LogTimePenalty(schedule)] if schedule is not None else []),
        *([LogSmoothness(smooth)] if smooth is not None else []),
        CheckpointCallback(every, os.path.join(args.out, "checkpoints"), save_vecnormalize=True),
        EvalCallback(eval_env, best_model_save_path=os.path.join(args.out, "best"),
                     callback_on_new_best=SaveNormalizerOnBest(os.path.join(args.out, "best", "vecnormalize.pkl")),
                     eval_freq=every, n_eval_episodes=1 if args.smoke else 5, deterministic=True),
    ]
    # EvalCallback syncs eval_env's normalization stats from env before each eval.
    model.learn(total_timesteps=args.timesteps, callback=callbacks)

    model.save(os.path.join(args.out, "final", "model"))
    env.save(os.path.join(args.out, "final", "vecnormalize.pkl"))
    print(f"saved to {args.out}/final and {args.out}/best")
    env.close()


def evaluate(args: argparse.Namespace) -> None:
    if max(args.cars) > 1:
        raise SystemExit("for a racing policy use: python3 examples/render_policy.py " + args.eval)
    render = "human" if args.render else None
    model = PPO.load(os.path.join(args.eval, "best_model" if "best" in args.eval else "model"))
    sim2real = model.observation_space.shape[0] == 19  # trained with Sim2RealWrapper
    env = DummyVecEnv([make_env(args.car, 0, args.seed, render, sim2real)])
    env = VecNormalize.load(os.path.join(args.eval, "vecnormalize.pkl"), env)
    env.training, env.norm_reward = False, False

    obs = env.reset()
    for ep in range(args.episodes):
        total, done = 0.0, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = env.step(action)
            total += float(env.get_original_reward()[0])
            done = bool(dones[0])
            if args.render:
                env.envs[0].unwrapped.render()
        info = infos[0]
        print(f"episode {ep}: reward={total:.2f}  laps={info['lap_frac']:.2f}  "
              f"end={info['termination_reason'] or 'time limit'}")
    env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--smoke", action="store_true", help="tiny run to check everything works")
    p.add_argument("--timesteps", type=int, default=5_000_000)
    p.add_argument("--num-envs", type=int, default=os.cpu_count() or 4)
    p.add_argument("--vehicle", choices=("github", "realistic", "legacy"), default="github",
                   help="github: the upstream car (RS-545 DC motor, DC servo, measured masses; "
                        "assets/neoracer_github.xml); realistic: the older car with this project's drive model; "
                        "legacy: the original car")
    p.add_argument("--car", default=None, help="car XML (default: from --vehicle)")
    p.add_argument("--sim2real", action="store_true", help="domain randomization, sensor noise, latency")
    p.add_argument("--smoothness-weight", type=float, default=0.0,
                   help="cost per step for changing the commands: weight x (d throttle^2 + d steering^2); "
                        "try 0.01 against jittery steering (0: off)")
    p.add_argument("--time-penalty-end", type=float, default=None,
                   help="ramp the time penalty from --time-penalty up to this value during training "
                        "(reward curriculum: safety first, then speed)")
    p.add_argument("--time-penalty-ramp", type=float, default=0.7,
                   help="fraction of training over which that ramp happens (the learning rate decays to 0, "
                        "so a ramp ending at 1.0 would barely matter)")
    p.add_argument("--time-penalty", type=float, default=0.015,
                   help="reward cost per step; rewards faster laps (0 disables)")
    p.add_argument("--track", choices=("loop", "procedural"), default="loop",
                   help="loop_corridor, or random generated tracks (ProceduralRacerEnv)")
    p.add_argument("--difficulty", type=parse_cars, default=[0, 1, 2, 3],
                   help="procedural: track difficulty 0 (gentle, wide) .. 3 (tight, narrow); e.g. 0-1 or 2")
    p.add_argument("--track-bank", type=int, default=0, metavar="N",
                   help="procedural: train on N pre-generated, saved tracks (0: generate as needed). "
                        "More tracks = less memorizing; a 10M-step run used to see ~1000+")
    p.add_argument("--track-bank-dir", default="track_banks", help="procedural: where track banks are saved")
    p.add_argument("--track-bank-seed", type=int, default=0, help="procedural: which bank to build/reuse")
    p.add_argument("--track-pool", type=int, default=8, help="procedural: tracks each env cycles through")
    p.add_argument("--track-swap", type=int, default=0, metavar="K",
                   help="procedural: replace one pool track every K episodes (default 5 with a bank, else 20)")
    p.add_argument("--cars", type=parse_cars, default=[1],
                   help="race with one shared policy: 3, or 2-4 / 2,3,4 for mixed race sizes")
    p.add_argument("--grip", type=float, default=0.8,
                   help="realistic car: sliding friction of every surface (0.8 = typical rubber on an indoor "
                        "floor; set a measured value). Randomized +-25%% with --sim2real")
    p.add_argument("--legacy-vehicle", action="store_true",
                   help="the original car: throttle = raw torque, +-0.4 rad steering, 360 deg LiDAR "
                        "(default: realistic drive matched to the real NeoRacer, see drive_model.py)")
    p.add_argument("--history", type=int, default=0, metavar="N",
                   help="memory: add the last N observations and N actions to the observation (0: off; try 3)")
    p.add_argument("--init-from", metavar="DIR",
                   help="warm start from a trained model folder (e.g. runs/s2r/ckpt3_2M)")
    p.add_argument("--start", choices=("grid", "opposite"), default="grid",
                   help="racing: grid (side by side) or opposite (2 cars half a lap apart, RoboRacer style)")
    p.add_argument("--laps", type=int, default=None, help="racing: laps to win (default 1 grid, 3 opposite)")
    p.add_argument("--crash-speed", type=float, default=None,
                   help="racing: closing speed (m/s) at which contact ends the race (default 0 grid, 0.5 opposite)")
    p.add_argument("--grid-lane", type=float, default=0.55,
                   help="racing: m between side-by-side cars (0.6 leaves only ~5 cm to the walls)")
    p.add_argument("--grid-row", type=float, default=1.2, help="racing: m between grid rows")
    p.add_argument("--overtake-reward", type=float, default=1.0, help="racing: reward per car passed")
    p.add_argument("--collision-penalty", type=float, default=10.0,
                   help="racing: penalty for the car at fault in a car-to-car crash (once, at impact)")
    p.add_argument("--respawn", action=argparse.BooleanOptionalAction, default=True,
                   help="racing: a crashed car respawns and car-to-car crashes don't end the race "
                        "(--no-respawn: any crash ends the race for everyone)")
    p.add_argument("--victim-share", type=float, default=0.2,
                   help="racing: fraction of the collision penalty the car that got hit pays")
    p.add_argument("--near-miss-weight", type=float, default=0.03,
                   help="racing: per-step cost at contact for a rival within 15 cm in the LiDAR (0 disables)")
    p.add_argument("--corner-weight", type=float, default=0.5,
                   help="extra progress reward for driving close to inner-corner apexes (0 disables)")
    p.add_argument("--terminate-on-wall", action=argparse.BooleanOptionalAction, default=True,
                   help="end the episode on any wall contact")
    p.add_argument("--wall-clearance-weight", type=float, default=0.02,
                   help="per-step cost at contact for running closer than --wall-margin to a wall (0 disables)")
    p.add_argument("--wall-margin", type=float, default=0.08, help="m of wall clearance below which that cost starts")
    p.add_argument("--stuck-penalty", type=float, default=5.0,
                   help="penalty when the stuck check ends the episode; raise it with --wall-penalty")
    p.add_argument("--rollover-penalty", type=float, default=10.0, help="penalty when the car rolls over")
    p.add_argument("--wall-penalty", type=float, default=5.0,
                   help="extra penalty when a wall ends the episode; keep <= --stuck-penalty or stalling beats driving")
    p.add_argument("--lr", type=float, default=3e-4, help="initial learning rate (decays to 0)")
    p.add_argument("--target-kl", type=float, default=0.02, help="early-stop each update past this KL")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/ppo")
    p.add_argument("--eval", metavar="DIR", help="replay a saved model dir (runs/ppo/best or /final)")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--render", action="store_true", help="viewer during --eval")
    args = p.parse_args()
    resolve_car(args)
    if args.smoke:
        args.timesteps, args.num_envs = 4_096, min(args.num_envs, 4)
    evaluate(args) if args.eval else train(args)
