"""
Train a PPO policy (Stable-Baselines3) on races of neoracer-rl's
NeoRacerRace: one car alone (--cars 1), or 2 to 4 cars on a grid, every one
driven by the one policy. Each flag of racing/setup.py's Setup is a flag
here.

    python3 scripts/train_ppo.py --out runs/loop
    python3 scripts/train_ppo.py --track procedural --difficulty 2 --sim2real --history 6 --out runs/proc
    python3 scripts/train_ppo.py --cars 2 --sim2real --out runs/race

Observations and rewards are normalized (VecNormalize), which the PPO needs:
the observations span from 1 (gravity) to tens (LiDAR, acceleration). A run
saves runs/<name>/model.zip, the normalization in vecnormalize.pkl, and its
Setup in setup.json, so render_policy.py replays it on the race it trained
in. Checkpoints go to runs/<name>/checkpoints.
"""

import argparse
import dataclasses
import os
import sys
import typing
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize

from racing.setup import Setup
from racing.vec_env import make


def add_setup_arguments(parser: argparse.ArgumentParser) -> None:
    """A flag for each field of Setup, with its default."""
    for field in dataclasses.fields(Setup):
        flag = "--" + field.name.replace("_", "-")
        if field.type is bool:
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=field.default)
        elif typing.get_origin(field.type) is typing.Literal:
            parser.add_argument(flag, choices=typing.get_args(field.type), default=field.default)
        else:
            parser.add_argument(flag, type=type(field.default), default=field.default)


def setup_of(args: argparse.Namespace) -> Setup:
    return Setup(**{f.name: getattr(args, f.name) for f in dataclasses.fields(Setup)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_setup_arguments(parser)
    parser.add_argument("--timesteps", type=int, default=5_000_000)
    parser.add_argument("--races", type=int, default=8, help="races stepped together")
    parser.add_argument("--nthread", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--lr", type=float, default=3e-4, help="decays to 0")
    parser.add_argument("--target-kl", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs/ppo"))
    args = parser.parse_args()
    setup = setup_of(args)
    args.out.mkdir(parents=True, exist_ok=True)
    setup.save(args.out / "setup.json")

    env = make(setup, args.races, args.seed, args.nthread)
    env = VecNormalize(VecMonitor(env), clip_obs=10.0, gamma=0.99)
    n_steps = 1024
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=n_steps,
        batch_size=min(512, n_steps * env.num_envs),
        n_epochs=10,
        learning_rate=lambda progress: args.lr * progress,
        target_kl=args.target_kl,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        policy_kwargs={"net_arch": {"pi": [128, 128], "vf": [128, 128]}},
        seed=args.seed,
        verbose=1,
    )
    every = max(1, 500_000 // env.num_envs)
    checkpoints = CheckpointCallback(every, str(args.out / "checkpoints"), save_vecnormalize=True)
    model.learn(args.timesteps, callback=checkpoints)
    model.save(args.out / "model")
    env.save(str(args.out / "vecnormalize.pkl"))
    env.close()


if __name__ == "__main__":
    main()
