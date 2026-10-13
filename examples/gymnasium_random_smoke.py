"""
Run random actions in neoracer_mujoco.envs.NeoRacerEnv for a fixed number of episodes.

This script is a smoke test of the environment interface. It does not train
a policy, and it does not measure performance.
The script runs many episodes, one after the other, with reset() and step().
It checks that no NaN appears and that each observation is in the observation space.
The output shows if each episode ended by termination, truncation or the step limit.
A random policy does not complete a lap.

Usage:
    python3 -m examples.gymnasium_random_smoke
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from neoracer_mujoco.envs import NeoRacerEnv
from neoracer_mujoco.envs.neoracer_env import TerminationReason

type Outcome = TerminationReason | Literal["truncated", "max_steps"]


@dataclass(frozen=True)
class SmokeParams:
    episodes: int = 10
    # The limit of this script. It is smaller than max_episode_steps of the env.
    max_steps_per_episode: int = 600


def main() -> None:
    params = SmokeParams()
    env = NeoRacerEnv()

    print(
        f"[gymnasium_random_smoke] {params.episodes} episodes, "
        f"<= {params.max_steps_per_episode} steps each, random actions"
    )

    for ep in range(params.episodes):
        obs, info = env.reset(seed=ep)
        env.action_space.seed(ep)  # the same random actions on every run
        assert env.observation_space.contains(obs), (
            "reset obs outside observation_space"
        )

        ep_return = 0.0
        outcome: Outcome = "max_steps"
        step_count = 0

        for step_count in range(1, params.max_steps_per_episode + 1):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            assert np.all(np.isfinite(obs)), (
                f"non-finite obs at ep {ep} step {step_count}"
            )
            assert np.isfinite(reward), (
                f"non-finite reward at ep {ep} step {step_count}"
            )
            assert env.observation_space.contains(obs), (
                f"obs outside observation_space at ep {ep} step {step_count}"
            )

            ep_return += reward

            if terminated:
                outcome = info["termination_reason"]
                break
            if truncated:
                outcome = "truncated"
                break

        print(
            f"  episode {ep:2d}: steps={step_count:4d}  return={ep_return:+7.2f}  "
            f"outcome={outcome:14s}  lap_frac={info['lap_frac']:+.3f}  "
            f"final_upright_cos={info['upright_cos']:+.3f}"
        )

    env.close()
    print(
        "[gymnasium_random_smoke] done -- no NaNs, no crashes, all observations valid."
    )


if __name__ == "__main__":
    main()
