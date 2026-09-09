"""
Bounded random-agent smoke test for neoracer_mujoco.envs.NeoRacerEnv
(issue #7).

This is NOT a performance benchmark and does not train anything -- it exists
to prove the environment interface itself holds up under repeated use:
reset/step work across many episodes back to back, no NaNs appear, actions
and observations stay valid, and termination/truncation both actually fire.
A random policy is not expected to complete a lap.

Usage:
    python3 -m examples.gymnasium_random_smoke
"""

from __future__ import annotations

import numpy as np

from neoracer_mujoco.envs import NeoRacerEnv

N_EPISODES = 10
MAX_STEPS_PER_EPISODE = 600  # bounded -- not the env's full max_episode_steps


def main() -> None:
    env = NeoRacerEnv()
    rng = np.random.default_rng(0)

    print(
        f"[gymnasium_random_smoke] {N_EPISODES} episodes, "
        f"<= {MAX_STEPS_PER_EPISODE} steps each, random actions"
    )

    for ep in range(N_EPISODES):
        obs, info = env.reset(seed=ep)
        assert env.observation_space.contains(obs), "reset obs outside observation_space"

        ep_return = 0.0
        outcome = "max_steps_reached"
        step_count = 0

        for step_count in range(1, MAX_STEPS_PER_EPISODE + 1):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            assert np.all(np.isfinite(obs)), f"non-finite obs at ep {ep} step {step_count}"
            assert np.isfinite(reward), f"non-finite reward at ep {ep} step {step_count}"
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
            f"outcome={outcome:14s}  lap_frac={info['lap_frac']:.3f}  "
            f"final_upright_cos={info['upright_cos']:+.3f}"
        )

    env.close()
    print("[gymnasium_random_smoke] done -- no NaNs, no crashes, all observations valid.")


if __name__ == "__main__":
    main()
