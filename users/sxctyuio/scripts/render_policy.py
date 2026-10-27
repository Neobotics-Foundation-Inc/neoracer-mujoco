"""
Score, watch or record a policy that train_ppo.py trained, on the race it
trained in (the run's setup.json), with the policy's mean action.

    python3 scripts/render_policy.py runs/race --headless --races 20
    mjpython scripts/render_policy.py runs/race         # a window (mjpython on macOS)
    python3 scripts/render_policy.py runs/race --record race.mp4

--headless runs the first episode of each of --races races and prints how
each car's ended, its laps, and the race's time. Otherwise one race runs
until --seconds pass, in a viewer, or into a video with --record.
"""

import argparse
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import mujoco.viewer
import numpy as np
from neoracer_rl.envs.neoracer_env import substeps_per_action
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv, VecNormalize

from racing.setup import Setup
from racing.vec_env import RaceVecEnv, make

FULL_PHYSICS = mujoco.mjtState.mjSTATE_FULLPHYSICS


def load(run: Path, races: int, seed: int, nthread: int) -> tuple[PPO, VecEnv, RaceVecEnv]:
    """The run's policy, its env as the policy sees it, and the races under it."""
    setup = Setup.load(run / "setup.json")
    venv = VecNormalize.load(str(run / "vecnormalize.pkl"), make(setup, races, seed, nthread))
    venv.training, venv.norm_reward = False, False
    inner = venv.unwrapped
    assert isinstance(inner, RaceVecEnv)
    return PPO.load(run / "model", device="cpu"), venv, inner


def control_period(inner: RaceVecEnv) -> float:
    """The time (s) of one env step."""
    model = inner.race.scene.model
    return substeps_per_action(inner.race.config.env, model) * float(model.opt.timestep)


def score(run: Path, races: int, seed: int, nthread: int) -> None:
    model, venv, inner = load(run, races, seed, nthread)
    cars = venv.num_envs // races
    dt = control_period(inner)
    obs, steps, ended = venv.reset(), 0, {}
    while len(ended) < venv.num_envs:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, infos = venv.step(action)
        steps += 1
        for i in np.flatnonzero(done):
            ended.setdefault(int(i), (infos[i]["outcome"], infos[i]["lap_frac"], steps * dt))
    print(f"[score] {run}: {races} races of {cars} cars")
    for r in range(races):
        row = [ended[r * cars + k] for k in range(cars)]
        cells = "  ".join(f"{o:12s} {lap:+.2f}" for o, lap, _ in row)
        print(f"  race {r:3d}  {row[0][2]:5.1f} s  {cells}")
    print(f"  outcomes {dict(Counter(o for o, _, _ in ended.values()))}")


def show(run: Path, seed: int, seconds: float, record: Path | None) -> None:
    model, venv, inner = load(run, 1, seed, 1)
    dt = control_period(inner)
    scene = inner.race.scene
    data = mujoco.MjData(scene.model)
    frames, renderer, viewer, shown = [], None, None, None
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = scene.model.bind_scalar(scene.car.body).id
    camera.distance, camera.elevation, camera.azimuth = 2.5, -35.0, 180.0
    rays_hidden = mujoco.MjvOption()
    rays_hidden.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False
    obs = venv.reset()
    for _ in range(round(seconds / dt)):
        start = time.perf_counter()
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, _ = venv.step(action)
        sim = inner.state.sim
        model_row = sim.models[0]
        mujoco.mj_setState(model_row, data, sim.rows[0], FULL_PHYSICS)
        mujoco.mj_forward(model_row, data)
        if record is not None:
            if renderer is None:
                renderer = mujoco.Renderer(model_row, 480, 640)
            renderer.update_scene(data, camera, rays_hidden)
            frames.append(renderer.render())
            continue
        if viewer is None or shown is not model_row:
            if viewer is not None:
                viewer.close()
            viewer, shown = mujoco.viewer.launch_passive(model_row, data), model_row
        viewer.sync()
        time.sleep(max(0.0, dt - (time.perf_counter() - start)))
    if record is not None:
        import imageio

        imageio.mimsave(record, frames, fps=round(1 / dt))
        print(f"[record] {len(frames)} frames to {record}")
    venv.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run", type=Path, help="a run folder of train_ppo.py")
    parser.add_argument("--headless", action="store_true", help="score, no window")
    parser.add_argument("--races", type=int, default=20)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--nthread", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--record", type=Path, help="write the race to this video")
    args = parser.parse_args()
    if args.headless:
        score(args.run, args.races, args.seed, args.nthread)
    else:
        show(args.run, args.seed, args.seconds, args.record)


if __name__ == "__main__":
    main()
