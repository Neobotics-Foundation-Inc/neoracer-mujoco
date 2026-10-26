"""
Watch a trained PPO policy drive NeoRacerEnv, or record it to a video.

    mjpython examples/render_policy.py                          # live viewer, runs/ppo/best
    mjpython examples/render_policy.py runs/ppo/final --speed 0.5
    python3 examples/render_policy.py --record lap.mp4 --episodes 1   # no window needed
    mjpython examples/render_policy.py runs/s2r/best --nominal         # sim2real policy, nominal car

Live viewing needs mjpython on macOS (installed with mujoco). Recording renders
offscreen, so plain python3 works; it needs `pip install "imageio[ffmpeg]"`.

A policy trained with --sim2real is detected automatically (19-D observation)
and runs inside Sim2RealWrapper, randomized like in training; --nominal turns
off the randomization, noise and delay but keeps its observation format.

A racing policy (train_ppo.py --cars N) is detected from the race.json that
train_ppo.py writes in the run folder: N cars race, the camera follows the
leader, and each race prints every car's result. A run trained with
--cars 2-4 races a random size from that range each race; --cars N fixes it. Cars: 0 red, 1 blue, 2 green, 3 yellow. A racing policy trained with
--sim2real races with every car randomized like in training; --nominal turns
that off.

--camera chase (default) stays behind the car, follow tracks it from a fixed
angle, free leaves the camera to your mouse. LiDAR rays are hidden unless you
pass --lidar. Close the viewer window or press Ctrl+C to stop.
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # users/sxctyuio

import argparse
import glob
import json
import math
import os
import time

import mujoco
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from neoracer_mujoco import assets
from neoracer_rl.envs import NeoRacerEnv, NeoRacerEnvConfig, TerminationConfig
from neoracer_rl.envs.multi_racer_env import OBS_DIM as RACE_OBS_DIM
from neoracer_rl.envs.multi_racer_env import SIM2REAL_OBS_DIM as RACE_S2R_OBS_DIM
from neoracer_rl.envs.multi_racer_env import MultiRacerConfig, MultiRacerVecEnv, RaceConfig, make_race
from neoracer_rl.envs.procedural_env import ProceduralEnvConfig, ProceduralRacerEnv
from neoracer_rl.envs.car_model import GITHUB_CAR, GITHUB_CAR_HELP
from neoracer_rl.envs.drive_model import DriveModelConfig
from neoracer_rl.envs.history import HistoryWrapper
from neoracer_rl.envs.sim2real import OBS_DIM as SIM2REAL_OBS_DIM
from neoracer_rl.envs.sim2real import Sim2RealConfig, Sim2RealWrapper


def find_model(model_dir: str) -> PPO:
    zips = sorted(glob.glob(os.path.join(model_dir, "*.zip")))
    if not zips:
        raise SystemExit(f"no model .zip in {model_dir}")
    return PPO.load(zips[0], device="cpu")


def load_normalizer(model_dir: str, env_fn) -> VecNormalize:
    """The observation normalizer saved by train_ppo.py next to the model."""
    stats_path = os.path.join(model_dir, "vecnormalize.pkl")
    if not os.path.exists(stats_path):
        raise SystemExit(f"missing {stats_path}: the policy needs the normalization stats it was trained with")
    norm = VecNormalize.load(stats_path, DummyVecEnv([env_fn]))
    norm.training = False
    return norm


TOP_DISTANCE = 8.0  # m above the track: loop_corridor (8 x 6 m) fills a 16:9 frame


def setup_camera(model: mujoco.MjModel, cam: mujoco.MjvCamera, mode: str, body: str = "car") -> None:
    if mode == "free":
        return
    if mode == "top":  # fixed, straight down over the track center
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = (0.0, 0.0, 0.0)
        cam.distance, cam.elevation, cam.azimuth = TOP_DISTANCE, -90.0, 90.0
        return
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = model.body(body).id
    cam.distance, cam.elevation = (1.2, -20.0) if mode == "chase" else (2.5, -45.0)
    cam.azimuth = 90.0


def update_camera(model: mujoco.MjModel, data: mujoco.MjData, cam: mujoco.MjvCamera, mode: str,
                  body: str = "car") -> None:
    """Chase mode: point the camera along the car's heading, so it sits behind it.
    MuJoCo's camera looks along its azimuth, so azimuth = yaw is directly behind."""
    if mode != "chase":
        return
    xmat = data.xmat[model.body(body).id]
    yaw = math.degrees(math.atan2(xmat[3], xmat[0]))
    diff = (yaw - cam.azimuth + 180.0) % 360.0 - 180.0
    cam.azimuth += 0.2 * diff  # smoothed, so wall bumps don't shake the view


def scene_options(show_lidar: bool) -> mujoco.MjvOption:
    opt = mujoco.MjvOption()
    opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = int(show_lidar)
    return opt


def run_episodes(env, model, norm, episodes, deterministic, on_step, on_reset=None):
    """Drive the policy; on_step() is called after every control step and
    returns False to stop. on_reset() after every reset (a procedural env
    may have switched to another track, i.e. another model)."""
    for ep in range(episodes):
        obs, info = env.reset()
        if on_reset is not None:
            on_reset()
        track = (f"  track {info['track_seed']} (difficulty {info['track_difficulty']}, "
                 f"{info['track_length_m']:.0f} m)" if "track_seed" in info else "")
        total, done, steps = 0.0, False, 0
        while not done:
            action, _ = model.predict(norm.normalize_obs(obs[None]), deterministic=deterministic)
            obs, reward, term, trunc, info = env.step(action[0])
            total += reward
            steps += 1
            done = term or trunc
            if not on_step(info, total):
                return
        print(f"episode {ep}: reward={total:7.2f}  laps={info['lap_frac']:.2f}  "
              f"time={info['episode_time_s']:.1f}s  end={info['termination_reason'] or 'time limit'}{track}")


def live(env, model, norm, args):
    from mujoco import viewer as mj_viewer

    base = env.unwrapped
    period = base.effective_control_dt / args.speed
    state = {"viewer": None, "model": None, "next_t": time.perf_counter()}

    def on_reset():
        if state["model"] is not base.model:  # first episode, or a new procedural track
            viewer = open_viewer(base.model, base.data, state["viewer"])
            setup_camera(base.model, viewer.cam, args.camera)
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = int(args.lidar)
            state.update(viewer=viewer, model=base.model, next_t=time.perf_counter())

    def on_step(info, total):
        viewer = state["viewer"]
        update_camera(base.model, base.data, viewer.cam, args.camera)
        viewer.sync()
        state["next_t"] += period
        time.sleep(max(0.0, state["next_t"] - time.perf_counter()))  # real-time pacing
        if state["next_t"] < time.perf_counter() - 0.5:  # fell behind (e.g. window dragged)
            state["next_t"] = time.perf_counter()
        return viewer.is_running()

    try:
        run_episodes(env, model, norm, args.episodes, not args.stochastic, on_step, on_reset)
    finally:
        if state["viewer"] is not None:
            state["viewer"].close()


def record(env, model, norm, args):
    import imageio

    base = env.unwrapped
    opt = scene_options(args.lidar)
    frames = []
    state = {"model": None}

    def on_reset():
        if state["model"] is not base.model:  # first episode, or a new procedural track
            # The offscreen framebuffer defaults to 640x480; grow it to the video size.
            base.model.vis.global_.offwidth = max(base.model.vis.global_.offwidth, args.width)
            base.model.vis.global_.offheight = max(base.model.vis.global_.offheight, args.height)
            cam = mujoco.MjvCamera()
            setup_camera(base.model, cam, "follow" if args.camera == "free" else args.camera)
            state.update(model=base.model, cam=cam,
                         renderer=mujoco.Renderer(base.model, height=args.height, width=args.width))

    def on_step(info, total):
        update_camera(base.model, base.data, state["cam"], args.camera)
        state["renderer"].update_scene(base.data, camera=state["cam"], scene_option=opt)
        frames.append(state["renderer"].render().copy())
        return True

    run_episodes(env, model, norm, args.episodes, not args.stochastic, on_step, on_reset)
    fps = round(1.0 / base.effective_control_dt * args.speed)
    imageio.mimsave(args.record, frames, fps=fps)
    print(f"saved {len(frames)} frames to {args.record} at {fps} fps")


COLORS = ("red", "blue", "green", "yellow")


def open_viewer(model, data, old=None):
    """Launch a passive viewer on model, closing old first. macOS mjpython
    allows one viewer at a time and closes it asynchronously on the UI thread,
    so wait for old to finish closing and retry the launch briefly."""
    from mujoco import viewer as mj_viewer

    if old is not None:
        old.close()
        deadline = time.perf_counter() + 5.0
        while old.is_running() and time.perf_counter() < deadline:
            time.sleep(0.02)
    for attempt in range(50):  # up to ~5 s
        try:
            return mj_viewer.launch_passive(model, data)
        except RuntimeError as e:
            if "already open" not in str(e) or attempt == 49:
                raise RuntimeError(
                    f"{e}. Switching models (race size or track) needs a new viewer window; "
                    "use --record, or --cars N for one race size.") from e
            time.sleep(0.1)


def lap_text(info: dict) -> str:
    laps = info.get("lap_times_s") or []
    if info.get("race_time_s") is not None:
        return f" [time {info['race_time_s']:.2f}s, best lap {min(laps):.2f}s]"
    return f" [{len(laps)} lap(s), best {min(laps):.2f}s]" if laps else ""


def run_races(get_env, sizes, model, norm, args, on_start, on_step):
    """Race the shared policy, a random size from sizes each race.
    on_start(env) when a race begins; on_step(leader_body) after every control
    step, returns False to stop."""
    rng = np.random.default_rng(args.seed)
    for race in range(args.episodes):
        env = get_env(int(rng.choice(sizes)))
        on_start(env)
        obs, infos = env.reset(seed=args.seed + race)
        totals = np.zeros(env.n)
        respawns = np.zeros(env.n, dtype=int)
        crashes = np.zeros(env.n, dtype=int)
        while True:
            actions, _ = model.predict(norm.normalize_obs(obs), deterministic=not args.stochastic)
            obs, rewards, term, trunc, infos = env.step(actions)
            totals += rewards
            crashes += [int(i.get("crashes_caused", 0) > 0 and i["fault"] >= 0.5) for i in infos]
            done = term | trunc
            if done.any() and not done.all():  # respawn mode: only these cars restart
                for i in np.where(done)[0]:
                    respawns[i] += 1
                    obs = env.respawn(int(i))
            leader = min(range(env.n), key=lambda i: infos[i]["place"])
            if not on_step(f"c{leader}_car"):
                return
            if done.all():
                break
        results = "  ".join(
            f"{COLORS[i]}: P{infos[i]['place']}"
            + (f" (lidar P{infos[i]['place_lidar']})" if infos[i]["place_lidar"] is not None else "") + " "
            f"{infos[i]['termination_reason'] or 'cut off'} {totals[i]:+.1f}" + lap_text(infos[i])
            + (f" respawned {respawns[i]}x" if respawns[i] else "") + (f" caused {crashes[i]} crash(es)" if crashes[i] else "")
            for i in range(env.n))
        print(f"race {race}, {env.n} cars ({infos[0]['episode_time_s']:.1f}s)  {results}")


def resolve_car(args, info: dict) -> str:
    """The car file a run trained on: --car if given, else from its "vehicle"."""
    if args.car:
        return args.car
    if info.get("vehicle") == "github-v1":
        if not os.path.exists(GITHUB_CAR):
            raise SystemExit(GITHUB_CAR_HELP)
        return GITHUB_CAR
    return assets.DEFAULT_CAR


def vehicle_drive(info: dict):
    """The car a run trained on, from its env.json / race.json:
    no "vehicle" entry or "legacy" -> the original car; "realistic-v1" -> the
    realistic drive on the MJCF's 1.5 grip (before surface_friction existed);
    "realistic-v2" -> the realistic drive at the run's "grip"; "github-v1" ->
    the GitHub car at the run's "grip" (its motor and servo are native)."""
    vehicle = info.get("vehicle", "legacy")
    if vehicle == "legacy":
        return None
    if vehicle == "realistic-v1":
        return DriveModelConfig(surface_friction=None)
    return DriveModelConfig(surface_friction=info.get("grip", 0.8))


def vehicle_label(info: dict, drive) -> str:
    if info.get("vehicle") == "github-v1":
        return f"GitHub car (RS-545 DC motor, DC servo), grip {drive.surface_friction}"
    if drive is None:
        return "original (raw torque) car"
    return f"realistic drive on the older car, grip {drive.surface_friction or 1.5}"


def base_obs_dim(model, history: int) -> int:
    """The env's own observation size, without the memory inputs."""
    d = model.observation_space.shape[0]
    return (d - 2 * history) // history if history > 0 else d


def find_race_info(model_dir: str) -> dict | None:
    """race.json written by train_ppo.py --cars N, in the run folder."""
    for d in (model_dir, os.path.dirname(os.path.normpath(model_dir))):
        path = os.path.join(d, "race.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    return None


def race_main(args, model, race_info: dict) -> None:
    info_cars = race_info.get("cars", [3])
    sizes = [args.cars] if args.cars else (info_cars if isinstance(info_cars, list) else [info_cars])
    history = race_info.get("history", 0)
    car = resolve_car(args, race_info)
    sim2real = base_obs_dim(model, history) == RACE_S2R_OBS_DIM
    s2r = None
    if sim2real:
        s2r = (Sim2RealConfig(randomize_physics=False, sensor_noise=False, action_delay=False)
               if args.nominal else Sim2RealConfig())
    term = TerminationConfig(terminate_on_wall=not args.allow_wall)
    start = race_info.get("start", "grid")
    laps = race_info.get("laps", 1)
    race = RaceConfig(start=start, laps=laps, crash_speed_ms=race_info.get("crash_speed", 0.0),
                      respawn=race_info.get("respawn", False),
                      grid_lane_m=race_info.get("grid_lane", 0.55), grid_row_m=race_info.get("grid_row", 1.2))
    if start == "opposite":
        sizes = [2]
    envs: dict = {}

    def get_env(n):
        if n not in envs:
            envs[n] = make_race(MultiRacerConfig(car_path=car, n_cars=n, termination=term, race=race,
                                                 drive=vehicle_drive(race_info),
                                                 max_episode_steps=max(1800, 600 * laps)), s2r, history)
            envs[n].model.vis.global_.offwidth = max(envs[n].model.vis.global_.offwidth, args.width)
            envs[n].model.vis.global_.offheight = max(envs[n].model.vis.global_.offheight, args.height)
        return envs[n]

    stats_path = os.path.join(args.model_dir, "vecnormalize.pkl")
    if not os.path.exists(stats_path):
        raise SystemExit(f"missing {stats_path}")
    norm = VecNormalize.load(stats_path, MultiRacerVecEnv(MultiRacerConfig(car_path=car, n_cars=2), sim2real=s2r,
                                                          history=history))
    norm.training = False
    mode = ("sim2real, nominal cars" if args.nominal else "sim2real, randomized per car") if sim2real else "plain"
    if history:
        mode += f", memory of {history} steps"
    mode += ", " + vehicle_label(race_info, vehicle_drive(race_info))
    print(f"racing policy: {'/'.join(map(str, sizes))} cars (red, blue, green, yellow), {mode}, "
          f"start {start}, {laps} lap(s)")
    if args.headless:  # results only: no window, no video
        run_races(get_env, sizes, model, norm, args, lambda env: None, lambda leader: True)
        return
    camera = args.camera or "top"  # races: whole track from above by default
    if camera == "free" and args.record:
        camera = "top"

    if args.record:
        import imageio

        frames, state = [], {}

        def on_start(env):
            if state.get("env") is not env:
                state.update(env=env, renderer=mujoco.Renderer(env.model, height=args.height, width=args.width))
            state["cam"] = mujoco.MjvCamera()
            if camera == "top":
                setup_camera(env.model, state["cam"], "top")

        def on_step(leader):
            env, cam = state["env"], state["cam"]
            if camera in ("chase", "follow") and cam.trackbodyid != env.model.body(leader).id:  # leader changed
                setup_camera(env.model, cam, camera, leader)
            update_camera(env.model, env.data, cam, camera, leader)
            state["renderer"].update_scene(env.data, camera=cam, scene_option=scene_options(args.lidar))
            frames.append(state["renderer"].render().copy())
            return True

        run_races(get_env, sizes, model, norm, args, on_start, on_step)
        fps = round(1.0 / state["env"].effective_control_dt * args.speed)
        imageio.mimsave(args.record, frames, fps=fps)
        print(f"saved {len(frames)} frames to {args.record} at {fps} fps")
        return

    from mujoco import viewer as mj_viewer  # not "import mujoco.viewer": that makes mujoco local here

    state: dict = {}

    def on_start(env):
        # A different race size is a different model: reopen the viewer on it.
        if state.get("env") is not env:
            state.update(env=env, viewer=open_viewer(env.model, env.data, state.get("viewer")),
                         next_t=time.perf_counter())
            if camera == "top":
                setup_camera(env.model, state["viewer"].cam, "top")
            state["viewer"].opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = int(args.lidar)

    def on_step(leader):
        env, viewer = state["env"], state["viewer"]
        if camera in ("chase", "follow"):
            if viewer.cam.trackbodyid != env.model.body(leader).id:
                setup_camera(env.model, viewer.cam, camera, leader)
            update_camera(env.model, env.data, viewer.cam, camera, leader)
        viewer.sync()
        state["next_t"] += env.effective_control_dt / args.speed
        time.sleep(max(0.0, state["next_t"] - time.perf_counter()))
        if state["next_t"] < time.perf_counter() - 0.5:
            state["next_t"] = time.perf_counter()
        return viewer.is_running()

    try:
        run_races(get_env, sizes, model, norm, args, on_start, on_step)
    finally:
        if state.get("viewer") is not None:
            state["viewer"].close()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model_dir", nargs="?", default="runs/ppo/best", help="runs/ppo/best, runs/ppo/final, ...")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--speed", type=float, default=1.0, help="playback speed (0.5 = half speed)")
    p.add_argument("--stochastic", action="store_true", help="sample actions instead of the mean")
    p.add_argument("--seed", type=int, default=None, help="default: random each run (printed)")
    p.add_argument("--car", default=None, help="car XML (default: the one the run trained on)")
    p.add_argument("--nominal", action="store_true", help="sim2real policy: no randomization, noise or delay")
    p.add_argument("--allow-wall", action="store_true", help="keep driving after a wall hit instead of ending the episode")
    p.add_argument("--camera", choices=("top", "chase", "follow", "free"), default=None,
                   help="top: whole track from above (default for races); chase: behind the car (default "
                        "single-car, follows the leader in races); follow: fixed angle; free: mouse")
    p.add_argument("--lidar", action="store_true", help="draw the LiDAR rays")
    p.add_argument("--record", metavar="FILE.mp4", help="render offscreen to a video instead")
    p.add_argument("--headless", action="store_true",
                   help="no window or video: just run the episodes/races and print the results (fast, plain python3)")
    p.add_argument("--cars", type=int, default=None, help="racing policy: number of cars (default: race.json)")
    p.add_argument("--track", choices=("loop", "procedural"), default=None,
                   help="single-car: loop_corridor or random generated tracks (default: the run's env.json)")
    p.add_argument("--difficulty", type=lambda t: list(range(int(t.split('-')[0]), int(t.split('-')[-1]) + 1)),
                   default=None, help="procedural: difficulty or range, e.g. 2 or 0-3 (default: the run's)")
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=540)
    args = p.parse_args()

    model = find_model(args.model_dir)
    race_info = find_race_info(args.model_dir)
    obs_dim = model.observation_space.shape[0]
    if race_info is None and obs_dim == RACE_OBS_DIM:
        race_info = {}  # plain racing model without race.json: 21-D is unambiguous
    if race_info is not None or args.cars:
        if args.seed is None:
            args.seed = int(np.random.default_rng().integers(1_000_000))
        print(f"model: {args.model_dir}  seed: {args.seed}")
        try:
            race_main(args, model, race_info or {})
        except KeyboardInterrupt:
            pass
        return
    if args.camera is None:
        args.camera = "chase"  # single car: behind it
    env_info = {}
    for d in (args.model_dir, os.path.dirname(os.path.normpath(args.model_dir))):
        if os.path.exists(os.path.join(d, "env.json")):
            with open(os.path.join(d, "env.json")) as f:
                env_info = json.load(f)
            break
    history = env_info.get("history", 0)
    car = resolve_car(args, env_info)
    sim2real = base_obs_dim(model, history) == SIM2REAL_OBS_DIM
    s2r_cfg = Sim2RealConfig(randomize_physics=False, sensor_noise=False, action_delay=False) if args.nominal else Sim2RealConfig()

    track = args.track or env_info.get("track", "loop")
    difficulty = args.difficulty or env_info.get("difficulty", [0, 1, 2, 3])

    def env_fn():
        base_cfg = NeoRacerEnvConfig(car_path=car, drive=vehicle_drive(env_info),
                                     termination=TerminationConfig(terminate_on_wall=not args.allow_wall))
        if track == "procedural":
            # A fresh, never-seen track every episode: an honest test (and the
            # time to generate one does not matter here).
            env = ProceduralRacerEnv(ProceduralEnvConfig(base=base_cfg, difficulties=tuple(difficulty),
                                                         pool_size=1, new_track_every=1))
        else:
            env = NeoRacerEnv(base_cfg)
        env = Sim2RealWrapper(env, s2r_cfg) if sim2real else env
        return HistoryWrapper(env, history) if history else env
    if track == "procedural":
        print(f"procedural tracks, difficulty {difficulty}: a new, never-seen track every episode")

    if args.seed is None:
        args.seed = int(np.random.default_rng().integers(1_000_000))
    print(f"model: {args.model_dir}  seed: {args.seed}")
    env = env_fn()
    env.reset(seed=args.seed)
    norm = load_normalizer(args.model_dir, env_fn)
    if sim2real:
        print("sim2real policy:", "nominal car, no noise" if args.nominal else "randomized like training")
    if history:
        print(f"memory: last {history} observations and actions")
    drive = vehicle_drive(env_info)
    print("vehicle:", vehicle_label(env_info, drive))
    if args.headless:
        run_episodes(env, model, norm, args.episodes, not args.stochastic, lambda info, total: True)
        return
    try:
        (record if args.record else live)(env, model, norm, args)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
