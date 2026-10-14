"""
Interactive MuJoCo viewer of the vehicle-space wall follower on
assets/tracks/loop_corridor.xml. It uses the same ray cast and controller as
loop_trials.py. Run it from the repository root. On macOS the passive viewer
needs mjpython, not python3:

    mjpython -m examples.wall_follower_vs.watch_loop
    mjpython -m examples.wall_follower_vs.watch_loop --n 1080 --seed 0
"""

import argparse
import time
from dataclasses import dataclass

import mujoco
import mujoco.viewer
import numpy as np

from examples.wall_follower_vs.dense_lidar import (
    DenseLidarParams,
    cast_dense_lidar,
    lidar_mount,
)
from examples.wall_follower_vs.scene_utils import load_track_scene, perturb_spawn
from examples.wall_follower_vs.vehicle_space_controller import (
    VsParams,
    VsState,
    car_geometry,
    label,
    step,
    turn_radius_m,
)
from neoracer_mujoco import sim


@dataclass(frozen=True)
class WatchParams:
    control_hz: float = 30.0  # Hz; controller rate
    camera_distance_m: float = 3.0  # m; tracking camera to car
    camera_azimuth_deg: float = 90.0  # deg
    camera_elevation_deg: float = -45.0  # deg; negative looks down
    print_period_s: float = 0.5  # s of sim time between two status lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1080, help="LiDAR beam count")
    parser.add_argument("--seed", type=int, default=0, help="spawn perturbation seed")
    args = parser.parse_args()
    params = WatchParams()

    scene = load_track_scene("loop_corridor.xml")
    model, car = scene.model, scene.car
    data = sim.settle(model)
    perturb_spawn(scene, data, np.random.default_rng(args.seed))

    mount = lidar_mount(scene)
    lidar = DenseLidarParams(n_beams=args.n)
    vs_params = VsParams(car=car_geometry(scene))
    state = VsState()
    dt = 1.0 / params.control_hz
    steps_per_tick = max(1, round(dt / float(model.opt.timestep)))
    print(f"[watch_loop] N={args.n} seed={args.seed}. Close the viewer window to stop.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = model.bind_scalar(car.body).id
        viewer.cam.distance = params.camera_distance_m
        viewer.cam.azimuth = params.camera_azimuth_deg
        viewer.cam.elevation = params.camera_elevation_deg

        last_print_s = 0.0
        while viewer.is_running():
            loop_start = time.time()
            angles, ranges = cast_dense_lidar(model, data, mount, lidar)
            state, result = step(vs_params, state, angles, ranges, dt)
            sim.apply_command(car, data, result.command)
            for _ in range(steps_per_tick):
                mujoco.mj_step(model, data)
            viewer.sync()

            if data.time - last_print_s > params.print_period_s:
                print(
                    f"t={data.time:6.2f}s target={result.target_steer_deg:+6.2f}deg "
                    f"R={turn_radius_m(vs_params, result.target_steer_deg):5.2f}m "
                    f"throttle={result.command.throttle:.4f}N·m {label(result)}"
                )
                last_print_s = data.time

            remaining = dt - (time.time() - loop_start)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
