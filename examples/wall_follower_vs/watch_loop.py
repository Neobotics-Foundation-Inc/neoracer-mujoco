"""
Interactive MuJoCo viewer of the vehicle-space wall follower on
assets/tracks/loop_corridor.xml. It drives the same control tick as
trials.py. Run it from the repository root. On macOS the passive viewer
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

from examples.wall_follower_vs.dense_lidar import DenseLidarParams
from examples.wall_follower_vs.scene_utils import load_track_scene
from examples.wall_follower_vs.trials import TRIALS, driver, tick
from examples.wall_follower_vs.vehicle_space_controller import (
    VsState,
    label,
    turn_radius_m,
)
from neoracer_mujoco import sim


@dataclass(frozen=True)
class WatchParams:
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

    track = "loop_corridor.xml"
    scene = load_track_scene(track)
    model, car = scene.model, scene.car
    data = sim.settle(model)
    sim.perturb_spawn(model, data, car, np.random.default_rng(args.seed))

    d = driver(scene, DenseLidarParams(n_beams=args.n), TRIALS[track].control_hz)
    state = VsState()
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
            t = tick(d, data, state)
            state, result = t.state, t.result
            viewer.sync()

            if data.time - last_print_s > params.print_period_s:
                print(
                    f"t={data.time:6.2f}s target={result.target_steer_deg:+6.2f}deg "
                    f"R={turn_radius_m(d.params, result.target_steer_deg):5.2f}m "
                    f"throttle={result.command.throttle:.4f} {label(result)}"
                )
                last_print_s = data.time

            remaining = d.dt - (time.time() - loop_start)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
