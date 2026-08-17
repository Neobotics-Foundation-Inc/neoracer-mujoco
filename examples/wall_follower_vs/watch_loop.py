"""
Interactive MuJoCo viewer for the vehicle-space wall follower on
assets/tracks/loop_corridor.xml -- so you can actually watch it drive, not
just read trial statistics. Experimental; not part of PR #27.

Drives the car with the same dense_lidar.cast_dense_lidar() +
vehicle_space_controller.VehicleSpaceController pipeline loop_trials.py
uses for the closed-loop validation runs -- this is a viewer around that
same controller, not a different one.

Usage (macOS requires mjpython for the passive viewer, not python3):
    mjpython examples/wall_follower_vs/watch_loop.py
    mjpython examples/wall_follower_vs/watch_loop.py --n 1080 --seed 0
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EXAMPLES_DIR)
sys.path.insert(0, os.path.dirname(_EXAMPLES_DIR))
import dense_lidar as dl  # noqa: E402
import vehicle_space_controller as vsc  # noqa: E402
from run_racecar_core import load_scene  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1080, help="dense LiDAR sample count")
    parser.add_argument("--seed", type=int, default=0, help="spawn perturbation seed")
    args = parser.parse_args()

    from neoracer_mujoco import sim as _sim

    model = load_scene("loop_corridor.xml")
    data = _sim.settle(model)

    rng = np.random.default_rng(args.seed)
    lateral = float(rng.uniform(-0.15, 0.15))
    yaw = float(rng.uniform(-0.15, 0.15))
    data.qpos[1] += lateral
    half = yaw / 2.0
    dq = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])
    q = data.qpos[3:7]
    w0, x0, y0, z0 = q
    w1, x1, y1, z1 = dq
    data.qpos[3:7] = [
        w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
        w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
        w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
        w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
    ]
    mujoco.mj_forward(model, data)
    for _ in range(50):
        mujoco.mj_step(model, data)

    ctrl = vsc.VehicleSpaceController()
    dt_control = 1.0 / vsc.CONTROL_HZ
    physics_dt = float(model.opt.timestep)
    ticks_per_control = max(1, round(dt_control / physics_dt))
    torque_limit = float(model.actuator("fl_motor").ctrlrange[1])
    steer_limit = float(model.actuator("steer_servo").ctrlrange[1])

    car_id = model.body("car").id
    print(
        f"[watch_loop] N={args.n} seed={args.seed} -- close the viewer window to stop"
    )

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = car_id
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -45

        last_print = 0.0
        while viewer.is_running():
            loop_start = time.time()

            angles, ranges = dl.cast_dense_lidar(model, data, args.n)
            result = ctrl.compute(angles, ranges, dt_control)
            data.ctrl[0:4] = result.speed * torque_limit
            data.ctrl[4] = -result.angle * steer_limit

            for _ in range(ticks_per_control):
                mujoco.mj_step(model, data)
            viewer.sync()

            if data.time - last_print > 0.5:
                r_str = (
                    "inf"
                    if np.isinf(result.turn_radius_m)
                    else f"{result.turn_radius_m:.2f}"
                )
                print(
                    f"t={data.time:6.2f}s delta={result.target_heading_deg:+6.2f}deg "
                    f"R={r_str:>5s}m speed={result.speed:.4f} state={result.state:10s}"
                )
                last_print = data.time

            remaining = dt_control - (time.time() - loop_start)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
