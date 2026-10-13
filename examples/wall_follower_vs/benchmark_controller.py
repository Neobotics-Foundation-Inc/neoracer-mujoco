"""
Measure the time of one step() of the vehicle-space wall follower on
synthetic scans. The time depends on the computer, so this script is not a
test. Run it from the repository root:

    python -m examples.wall_follower_vs.benchmark_controller --n 270 540 1080

The script uses two scenes. In "corner", a wall 1 m ahead blocks the straight
path and the left side. In "corridor", two walls 0.5 m to each side give
about four times more obstacle points, as on the loop track.
"""

import argparse
import time

import numpy as np

from examples.wall_follower_vs.dense_lidar import DenseLidarParams
from examples.wall_follower_vs.synthetic_scenes import (
    Segment,
    cast_scan,
    corridor_walls,
)
from examples.wall_follower_vs.vehicle_space_controller import (
    VsParams,
    VsState,
    car_geometry,
    step,
)
from neoracer_mujoco import assets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n", type=int, nargs="+", default=[1080], help="LiDAR beam counts"
    )
    parser.add_argument("--iters", type=int, default=200, help="timed steps")
    parser.add_argument("--control-hz", type=float, default=30.0, help="Hz")
    args = parser.parse_args()

    params = VsParams(car=car_geometry(assets.load()))
    dt = 1.0 / args.control_hz
    scenes: tuple[tuple[str, list[Segment]], ...] = (
        ("corner", [((1.0, -0.1), (1.0, 2.0))]),
        ("corridor", corridor_walls(half_width_m=0.5)),
    )
    print(f"control period {1000.0 * dt:.1f} ms, {args.iters} timed steps")
    for name, walls in scenes:
        for n in args.n:
            angles, ranges = cast_scan(walls, DenseLidarParams(n_beams=n))
            state, _ = step(params, VsState(), angles, ranges, dt)  # builds the table
            times_ms = np.empty(args.iters)
            for i in range(args.iters):
                t0 = time.perf_counter()
                state, _ = step(params, state, angles, ranges, dt)
                times_ms[i] = (time.perf_counter() - t0) * 1000.0
            print(
                f"{name:8s} N={n:5d}: mean {times_ms.mean():7.3f} ms  "
                f"p95 {np.percentile(times_ms, 95):7.3f} ms  "
                f"max {times_ms.max():7.3f} ms"
            )


if __name__ == "__main__":
    main()
