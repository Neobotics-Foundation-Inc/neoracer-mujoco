"""
Measure the time of one step() of the vehicle-space wall follower on a
synthetic scan. The time depends on the computer, so this script is not a
test. Run it from the repository root:

    python -m examples.wall_follower_vs.benchmark_controller --n 1080
"""

import argparse
import time

from examples.wall_follower_vs.dense_lidar import DenseLidarParams
from examples.wall_follower_vs.synthetic_scenes import cast_scan
from examples.wall_follower_vs.vehicle_space_controller import (
    VsParams,
    VsState,
    car_geometry,
    step,
)
from neoracer_mujoco import assets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1080, help="LiDAR beam count")
    parser.add_argument("--iters", type=int, default=50, help="timed steps")
    parser.add_argument("--control-hz", type=float, default=30.0, help="Hz")
    args = parser.parse_args()

    params = VsParams(car=car_geometry(assets.load()))
    dt = 1.0 / args.control_hz
    # A wall 1 m ahead that blocks the straight path and the left side.
    angles, ranges = cast_scan(
        [((1.0, -0.1), (1.0, 2.0))], DenseLidarParams(n_beams=args.n)
    )
    state, _ = step(params, VsState(), angles, ranges, dt)  # builds the arc table
    t0 = time.perf_counter()
    for _ in range(args.iters):
        state, _ = step(params, state, angles, ranges, dt)
    mean_ms = (time.perf_counter() - t0) / args.iters * 1000.0
    print(
        f"N={args.n}: mean step time {mean_ms:.3f} ms "
        f"(control period {1000.0 * dt:.1f} ms)"
    )


if __name__ == "__main__":
    main()
