"""
Closed-loop trials of the vehicle-space wall follower on
assets/tracks/straight_corridor.xml. Each trial perturbs the spawn pose with
its own seed. A trial is a success when the car drives corridor_success_m
along +X. The trials are not part of the test suite, which must stay fast and
deterministic (AGENTS.md). Run them by hand from the repository root:

    python -m examples.wall_follower_vs.closed_loop_trials --n 1080 --trials 10

Frame: +X forward, +Y left, +Z up.
"""

import argparse
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import assert_never

import mujoco
import numpy as np

from examples.wall_follower_vs.dense_lidar import (
    DenseLidarParams,
    cast_dense_lidar,
    lidar_mount,
)
from examples.wall_follower_vs.report_utils import Outcome, outcome_header
from examples.wall_follower_vs.scene_utils import load_track_scene, perturb_spawn
from examples.wall_follower_vs.vehicle_space_controller import (
    Driving,
    NoPath,
    VsParams,
    VsState,
    car_geometry,
    step,
)
from neoracer_mujoco import collision, sim
from neoracer_mujoco.scene import Scene


@dataclass(frozen=True)
class TrialParams:
    control_hz: float = 30.0  # Hz; controller rate
    # m along +X; near the far end of straight_corridor.xml (x from -2 m to 12 m)
    corridor_success_m: float = 10.0
    max_sim_time_s: float = 35.0  # s
    stuck_window_s: float = 3.0  # s
    stuck_progress_m: float = 0.05  # m; minimum progress along +X in stuck_window_s
    upright_rollover_cos: float = 0.5  # cos(tilt); below this the car has rolled over


@dataclass(frozen=True)
class TrialResult:
    outcome: Outcome
    sim_time_s: float  # s
    dx_m: float  # m; progress along +X
    contact_steps: int  # physics steps in which the car touches a wall
    # m; the smallest safe arc length of the 0 deg candidate. This value is a
    # controller output, not a measured distance.
    min_straight_clear_m: float
    no_path_ticks: int
    longest_no_path_ticks: int
    steer_std_rad: float  # rad; standard deviation of the steer command
    max_target_jump_deg: float  # deg; largest change of the target in one tick
    region_switches: int  # ticks at which the start of the selected region changed
    mean_tick_ms: float  # ms; wall-clock time of ray cast plus step()
    max_tick_ms: float  # ms


def run_trial(
    scene: Scene,
    lidar: DenseLidarParams,
    seed: int,
    params: TrialParams = TrialParams(),
) -> TrialResult:
    """Drive one trial from a spawn pose that rng(seed) perturbs."""
    model, car = scene.model, scene.car
    data = sim.settle(model)
    perturb_spawn(scene, data, np.random.default_rng(seed))

    car_id = car.body
    walls = collision.wall_geom_ids(scene.spec, model)
    mount = lidar_mount(scene)
    vs_params = VsParams(car=car_geometry(scene))
    dt = 1.0 / params.control_hz
    steps_per_tick = max(1, round(dt / float(model.opt.timestep)))
    max_ticks = int(params.max_sim_time_s / dt)

    x0_m = float(data.xpos[car_id][0])
    state = VsState()
    contact_steps = 0
    min_straight_clear_m = math.inf
    no_path_ticks = 0
    longest_no_path_ticks = 0
    no_path_run = 0
    steer_cmds: list[float] = []
    targets: list[float] = []
    region_switches = 0
    prev_region_key: int | None = None
    tick_times_ms: list[float] = []
    x_history: list[tuple[float, float]] = []  # (sim time, x)
    outcome: Outcome = "timeout"
    sim_time_s = 0.0

    for tick in range(max_ticks):
        t0 = time.perf_counter()
        angles, ranges = cast_dense_lidar(model, data, mount, lidar)
        state, result = step(vs_params, state, angles, ranges, dt)
        tick_times_ms.append((time.perf_counter() - t0) * 1000.0)

        sim.apply_command(car, data, result.command)
        for _ in range(steps_per_tick):
            mujoco.mj_step(model, data)
            if collision.car_touches_wall(model, data, car_id, walls):
                contact_steps += 1

        sim_time_s = (tick + 1) * dt
        min_straight_clear_m = min(min_straight_clear_m, result.straight_clear_m)
        steer_cmds.append(result.command.steer)
        targets.append(result.target_steer_deg)
        match result:
            case Driving(region=region):
                no_path_run = 0
                region_key = round(region.start_deg)
            case NoPath():
                no_path_ticks += 1
                no_path_run += 1
                longest_no_path_ticks = max(longest_no_path_ticks, no_path_run)
                region_key = None
            case _:
                assert_never(result)
        if region_key != prev_region_key and prev_region_key is not None:
            region_switches += 1
        prev_region_key = region_key

        if sim.car_upright_cos(data, car_id) < params.upright_rollover_cos:
            outcome = "rollover"
            break
        x_m = float(data.xpos[car_id][0])
        x_history.append((sim_time_s, x_m))
        if x_m - x0_m >= params.corridor_success_m:
            outcome = "success"
            break
        if sim_time_s >= params.stuck_window_s:
            cutoff = sim_time_s - params.stuck_window_s
            past_x = next((px for pt, px in x_history if pt >= cutoff), x_history[0][1])
            if x_m - past_x < params.stuck_progress_m:
                outcome = "stuck"
                break

    return TrialResult(
        outcome=outcome,
        sim_time_s=sim_time_s,
        dx_m=float(data.xpos[car_id][0]) - x0_m,
        contact_steps=contact_steps,
        min_straight_clear_m=min_straight_clear_m,
        no_path_ticks=no_path_ticks,
        longest_no_path_ticks=longest_no_path_ticks,
        steer_std_rad=float(np.std(steer_cmds)),
        max_target_jump_deg=(
            float(np.max(np.abs(np.diff(targets)))) if len(targets) > 1 else 0.0
        ),
        region_switches=region_switches,
        mean_tick_ms=float(np.mean(tick_times_ms)),
        max_tick_ms=float(np.max(tick_times_ms)),
    )


def summarize(results: Sequence[TrialResult]) -> None:
    print(outcome_header([r.outcome for r in results]))
    for i, r in enumerate(results):
        print(
            f"  trial {i}: {r.outcome:9s} dx={r.dx_m:6.2f}m t={r.sim_time_s:5.1f}s "
            f"contact_steps={r.contact_steps:4d} "
            f"min_straight_clear={r.min_straight_clear_m:5.2f}m "
            f"no_path={r.no_path_ticks:4d} longest_np={r.longest_no_path_ticks:3d} "
            f"steer_std={r.steer_std_rad:.3f}rad "
            f"max_jump={r.max_target_jump_deg:5.2f}deg "
            f"switches={r.region_switches:2d} "
            f"tick={r.mean_tick_ms:.2f}/{r.max_tick_ms:.2f}ms"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, required=True, help="LiDAR beam count")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed0", type=int, default=0, help="seed of the first trial")
    args = parser.parse_args()
    scene = load_track_scene("straight_corridor.xml")
    lidar = DenseLidarParams(n_beams=args.n)
    summarize([run_trial(scene, lidar, args.seed0 + i) for i in range(args.trials)])


if __name__ == "__main__":
    main()
