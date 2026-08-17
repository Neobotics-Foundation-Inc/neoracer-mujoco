"""
Closed-loop MuJoCo trial harness for the vehicle-space wall follower --
experimental, run manually (not part of `pytest validation/`, which stays a
fast deterministic conformance gate per CLAUDE.md). Drives the car down
assets/tracks/straight_corridor.xml at a fixed sensor resolution N, with a
seeded random spawn perturbation per trial, and reports success/stuck/
rollover/contacts/lap-time/clearance/no-path/steering/compute metrics.

Usage:
    python3 examples/wall_follower_vs/closed_loop_trials.py --n 1080 --trials 10
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dense_lidar as dl  # noqa: E402
import vehicle_space_controller as vsc  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CORRIDOR_X_SUCCESS = 10.0  # near the far end of straight_corridor.xml (x: -2..12)
MAX_SIM_TIME_S = 35.0
STUCK_WINDOW_S = 3.0
STUCK_PROGRESS_M = 0.05
UPRIGHT_ROLLOVER_COS = 0.5  # cos(tilt) below this = rolled over


def _load_model() -> mujoco.MjModel:
    scene = mujoco.MjSpec.from_file(
        os.path.join(_REPO, "assets", "tracks", "straight_corridor.xml")
    )
    car = mujoco.MjSpec.from_file(os.path.join(_REPO, "assets", "neoracer.xml"))
    scene.worldbody.add_frame().attach_body(car.body("car"), "", "")
    return scene.compile()


@dataclass
class TrialResult:
    outcome: str  # "success" | "stuck" | "rollover"
    sim_time_s: float
    dx_m: float
    contacts_ticks: int
    min_clearance_m: float
    no_path_ticks: int
    longest_no_path_run: int
    steering_std: float
    max_target_jump_deg: float
    region_switches: int
    mean_tick_ms: float
    max_tick_ms: float


def run_trial(model: mujoco.MjModel, n: int, seed: int) -> TrialResult:
    from neoracer_mujoco import collision, sim as _sim

    data = _sim.settle(model)

    rng = np.random.default_rng(seed)
    lateral = float(rng.uniform(-0.15, 0.15))
    yaw = float(rng.uniform(-0.15, 0.15))
    data.qpos[1] += lateral
    # yaw about +Z, applied as an extra quaternion rotation composed onto the
    # settled orientation.
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
    for _ in range(50):  # re-settle suspension after the perturbation
        mujoco.mj_step(model, data)

    x0_pos = float(data.body("car").xpos[0])

    ctrl = vsc.VehicleSpaceController()
    dt_control = 1.0 / vsc.CONTROL_HZ
    physics_dt = float(model.opt.timestep)
    ticks_per_control = max(1, round(dt_control / physics_dt))
    max_ticks = int(MAX_SIM_TIME_S / dt_control)

    contacts_ticks = 0
    min_clearance = float("inf")
    no_path_ticks = 0
    longest_no_path_run = 0
    cur_no_path_run = 0
    steering_cmds: list[float] = []
    targets: list[float] = []
    region_switches = 0
    prev_region_key = None
    tick_times_ms: list[float] = []
    x_history: list[tuple[float, float]] = []  # (sim_time, x)
    outcome = "stuck"
    sim_time_s = 0.0

    for tick in range(max_ticks):
        t0 = time.perf_counter()
        angles, ranges = dl.cast_dense_lidar(model, data, n)
        result = ctrl.compute(angles, ranges, dt_control)
        tick_times_ms.append((time.perf_counter() - t0) * 1000.0)

        torque_limit = float(model.actuator("fl_motor").ctrlrange[1])
        steer_limit = float(model.actuator("steer_servo").ctrlrange[1])
        data.ctrl[0:4] = result.speed * torque_limit
        data.ctrl[4] = -result.angle * steer_limit  # racecar_core angle -> ctrl[4]

        for _ in range(ticks_per_control):
            mujoco.mj_step(model, data)
            if collision.car_hit_wall(model, data):
                contacts_ticks += 1

        sim_time_s = (tick + 1) * dt_control
        min_clearance = min(min_clearance, result.forward_clear_m)
        steering_cmds.append(result.angle)
        targets.append(result.target_heading_deg)
        if result.state != "NORMAL":
            no_path_ticks += 1
            cur_no_path_run += 1
            longest_no_path_run = max(longest_no_path_run, cur_no_path_run)
        else:
            cur_no_path_run = 0
        region_key = (
            round(result.chosen_region["start_deg"]) if result.chosen_region else None
        )
        if region_key != prev_region_key and prev_region_key is not None:
            region_switches += 1
        prev_region_key = region_key

        upright = _sim.car_upright_cos(data)
        if upright < UPRIGHT_ROLLOVER_COS:
            outcome = "rollover"
            break

        x = float(data.body("car").xpos[0])
        x_history.append((sim_time_s, x))
        if x - x0_pos >= CORRIDOR_X_SUCCESS:
            outcome = "success"
            break
        if sim_time_s >= STUCK_WINDOW_S:
            cutoff = sim_time_s - STUCK_WINDOW_S
            past_x = next((px for pt, px in x_history if pt >= cutoff), x_history[0][1])
            if x - past_x < STUCK_PROGRESS_M:
                outcome = "stuck"
                break

    dx = float(data.body("car").xpos[0]) - x0_pos
    max_jump = float(np.max(np.abs(np.diff(targets)))) if len(targets) > 1 else 0.0

    return TrialResult(
        outcome=outcome,
        sim_time_s=sim_time_s,
        dx_m=dx,
        contacts_ticks=contacts_ticks,
        min_clearance_m=min_clearance if np.isfinite(min_clearance) else -1.0,
        no_path_ticks=no_path_ticks,
        longest_no_path_run=longest_no_path_run,
        steering_std=float(np.std(steering_cmds)) if steering_cmds else 0.0,
        max_target_jump_deg=max_jump,
        region_switches=region_switches,
        mean_tick_ms=float(np.mean(tick_times_ms)) if tick_times_ms else 0.0,
        max_tick_ms=float(np.max(tick_times_ms)) if tick_times_ms else 0.0,
    )


def run_trials(n: int, n_trials: int = 10, seed0: int = 0) -> list[TrialResult]:
    model = _load_model()
    return [run_trial(model, n, seed0 + i) for i in range(n_trials)]


def summarize(results: list[TrialResult]) -> None:
    n_success = sum(r.outcome == "success" for r in results)
    n_stuck = sum(r.outcome == "stuck" for r in results)
    n_rollover = sum(r.outcome == "rollover" for r in results)
    print(f"success={n_success}/{len(results)}  stuck={n_stuck}  rollover={n_rollover}")
    for i, r in enumerate(results):
        print(
            f"  trial {i}: {r.outcome:9s} dx={r.dx_m:6.2f}m t={r.sim_time_s:5.1f}s "
            f"contacts={r.contacts_ticks:4d} min_clear={r.min_clearance_m:5.2f}m "
            f"no_path={r.no_path_ticks:4d} longest_np={r.longest_no_path_run:3d} "
            f"steer_std={r.steering_std:.3f} max_jump={r.max_target_jump_deg:5.2f}deg "
            f"switches={r.region_switches:2d} tick={r.mean_tick_ms:.2f}/{r.max_tick_ms:.2f}ms"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed0", type=int, default=0)
    args = parser.parse_args()
    results = run_trials(args.n, args.trials, args.seed0)
    summarize(results)
