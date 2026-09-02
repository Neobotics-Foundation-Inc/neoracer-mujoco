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
import time
from dataclasses import dataclass

import dense_lidar as dl
import mujoco
import numpy as np
import vehicle_space_controller as vsc
from math_utils import compose_yaw_perturbation
from report_utils import outcome_header

from neoracer_mujoco import collision
from neoracer_mujoco import sim as _sim

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass(frozen=True)
class TrialParams:
    corridor_x_success: float = (
        10.0  # near the far end of straight_corridor.xml (x: -2..12)
    )
    max_sim_time_s: float = 35.0
    stuck_window_s: float = 3.0
    stuck_progress_m: float = 0.05
    upright_rollover_cos: float = 0.5  # cos(tilt) below this = rolled over


_DEFAULT_TRIAL_PARAMS = TrialParams()


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


def run_trial(
    model: mujoco.MjModel,
    n: int,
    seed: int,
    params: TrialParams = _DEFAULT_TRIAL_PARAMS,
) -> TrialResult:
    data = _sim.settle(model)

    rng = np.random.default_rng(seed)
    lateral = float(rng.uniform(-0.15, 0.15))
    yaw = float(rng.uniform(-0.15, 0.15))
    data.qpos[1] += lateral
    # yaw about +Z, applied as an extra quaternion rotation composed onto the
    # settled orientation.
    data.qpos[3:7] = compose_yaw_perturbation(data.qpos[3:7], yaw)
    mujoco.mj_forward(model, data)
    for _ in range(50):  # re-settle suspension after the perturbation
        mujoco.mj_step(model, data)

    x0_pos = float(data.body("car").xpos[0])

    ctrl = vsc.VehicleSpaceController()
    dt_control = 1.0 / vsc.CONTROL_HZ
    physics_dt = float(model.opt.timestep)
    ticks_per_control = max(1, round(dt_control / physics_dt))
    max_ticks = int(params.max_sim_time_s / dt_control)

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
        if upright < params.upright_rollover_cos:
            outcome = "rollover"
            break

        x = float(data.body("car").xpos[0])
        x_history.append((sim_time_s, x))
        if x - x0_pos >= params.corridor_x_success:
            outcome = "success"
            break
        if sim_time_s >= params.stuck_window_s:
            cutoff = sim_time_s - params.stuck_window_s
            past_x = next((px for pt, px in x_history if pt >= cutoff), x_history[0][1])
            if x - past_x < params.stuck_progress_m:
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


def run_trials(
    n: int,
    n_trials: int = 10,
    seed0: int = 0,
    params: TrialParams = _DEFAULT_TRIAL_PARAMS,
) -> list[TrialResult]:
    model = _load_model()
    return [run_trial(model, n, seed0 + i, params) for i in range(n_trials)]


def summarize(results: list[TrialResult]) -> None:
    print(outcome_header(results))
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
