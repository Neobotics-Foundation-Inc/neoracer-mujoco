"""
Closed-loop MuJoCo trial harness for the vehicle-space wall follower on
assets/tracks/loop_corridor.xml -- the chamfered-corner ring track that
previously caused wall-following controllers to graze walls, corner too
late, wedge, spin, or get stuck (see loop_corridor.xml's own docstring).
Experimental; run manually, not part of `pytest validation/`.

Reuses examples/run_racecar_core.py's load_scene() so the car spawns at the
same (0, -2.5, 0) offset -- centered in the bottom straight, facing +X --
that every other loop_corridor.xml consumer in this repo uses.

Usage:
    python3 examples/wall_follower_vs/loop_trials.py --n 1080 --trials 1 --verbose
    python3 examples/wall_follower_vs/loop_trials.py --n 1080 --trials 10
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass

import mujoco
import numpy as np

_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EXAMPLES_DIR)
sys.path.insert(0, os.path.dirname(_EXAMPLES_DIR))
import dense_lidar as dl
import vehicle_space_controller as vsc
from run_racecar_core import load_scene

MAX_SIM_TIME_S = 60.0  # ~1 lap (~23m centerline) at ~0.3-1.0 m/s + cornering slowdown
STUCK_WINDOW_S = 4.0
STUCK_PROGRESS_RAD = 0.05  # minimum angular progress required over the window
UPRIGHT_ROLLOVER_COS = 0.5
LAP_TARGET_RAD = 2.0 * np.pi


def _settle(model: mujoco.MjModel) -> mujoco.MjData:
    from neoracer_mujoco import sim as _sim

    return _sim.settle(model)


def _quat_yaw(data: mujoco.MjData) -> float:
    xmat = data.body("car").xmat.reshape(3, 3)
    return float(np.arctan2(xmat[1, 0], xmat[0, 0]))


@dataclass
class LoopTrialResult:
    seed: int
    outcome: str  # "success" | "stuck" | "rollover"
    sim_time_s: float
    lap_frac: float
    # Contacts: from actual MuJoCo contact state (collision.car_hit_wall),
    # NOT the controller's forward_clear_m diagnostic.
    contact_ticks: int
    contact_events: int  # contiguous contact runs, not raw ticks
    # Physically meaningful clearance: the minimum VALID (>0) raw LiDAR
    # range this trial ever saw -- a real sensed distance to an obstacle,
    # independent of the controller's own arc-model bookkeeping (which is
    # reported separately as diagnostic_min_forward_clear_m below).
    min_sensed_range_m: float
    diagnostic_min_forward_clear_m: float  # forward_clear_m artifact; NOT physical
    no_path_ticks: int
    no_path_events: int
    no_path_durations_ticks: list  # length of each NO_PATH event, in ticks
    longest_no_path_run: int
    no_path_positions: list  # (t_s, x, y) at the start of each NO_PATH event
    no_path_caused_contact: bool  # any contact tick within 1s after a NO_PATH event
    max_target_jump_deg: float
    region_switches: int
    # Timing, split so raycasting and controller compute can be analyzed
    # separately (Stage D's component-timing request) without touching
    # vehicle_space_controller.py: raycast_ms is dense_lidar.cast_dense_lidar
    # alone; compute_ms is ctrl.compute() alone -- the SAME call that drives
    # the actual ctrl.speed/ctrl.angle sent to MuJoCo, just timed around.
    raycast_times_ms: list
    compute_times_ms: list
    obstacle_point_counts: list  # len(x) from the same preprocessing
    # ctrl.compute() already ran internally -- measured via a SEPARATE call
    # to the public preprocessing function AFTER control was already applied
    # this tick, so it never affects what the car actually did.


def run_loop_trial(
    model: mujoco.MjModel, n: int, seed: int, verbose: bool = False, log_every: int = 1
) -> LoopTrialResult:
    from neoracer_mujoco import collision
    from neoracer_mujoco import sim as _sim

    data = _settle(model)

    rng = np.random.default_rng(seed)
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

    x0p, y0p = float(data.body("car").xpos[0]), float(data.body("car").xpos[1])
    prev_angle = float(np.arctan2(y0p, x0p))
    cum_angle = 0.0

    ctrl = vsc.VehicleSpaceController()
    dt_control = 1.0 / vsc.CONTROL_HZ
    physics_dt = float(model.opt.timestep)
    ticks_per_control = max(1, round(dt_control / physics_dt))
    max_ticks = int(MAX_SIM_TIME_S / dt_control)

    torque_limit = float(model.actuator("fl_motor").ctrlrange[1])
    steer_limit = float(model.actuator("steer_servo").ctrlrange[1])

    contact_ticks = 0
    contact_events = 0
    cur_contact_run = False
    contact_tick_indices: set[int] = set()
    diagnostic_min_forward_clear = float("inf")
    min_sensed_range = float("inf")
    no_path_ticks = 0
    no_path_events = 0
    no_path_durations_ticks: list[int] = []
    no_path_event_start_ticks: list[int] = []
    longest_no_path_run = 0
    cur_no_path_run = 0
    no_path_positions: list[tuple[float, float, float]] = []
    targets: list[float] = []
    region_switches = 0
    prev_region_key = None
    raycast_times_ms: list[float] = []
    compute_times_ms: list[float] = []
    obstacle_point_counts: list[int] = []
    angle_history: list[tuple[float, float]] = [(0.0, 0.0)]  # (sim_time, cum_angle)
    outcome = "stuck"
    sim_time_s = 0.0
    prev_state = "NORMAL"

    for tick in range(max_ticks):
        t0 = time.perf_counter()
        angles, ranges = dl.cast_dense_lidar(model, data, n)
        t1 = time.perf_counter()
        valid_ranges = ranges[ranges > 0]
        if len(valid_ranges) > 0:
            min_sensed_range = min(min_sensed_range, float(valid_ranges.min()))
        result = ctrl.compute(angles, ranges, dt_control)
        t2 = time.perf_counter()
        raycast_times_ms.append((t1 - t0) * 1000.0)
        compute_times_ms.append((t2 - t1) * 1000.0)
        # Point count via the SAME public preprocessing function compute()
        # used internally, called again here purely to measure -- control
        # for this tick was already decided and is about to be applied
        # below, so this cannot change vehicle behavior.
        obs_x, _ = vsc._obstacle_points_with_midpoints(angles, ranges)
        obstacle_point_counts.append(len(obs_x))

        data.ctrl[0:4] = result.speed * torque_limit
        data.ctrl[4] = -result.angle * steer_limit

        tick_had_contact = False
        for _ in range(ticks_per_control):
            mujoco.mj_step(model, data)
            if collision.car_hit_wall(model, data):
                contact_ticks += 1
                tick_had_contact = True
        if tick_had_contact:
            contact_tick_indices.add(tick)
        if tick_had_contact and not cur_contact_run:
            contact_events += 1
        cur_contact_run = tick_had_contact

        sim_time_s = (tick + 1) * dt_control
        diagnostic_min_forward_clear = min(
            diagnostic_min_forward_clear, result.forward_clear_m
        )
        targets.append(result.target_heading_deg)
        xp, yp = float(data.body("car").xpos[0]), float(data.body("car").xpos[1])
        if result.state != "NORMAL":
            if cur_no_path_run == 0:
                no_path_events += 1
                no_path_positions.append((sim_time_s, xp, yp))
                no_path_event_start_ticks.append(tick)
            no_path_ticks += 1
            cur_no_path_run += 1
            longest_no_path_run = max(longest_no_path_run, cur_no_path_run)
        else:
            if cur_no_path_run > 0:
                no_path_durations_ticks.append(cur_no_path_run)
            cur_no_path_run = 0
        region_key = (
            round(result.chosen_region["start_deg"]) if result.chosen_region else None
        )
        if region_key != prev_region_key and prev_region_key is not None:
            region_switches += 1
        prev_region_key = region_key

        angle = float(np.arctan2(yp, xp))
        delta = angle - prev_angle
        if delta > np.pi:
            delta -= 2 * np.pi
        elif delta < -np.pi:
            delta += 2 * np.pi
        cum_angle += delta
        prev_angle = angle
        angle_history.append((sim_time_s, cum_angle))

        if verbose and tick % log_every == 0:
            yaw_deg = np.degrees(_quat_yaw(data))
            region = result.chosen_region
            region_str = (
                f"[{region['start_deg']:.0f},{region['end_deg']:.0f}]"
                if region
                else "None"
            )
            r_str = (
                "inf"
                if np.isinf(result.turn_radius_m)
                else f"{result.turn_radius_m:.2f}"
            )
            print(
                f"t={sim_time_s:5.2f}s pos=({xp:+.2f},{yp:+.2f}) yaw={yaw_deg:+6.1f} "
                f"delta={result.target_heading_deg:+6.2f}deg R={r_str:>5s}m "
                f"region={region_str:>12s} "
                f"straight_clear={result.forward_clear_m:.2f} sel_clear={result.selected_path_clear_m:.2f} "
                f"steer={result.angle:+.3f} speed={result.speed:.4f} state={result.state:10s} "
                f"cum_angle={np.degrees(cum_angle):+7.1f}deg"
            )
        if result.state != prev_state and verbose:
            print(
                f"  -- state transition {prev_state} -> {result.state} at t={sim_time_s:.2f}s"
            )
        prev_state = result.state

        upright = _sim.car_upright_cos(data)
        if upright < UPRIGHT_ROLLOVER_COS:
            outcome = "rollover"
            if verbose:
                print(f"  ROLLOVER at t={sim_time_s:.2f}s upright_cos={upright:.3f}")
            break

        if abs(cum_angle) >= LAP_TARGET_RAD:
            outcome = "success"
            break
        if sim_time_s >= STUCK_WINDOW_S:
            cutoff = sim_time_s - STUCK_WINDOW_S
            past_angle = next(
                (pa for pt, pa in angle_history if pt >= cutoff), angle_history[0][1]
            )
            if abs(cum_angle - past_angle) < STUCK_PROGRESS_RAD:
                outcome = "stuck"
                if verbose:
                    print(f"  STUCK detected at t={sim_time_s:.2f}s")
                break

    if cur_no_path_run > 0:  # trial ended mid-event (success/rollover/stuck)
        no_path_durations_ticks.append(cur_no_path_run)

    no_path_caused_contact = any(
        any((start + w) in contact_tick_indices for w in range(30))  # ~1s window
        for start in no_path_event_start_ticks
    )

    lap_frac = abs(cum_angle) / LAP_TARGET_RAD
    max_jump = float(np.max(np.abs(np.diff(targets)))) if len(targets) > 1 else 0.0

    return LoopTrialResult(
        seed=seed,
        outcome=outcome,
        sim_time_s=sim_time_s,
        lap_frac=lap_frac,
        contact_ticks=contact_ticks,
        contact_events=contact_events,
        min_sensed_range_m=min_sensed_range if np.isfinite(min_sensed_range) else -1.0,
        diagnostic_min_forward_clear_m=(
            diagnostic_min_forward_clear
            if np.isfinite(diagnostic_min_forward_clear)
            else -1.0
        ),
        no_path_ticks=no_path_ticks,
        no_path_events=no_path_events,
        no_path_durations_ticks=no_path_durations_ticks,
        longest_no_path_run=longest_no_path_run,
        no_path_positions=no_path_positions,
        no_path_caused_contact=no_path_caused_contact,
        max_target_jump_deg=max_jump,
        region_switches=region_switches,
        raycast_times_ms=raycast_times_ms,
        compute_times_ms=compute_times_ms,
        obstacle_point_counts=obstacle_point_counts,
    )


def run_loop_trials(
    n: int, n_trials: int = 10, seed0: int = 0
) -> list[LoopTrialResult]:
    model = load_scene("loop_corridor.xml")
    return [run_loop_trial(model, n, seed0 + i) for i in range(n_trials)]


def _tick_total_ms(r: LoopTrialResult) -> np.ndarray:
    """Per-tick TOTAL time (raycast + compute) -- what actually has to fit
    inside the 33.3ms/30Hz budget in the real control loop."""
    return np.asarray(r.raycast_times_ms) + np.asarray(r.compute_times_ms)


def summarize(results: list[LoopTrialResult]) -> None:
    n_success = sum(r.outcome == "success" for r in results)
    n_stuck = sum(r.outcome == "stuck" for r in results)
    n_rollover = sum(r.outcome == "rollover" for r in results)
    print(f"success={n_success}/{len(results)}  stuck={n_stuck}  rollover={n_rollover}")
    for r in results:
        total_ms = _tick_total_ms(r)
        mean_ms = float(np.mean(total_ms)) if len(total_ms) else 0.0
        max_ms = float(np.max(total_ms)) if len(total_ms) else 0.0
        print(
            f"  seed {r.seed}: {r.outcome:9s} lap={r.lap_frac:5.2f} t={r.sim_time_s:5.1f}s "
            f"contact_evt={r.contact_events:2d} contact_ticks={r.contact_ticks:4d} "
            f"min_sensed={r.min_sensed_range_m:5.2f}m "
            f"no_path_evt={r.no_path_events:2d} no_path_ticks={r.no_path_ticks:4d} "
            f"longest_np={r.longest_no_path_run:3d} max_jump={r.max_target_jump_deg:5.2f}deg "
            f"switches={r.region_switches:3d} tick={mean_ms:.2f}/{max_ms:.2f}ms"
        )
        for t_s, x, y in r.no_path_positions:
            print(f"      NO_PATH event at t={t_s:.2f}s pos=({x:+.2f},{y:+.2f})")


def compute_distribution(results: list[LoopTrialResult]) -> dict:
    """Aggregate every controller tick's TOTAL wall-clock time (raycast +
    compute) across ALL trials into one distribution."""
    all_ms = np.concatenate(
        [_tick_total_ms(r) for r in results if len(r.compute_times_ms)]
    )
    return {
        "n_ticks": len(all_ms),
        "mean": float(np.mean(all_ms)),
        "median": float(np.median(all_ms)),
        "p90": float(np.percentile(all_ms, 90)),
        "p95": float(np.percentile(all_ms, 95)),
        "p99": float(np.percentile(all_ms, 99)),
        "p99_9": float(np.percentile(all_ms, 99.9)),
        "max": float(np.max(all_ms)),
        "pct_over_33_3ms": float(np.mean(all_ms > 33.3) * 100.0),
        "pct_over_50ms": float(np.mean(all_ms > 50.0) * 100.0),
        "pct_over_100ms": float(np.mean(all_ms > 100.0) * 100.0),
    }


def deadline_miss_analysis(results: list[LoopTrialResult]) -> dict:
    """Correlate ticks that exceed the 33.3ms budget with obstacle point
    count, NO_PATH state, and which seed/wall-clock position they occurred
    at -- to distinguish an algorithmic worst case (correlates with point
    count) from external jitter (scattered, uncorrelated)."""
    miss_points: list[int] = []
    miss_seeds: list[int] = []
    all_points: list[int] = []
    all_ms: list[float] = []
    for r in results:
        total_ms = _tick_total_ms(r)
        for i, ms in enumerate(total_ms):
            all_points.append(r.obstacle_point_counts[i])
            all_ms.append(float(ms))
            if ms > 33.3:
                miss_points.append(r.obstacle_point_counts[i])
                miss_seeds.append(r.seed)

    all_points_arr = np.asarray(all_points)
    all_ms_arr = np.asarray(all_ms)
    correlation = (
        float(np.corrcoef(all_points_arr, all_ms_arr)[0, 1])
        if len(all_ms_arr) > 1
        else 0.0
    )
    seeds_with_misses = sorted(set(miss_seeds))
    return {
        "n_misses": len(miss_points),
        "miss_seed_count": len(seeds_with_misses),
        "seeds_with_misses": seeds_with_misses,
        "mean_points_at_miss": float(np.mean(miss_points)) if miss_points else 0.0,
        "mean_points_overall": float(np.mean(all_points_arr))
        if len(all_points_arr)
        else 0.0,
        "pearson_r_points_vs_ms": correlation,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed0", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log-every", type=int, default=1)
    args = parser.parse_args()
    if args.trials == 1 and args.verbose:
        model = load_scene("loop_corridor.xml")
        r = run_loop_trial(
            model, args.n, args.seed0, verbose=True, log_every=args.log_every
        )
        summarize([r])
    else:
        results = run_loop_trials(args.n, args.trials, args.seed0)
        summarize(results)
        dist = compute_distribution(results)
        print(
            f"compute (N={args.n}, {dist['n_ticks']} ticks): mean={dist['mean']:.2f}ms "
            f"median={dist['median']:.2f}ms p95={dist['p95']:.2f}ms p99={dist['p99']:.2f}ms "
            f"max={dist['max']:.2f}ms  over_33.3ms={dist['pct_over_33_3ms']:.2f}%"
        )
