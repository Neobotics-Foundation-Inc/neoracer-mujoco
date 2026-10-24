"""
Robustness report of the vehicle-space wall follower on loop_corridor.xml.
The script runs many seeded trials of trials.run_trial(), saves the
raw results in a pickle file, and prints the failures, contacts, NoPath
events, lap times and tick times. Run it by hand from the repository root:

    python -m examples.wall_follower_vs.loop_robustness --n 1080 --trials 100
"""

import argparse
import pickle
from collections import Counter
from dataclasses import dataclass
from typing import Literal, assert_never

import numpy as np

from examples.wall_follower_vs.dense_lidar import DenseLidarParams
from examples.wall_follower_vs.report_utils import outcome_header
from examples.wall_follower_vs.scene_utils import load_track_scene
from examples.wall_follower_vs.trials import (
    TRIALS,
    TrialParams,
    TrialResult,
    deadline_misses,
    run_trial,
    tick_time_stats,
)

type FailureClass = Literal[
    "rollover",
    "no_path_with_contact",  # a long NoPath run, and the car touched a wall
    "contact",  # the car touched a wall without a long NoPath run
    "no_path",  # a long NoPath run without contact
    "target_switching",  # many changes of the selected region
    "other",
]
type ContactClass = Literal[
    "brief_contact_lap",
    "sustained_contact_lap",
    "contact_then_stuck",
    "contact_then_timeout",
    "rollover",
]


@dataclass(frozen=True)
class ClassifyParams:
    """Thresholds of the failure and contact classes."""

    long_no_path_ticks: int = 30  # ticks; 1 s at 30 Hz
    many_region_switches: int = 100
    sustained_contact_steps: int = 30  # physics steps; 0.06 s at 2 ms per step


def classify_failure(r: TrialResult, params: ClassifyParams) -> FailureClass | None:
    """The failure class of a trial, or None for a success. The class uses
    only the values in TrialResult."""
    long_no_path = r.longest_no_path_ticks >= params.long_no_path_ticks
    match r.outcome:
        case "success":
            return None
        case "rollover":
            return "rollover"
        case "stuck" | "timeout":
            if r.contact_steps > 0:
                return "no_path_with_contact" if long_no_path else "contact"
            if long_no_path:
                return "no_path"
            if r.region_switches > params.many_region_switches:
                return "target_switching"
            return "other"
        case _:
            assert_never(r.outcome)


def classify_contact(r: TrialResult, params: ClassifyParams) -> ContactClass:
    """The contact class of a trial with at least one contact."""
    match r.outcome:
        case "success":
            if r.contact_steps > params.sustained_contact_steps:
                return "sustained_contact_lap"
            return "brief_contact_lap"
        case "stuck":
            return "contact_then_stuck"
        case "timeout":
            return "contact_then_timeout"
        case "rollover":
            return "rollover"
        case _:
            assert_never(r.outcome)


def report(
    results: list[TrialResult],
    trial_params: TrialParams,
    params: ClassifyParams,
) -> None:
    n = len(results)
    print(f"\n=== OUTCOMES ===\n{outcome_header([r.outcome for r in results])}")

    print("\n=== FAILURES ===")
    for r in results:
        failure = classify_failure(r, params)
        if failure is not None:
            print(
                f"  seed {r.seed}: {r.outcome} lap={r.fraction:.2f} class={failure} "
                f"contact_steps={r.contact_steps} no_path_ticks={r.no_path_ticks} "
                f"longest_np={r.longest_no_path_ticks}"
            )

    with_contact = [r for r in results if r.contact_steps > 0]
    print("\n=== CONTACTS ===")
    print(f"trials with any contact: {len(with_contact)}/{n}")
    print(f"total contact events: {sum(r.contact_events for r in results)}")
    print(f"total contact steps: {sum(r.contact_steps for r in results)}")
    if with_contact:
        worst = max(with_contact, key=lambda r: r.contact_steps)
        print(
            f"worst contact trial: seed {worst.seed}, {worst.contact_steps} steps, "
            f"outcome={worst.outcome}"
        )
    for r in with_contact:
        print(
            f"  seed {r.seed}: {classify_contact(r, params)} "
            f"({r.contact_steps} steps, {r.contact_events} events)"
        )

    with_no_path = [r for r in results if r.no_path_events]
    events = [e for r in results for e in r.no_path_events]
    print("\n=== NO PATH ===")
    print(
        f"trials with NoPath: {len(with_no_path)}/{n} "
        f"({100 * len(with_no_path) / n:.1f}%)"
    )
    print(f"total NoPath events: {len(events)}")
    print(f"total NoPath ticks: {sum(r.no_path_ticks for r in results)}")
    if events:
        durations = np.asarray([e.duration_ticks for e in events])
        print(
            f"event duration (ticks): median={np.median(durations):.1f} "
            f"p95={np.percentile(durations, 95):.1f} max={durations.max()}"
        )
    print(
        "longest NoPath run of all trials: "
        f"{max((r.longest_no_path_ticks for r in results), default=0)}"
    )
    print(
        "trials with a contact within "
        f"{trial_params.contact_after_no_path_s:.1f}s after a NoPath start: "
        f"{sum(r.no_path_caused_contact for r in results)}"
    )
    print(
        "trials with NoPath that failed: "
        f"{sum(r.outcome != 'success' for r in with_no_path)}/{len(with_no_path)}"
    )
    cells = Counter((round(e.x_m * 2) / 2, round(e.y_m * 2) / 2) for e in events)
    print("NoPath start positions (0.5 m grid, top 8):")
    for (x, y), count in cells.most_common(8):
        print(f"  ({x:+.1f}, {y:+.1f}): {count}")

    successes = [r for r in results if r.outcome == "success"]
    print("\n=== SUCCESSFUL TRIALS ===")
    if successes:
        lap_times = np.asarray([r.sim_time_s for r in successes])
        print(
            f"lap time: mean={lap_times.mean():.2f}s median={np.median(lap_times):.2f}s "
            f"p5={np.percentile(lap_times, 5):.2f}s "
            f"p95={np.percentile(lap_times, 95):.2f}s "
            f"min={lap_times.min():.2f}s max={lap_times.max():.2f}s"
        )
        min_ranges = np.asarray([r.min_sensed_range_m for r in successes])
        print(
            f"min sensed range: mean={min_ranges.mean():.3f}m "
            f"median={np.median(min_ranges):.3f}m "
            f"p5={np.percentile(min_ranges, 5):.3f}m min={min_ranges.min():.3f}m"
        )
        per_trial = np.asarray([len(r.no_path_events) for r in successes])
        print(
            f"NoPath events per trial: mean={per_trial.mean():.2f} "
            f"median={np.median(per_trial):.1f} max={per_trial.max()}"
        )

    stats = tick_time_stats(results, trial_params)
    print("\n=== TICK TIME (ray cast + step) ===")
    print(f"total ticks: {stats.n_ticks}")
    print(
        f"mean={stats.mean_ms:.2f}ms median={stats.median_ms:.2f}ms "
        f"p90={stats.p90_ms:.2f}ms p95={stats.p95_ms:.2f}ms p99={stats.p99_ms:.2f}ms "
        f"p99.9={stats.p99_9_ms:.2f}ms max={stats.max_ms:.2f}ms"
    )
    print(
        f">{trial_params.deadline_ms:.1f}ms: {stats.pct_over_deadline:.3f}%  "
        f">{trial_params.slow_tick_ms:.0f}ms: {stats.pct_over_slow:.3f}%  "
        f">{trial_params.very_slow_tick_ms:.0f}ms: {stats.pct_over_very_slow:.3f}%"
    )
    raycast = np.concatenate([np.asarray(r.raycast_times_ms) for r in results])
    compute = np.concatenate([np.asarray(r.compute_times_ms) for r in results])
    print(
        f"ray cast only: mean={raycast.mean():.2f}ms "
        f"median={np.median(raycast):.2f}ms max={raycast.max():.2f}ms"
    )
    print(
        f"step only: mean={compute.mean():.2f}ms "
        f"median={np.median(compute):.2f}ms max={compute.max():.2f}ms"
    )

    misses = deadline_misses(results, trial_params)
    seeds = misses.seeds_with_misses
    print("\n=== DEADLINE MISSES ===")
    print(f"ticks over {trial_params.deadline_ms:.1f}ms: {misses.n_misses}")
    print(
        f"seeds with a miss ({len(seeds)}): {list(seeds[:20])}"
        f"{'...' if len(seeds) > 20 else ''}"
    )
    if misses.mean_points_at_miss is not None:
        print(f"mean obstacle points at a miss: {misses.mean_points_at_miss:.1f}")
    print(f"mean obstacle points overall: {misses.mean_points_overall:.1f}")
    print(f"Pearson r(point count, tick time): {misses.pearson_r_points_vs_ms:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1080, help="LiDAR beam count")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed0", type=int, default=0, help="seed of the first trial")
    parser.add_argument(
        "--out",
        default="loop_robustness_results.pkl",
        help="pickle file of the results",
    )
    args = parser.parse_args()

    scene = load_track_scene("loop_corridor.xml")
    lidar = DenseLidarParams(n_beams=args.n)
    trial_params = TRIALS["loop_corridor.xml"]
    results: list[TrialResult] = []
    for seed in range(args.seed0, args.seed0 + args.trials):
        r = run_trial(scene, lidar, seed, trial_params)
        results.append(r)
        print(
            f"seed {seed:3d}: {r.outcome:9s} lap={r.fraction:5.2f} "
            f"contact_evt={r.contact_events} no_path_evt={len(r.no_path_events)} "
            f"longest_np={r.longest_no_path_ticks}",
            flush=True,
        )

    with open(args.out, "wb") as f:
        pickle.dump(results, f)
    print(f"\nsaved the raw results in {args.out}")
    report(results, trial_params, ClassifyParams())


if __name__ == "__main__":
    main()
