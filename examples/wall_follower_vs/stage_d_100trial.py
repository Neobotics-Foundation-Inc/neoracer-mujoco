"""
Stage D: 100-trial N=1080 robustness measurement for the (frozen) vehicle-
space wall follower on loop_corridor.xml. Measurement only -- does not call
into or modify vehicle_space_controller.py's decision logic; only times and
counts around the same public functions/entry points loop_trials.py already
uses for the 10-trial Stage B/C runs.

Usage:
    python3 examples/wall_follower_vs/stage_d_100trial.py
"""

import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loop_trials as lt  # noqa: E402

N = 1080
SEED0 = 0
N_TRIALS = 100
OUT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "stage_d_results.pkl"
)


def classify_failure(r: lt.LoopTrialResult) -> str:
    """A/B/C/D/E/F/G/H/I per Stage D's taxonomy, best-effort from the
    collected signals (no controller internals beyond what LoopTrialResult
    already records)."""
    if r.outcome == "rollover":
        return "G. rollover"
    if r.outcome == "success":
        return "none"
    # outcome == "stuck"
    if r.contact_ticks > 0:
        if r.no_path_events > 0 and r.longest_no_path_run >= 30:
            return "A. prolonged NO_PATH (with contact)"
        return "B. collision despite planner"
    if r.longest_no_path_run >= 30:  # ~1s+ of continuous NO_PATH
        return "A. prolonged NO_PATH"
    if r.region_switches > 100:
        return "C. target/trajectory switching"
    return "I. other (stuck, no contact, no prolonged NO_PATH)"


def main() -> None:
    model = lt.load_scene("loop_corridor.xml")
    results: list[lt.LoopTrialResult] = []
    for i in range(N_TRIALS):
        seed = SEED0 + i
        r = lt.run_loop_trial(model, N, seed)
        results.append(r)
        print(
            f"seed {seed:3d}: {r.outcome:9s} lap={r.lap_frac:5.2f} "
            f"contact_evt={r.contact_events} no_path_evt={r.no_path_events} "
            f"longest_np={r.longest_no_path_run}",
            flush=True,
        )

    with open(OUT_PATH, "wb") as f:
        pickle.dump(results, f)
    print(f"\nsaved raw results to {OUT_PATH}")

    # --- summary ---------------------------------------------------------
    n_success = sum(r.outcome == "success" for r in results)
    n_stuck = sum(r.outcome == "stuck" for r in results)
    n_rollover = sum(r.outcome == "rollover" for r in results)
    print(
        f"\n=== SUCCESS: {n_success}/{N_TRIALS}  stuck={n_stuck}  rollover={n_rollover} ==="
    )

    failed = [r for r in results if r.outcome != "success"]
    if failed:
        print("\n=== FAILURES ===")
        for r in failed:
            cls = classify_failure(r)
            print(
                f"  seed {r.seed}: {r.outcome} lap={r.lap_frac:.2f} class={cls} "
                f"contact_ticks={r.contact_ticks} no_path_ticks={r.no_path_ticks} "
                f"longest_np={r.longest_no_path_run}"
            )

    # --- contact analysis --------------------------------------------------
    with_contact = [r for r in results if r.contact_ticks > 0]
    print("\n=== CONTACTS ===")
    print(f"trials with any contact: {len(with_contact)}/{N_TRIALS}")
    print(f"total contact events: {sum(r.contact_events for r in results)}")
    print(f"total contact ticks: {sum(r.contact_ticks for r in results)}")
    if with_contact:
        worst = max(with_contact, key=lambda r: r.contact_ticks)
        print(
            f"worst contact trial: seed {worst.seed}, {worst.contact_ticks} ticks, outcome={worst.outcome}"
        )
    for r in with_contact:
        category = (
            "5. rollover"
            if r.outcome == "rollover"
            else "4. contact leading to stuck"
            if r.outcome == "stuck"
            else "3. sustained contact but lap completed"
            if r.contact_ticks > 30
            else "2. successful lap with brief contact"
        )
        print(
            f"  seed {r.seed}: {category} ({r.contact_ticks} ticks, {r.contact_events} events)"
        )

    # --- NO_PATH analysis ----------------------------------------------------
    with_no_path = [r for r in results if r.no_path_events > 0]
    all_durations = [d for r in results for d in r.no_path_durations_ticks]
    print("\n=== NO_PATH ===")
    print(
        f"trials with NO_PATH: {len(with_no_path)}/{N_TRIALS} ({100 * len(with_no_path) / N_TRIALS:.1f}%)"
    )
    print(f"total NO_PATH events: {sum(r.no_path_events for r in results)}")
    print(f"total NO_PATH ticks: {sum(r.no_path_ticks for r in results)}")
    if all_durations:
        arr = np.asarray(all_durations)
        print(
            f"event duration (ticks): median={np.median(arr):.1f} p95={np.percentile(arr, 95):.1f} max={arr.max()}"
        )
    print(
        f"max longest_no_path_run across all trials: {max((r.longest_no_path_run for r in results), default=0)}"
    )
    caused_contact = [r for r in results if r.no_path_caused_contact]
    print(
        f"trials where a NO_PATH event was followed by contact within ~1s: {len(caused_contact)}"
    )
    print(
        f"trials where NO_PATH correlates with failure (outcome != success): "
        f"{sum(1 for r in with_no_path if r.outcome != 'success')}/{len(with_no_path)}"
    )
    # location clustering: round position to 0.5m grid cells
    from collections import Counter

    cells = Counter()
    for r in results:
        for _, x, y in r.no_path_positions:
            cells[(round(x * 2) / 2, round(y * 2) / 2)] += 1
    print("NO_PATH location clusters (0.5m grid, top 8):")
    for (x, y), count in cells.most_common(8):
        print(f"  ({x:+.1f}, {y:+.1f}): {count}")

    # --- lap time / min-sensed-range distributions (successes only) ------
    successes = [r for r in results if r.outcome == "success"]
    print("\n=== ROBUSTNESS DISTRIBUTIONS (successful trials) ===")
    if successes:
        lap_times = np.asarray([r.sim_time_s for r in successes])
        print(
            f"lap time: mean={lap_times.mean():.2f}s median={np.median(lap_times):.2f}s "
            f"p5={np.percentile(lap_times, 5):.2f}s p95={np.percentile(lap_times, 95):.2f}s "
            f"min={lap_times.min():.2f}s max={lap_times.max():.2f}s"
        )
        min_ranges = np.asarray([r.min_sensed_range_m for r in successes])
        print(
            f"min sensed range: mean={min_ranges.mean():.3f}m median={np.median(min_ranges):.3f}m "
            f"p5={np.percentile(min_ranges, 5):.3f}m min={min_ranges.min():.3f}m"
        )
        events_per_trial = np.asarray([r.no_path_events for r in successes])
        print(
            f"NO_PATH events/trial: mean={events_per_trial.mean():.2f} "
            f"median={np.median(events_per_trial):.1f} max={events_per_trial.max()}"
        )

    # --- performance --------------------------------------------------------
    dist = lt.compute_distribution(results)
    print("\n=== PERFORMANCE (total = raycast + compute) ===")
    print(f"total ticks: {dist['n_ticks']}")
    print(
        f"mean={dist['mean']:.2f}ms median={dist['median']:.2f}ms p90={dist['p90']:.2f}ms "
        f"p95={dist['p95']:.2f}ms p99={dist['p99']:.2f}ms p99.9={dist['p99_9']:.2f}ms "
        f"max={dist['max']:.2f}ms"
    )
    print(
        f">33.3ms: {dist['pct_over_33_3ms']:.3f}%  >50ms: {dist['pct_over_50ms']:.3f}%  "
        f">100ms: {dist['pct_over_100ms']:.3f}%"
    )

    all_raycast = np.concatenate([np.asarray(r.raycast_times_ms) for r in results])
    all_compute = np.concatenate([np.asarray(r.compute_times_ms) for r in results])
    print(
        f"raycast-only: mean={all_raycast.mean():.2f}ms median={np.median(all_raycast):.2f}ms "
        f"max={all_raycast.max():.2f}ms"
    )
    print(
        f"compute-only: mean={all_compute.mean():.2f}ms median={np.median(all_compute):.2f}ms "
        f"max={all_compute.max():.2f}ms"
    )

    miss = lt.deadline_miss_analysis(results)
    print("\n=== DEADLINE MISS ANALYSIS ===")
    print(f"misses (>33.3ms): {miss['n_misses']}")
    print(
        f"misses occurred across {miss['miss_seed_count']} distinct seeds: {miss['seeds_with_misses'][:20]}"
        f"{'...' if len(miss['seeds_with_misses']) > 20 else ''}"
    )
    print(f"mean obstacle point count at miss ticks: {miss['mean_points_at_miss']:.1f}")
    print(f"mean obstacle point count overall: {miss['mean_points_overall']:.1f}")
    print(f"Pearson r(point count, tick time): {miss['pearson_r_points_vs_ms']:.3f}")


if __name__ == "__main__":
    main()
