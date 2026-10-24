"""
Closed-loop trials of the vehicle-space wall follower. Each trial perturbs
the spawn pose with its own seed and drives until the car reaches the
track's goal: 10 m along +X on assets/tracks/straight_corridor.xml, or one
lap of assets/tracks/loop_corridor.xml, a ring track with chamfered corners.
The trials are not part of the test suite, which must stay fast and
deterministic (AGENTS.md). Run them by hand from the repository root:

    python -m examples.wall_follower_vs.trials --track loop_corridor.xml --n 1080 --trials 1 --verbose
    python -m examples.wall_follower_vs.trials --track straight_corridor.xml --n 1080 --trials 10

Frame: +X forward, +Y left, +Z up. A lap is measured by the angle of the car's
position about the world origin, + = counterclockwise.
"""

import argparse
import itertools
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import assert_never, get_args

import mujoco
import numpy as np

from examples.wall_follower_vs.dense_lidar import (
    DenseLidarParams,
    LidarMount,
    Scan,
    cast_dense_lidar,
    lidar_mount,
)
from examples.wall_follower_vs.report_utils import Outcome, outcome_header
from examples.wall_follower_vs.scene_utils import (
    TrackName,
    load_track_scene,
    perturb_spawn,
)
from examples.wall_follower_vs.vehicle_space_controller import (
    Driving,
    NoPath,
    VsParams,
    VsState,
    car_geometry,
    label,
    obstacle_points,
    step,
    turn_radius_m,
)
from neoracer_mujoco import collision, sim
from neoracer_mujoco.scene import Car, Scene

# --- the control tick, which watch_loop.py shares -----------------------------


@dataclass(frozen=True)
class Driver:
    """The wall follower on a scene: its LiDAR, its params, and the physics
    steps of one control period."""

    scene: Scene
    mount: LidarMount
    lidar: DenseLidarParams
    params: VsParams
    dt: float  # s, one control period
    steps: int  # physics steps in a control period


def driver(scene: Scene, lidar: DenseLidarParams, control_hz: float) -> Driver:
    dt = 1.0 / control_hz
    return Driver(
        scene=scene,
        mount=lidar_mount(scene),
        lidar=lidar,
        params=VsParams(car=car_geometry(scene)),
        dt=dt,
        steps=max(1, round(dt / float(scene.model.opt.timestep))),
    )


@dataclass(frozen=True)
class Tick:
    """One control period: the scan, the controller's state and result, the
    physics steps in which the car touched a wall, and the wall-clock time
    of the ray cast and of step()."""

    scan: Scan
    state: VsState
    result: Driving | NoPath
    contact_steps: int
    raycast_ms: float
    compute_ms: float


def tick(d: Driver, data: mujoco.MjData, state: VsState) -> Tick:
    """Scan, run the controller, and step the physics for one control
    period with its command."""
    model, car = d.scene.model, d.scene.car
    t0 = time.perf_counter()
    scan = cast_dense_lidar(model, data, d.mount, d.lidar)
    t1 = time.perf_counter()
    state, result = step(d.params, state, *scan, d.dt)
    t2 = time.perf_counter()
    sim.apply_command(car, data, result.command)
    contact_steps = 0
    for _ in range(d.steps):
        mujoco.mj_step(model, data)
        if collision.car_touches_wall(data, car):
            contact_steps += 1
    return Tick(
        scan=scan,
        state=state,
        result=result,
        contact_steps=contact_steps,
        raycast_ms=(t1 - t0) * 1000.0,
        compute_ms=(t2 - t1) * 1000.0,
    )


# --- the trials ---------------------------------------------------------------


@dataclass(frozen=True)
class AlongX:
    """Drive success_m along +X."""

    success_m: float  # m


@dataclass(frozen=True)
class Lap:
    """Drive one lap about the world origin, either way."""

    target_rad: float = 2.0 * math.pi  # rad


type Goal = AlongX | Lap


@dataclass(frozen=True)
class TrialParams:
    """Settings of one trial and of the timing report. Progress is in the
    goal's unit: m along +X, or rad about the origin."""

    goal: Goal
    max_sim_time_s: float  # s
    stuck_window_s: float  # s
    stuck_progress: float  # minimum progress in stuck_window_s
    control_hz: float = 30.0  # Hz; controller rate
    upright_rollover_cos: float = 0.5  # cos(tilt); below this the car has rolled over
    # s; a contact in this time after the start of a NoPath event counts as
    # caused by that event
    contact_after_no_path_s: float = 1.0
    slow_tick_ms: float = 50.0  # ms; a tick time that the timing report counts
    very_slow_tick_ms: float = 100.0  # ms; a tick time that the timing report counts

    @property
    def deadline_ms(self) -> float:
        """The time (ms) of one control period."""
        return 1000.0 / self.control_hz


# Each track's goal and limits.
TRIALS: Mapping[TrackName, TrialParams] = MappingProxyType(
    {
        "straight_corridor.xml": TrialParams(
            # Near the far end of the corridor, which runs from x = -2 m to 12 m.
            goal=AlongX(success_m=10.0),
            max_sim_time_s=35.0,
            stuck_window_s=3.0,
            stuck_progress=0.05,  # m
        ),
        "loop_corridor.xml": TrialParams(
            goal=Lap(),
            # One lap of the 23 m centerline at 0.3 to 1.0 m/s, with slower corners.
            max_sim_time_s=60.0,
            stuck_window_s=4.0,
            stuck_progress=0.05,  # rad
        ),
    }
)


def _position(goal: Goal, data: mujoco.MjData, car: Car) -> float:
    """Where the car is toward goal: its x (m), or the angle (rad) of its
    position about the world origin."""
    xpos = data.bind_scalar(car.body).xpos
    match goal:
        case AlongX():
            return float(xpos[0])
        case Lap():
            return float(np.arctan2(xpos[1], xpos[0]))
        case _:
            assert_never(goal)


def _advance(goal: Goal, before: float, after: float) -> float:
    """The progress from position before to after; about the origin, the
    short way round."""
    match goal:
        case AlongX():
            return after - before
        case Lap():
            return math.remainder(after - before, 2.0 * math.pi)
        case _:
            assert_never(goal)


def _forward(goal: Goal, progress: float) -> float:
    """How far progress takes the car toward goal: along +X, or about the
    origin either way."""
    match goal:
        case AlongX():
            return progress
        case Lap():
            return abs(progress)
        case _:
            assert_never(goal)


def _target(goal: Goal) -> float:
    match goal:
        case AlongX(success_m=m):
            return m
        case Lap(target_rad=rad):
            return rad
        case _:
            assert_never(goal)


def _progress_text(goal: Goal, progress: float) -> str:
    match goal:
        case AlongX():
            return f"dx={progress:+6.2f}m"
        case Lap():
            return f"lap_angle={math.degrees(progress):+7.1f}deg"
        case _:
            assert_never(goal)


def _car_yaw_rad(data: mujoco.MjData, car: Car) -> float:
    """Yaw (rad) of the car body about world +Z."""
    xmat = data.bind_scalar(car.body).xmat.reshape(3, 3)
    return float(np.arctan2(xmat[1, 0], xmat[0, 0]))


@dataclass(frozen=True)
class NoPathEvent:
    """A run of adjacent ticks with a NoPath result."""

    t_s: float  # s; sim time at the end of the first tick of the run
    x_m: float  # m; world x of the car at that time
    y_m: float  # m; world y of the car at that time
    duration_ticks: int


@dataclass(frozen=True)
class TrialResult:
    seed: int
    outcome: Outcome
    sim_time_s: float  # s
    fraction: float  # of the goal reached
    contact_steps: int  # physics steps in which the car touches a wall
    contact_events: int  # runs of adjacent ticks with a contact
    # m; the smallest valid LiDAR range in the trial; inf if no beam had a return
    min_sensed_range_m: float
    # m; the smallest safe arc length of the 0 deg candidate. This value is a
    # controller output, not a measured distance.
    min_straight_clear_m: float
    no_path_events: tuple[NoPathEvent, ...]
    no_path_caused_contact: bool  # a contact came soon after a NoPath start
    steer_std_rad: float  # rad; standard deviation of the steer command
    max_target_jump_deg: float  # deg; largest change of the target in one tick
    region_switches: int  # ticks at which the start of the selected region changed
    # Wall-clock time of the ray cast and of step() at each tick.
    raycast_times_ms: tuple[float, ...]
    compute_times_ms: tuple[float, ...]
    # Obstacle points at each tick (see vehicle_space_controller.obstacle_points).
    obstacle_point_counts: tuple[int, ...]

    @property
    def no_path_ticks(self) -> int:
        return sum(e.duration_ticks for e in self.no_path_events)

    @property
    def longest_no_path_ticks(self) -> int:
        return max((e.duration_ticks for e in self.no_path_events), default=0)

    @property
    def tick_times_ms(self) -> np.ndarray:
        """Ray cast time plus step() time at each tick. Both must fit in one
        control period."""
        return np.asarray(self.raycast_times_ms) + np.asarray(self.compute_times_ms)


def _no_path_events(
    no_path: Sequence[bool], xy_m: Sequence[tuple[float, float]], dt: float
) -> tuple[NoPathEvent, ...]:
    """The runs of True in no_path, with the time and position of the start
    of each run."""
    events = []
    for is_no_path, run in itertools.groupby(range(len(no_path)), no_path.__getitem__):
        if is_no_path:
            ticks = list(run)
            x_m, y_m = xy_m[ticks[0]]
            events.append(
                NoPathEvent(
                    t_s=(ticks[0] + 1) * dt, x_m=x_m, y_m=y_m, duration_ticks=len(ticks)
                )
            )
    return tuple(events)


def _starts(flags: Sequence[bool]) -> list[int]:
    """The indices at which a run of True begins."""
    return [i for i, f in enumerate(flags) if f and (i == 0 or not flags[i - 1])]


def run_trial(
    scene: Scene,
    lidar: DenseLidarParams,
    seed: int,
    params: TrialParams,
    verbose: bool = False,
    log_every: int = 1,
) -> TrialResult:
    """Drive one trial from a spawn pose that rng(seed) perturbs. If verbose
    is true, print one line every log_every ticks."""
    model, car = scene.model, scene.car
    goal = params.goal
    data = sim.settle(model)
    perturb_spawn(scene, data, np.random.default_rng(seed))

    d = driver(scene, lidar, params.control_hz)
    max_ticks = int(params.max_sim_time_s / d.dt)

    state = VsState()
    position = _position(goal, data, car)
    progress = 0.0
    history: list[tuple[float, float]] = [(0.0, 0.0)]  # (sim time, progress)
    contact_steps = 0
    contact: list[bool] = []  # one value per tick
    no_path: list[bool] = []  # one value per tick
    xy_m: list[tuple[float, float]] = []  # car position after each tick
    steer_cmds: list[float] = []
    targets: list[float] = []
    region_switches = 0
    prev_region_key: int | None = None
    min_sensed_range_m = math.inf
    min_straight_clear_m = math.inf
    raycast_times_ms: list[float] = []
    compute_times_ms: list[float] = []
    obstacle_point_counts: list[int] = []
    prev_label = "driving"
    outcome: Outcome = "timeout"
    sim_time_s = 0.0

    for i in range(max_ticks):
        t = tick(d, data, state)
        state, result = t.state, t.result
        raycast_times_ms.append(t.raycast_ms)
        compute_times_ms.append(t.compute_ms)
        # step() computes the same points. This second call only counts them.
        obstacle_point_counts.append(len(obstacle_points(d.params, *t.scan)[0]))
        ranges = t.scan[1]
        valid_ranges = ranges[ranges > 0]
        if len(valid_ranges) > 0:
            min_sensed_range_m = min(min_sensed_range_m, float(valid_ranges.min()))
        contact_steps += t.contact_steps
        contact.append(t.contact_steps > 0)

        sim_time_s = (i + 1) * d.dt
        xpos = data.bind_scalar(car.body).xpos
        x_m, y_m = float(xpos[0]), float(xpos[1])
        xy_m.append((x_m, y_m))
        min_straight_clear_m = min(min_straight_clear_m, result.straight_clear_m)
        steer_cmds.append(result.command.steer)
        targets.append(result.target_steer_deg)
        match result:
            case Driving(region=region):
                no_path.append(False)
                region_key = round(region.start_deg)
            case NoPath():
                no_path.append(True)
                region_key = None
            case _:
                assert_never(result)
        if region_key != prev_region_key and prev_region_key is not None:
            region_switches += 1
        prev_region_key = region_key

        now = _position(goal, data, car)
        progress += _advance(goal, position, now)
        position = now
        history.append((sim_time_s, progress))

        if verbose and i % log_every == 0:
            match result:
                case Driving(region=region, path_clear_m=path_clear_m):
                    detail = (
                        f"region=[{region.start_deg:.0f},{region.end_deg:.0f}] "
                        f"path_clear={path_clear_m:.2f}m"
                    )
                case NoPath():
                    detail = "region=None"
                case _:
                    assert_never(result)
            print(
                f"t={sim_time_s:5.2f}s pos=({x_m:+.2f},{y_m:+.2f}) "
                f"yaw={math.degrees(_car_yaw_rad(data, car)):+6.1f}deg "
                f"target={result.target_steer_deg:+6.2f}deg "
                f"R={turn_radius_m(d.params, result.target_steer_deg):5.2f}m "
                f"straight_clear={result.straight_clear_m:.2f}m {detail} "
                f"steer={result.command.steer:+.3f}rad "
                f"throttle={result.command.throttle:.4f} "
                f"{_progress_text(goal, progress)}"
            )
        if verbose and label(result) != prev_label:
            print(f"  -- {prev_label} -> {label(result)} at t={sim_time_s:.2f}s")
        prev_label = label(result)

        upright = sim.car_upright_cos(data, car)
        if upright < params.upright_rollover_cos:
            outcome = "rollover"
            if verbose:
                print(f"  ROLLOVER at t={sim_time_s:.2f}s upright_cos={upright:.3f}")
            break
        if _forward(goal, progress) >= _target(goal):
            outcome = "success"
            break
        if sim_time_s >= params.stuck_window_s:
            cutoff = sim_time_s - params.stuck_window_s
            past = next((p for pt, p in history if pt >= cutoff), history[0][1])
            if _forward(goal, progress - past) < params.stuck_progress:
                outcome = "stuck"
                if verbose:
                    print(f"  STUCK at t={sim_time_s:.2f}s")
                break

    window_ticks = round(params.contact_after_no_path_s * params.control_hz)
    return TrialResult(
        seed=seed,
        outcome=outcome,
        sim_time_s=sim_time_s,
        fraction=_forward(goal, progress) / _target(goal),
        contact_steps=contact_steps,
        contact_events=len(_starts(contact)),
        min_sensed_range_m=min_sensed_range_m,
        min_straight_clear_m=min_straight_clear_m,
        no_path_events=_no_path_events(no_path, xy_m, d.dt),
        no_path_caused_contact=any(
            any(contact[s : s + window_ticks]) for s in _starts(no_path)
        ),
        steer_std_rad=float(np.std(steer_cmds)),
        max_target_jump_deg=(
            float(np.max(np.abs(np.diff(targets)))) if len(targets) > 1 else 0.0
        ),
        region_switches=region_switches,
        raycast_times_ms=tuple(raycast_times_ms),
        compute_times_ms=tuple(compute_times_ms),
        obstacle_point_counts=tuple(obstacle_point_counts),
    )


def summarize(results: Sequence[TrialResult]) -> None:
    print(outcome_header([r.outcome for r in results]))
    for r in results:
        tick_ms = r.tick_times_ms
        print(
            f"  seed {r.seed}: {r.outcome:9s} goal={r.fraction:5.2f} "
            f"t={r.sim_time_s:5.1f}s "
            f"contact_evt={r.contact_events:2d} contact_steps={r.contact_steps:4d} "
            f"min_sensed={r.min_sensed_range_m:5.2f}m "
            f"min_straight_clear={r.min_straight_clear_m:5.2f}m "
            f"no_path_evt={len(r.no_path_events):2d} no_path_ticks={r.no_path_ticks:4d} "
            f"longest_np={r.longest_no_path_ticks:3d} "
            f"steer_std={r.steer_std_rad:.3f}rad "
            f"max_jump={r.max_target_jump_deg:5.2f}deg "
            f"switches={r.region_switches:3d} "
            f"tick={tick_ms.mean():.2f}/{tick_ms.max():.2f}ms"
        )
        for e in r.no_path_events:
            print(f"      NoPath at t={e.t_s:.2f}s pos=({e.x_m:+.2f},{e.y_m:+.2f})")


@dataclass(frozen=True)
class TickTimeStats:
    """Distribution of the tick times (ms) of all trials."""

    n_ticks: int
    mean_ms: float
    median_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    p99_9_ms: float
    max_ms: float
    pct_over_deadline: float  # % of ticks over TrialParams.deadline_ms
    pct_over_slow: float  # % of ticks over TrialParams.slow_tick_ms
    pct_over_very_slow: float  # % of ticks over TrialParams.very_slow_tick_ms


def tick_time_stats(
    results: Sequence[TrialResult], params: TrialParams
) -> TickTimeStats:
    all_ms = np.concatenate([r.tick_times_ms for r in results])
    return TickTimeStats(
        n_ticks=len(all_ms),
        mean_ms=float(np.mean(all_ms)),
        median_ms=float(np.median(all_ms)),
        p90_ms=float(np.percentile(all_ms, 90)),
        p95_ms=float(np.percentile(all_ms, 95)),
        p99_ms=float(np.percentile(all_ms, 99)),
        p99_9_ms=float(np.percentile(all_ms, 99.9)),
        max_ms=float(np.max(all_ms)),
        pct_over_deadline=float(np.mean(all_ms > params.deadline_ms) * 100.0),
        pct_over_slow=float(np.mean(all_ms > params.slow_tick_ms) * 100.0),
        pct_over_very_slow=float(np.mean(all_ms > params.very_slow_tick_ms) * 100.0),
    )


@dataclass(frozen=True)
class DeadlineMisses:
    """The ticks over the deadline, and how they relate to the obstacle point
    count. A strong correlation shows that the algorithm causes the misses.
    A weak correlation shows that other load on the computer causes them."""

    n_misses: int
    seeds_with_misses: tuple[int, ...]
    mean_points_at_miss: float | None  # None if no tick missed the deadline
    mean_points_overall: float
    pearson_r_points_vs_ms: float


def deadline_misses(
    results: Sequence[TrialResult], params: TrialParams
) -> DeadlineMisses:
    points = np.concatenate([np.asarray(r.obstacle_point_counts) for r in results])
    ms = np.concatenate([r.tick_times_ms for r in results])
    seeds = np.concatenate([np.full(len(r.compute_times_ms), r.seed) for r in results])
    miss = ms > params.deadline_ms
    return DeadlineMisses(
        n_misses=int(miss.sum()),
        seeds_with_misses=tuple(sorted({int(s) for s in seeds[miss]})),
        mean_points_at_miss=float(points[miss].mean()) if miss.any() else None,
        mean_points_overall=float(points.mean()),
        pearson_r_points_vs_ms=float(np.corrcoef(points, ms)[0, 1]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", required=True, choices=get_args(TrackName.__value__))
    parser.add_argument("--n", type=int, required=True, help="LiDAR beam count")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed0", type=int, default=0, help="seed of the first trial")
    parser.add_argument("--verbose", action="store_true", help="print each tick")
    parser.add_argument("--log-every", type=int, default=1, help="ticks per line")
    args = parser.parse_args()

    scene = load_track_scene(args.track)
    lidar = DenseLidarParams(n_beams=args.n)
    params = TRIALS[args.track]
    results = [
        run_trial(scene, lidar, args.seed0 + i, params, args.verbose, args.log_every)
        for i in range(args.trials)
    ]
    summarize(results)
    stats = tick_time_stats(results, params)
    print(
        f"tick time (N={args.n}, {stats.n_ticks} ticks): mean={stats.mean_ms:.2f}ms "
        f"median={stats.median_ms:.2f}ms p95={stats.p95_ms:.2f}ms "
        f"p99={stats.p99_ms:.2f}ms max={stats.max_ms:.2f}ms "
        f"over_{params.deadline_ms:.1f}ms={stats.pct_over_deadline:.2f}%"
    )


if __name__ == "__main__":
    main()
