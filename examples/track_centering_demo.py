"""
Run the track-centering controller in a straight corridor and report the result.

By default the demo runs headless. With --viewer, it opens the MuJoCo viewer.

The demo puts the car (neoracer.xml) on assets/tracks/straight_corridor.xml with assets.compose.
The two XML files do not name each other.
The demo lets the car settle, moves it toward the left wall, and then runs the controller.

The repo has no corridor with a labeled centerline.
So the demo measures lateral_error, not the distance from a centerline.
lateral_error is 0 when the left and right LiDAR clearances are equal.
track_centering.py gives the sign convention.

The viewer code (run_viewer, _side_clearance, _print_status) uses only the
public compute_signals() and step() functions, the same as run().

Usage:
    python3 -m examples.track_centering_demo                       # headless
    mjpython -m examples.track_centering_demo --viewer             # viewer, 6 s
    mjpython -m examples.track_centering_demo --viewer --no-limit  # until you close it

On macOS, the passive viewer needs mjpython, not plain python3.
mjpython runs the script on a second thread, so the viewer can use the main thread.

Frame: +X forward, +Y left, +Z up.
"""

import argparse
import os
import time
from dataclasses import dataclass

import mujoco
import mujoco.viewer
import numpy as np

from neoracer_mujoco import assets, sensors, sim
from neoracer_mujoco.control import PDState, track_centering
from neoracer_mujoco.control.steering import ControlCommand
from neoracer_mujoco.control.track_centering import (
    CenteringSignals,
    TrackCenteringConfig,
    compute_signals,
)
from neoracer_mujoco.scene import Scene


@dataclass(frozen=True)
class DemoParams:
    steps: int = 3000  # physics steps of the headless run; 6 s at a 0.002 s timestep
    # m, + = toward the left wall. The demo moves the car by this offset after
    # the settle. The corridor half-width is 0.5 m. Without the offset, the car
    # starts at the center, and the run does not test the correction.
    initial_y_offset_m: float = 0.25
    # s of simulation time in viewer mode. It is the same as the headless run.
    # In 6 s the car travels about 9 to 10 m (measured). The corridor goes from
    # x = -2 m to x = 14 m, so the car stays in the corridor.
    viewer_duration_s: float = 6.0
    print_every_steps: int = 50  # physics steps between two status lines; 0.1 s
    # The car is upright while the cosine of its tilt is above this value.
    min_upright_cos: float = 0.7

    def __post_init__(self) -> None:
        # The report compares the first and the last quarter of the run, so
        # each quarter needs at least one step.
        if self.steps < 4:
            raise ValueError(f"steps must be >= 4, got {self.steps}")
        if self.print_every_steps < 1:
            raise ValueError(
                f"print_every_steps must be >= 1, got {self.print_every_steps}"
            )
        if not self.viewer_duration_s > 0:
            raise ValueError(
                f"viewer_duration_s must be > 0, got {self.viewer_duration_s}"
            )


@dataclass(frozen=True)
class CenteringReport:
    """The result of the headless run. Distances in m, steer in rad, throttle in N·m."""

    distance_traveled_x: float
    mean_abs_lateral_error: float
    max_abs_lateral_error: float
    first_quarter_mean_abs_lateral_error: float
    last_quarter_mean_abs_lateral_error: float
    mean_abs_steer: float
    max_abs_steer: float
    steer_within_limits: bool
    mean_throttle: float
    min_front_distance: float
    upright_throughout: bool
    finite_throughout: bool


def load_model() -> Scene:
    return assets.compose(
        mujoco.MjSpec.from_file(os.path.join(assets.TRACK_DIR, "straight_corridor.xml"))
    )


def _spawn(scene: Scene, params: DemoParams) -> mujoco.MjData:
    """Settle the car, then move it sideways by params.initial_y_offset_m."""
    data = sim.settle(scene.model)
    qpos = data.qpos[scene.car.qpos]  # a view: x, y, z, then the quaternion
    qpos[1] += params.initial_y_offset_m
    mujoco.mj_forward(scene.model, data)
    return data


def _side_clearance(
    readings: sensors.SensorReadings,
    beams: tuple[int, ...],
    cfg: TrackCenteringConfig,
) -> float:
    """The mean clearance (m) of the given beams, for the status line only.
    It uses the same beam_distance as the controller."""
    ranges = sensors.lidar_ranges(readings)
    values = [
        sensors.beam_distance(sensors.parse_beam(float(ranges[i])), cfg.max_beam_range)
        for i in beams
    ]
    return sum(values) / len(values)


def run(params: DemoParams) -> CenteringReport:
    """Run the controller headless for params.steps steps and measure the result."""
    scene = load_model()
    model, car = scene.model, scene.car
    data = _spawn(scene, params)
    car_id = car.body
    dt = model.opt.timestep

    cfg = TrackCenteringConfig()
    state = PDState()

    lateral_errors, steers, throttles, front_dists = [], [], [], []
    upright_all, finite_all = True, True

    for _ in range(params.steps):
        readings = sensors.read(car.sensors, data.sensordata)
        signals = compute_signals(readings, cfg)
        state, control = track_centering.step(cfg, state, readings, dt)

        sim.apply_command(car, data, control)
        mujoco.mj_step(model, data)

        lateral_errors.append(signals.lateral_error)
        steers.append(control.steer)
        throttles.append(control.throttle)
        front_dists.append(signals.front_distance)

        if sim.car_upright_cos(data, car_id) < params.min_upright_cos:
            upright_all = False
        if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
            finite_all = False

    lateral_errors = np.array(lateral_errors)
    steers = np.array(steers)
    throttles = np.array(throttles)
    front_dists = np.array(front_dists)
    quarter = len(lateral_errors) // 4

    return CenteringReport(
        distance_traveled_x=float(data.xpos[car_id][0]),
        mean_abs_lateral_error=float(np.mean(np.abs(lateral_errors))),
        max_abs_lateral_error=float(np.max(np.abs(lateral_errors))),
        first_quarter_mean_abs_lateral_error=float(
            np.mean(np.abs(lateral_errors[:quarter]))
        ),
        last_quarter_mean_abs_lateral_error=float(
            np.mean(np.abs(lateral_errors[-quarter:]))
        ),
        mean_abs_steer=float(np.mean(np.abs(steers))),
        max_abs_steer=float(np.max(np.abs(steers))),
        steer_within_limits=bool(
            np.all(np.abs(steers) <= cfg.shaping.steer_limit + 1e-9)
        ),
        mean_throttle=float(np.mean(throttles)),
        min_front_distance=float(np.min(front_dists)),
        upright_throughout=upright_all,
        finite_throughout=finite_all,
    )


def _print_status(
    data: mujoco.MjData,
    control: ControlCommand,
    signals: CenteringSignals,
    left: float,
    right: float,
) -> None:
    """Print one status line in viewer mode."""
    print(
        f"t={data.time:6.2f}s "
        f"steer={control.steer:+.3f} thr={control.throttle:+.3f} "
        f"lat_err={signals.lateral_error:+.3f} "
        f"L={left:.3f} R={right:.3f} fwd={signals.front_distance:.3f}"
    )


def run_viewer(params: DemoParams, duration_s: float | None) -> None:
    """Run the same loop as run() in the passive viewer, in real time.

    The loop stops after duration_s seconds of simulation time, or when you
    close the window. If duration_s is None, only the window close stops it.
    On macOS, run it with mjpython (see the module docstring).
    """
    scene = load_model()
    model, car = scene.model, scene.car
    data = _spawn(scene, params)
    car_id = car.body
    dt = model.opt.timestep
    max_steps = None if duration_s is None else int(duration_s / dt)

    cfg = TrackCenteringConfig()
    state = PDState()

    if duration_s is None:
        print(
            "NeoRacer track-centering demo — viewer mode. Close the viewer window to exit."
        )
    else:
        print(
            f"NeoRacer track-centering demo — viewer mode, stops automatically "
            f"after {duration_s:.1f}s (or close the window to exit early)."
        )

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = car_id
        viewer.cam.distance = 1.6  # m from the car
        viewer.cam.azimuth = 90  # deg
        viewer.cam.elevation = -30  # deg; negative = the camera looks down

        step = 0
        while viewer.is_running():
            if max_steps is not None and step >= max_steps:
                print(f"Reached the {duration_s:.1f}s duration limit — stopping.")
                break

            loop_start = time.time()

            readings = sensors.read(car.sensors, data.sensordata)
            signals = compute_signals(readings, cfg)
            state, control = track_centering.step(cfg, state, readings, dt)

            sim.apply_command(car, data, control)
            mujoco.mj_step(model, data)
            viewer.sync()

            step += 1
            if step % params.print_every_steps == 0:
                left = _side_clearance(readings, sensors.LEFT_BEAMS, cfg)
                right = _side_clearance(readings, sensors.RIGHT_BEAMS, cfg)
                _print_status(data, control, signals, left, right)

            # Sleep only for the rest of this timestep, so the simulation
            # keeps pace with real time.
            remaining = dt - (time.time() - loop_start)
            if remaining > 0:
                time.sleep(remaining)


def main() -> None:
    params = DemoParams()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="open an interactive MuJoCo passive viewer instead of running headless",
    )
    limit = parser.add_mutually_exclusive_group()
    limit.add_argument(
        "--duration",
        type=float,
        default=params.viewer_duration_s,
        help=(
            "viewer mode only: seconds of simulated time before auto-stopping "
            f"(default {params.viewer_duration_s:.1f}s). "
            "Ignored in headless mode, which always runs its fixed steps."
        ),
    )
    limit.add_argument(
        "--no-limit",
        action="store_true",
        help="viewer mode only: run until you close the window",
    )
    args = parser.parse_args()

    if args.viewer:
        run_viewer(params, None if args.no_limit else args.duration)
        return

    report = run(params)
    print(
        "NeoRacer track-centering demo — straight_corridor.xml, "
        f"{params.steps} steps after {sim.SETTLE_STEPS}-step settle"
    )
    print()
    print(f"  initial lateral offset        : {params.initial_y_offset_m:+.3f} m")
    print(f"  distance traveled (x)        : {report.distance_traveled_x:+.3f} m")
    print(f"  mean |lateral_error|          : {report.mean_abs_lateral_error:.3f} m")
    print(f"  max  |lateral_error|          : {report.max_abs_lateral_error:.3f} m")
    print(
        f"  first-quarter mean |error|    : {report.first_quarter_mean_abs_lateral_error:.3f} m"
    )
    print(
        f"  last-quarter  mean |error|    : {report.last_quarter_mean_abs_lateral_error:.3f} m"
    )
    print(f"  mean |steer|                  : {report.mean_abs_steer:.3f} rad")
    print(f"  max  |steer|                  : {report.max_abs_steer:.3f} rad")
    print(f"  steer stayed within ±limits   : {report.steer_within_limits}")
    print(f"  mean throttle                 : {report.mean_throttle:+.3f} N·m")
    print(f"  min front clearance seen      : {report.min_front_distance:.3f} m")
    print(f"  upright for entire run        : {report.upright_throughout}")
    print(f"  all qpos/qvel finite          : {report.finite_throughout}")
    print()
    print("  NOTE: lateral_error is a PROXY metric (balanced left/right LiDAR")
    print("  clearance), not distance from a labeled track centerline — no such")
    print("  centerline exists in this repo. Its MAGNITUDE is not directly")
    print("  comparable to the physical spawn offset in meters (diagonal beams")
    print(
        f"  amplify it ~1.41x) — the {params.initial_y_offset_m:.2f} m offset above produces a"
    )
    print("  much larger initial lateral_error; track its trend, not its raw value.")
    print("  Results are for this specific 1.0 m-wide straight corridor only;")
    print("  do not extrapolate to curves.")


if __name__ == "__main__":
    main()
