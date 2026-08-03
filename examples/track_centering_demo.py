"""
Demonstration/validation run of TrackCenteringController — headless by
default, or with an optional interactive MuJoCo viewer.

Composes neoracer.xml onto assets/tracks/straight_corridor.xml at runtime
(same MjSpec.attach_body pattern as manual_drive.py's load_model — neither
XML file names or depends on the other), settles the car, then drives the
controller for a fixed number of steps and reports centering performance.

No walled corridor with a "true" centerline exists elsewhere in the repo, so
the metric used here is a PROXY: how close the car stays to lateral_error==0
(balanced left/right LiDAR clearance, see track_centering.py's docstring for
the exact sign convention), not literal distance from a labeled center point.

Viewer code (run_viewer, _side_clearance, _print_status) is isolated from
the controller: it only calls the same public compute_signals()/compute()
API the headless run() uses, and never touches track_centering.py.

Usage:
    python3 -m examples.track_centering_demo              # headless (default)
    mjpython -m examples.track_centering_demo --viewer     # interactive viewer

NOTE: on macOS, the passive viewer requires mjpython, not plain python3 (same
constraint as examples/run.py) — mjpython runs the script on a secondary
thread, keeping the main thread free for the viewer's Cocoa event loop.
"""

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from neoracer_mujoco import sensors as sl
from neoracer_mujoco.control.track_centering import (
    LEFT_BEAMS,
    RIGHT_BEAMS,
    TrackCenteringConfig,
    TrackCenteringController,
    compute_signals,
)

_PROJECT_DIR = Path(__file__).resolve().parent.parent

STEPS = 3000
SETTLE_STEPS = 400
# Deliberate initial lateral offset (m), applied AFTER settling, toward the
# left wall (corridor half-width is 0.5 m) — without this the car spawns
# already centered in this symmetric corridor and the run would never
# exercise the controller's correction behavior, only its steady-state one.
INITIAL_Y_OFFSET = 0.25
VIEWER_PRINT_STEPS = 50  # ~0.1s at the model's 0.002s timestep
# Default --viewer session length. 6.0s matches STEPS*timestep (3000*0.002)
# in the headless run() below, and the car stays well inside the 16 m
# corridor over that window (measured: ~9-10 m traveled by t=6s, corridor
# spans x=-2..14) — the viewer would otherwise run indefinitely until the
# car eventually exits the far end and LiDAR has nothing left to measure.
DEFAULT_VIEWER_DURATION_S = 6.0


def load_model() -> mujoco.MjModel:
    scene = mujoco.MjSpec.from_file(
        str(_PROJECT_DIR / "assets" / "tracks" / "straight_corridor.xml")
    )
    car = mujoco.MjSpec.from_file(str(_PROJECT_DIR / "assets" / "neoracer.xml"))
    scene.worldbody.add_frame().attach_body(car.body("car"), "", "")
    return scene.compile()


def _side_clearance(
    sensors: sl.SensorReadings, beam_names, cfg: TrackCenteringConfig
) -> float:
    """Mean clearance across the given beams, for DISPLAY only — mirrors
    (does not modify) the controller's no-hit-as-open convention. The
    controller's own internal beam handling lives entirely in
    track_centering.py; this is a read-only echo of it for the overlay."""
    values = []
    for name in beam_names:
        raw = float(getattr(sensors, name)[0])
        values.append(cfg.max_beam_range if raw < 0 else min(raw, cfg.max_beam_range))
    return sum(values) / len(values)


def run() -> dict:
    model = load_model()
    data = mujoco.MjData(model)
    car_id = model.body("car").id
    dt = model.opt.timestep

    data.ctrl[:] = 0
    for _ in range(SETTLE_STEPS):
        mujoco.mj_step(model, data)

    data.qpos[1] += INITIAL_Y_OFFSET
    mujoco.mj_forward(model, data)

    cfg = TrackCenteringConfig()
    controller = TrackCenteringController(cfg)

    lateral_errors, steers, throttles, front_dists = [], [], [], []
    upright_all, finite_all = True, True

    for _ in range(STEPS):
        sensors = sl.read(model, data)
        signals = compute_signals(sensors, cfg)
        control = controller.compute(sensors, dt)

        data.ctrl[0:4] = control.throttle
        data.ctrl[4] = control.steer
        mujoco.mj_step(model, data)

        lateral_errors.append(signals.lateral_error)
        steers.append(control.steer)
        throttles.append(control.throttle)
        front_dists.append(signals.front_distance)

        upright_cos = float(data.body("car").xmat[8])
        if upright_cos < 0.7:
            upright_all = False
        if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
            finite_all = False

    lateral_errors = np.array(lateral_errors)
    steers = np.array(steers)
    throttles = np.array(throttles)
    front_dists = np.array(front_dists)
    quarter = len(lateral_errors) // 4

    return {
        "final_pos": data.xpos[car_id].copy(),
        "distance_traveled_x": float(data.xpos[car_id][0]),
        "initial_y_offset": INITIAL_Y_OFFSET,
        "mean_abs_lateral_error": float(np.mean(np.abs(lateral_errors))),
        "max_abs_lateral_error": float(np.max(np.abs(lateral_errors))),
        "first_quarter_mean_abs_lateral_error": float(
            np.mean(np.abs(lateral_errors[:quarter]))
        ),
        "last_quarter_mean_abs_lateral_error": float(
            np.mean(np.abs(lateral_errors[-quarter:]))
        ),
        "mean_abs_steer": float(np.mean(np.abs(steers))),
        "max_abs_steer": float(np.max(np.abs(steers))),
        "steer_within_limits": bool(np.all(np.abs(steers) <= cfg.steer_limit + 1e-9)),
        "mean_throttle": float(np.mean(throttles)),
        "min_front_distance": float(np.min(front_dists)),
        "upright_throughout": upright_all,
        "finite_throughout": finite_all,
    }


def _print_status(
    data: mujoco.MjData, control, signals, left: float, right: float
) -> None:
    """One diagnostic line, printed periodically in viewer mode (mirrors
    run.py's _log pattern)."""
    print(
        f"t={data.time:6.2f}s "
        f"steer={control.steer:+.3f} thr={control.throttle:+.3f} "
        f"lat_err={signals.lateral_error:+.3f} "
        f"L={left:.3f} R={right:.3f} fwd={signals.front_distance:.3f}"
    )


def run_viewer(duration: float = DEFAULT_VIEWER_DURATION_S) -> None:
    """Same settle -> offset -> controller loop as run(), but synced to an
    interactive mujoco.viewer.launch_passive window instead of collecting
    metrics headlessly. Requires mjpython on macOS (see module docstring).

    Exits when EITHER `duration` seconds of simulated time have elapsed OR
    the viewer window is closed, whichever comes first. Pass duration<=0
    for no automatic limit (run until the window is closed)."""
    model = load_model()
    data = mujoco.MjData(model)
    car_id = model.body("car").id
    dt = model.opt.timestep
    max_steps = int(duration / dt) if duration and duration > 0 else None

    data.ctrl[:] = 0
    for _ in range(SETTLE_STEPS):
        mujoco.mj_step(model, data)

    data.qpos[1] += INITIAL_Y_OFFSET
    mujoco.mj_forward(model, data)

    cfg = TrackCenteringConfig()
    controller = TrackCenteringController(cfg)

    if max_steps is not None:
        print(
            f"NeoRacer track-centering demo — viewer mode, stops automatically "
            f"after {duration:.1f}s (or close the window to exit early)."
        )
    else:
        print(
            "NeoRacer track-centering demo — viewer mode. Close the viewer window to exit."
        )

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = car_id
        viewer.cam.distance = 1.6
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -30

        step = 0
        while viewer.is_running():
            if max_steps is not None and step >= max_steps:
                print(f"Reached the {duration:.1f}s duration limit — stopping.")
                break

            loop_start = time.time()

            sensors = sl.read(model, data)
            signals = compute_signals(sensors, cfg)
            control = controller.compute(sensors, dt)

            data.ctrl[0:4] = control.throttle
            data.ctrl[4] = control.steer
            mujoco.mj_step(model, data)
            viewer.sync()

            step += 1
            if step % VIEWER_PRINT_STEPS == 0:
                left = _side_clearance(sensors, LEFT_BEAMS, cfg)
                right = _side_clearance(sensors, RIGHT_BEAMS, cfg)
                _print_status(data, control, signals, left, right)

            # sleep only the time left in this timestep, so the sim tracks
            # real time instead of running faster/slower than real-world pace.
            remaining = dt - (time.time() - loop_start)
            if remaining > 0:
                time.sleep(remaining)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="open an interactive MuJoCo passive viewer instead of running headless",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_VIEWER_DURATION_S,
        help=(
            "viewer mode only: seconds of simulated time before auto-stopping "
            f"(default {DEFAULT_VIEWER_DURATION_S:.1f}s; <=0 disables the limit). "
            "Ignored in headless mode, which always runs its fixed STEPS."
        ),
    )
    args = parser.parse_args()

    if args.viewer:
        run_viewer(duration=args.duration)
        return

    result = run()
    print(
        "NeoRacer track-centering demo — straight_corridor.xml, "
        f"{STEPS} steps after {SETTLE_STEPS}-step settle"
    )
    print()
    print(f"  initial lateral offset        : {result['initial_y_offset']:+.3f} m")
    print(f"  distance traveled (x)        : {result['distance_traveled_x']:+.3f} m")
    print(f"  mean |lateral_error|          : {result['mean_abs_lateral_error']:.3f} m")
    print(f"  max  |lateral_error|          : {result['max_abs_lateral_error']:.3f} m")
    print(
        f"  first-quarter mean |error|    : {result['first_quarter_mean_abs_lateral_error']:.3f} m"
    )
    print(
        f"  last-quarter  mean |error|    : {result['last_quarter_mean_abs_lateral_error']:.3f} m"
    )
    print(f"  mean |steer|                  : {result['mean_abs_steer']:.3f} rad")
    print(f"  max  |steer|                  : {result['max_abs_steer']:.3f} rad")
    print(f"  steer stayed within ±limits   : {result['steer_within_limits']}")
    print(f"  mean throttle                 : {result['mean_throttle']:+.3f} N·m")
    print(f"  min front clearance seen      : {result['min_front_distance']:.3f} m")
    print(f"  upright for entire run        : {result['upright_throughout']}")
    print(f"  all qpos/qvel finite          : {result['finite_throughout']}")
    print()
    print("  NOTE: lateral_error is a PROXY metric (balanced left/right LiDAR")
    print("  clearance), not distance from a labeled track centerline — no such")
    print("  centerline exists in this repo. Its MAGNITUDE is not directly")
    print("  comparable to the physical spawn offset in meters (diagonal beams")
    print(
        f"  amplify it ~1.41x) — the {INITIAL_Y_OFFSET:.2f} m offset above produces a"
    )
    print("  much larger initial lateral_error; track its trend, not its raw value.")
    print("  Results are for this specific 1.0 m-wide straight corridor only;")
    print("  do not extrapolate to curves.")


if __name__ == "__main__":
    main()
