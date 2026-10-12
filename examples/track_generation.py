"""
Generate procedural racetracks and drive one.

Usage:
    python3 -m examples.track_generation                 # describe a few tracks
    python3 -m examples.track_generation --save out.xml  # write one as MJCF
    mjpython -m examples.track_generation --drive        # drive it in the viewer
    mjpython -m examples.track_generation --drive --seed 7 --difficulty 3

Two calls are the whole API: generate_track gives you geometry, assets.compose
puts the car on it. Everything below is reporting and a control loop around
those.

NOTE: --drive needs mjpython on macOS, not plain python3 (the passive viewer
must own the main thread). The default headless mode runs under either.
"""

import argparse
import time
from typing import assert_never

import mujoco
import mujoco.viewer
import numpy as np

from neoracer_mujoco import assets, sensors
from neoracer_mujoco.collision import reset_if_wall_hit
from neoracer_mujoco.contract import DRIVE, STEER
from neoracer_mujoco.control import wall_following
from neoracer_mujoco.control.steering import PDState
from neoracer_mujoco.track_generation import (
    Difficulty,
    NoValidTrack,
    Track,
    generate_track,
    settings_for_difficulty,
    to_mjcf,
    to_spec,
)

DIFFICULTIES: frozendict[str, Difficulty] = frozendict({"0": 0, "1": 1, "2": 2, "3": 3})

# Follow the right-hand wall at roughly a quarter of the corridor width, so the
# car has room on both sides even on the narrowest stretches (half_width_min is
# 0.35 m at the hardest difficulty).
DRIVE_CONFIG = wall_following.WallFollowingConfig(
    follow_side="right", target_distance=0.25
)


def make_track(seed: int, difficulty: Difficulty) -> Track:
    """The generated track, or exit with a message if there is none."""
    match generate_track(seed, settings_for_difficulty(difficulty)):
        case Track() as track:
            return track
        case NoValidTrack(attempts=n):
            raise SystemExit(f"no valid track for seed {seed} after {n} attempts")
        case other:
            assert_never(other)


def describe(seed: int, difficulty: Difficulty) -> None:
    """Generate one track and print what came out."""
    track = make_track(seed, difficulty)
    model = assets.compose(to_spec(track))
    # Curvature is 1/m, so its reciprocal is the radius of the tightest corner.
    tightest_radius = 1.0 / np.abs(track.curvature).max()
    print(
        f"  seed {seed} difficulty {difficulty}: "
        f"{track.total_length:5.1f} m loop, "
        f"corridor {2 * track.half_width.min():.2f}-{2 * track.half_width.max():.2f} m, "
        f"tightest corner {tightest_radius:.2f} m radius, "
        f"{model.ngeom} geoms"
    )


def drive(seed: int, difficulty: Difficulty) -> None:
    """Put the car on a generated track and let the wall follower run it."""
    track = make_track(seed, difficulty)
    model = assets.compose(to_spec(track))
    data = mujoco.MjData(model)
    layout = sensors.sensor_layout(model)
    dt = model.opt.timestep
    print(
        f"seed {seed}, difficulty {difficulty}: {track.total_length:.1f} m loop, "
        f"{model.ngeom} geoms. Close the viewer window to exit."
    )

    state = PDState()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = model.body("car").id
        viewer.cam.distance = 2.5
        viewer.cam.elevation = -35

        while viewer.is_running():
            step_start = time.time()
            readings = sensors.read_layout(layout, data)
            state, command = wall_following.step(DRIVE_CONFIG, state, readings, dt)
            data.ctrl[DRIVE] = command.throttle
            data.ctrl[STEER] = command.steer
            mujoco.mj_step(model, data)

            # Touching a wall resets the car to its spawn pose. Clear the
            # controller state too, or its D-term carries across the jump.
            if reset_if_wall_hit(model, data):
                state = PDState()

            viewer.sync()
            time.sleep(max(0.0, dt - (time.time() - step_start)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--difficulty", default="1", choices=DIFFICULTIES)
    parser.add_argument("--drive", action="store_true", help="open the viewer")
    parser.add_argument("--save", metavar="PATH", help="write the track as MJCF")
    args = parser.parse_args()
    difficulty = DIFFICULTIES[args.difficulty]

    if args.drive:
        drive(args.seed, difficulty)
        return

    if args.save:
        track = make_track(args.seed, difficulty)
        with open(args.save, "w") as handle:
            handle.write(to_mjcf(track))
        print(f"wrote {args.save} ({track.total_length:.1f} m loop)")
        return

    # Same seed, four difficulties: the loop keeps its family resemblance while
    # the corners tighten and the corridor narrows.
    print(f"one seed across the difficulty range (seed {args.seed}):")
    for d in DIFFICULTIES.values():
        describe(args.seed, d)

    print(f"\nfour seeds at difficulty {difficulty}:")
    for seed in range(args.seed, args.seed + 4):
        describe(seed, difficulty)


if __name__ == "__main__":
    main()
