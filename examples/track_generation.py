"""
Generate procedural racetracks and drive one.

Usage:
    python3 -m examples.track_generation                 # describe a few tracks
    python3 -m examples.track_generation --save out.xml  # write one as MJCF
    mjpython -m examples.track_generation --drive        # drive it in the viewer
    mjpython -m examples.track_generation --drive --seed 7 --difficulty 3

Two calls are the whole API: generate_track gives you geometry, compose puts
the car on it. Everything below is reporting and a control loop around those.

NOTE: --drive needs mjpython on macOS, not plain python3 (the passive viewer
must own the main thread). The default headless mode runs under either.
"""

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from neoracer_mujoco import compose, generate_track
from neoracer_mujoco import sensors as sl
from neoracer_mujoco.collision import reset_if_wall_hit
from neoracer_mujoco.control.wall_following import (
    WallFollowingConfig,
    WallFollowingController,
)
from neoracer_mujoco.track_generation import to_mjcf

# Follow the right-hand wall at roughly a quarter of the corridor width, so the
# car has room on both sides even on the narrowest stretches (half_width_min is
# 0.35 m at the hardest difficulty).
DRIVE_CONFIG = WallFollowingConfig(follow_side="right", target_distance=0.25)


def describe(seed: int, difficulty: int) -> None:
    """Generate one track and print what came out."""
    track = generate_track(seed, difficulty=difficulty)
    model = compose(track)
    # Curvature is 1/m, so its reciprocal is the radius of the tightest corner.
    tightest_radius = 1.0 / np.abs(track.curvature).max()
    print(
        f"  seed {seed} difficulty {difficulty}: "
        f"{track.total_length:5.1f} m loop, "
        f"corridor {2 * track.half_width.min():.2f}-{2 * track.half_width.max():.2f} m, "
        f"tightest corner {tightest_radius:.2f} m radius, "
        f"{model.ngeom} geoms"
    )


def drive(seed: int, difficulty: int) -> None:
    """Put the car on a generated track and let the wall-follower run it."""
    track = generate_track(seed, difficulty=difficulty)
    model = compose(track)
    data = mujoco.MjData(model)
    dt = model.opt.timestep
    print(
        f"seed {seed}, difficulty {difficulty}: {track.total_length:.1f} m loop, "
        f"{model.ngeom} geoms. Close the viewer window to exit."
    )

    controller = WallFollowingController(DRIVE_CONFIG)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = model.body("car").id
        viewer.cam.distance = 2.5
        viewer.cam.elevation = -35

        while viewer.is_running():
            step_start = time.time()

            control = controller.compute(sl.read(model, data), dt)
            data.ctrl[0:4] = control.throttle
            data.ctrl[4] = control.steer
            mujoco.mj_step(model, data)

            # Touching a wall resets the car to its spawn pose; clear the
            # controller's history too or its D-term carries across the jump.
            if reset_if_wall_hit(model, data):
                controller.reset()

            viewer.sync()
            time.sleep(max(0.0, dt - (time.time() - step_start)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--difficulty", type=int, default=1, choices=range(4))
    parser.add_argument("--drive", action="store_true", help="open the viewer")
    parser.add_argument("--save", metavar="PATH", help="write the track as MJCF")
    args = parser.parse_args()

    if args.drive:
        drive(args.seed, args.difficulty)
        return

    if args.save:
        track = generate_track(args.seed, difficulty=args.difficulty)
        with open(args.save, "w") as handle:
            handle.write(to_mjcf(track))
        print(f"wrote {args.save} ({track.total_length:.1f} m loop)")
        return

    # Same seed, four difficulties: the loop keeps its family resemblance while
    # the corners tighten and the corridor narrows.
    print(f"one seed across the difficulty range (seed {args.seed}):")
    for difficulty in range(4):
        describe(args.seed, difficulty)

    print(f"\nfour seeds at difficulty {args.difficulty}:")
    for seed in range(args.seed, args.seed + 4):
        describe(seed, args.difficulty)


if __name__ == "__main__":
    main()
