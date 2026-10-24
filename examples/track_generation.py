"""
Generate procedural racetracks and drive one.

Usage:
    python3 -m examples.track_generation                 # describe a few tracks
    python3 -m examples.track_generation --save out.xml  # write one as MJCF
    mjpython -m examples.track_generation --drive        # drive it in the viewer
    mjpython -m examples.track_generation --drive --seed 7 --difficulty 3

generate_track makes the track geometry. assets.compose attaches the car to it.
The rest of this script prints reports and runs a control loop.

On macOS, --drive needs mjpython, not plain python3, because the passive
viewer must use the main thread. The default headless mode runs with either.

Frame: +X forward, +Y left, +Z up.
"""

import argparse
import time
from collections.abc import Mapping
from types import MappingProxyType
from typing import assert_never

import mujoco
import mujoco.viewer
import numpy as np

from neoracer_mujoco import assets, collision, sensors, sim
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

DIFFICULTIES: Mapping[str, Difficulty] = MappingProxyType(
    {"0": 0, "1": 1, "2": 2, "3": 3}
)

DRIVE_CONFIG = wall_following.WallFollowingConfig(
    follow_side="right", target_distance=0.25
)
"""The controller follows the right-hand wall at roughly a quarter of the
corridor width, so the car has room on both sides even on the narrowest
stretches (half_width_min is 0.35 m at the hardest difficulty)."""


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
    model = assets.compose(to_spec(track)).model
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
    scene = assets.compose(to_spec(track))
    model, car = scene.model, scene.car
    data = mujoco.MjData(model)
    dt = model.opt.timestep
    print(
        f"seed {seed}, difficulty {difficulty}: {track.total_length:.1f} m loop, "
        f"{model.ngeom} geoms. Close the viewer window to exit."
    )

    state = PDState()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = model.bind_scalar(car.body).id
        viewer.cam.distance = 2.5  # m from the car
        viewer.cam.elevation = -35  # deg; negative = the camera looks down

        while viewer.is_running():
            step_start = time.time()
            readings = sensors.read(car.sensors, data)
            state, command = wall_following.step(DRIVE_CONFIG, state, readings, dt)
            sim.apply_command(car, data, command)
            mujoco.mj_step(model, data)

            # Touching a wall resets the car to its spawn pose. Clear the
            # controller state too, or its D-term carries across the jump.
            if collision.car_touches_wall(data, car):
                collision.reset_car(model, data)
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

    # One seed at each difficulty. A harder difficulty allows tighter corners
    # and a narrower corridor. It can also reject other attempts, so the
    # loop shape can differ between difficulties.
    print(f"one seed across the difficulty range (seed {args.seed}):")
    for d in DIFFICULTIES.values():
        describe(args.seed, d)

    print(f"\nfour seeds at difficulty {difficulty}:")
    for seed in range(args.seed, args.seed + 4):
        describe(seed, difficulty)


if __name__ == "__main__":
    main()
