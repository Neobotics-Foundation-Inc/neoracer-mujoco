"""
Runtime bootstrap: run an unmodified racecar_core controller script (e.g.
neoracer-labs/ultimate-wall-follower/wall_follower.py) against NeoRacer
MuJoCo, outside pytest.

neoracer_mujoco.racecar_core_adapter.MujocoRacecar backs the same rc.drive /
rc.lidar / rc.controller / rc.display / rc.set_start_update / rc.go interface
every racecar_core controller is written against (see racecar_core_adapter.py
for the full contract and the sign/unit conventions it crosses). This script
is the process-level seam that lets a real controller *file* run against that
adapter -- validation/test_wall_follower_example.py proves the same claim
inside a pytest fixture; this is the same seam, runnable directly:

  1. Compose a MuJoCo scene the controller can actually drive in (the bare
     car has no walls for a wall-follower's LiDAR to see) and settle it.
  2. Construct a MujocoRacecar over that model/data.
  3. Inject a fake `racecar_core` module into sys.modules whose
     create_racecar() returns that MujocoRacecar -- the ONLY seam. The
     controller file itself runs unmodified via runpy.run_path().
  4. Put the controller's own directory on sys.path so its sibling imports
     (e.g. `import control`) resolve exactly as running it from that
     directory directly would resolve them.

Usage:
    python3 examples/run_racecar_core.py <path/to/controller.py> [--steps N]

Example:
    python3 examples/run_racecar_core.py examples/ultimate_wall_follower/wall_follower.py
"""

import argparse
import runpy
import sys
import types
from pathlib import Path

import mujoco

from neoracer_mujoco import sim as _sim
from neoracer_mujoco.racecar_core_adapter import MujocoRacecar

_PROJECT_DIR = Path(__file__).resolve().parent.parent

# 1.2s of sim time at the model's 0.002s timestep. straight_corridor.xml is
# only ~16m long and the vendored wall_follower.yaml drives at speed=1.0 (its
# own tuned config, left untouched per SOURCE.md), so a longer default run
# drives the car straight out the far end of the corridor and into
# uninteresting "no_data" LiDAR for the rest of the run. This default keeps
# the out-of-the-box demo inside the corridor, actually wall/gap-following.
# Pass --steps for a longer run (e.g. on a bigger track).
DEFAULT_STEPS = 600


def load_scene() -> mujoco.MjModel:
    """Car centered in a walled corridor (assets/tracks/straight_corridor.xml)
    so a wall-following controller's side LiDAR beams get real returns --
    the same composition validation/test_wall_follower_example.py uses."""
    scene = mujoco.MjSpec.from_file(
        str(_PROJECT_DIR / "assets" / "tracks" / "straight_corridor.xml")
    )
    car = mujoco.MjSpec.from_file(str(_PROJECT_DIR / "assets" / "neoracer.xml"))
    scene.worldbody.add_frame().attach_body(car.body("car"), "", "")
    return scene.compile()


def build_racecar(
    steps: int = DEFAULT_STEPS,
) -> tuple[mujoco.MjModel, mujoco.MjData, MujocoRacecar]:
    """Compile + settle the scene, then wrap it in a MujocoRacecar. RB is
    held by default (MujocoController's own default) so a deadman-gated
    controller -- e.g. the wall follower's RB check -- drives immediately in
    this headless runner, matching the pytest fixture's behavior."""
    model = load_scene()
    data = _sim.settle(model)
    rc = MujocoRacecar(model, data, max_steps=steps)
    return model, data, rc


def install_racecar_core_shim(rc: MujocoRacecar) -> None:
    """Inject a fake `racecar_core` module so a controller's own
    `import racecar_core; racecar_core.create_racecar()` resolves to `rc`
    with no real racecar_core install anywhere on this machine."""
    shim = types.ModuleType("racecar_core")
    shim.create_racecar = lambda *args, **kwargs: rc
    sys.modules["racecar_core"] = shim


def run_controller(controller_path: str, steps: int = DEFAULT_STEPS) -> None:
    """Validate the path, build the sim + shim, then execute the controller
    file unchanged with runpy so its own `if __name__ == "__main__":` block
    (typically set_start_update(...) + go()) runs exactly as it would running
    the file directly -- only its `import racecar_core` is intercepted."""
    path = Path(controller_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"controller path is not a file: {path}")
    if path.suffix != ".py":
        raise ValueError(f"controller path must be a .py file, got: {path}")

    model, data, rc = build_racecar(steps)
    install_racecar_core_shim(rc)
    x0 = float(data.body("car").xpos[0])

    controller_dir = str(path.parent)
    sys.path.insert(0, controller_dir)
    try:
        runpy.run_path(str(path), run_name="__main__")
    finally:
        sys.path.remove(controller_dir)
        sys.modules.pop("racecar_core", None)

    x1 = float(data.body("car").xpos[0])
    print(
        f"[run_racecar_core] {steps} steps done -- car moved dx={x1 - x0:+.2f}m "
        f"(x: {x0:.2f} -> {x1:.2f})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("controller", help="path to a racecar_core controller .py file")
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help=f"physics steps to run before stopping (default {DEFAULT_STEPS})",
    )
    args = parser.parse_args()

    try:
        run_controller(args.controller, args.steps)
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(f"error: {exc}")
    except Exception as exc:
        sys.exit(f"error: controller failed during initialization/run: {exc!r}")


if __name__ == "__main__":
    main()
