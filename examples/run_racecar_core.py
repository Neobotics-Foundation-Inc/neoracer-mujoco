"""
Run a racecar_core controller script against the NeoRacer MuJoCo simulation.

This script runs a controller file with a fake `racecar_core` module that returns a MujocoRacecar.
The controller file runs unchanged, for example
examples/ultimate_wall_follower/wall_follower.py.
MujocoRacecar (neoracer_mujoco.racecar_core_adapter) gives the controller
rc.drive, rc.lidar, rc.controller, rc.display, rc.set_start_update and rc.go.
racecar_core_adapter.py gives the units and the sign conventions.

The script does these steps:
  1. It puts the car in a corridor with walls and lets the car settle.
     The LiDAR of a wall follower needs walls to measure.
  2. It makes a MujocoRacecar for the scene and data.
  3. It adds a fake `racecar_core` module to sys.modules.
     The create_racecar() function of this module returns the MujocoRacecar.
  4. It puts the directory of the controller first on sys.path.
     The controller can then import its own modules, for example `import control`.
  5. It runs the controller file with runpy.run_path() as __main__.

Usage:
    python3 examples/run_racecar_core.py <path/to/controller.py> [--steps N]

Example:
    python3 examples/run_racecar_core.py examples/ultimate_wall_follower/wall_follower.py
"""

import argparse
import contextlib
import os
import runpy
import sys
import time
import types
from collections.abc import Generator
from pathlib import Path

import mujoco

from neoracer_mujoco import assets, sim
from neoracer_mujoco.racecar_core_adapter import MujocoRacecar
from neoracer_mujoco.scene import Scene

# 1.2 s of simulation time at the 0.002 s timestep of the model.
# straight_corridor.xml is about 16 m long. The wall_follower.yaml config of the
# example drives at speed=1.0. In a longer run, the car leaves the far end of
# the corridor, and the LiDAR then measures nothing. For a longer run on a
# bigger track, pass --steps.
DEFAULT_STEPS = 600


def load_scene() -> Scene:
    """The car at the center of the corridor in assets/tracks/straight_corridor.xml.
    The side LiDAR beams of a wall follower then measure the walls."""
    return assets.compose(
        mujoco.MjSpec.from_file(os.path.join(assets.TRACK_DIR, "straight_corridor.xml"))
    )


def build_racecar(scene: Scene, steps: int) -> tuple[mujoco.MjData, MujocoRacecar]:
    """Settle the car in scene, and make a MujocoRacecar that stops after steps.

    The MujocoRacecar holds the RB button down by default. A controller that
    drives only while RB is held, like the wall follower, then drives at once.
    """
    data = sim.settle(scene.model)
    return data, MujocoRacecar(scene, data, max_steps=steps)


class _SimTime(types.ModuleType):
    """The time module, with its clocks reading the simulated time of rc."""

    def __init__(self, rc: MujocoRacecar) -> None:
        super().__init__("time")
        self._rc = rc

    def __getattr__(self, name: str) -> object:
        return getattr(time, name)

    def time(self) -> float:
        return self._rc.clock()

    def monotonic(self) -> float:
        return self._rc.clock()

    def perf_counter(self) -> float:
        return self._rc.clock()


@contextlib.contextmanager
def controller_imports(rc: MujocoRacecar, controller_dir: Path) -> Generator[None]:
    """Make a controller in controller_dir importable with rc in the with block.

    In the block, `import racecar_core` gives a fake module whose
    create_racecar() returns rc, and controller_dir is first on sys.path.
    `import time` gives the time module with its clocks (time, monotonic and
    perf_counter) reading the simulated time, rc.clock(). A controller that
    measures its frame time with them then sees the physics step, whatever
    the speed of the machine.

    A cached module in sys.modules wins over sys.path. So before the block,
    the function moves aside every cached module that has the name of a
    module in controller_dir, and racecar_core. On exit, it removes the fake
    module, the sys.path entry and each module that the block imported from
    controller_dir. Then it puts back the modules that it moved aside, so the
    import state of the caller is the same as before.
    """
    local = {p.stem for p in controller_dir.glob("*.py")} | {
        p.name for p in controller_dir.iterdir() if (p / "__init__.py").is_file()
    }
    hidden = {
        name: module
        for name, module in sys.modules.items()
        if name in ("racecar_core", "time") or name.split(".")[0] in local
    }
    for name in hidden:
        del sys.modules[name]
    shim = types.ModuleType("racecar_core")
    shim.__dict__["create_racecar"] = lambda *args, **kwargs: rc
    sys.modules["racecar_core"] = shim
    sys.modules["time"] = _SimTime(rc)
    sys.path.insert(0, str(controller_dir))
    try:
        yield
    finally:
        sys.path.remove(str(controller_dir))
        sys.modules.pop("racecar_core", None)
        sys.modules.pop("time", None)
        for name in list(sys.modules):
            file = getattr(sys.modules[name], "__file__", None)
            if file is not None and Path(file).is_relative_to(controller_dir):
                del sys.modules[name]
        sys.modules.update(hidden)


def run_controller(controller_path: str, steps: int = DEFAULT_STEPS) -> float:
    """Run the controller file at controller_path for steps physics steps.

    runpy runs the file as __main__, so its own `if __name__ == "__main__":`
    block runs (usually set_start_update(...) and go()).
    Returns the distance (m) that the car moved along +X.
    Raises FileNotFoundError or ValueError if the path is not a .py file.
    """
    path = Path(controller_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"controller path is not a file: {path}")
    if path.suffix != ".py":
        raise ValueError(f"controller path must be a .py file, got: {path}")

    scene = load_scene()
    data, rc = build_racecar(scene, steps)
    x0 = float(data.bind_scalar(scene.car.body).xpos[0])

    with controller_imports(rc, path.parent):
        runpy.run_path(str(path), run_name="__main__")

    x1 = float(data.bind_scalar(scene.car.body).xpos[0])
    print(
        f"[run_racecar_core] {steps} steps done -- car moved dx={x1 - x0:+.2f}m "
        f"(x: {x0:.2f} -> {x1:.2f})"
    )
    return x1 - x0


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


if __name__ == "__main__":
    main()
