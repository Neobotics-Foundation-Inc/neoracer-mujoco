"""
Acceptance test for issue #28: the canonical racecar_core wall-following
example must run against MuJoCo with no changes to its controller logic.

Uses the vendored, byte-for-byte copy of neoracer-labs/ultimate-wall-follower
(see testdata/ultimate_wall_follower/SOURCE.md) -- not a rewritten imitation.
wall_follower.py's own `import racecar_core` is satisfied by injecting a
fake `racecar_core` module into sys.modules whose create_racecar() returns a
MujocoRacecar; the vendored files themselves are never touched.
"""

import importlib.util
import os
import sys
import types

import mujoco
import numpy as np

from neoracer_mujoco import sim as _sim
from neoracer_mujoco.racecar_core_adapter import MujocoRacecar

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "testdata", "ultimate_wall_follower"
)


def _load_corridor_model() -> mujoco.MjModel:
    """Car centered in a walled corridor -- see assets/tracks/straight_corridor.xml
    -- so the side LiDAR beams the wall follower reads get real returns."""
    scene = mujoco.MjSpec.from_file(
        os.path.join(_REPO, "assets", "tracks", "straight_corridor.xml")
    )
    car = mujoco.MjSpec.from_file(os.path.join(_REPO, "assets", "neoracer.xml"))
    scene.worldbody.add_frame().attach_body(car.body("car"), "", "")
    return scene.compile()


def _import_vendored_wall_follower(rc, monkeypatch):
    """Load the vendored wall_follower.py as a real module and let it run
    its own top-level code (`rc = racecar_core.create_racecar()`), backed
    by a fake racecar_core module that just returns `rc`. This is the only
    seam: everything downstream (control.py, wall_follower.py) runs
    unmodified."""
    shim = types.ModuleType("racecar_core")
    shim.create_racecar = lambda *args, **kwargs: rc
    monkeypatch.setitem(sys.modules, "racecar_core", shim)
    # wall_follower.py does `import control`, resolved by bare module name
    # from whatever directory it lives in -- put that directory on sys.path
    # exactly as running it from its own folder would.
    monkeypatch.syspath_prepend(_FIXTURE_DIR)

    path = os.path.join(_FIXTURE_DIR, "wall_follower.py")
    spec = importlib.util.spec_from_file_location("_vendored_wall_follower", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
        sys.modules.pop("control", None)
    return module


def test_canonical_wall_follower_runs_unmodified_against_mujoco(monkeypatch):
    model = _load_corridor_model()
    data = _sim.settle(model)
    x0 = float(data.body("car").xpos[0])

    rc = MujocoRacecar(model, data, max_steps=400)

    # rc.drive / rc.lidar / rc.controller / rc.display each work before the
    # canonical controller ever touches them.
    assert rc.lidar.get_samples().shape == (8,)
    rc.drive.stop()
    assert rc.controller.is_down(rc.controller.Button.RB) is True
    rc.display.show_text("ok")  # no-op; must not raise

    wf = _import_vendored_wall_follower(rc, monkeypatch)

    # set_start_update() + go() run the callbacks: no MuJoCo-specific code
    # inside start/update/update_slow, nor inside control.py.
    rc.set_start_update(wf.start, wf.update, wf.update_slow)
    rc.go()

    scan = rc.lidar.get_samples()
    assert np.all(np.isfinite(scan))
    assert np.all(scan >= 0.0)

    assert np.isfinite(data.ctrl).all()
    torque_limit = model.actuator("fl_motor").ctrlrange[1]
    steer_limit = model.actuator("steer_servo").ctrlrange[1]
    assert np.all(np.abs(data.ctrl[0:4]) <= torque_limit + 1e-6)
    assert abs(data.ctrl[4]) <= steer_limit + 1e-6

    x1 = float(data.body("car").xpos[0])
    assert x1 - x0 > 0.2, (
        f"expected forward progress down the corridor, got dx={x1 - x0:.3f}"
    )
