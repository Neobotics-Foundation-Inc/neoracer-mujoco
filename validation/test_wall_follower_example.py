"""
The example racecar_core wall follower runs against MuJoCo with no changes to its controller code.

The test uses the byte-for-byte copy of neoracer-labs/ultimate-wall-follower
in testdata/ultimate_wall_follower/ (see its SOURCE.md).
wall_follower.py runs `import racecar_core`. The test gives it a fake
`racecar_core` module whose create_racecar() returns a MujocoRacecar.
The test does not change the copied files.

This test checks only that the controller runs on the adapter.
It does not check that every controller follows a wall in every scene.
"""

import os
import runpy
from pathlib import Path

import numpy as np

from examples import run_racecar_core
from neoracer_mujoco import contract
from neoracer_mujoco.racecar_core_adapter import neoracer_params

_FIXTURE_DIR = Path(__file__).resolve().parent / "testdata" / "ultimate_wall_follower"


def test_canonical_wall_follower_runs_unmodified_against_mujoco():
    scene = run_racecar_core.load_scene()
    car = scene.car
    data, rc = run_racecar_core.build_racecar(scene, steps=600)
    x0 = float(data.bind_scalar(car.body).xpos[0])

    # rc.drive, rc.lidar, rc.controller and rc.display work before the
    # controller uses them.
    assert rc.lidar.get_samples().shape == (neoracer_params().lidar.slots,)
    rc.drive.stop()
    assert rc.controller.is_down(rc.controller.Button.RB) is True
    rc.display.show_text("ok")  # no-op; must not raise

    # Run the top-level code of wall_follower.py (`rc = racecar_core.create_racecar()`),
    # but not its __main__ block. wf holds the globals of the module.
    with run_racecar_core.controller_imports(rc, _FIXTURE_DIR):
        wf = runpy.run_path(
            os.path.join(_FIXTURE_DIR, "wall_follower.py"), run_name="wall_follower"
        )

    # start, update, update_slow and control.py contain no MuJoCo code.
    rc.set_start_update(wf["start"], wf["update"], wf["update_slow"])
    rc.go()

    scan = rc.lidar.get_samples()
    assert np.all(np.isfinite(scan))
    assert np.all(scan >= 0.0)

    limits = contract.actuator_limits(scene.model, car)
    assert np.isfinite(data.ctrl).all()
    assert abs(data.bind_scalar(car.drive).ctrl[0]) <= limits.voltage + 1e-6
    assert abs(data.bind_scalar(car.steer).ctrl[0]) <= limits.steer_rad + 1e-6

    x1 = float(data.bind_scalar(car.body).xpos[0])
    assert x1 - x0 > 0.2, (
        f"expected forward progress down the corridor, got dx={x1 - x0:.3f}"
    )
