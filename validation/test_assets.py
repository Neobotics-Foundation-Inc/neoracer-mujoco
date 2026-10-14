"""
Laws for the car and track XML files in assets/.
"""

import os
import tomllib
from pathlib import Path

import mujoco
import numpy as np
import pytest
from scenes import assert_car_rests, assert_keeps_car_solver_options

from neoracer_mujoco import assets, sim

TRACKS = assets.tracks()


@pytest.mark.parametrize("track", TRACKS, ids=os.path.basename)
@pytest.mark.parametrize("car_path", assets.cars(), ids=os.path.basename)
def test_car_on_track_keeps_the_car_solver_options(car_path, track):
    assert_keeps_car_solver_options(mujoco.MjSpec.from_file(track), car_path)


@pytest.mark.parametrize("track", TRACKS, ids=os.path.basename)
def test_car_comes_to_rest_on_every_track(track):
    """Each track has its own floor, and the car rests on it. compose
    attaches only the car body, so the floor of neoracer.xml is not in the
    scene."""
    assert_car_rests(assets.compose(mujoco.MjSpec.from_file(track)))


def test_the_wheel_puts_assets_where_assets_py_looks():
    """pyproject.toml copies assets/ into the wheel, and assets.py finds it
    there by name. If one of them changes the name, an installed package
    cannot find its car."""
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    wheel = tomllib.loads(pyproject.read_text())["tool"]["hatch"]["build"]["targets"][
        "wheel"
    ]
    assert wheel["force-include"] == {
        "assets": f"neoracer_mujoco/{assets.PACKAGED_DIR}"
    }


@pytest.mark.parametrize("track", TRACKS, ids=os.path.basename)
def test_a_scene_without_visuals_senses_the_same(track):
    """visuals=False deletes only meshes that collide with nothing, so every
    sensor reads the same, bit for bit, while the car drives and steers."""
    spec = mujoco.MjSpec.from_file(track)
    full, bare = assets.compose(spec), assets.compose(spec, visuals=False)
    assert bare.model.nmesh < full.model.nmesh
    readings = []
    for scene in (full, bare):
        data = sim.settle(scene.model)
        for motor in scene.car.drive:
            data.bind_scalar(motor).ctrl = 0.3  # N·m
        data.bind_scalar(scene.car.steer).ctrl = 0.3  # rad
        steps = []
        for _ in range(300):  # 0.6 s
            mujoco.mj_step(scene.model, data)
            steps.append(data.sensordata.copy())
        readings.append(np.array(steps))
    np.testing.assert_array_equal(readings[0], readings[1])
