"""
Laws for the car and track XML files in assets/.
"""

import os
import tomllib
from pathlib import Path

import mujoco
import pytest
from scenes import assert_car_rests, assert_keeps_car_solver_options

from neoracer_mujoco import assets

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
