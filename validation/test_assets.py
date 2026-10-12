"""
Laws for the car and track XML files in assets/.
"""

import os

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
