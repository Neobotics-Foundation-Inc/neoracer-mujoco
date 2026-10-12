"""
Laws for the car and track XML files in assets/.
"""

import os
import warnings

import mujoco
import numpy as np
import pytest

from neoracer_mujoco import assets, sim

TRACKS = assets.tracks()


@pytest.mark.parametrize("track", TRACKS, ids=os.path.basename)
@pytest.mark.parametrize("car_path", assets.cars(), ids=os.path.basename)
def test_car_on_track_keeps_the_car_solver_options(car_path, track):
    """When a car is attached to a track, MuJoCo keeps the track's <option>.
    So the composed scene must have the car's own friction cone and impratio,
    and the attach must give no conflict warning."""
    car = assets.load(car_path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        composed = assets.compose(mujoco.MjSpec.from_file(track), car_path)
    assert [str(w.message) for w in caught] == []
    assert composed.opt.cone == car.opt.cone
    assert composed.opt.impratio == car.opt.impratio


@pytest.mark.parametrize("track", TRACKS, ids=os.path.basename)
def test_car_comes_to_rest_on_every_track(track):
    """Each track has its own floor: the car settles on it and does not fall
    through the world. compose attaches only the car body, so the floor of
    neoracer.xml is not in the scene."""
    model = assets.compose(mujoco.MjSpec.from_file(track))
    data = mujoco.MjData(model)
    sim.apply_ctrl_and_step(model, data, 0.0, sim.SETTLE_STEPS)
    car = data.body("car")
    assert 0.05 < float(car.xpos[2]) < 0.2, f"car z = {car.xpos[2]:.3f} m"
    assert float(np.linalg.norm(data.qvel[:3])) < 0.05
