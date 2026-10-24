"""
Laws for sim's settle and perturb_spawn: the car at rest, and moved off its
start at random.
"""

import math
import os

import mujoco
import numpy as np
import pytest
from cases import case

from neoracer_mujoco import assets, sim

LOOP = os.path.join(assets.TRACK_DIR, "loop_corridor.xml")


def _heading(data: mujoco.MjData, car) -> float:
    w, x, y, z = data.bind_scalar(car.joint).qpos[3:7]
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def test_settling_data_again_is_settling_new_data():
    """settle resets the data that it is given, so a used one settles to the
    state of a new one."""
    scene = assets.compose(mujoco.MjSpec.from_file(LOOP), visuals=False)
    used = sim.settle(scene.model)
    sim.perturb_spawn(scene.model, used, scene.car, np.random.default_rng(0))
    again = sim.settle(scene.model, used)
    assert again is used
    np.testing.assert_array_equal(again.qpos, sim.settle(scene.model).qpos)


def test_a_spawn_range_that_is_not_a_range_is_refused():
    """A bound or a resettle time that is negative, infinite or NaN raises
    before any spawn; 0 is no perturbation."""
    sim.SpawnRange(lateral=0.0, yaw=0.0, resettle=0.0)
    for field in ("lateral", "yaw", "resettle"):
        for bad in (-0.1, math.inf, math.nan):
            with case((field, bad)), pytest.raises(ValueError, match=field):
                sim.SpawnRange(**{field: bad})


def test_a_spawn_moves_the_car_square_to_its_heading_and_turns_it():
    """The car moves by the first draw from rng along its left, whichever
    way its start faces, and turns by the second."""
    for turn in (0.0, 90.0, -135.0):  # degrees, the start's heading
        with case(turn):
            track = mujoco.MjSpec.from_file(LOOP)
            half = math.radians(turn) / 2
            quat = np.array([math.cos(half), 0.0, 0.0, math.sin(half)])
            track.site(assets.START_SITE).quat = quat
            scene = assets.compose(track, visuals=False)
            data = sim.settle(scene.model)
            before = data.bind_scalar(scene.car.joint).qpos.copy()
            heading = _heading(data, scene.car)
            spawn = sim.SpawnRange(resettle=0.0)
            sim.perturb_spawn(
                scene.model, data, scene.car, np.random.default_rng(3), spawn
            )
            draws = np.random.default_rng(3)
            lateral = draws.uniform(-spawn.lateral, spawn.lateral)
            yaw = draws.uniform(-spawn.yaw, spawn.yaw)
            after = data.bind_scalar(scene.car.joint).qpos
            left = np.array([-math.sin(heading), math.cos(heading)])
            np.testing.assert_allclose(
                after[:2] - before[:2], lateral * left, atol=1e-12
            )
            assert math.remainder(
                _heading(data, scene.car) - heading - yaw, 2 * math.pi
            ) == pytest.approx(0.0, abs=1e-6)  # the turn is about the car's own Z
