"""
Laws for sim's settle and perturb_spawn: the car at rest, and moved off its
start at random.
"""

import itertools
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


def test_a_spawn_moves_no_point_of_the_car_farther_than_its_reach():
    """Every point within a colliding geom's bounding sphere moves by at
    most spawn_reach, in the plane, whichever way the start faces, and for a
    yaw bound past half a turn too; and some point moves by more than half
    of it, so the bound is not loose."""
    spawns = (
        sim.SpawnRange(lateral=0.1, yaw=0.3, resettle=0.0),
        sim.SpawnRange(lateral=0.0, yaw=6.0, resettle=0.0),
    )
    angles = np.linspace(0, 2 * math.pi, 8, endpoint=False)
    directions = np.stack([np.cos(angles), np.sin(angles), np.zeros(8)], axis=1)
    for turn, spawn in itertools.product(
        (0.0, 120.0), spawns
    ):  # turn: the start's, degrees
        with case((turn, spawn)):
            track = mujoco.MjSpec.from_file(LOOP)
            half = math.radians(turn) / 2
            track.site(assets.START_SITE).quat = np.array(
                [math.cos(half), 0.0, 0.0, math.sin(half)]
            )
            scene = assets.compose(track, visuals=False)
            model, car = scene.model, scene.car
            reach = sim.spawn_reach(model, car, spawn)
            root = model.bind_scalar(car.body).id
            geoms = np.flatnonzero(
                (model.body_rootid[model.geom_bodyid] == root)
                & ((model.geom_contype != 0) | (model.geom_conaffinity != 0))
            )
            farthest = 0.0
            for seed in range(40):
                data = sim.settle(model)
                before = (data.xpos[root].copy(), data.xmat[root].reshape(3, 3).copy())
                points = np.concatenate(
                    [
                        data.geom_xpos[g] + model.geom_rbound[g] * directions
                        for g in geoms
                    ]
                )
                local = (points - before[0]) @ before[1]
                sim.perturb_spawn(model, data, car, np.random.default_rng(seed), spawn)
                moved = local @ data.xmat[root].reshape(3, 3).T + data.xpos[root]
                gaps = np.linalg.norm((moved - points)[:, :2], axis=1)
                farthest = max(farthest, float(gaps.max()))
            assert farthest <= reach + 1e-9
            assert farthest > reach / 2


def test_no_move_of_a_spawn_that_clears_the_walls_touches_a_wall():
    """Where spawn_clears_walls holds, no move of perturb_spawn puts a car
    against a wall, and a car near a wall does not clear it."""
    spawn = sim.SpawnRange(lateral=0.2, yaw=0.3, resettle=0.0)
    track = mujoco.MjSpec.from_file(LOOP)
    scene = assets.compose_cars(
        track, [assets.Entry("a/"), assets.Entry("b/", (-1.5, 0.0))], visuals=False
    )
    assert sim.spawn_clears_walls(scene, spawn)
    for seed in range(40):
        data = mujoco.MjData(scene.model)
        mujoco.mj_forward(scene.model, data)
        rng = np.random.default_rng(seed)
        for car in scene.cars:
            sim.perturb_spawn(scene.model, data, car, rng, spawn)
        with case(seed):
            for car in scene.cars:
                assert data.bind_scalar(car.truth.wall_contact).data[0] == 0
    for across in (0.3, -0.3):  # b/ beside the inner wall, then the outer
        with case(across):
            entries = [assets.Entry("a/"), assets.Entry("b/", (-1.5, across))]
            scene = assets.compose_cars(track, entries, visuals=False)
            assert not sim.spawn_clears_walls(scene, spawn)
