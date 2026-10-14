"""
Tests for collision.py: wall detection, and the reset after a wall hit.
"""

import os

import mujoco
import numpy as np
from scenes import corridor_scene

from neoracer_mujoco import Scene, assets, sim
from neoracer_mujoco.collision import car_touches_wall, reset_car, wall_geom_ids


def _touches_wall(scene: Scene, data: mujoco.MjData) -> bool:
    walls = wall_geom_ids(scene.spec, scene.model)
    return car_touches_wall(scene.model, data, scene.car.body, walls)


def _wall_names(spec: mujoco.MjSpec, model: mujoco.MjModel) -> set[str]:
    return {model.geom(i).name for i in wall_geom_ids(spec, model)}


# --- wall_geom_ids -----------------------------------------------------------


def test_wall_geom_ids_finds_corridor_walls():
    scene = corridor_scene()
    names = _wall_names(scene.spec, scene.model)
    assert names == {"corridor_wall_left", "corridor_wall_right"}


def test_wall_geom_ids_excludes_ramp_and_floor():
    """The ramp in ramp_course.xml is drivable. It uses another material, so
    it is not a wall."""
    ramp_course = os.path.join(assets.TRACK_DIR, "ramp_course.xml")
    scene = assets.compose(mujoco.MjSpec.from_file(ramp_course))
    assert _wall_names(scene.spec, scene.model) == {"turn1", "turn2"}


def test_wall_geom_ids_counts_wall_prefixed_materials():
    """f1_track.xml colors its walls with materials wall_outer and wall_inner.
    Every panel and post is a wall; the floor is not."""
    spec = mujoco.MjSpec.from_file(os.path.join(assets.TRACK_DIR, "f1_track.xml"))
    model = spec.compile()
    names = _wall_names(spec, model)
    assert len(names) == model.ngeom - 1
    assert "floor" not in names


def test_wall_geom_ids_empty_when_no_walls():
    """The bare car on its own plane has no wall material."""
    scene = assets.load()
    assert wall_geom_ids(scene.spec, scene.model) == frozenset()


# --- car_touches_wall / reset_car --------------------------------------------


def test_no_contact_at_rest_in_corridor_center():
    scene = corridor_scene()
    assert not _touches_wall(scene, sim.settle(scene.model))


def test_reset_after_wall_penetration_restores_the_spawn_state():
    scene = corridor_scene()
    model = scene.model
    data = sim.settle(model)

    # The corridor half-width is 0.5 m: this moves the car into the left wall.
    data.qpos[scene.car.qpos][1] = 0.6
    mujoco.mj_forward(model, data)
    assert _touches_wall(scene, data)

    reset_car(model, data)
    assert np.array_equal(data.qpos, model.qpos0)
    assert not _touches_wall(scene, data)

    # A second reset changes nothing.
    before = data.qpos.copy(), data.qvel.copy(), data.time
    reset_car(model, data)
    assert np.array_equal(data.qpos, before[0])
    assert np.array_equal(data.qvel, before[1])
    assert data.time == before[2]


def test_no_contact_when_driving_down_the_center():
    """A car that drives straight down the center never touches a wall."""
    scene = corridor_scene()
    model = scene.model
    data = sim.settle(model)
    walls = wall_geom_ids(scene.spec, model)
    data.ctrl[list(scene.car.drive)] = 0.1  # N·m per wheel
    for _ in range(500):  # 1 s at the 2 ms timestep
        mujoco.mj_step(model, data)
        assert not car_touches_wall(model, data, scene.car.body, walls)


def test_bare_car_never_touches_a_wall():
    """The bare car model has no wall material, so no contact is a wall
    contact."""
    scene = assets.load()
    assert not _touches_wall(scene, sim.settle(scene.model))
