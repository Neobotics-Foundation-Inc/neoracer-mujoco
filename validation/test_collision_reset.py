"""
Tests for collision.py: which geoms are walls, wall detection, and the reset
after a wall hit.
"""

import os

import mujoco
import numpy as np
from scenes import corridor_scene

from neoracer_mujoco import assets, sim
from neoracer_mujoco.collision import car_touches_wall, reset_car
from neoracer_mujoco.control.steering import ControlCommand
from neoracer_mujoco.truth import WALLS_BODY


def _wall_names(spec: mujoco.MjSpec) -> set[str]:
    walls = spec.body(WALLS_BODY)
    assert walls is not None
    return {geom.name for geom in walls.geoms}


# --- the walls body ----------------------------------------------------------


def test_corridor_walls_are_in_the_walls_body():
    assert _wall_names(corridor_scene().spec) == {
        "corridor_wall_left",
        "corridor_wall_right",
    }


def test_ramp_and_floor_are_not_walls():
    """The ramp in ramp_course.xml is drivable, so it is not in the walls
    body."""
    ramp_course = os.path.join(assets.TRACK_DIR, "ramp_course.xml")
    assert _wall_names(mujoco.MjSpec.from_file(ramp_course)) == {"turn1", "turn2"}


def test_every_f1_geom_but_the_floor_is_a_wall():
    """Every panel and post of f1_track.xml is a wall; the floor is not."""
    spec = mujoco.MjSpec.from_file(os.path.join(assets.TRACK_DIR, "f1_track.xml"))
    names = _wall_names(spec)
    assert len(names) == spec.compile().ngeom - 1
    assert "floor" not in names


def test_the_bare_car_gets_an_empty_walls_body():
    """neoracer.xml has no walls. Its scene gets an empty walls body, so it
    has the same truth sensors as a track."""
    assert _wall_names(assets.load().spec) == set()


# --- car_touches_wall / reset_car --------------------------------------------


def test_no_contact_at_rest_in_corridor_center():
    scene = corridor_scene()
    assert not car_touches_wall(sim.settle(scene.model), scene.car)


def test_reset_after_wall_penetration_restores_the_spawn_state():
    scene = corridor_scene()
    model = scene.model
    data = sim.settle(model)

    # The corridor half-width is 0.5 m: this moves the car into the left wall.
    data.bind_scalar(scene.car.joint).qpos[1] = 0.6
    mujoco.mj_forward(model, data)
    assert car_touches_wall(data, scene.car)

    reset_car(model, data)
    assert np.array_equal(data.qpos, model.qpos0)
    assert not car_touches_wall(data, scene.car)

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
    sim.apply_command(scene.car, data, ControlCommand(throttle=0.5, steer=0.0))
    for _ in range(500):  # 1 s at the 2 ms timestep
        mujoco.mj_step(model, data)
        assert not car_touches_wall(data, scene.car)


def test_bare_car_never_touches_a_wall():
    """The bare car model has no walls, so no contact is a wall contact."""
    scene = assets.load()
    assert not car_touches_wall(sim.settle(scene.model), scene.car)
