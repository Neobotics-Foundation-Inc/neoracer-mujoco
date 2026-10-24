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


def test_the_walls_body_holds_exactly_the_walls_of_each_track():
    """The ramp of ramp_course.xml is drivable, so it is not a wall; every
    panel and post of f1_track.xml is a wall, and the floor is not. The bare
    car of neoracer.xml has no walls, and its scene gets an empty walls body,
    so it has the same truth sensors as a track."""
    assert _wall_names(corridor_scene().spec) == {
        "corridor_wall_left",
        "corridor_wall_right",
    }
    ramp_course = os.path.join(assets.TRACK_DIR, "ramp_course.xml")
    assert _wall_names(mujoco.MjSpec.from_file(ramp_course)) == {"turn1", "turn2"}
    f1 = mujoco.MjSpec.from_file(os.path.join(assets.TRACK_DIR, "f1_track.xml"))
    names = _wall_names(f1)
    assert len(names) == f1.compile().ngeom - 1
    assert "floor" not in names
    assert _wall_names(assets.load().spec) == set()


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
    for _ in range(round(1.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)
        assert not car_touches_wall(data, scene.car)
