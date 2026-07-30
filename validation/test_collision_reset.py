"""
Tests for examples/collision_reset.py: the auto-reset-on-wall-collision
"driver's test" (issue #5).

Uses the same straight_corridor.xml + neoracer.xml composition pattern as
test_track_centering.py's physics-integration test.
"""

import os

import mujoco
from collision_reset import car_hit_wall, reset_if_wall_hit, wall_geom_ids


def _load_corridor_model() -> mujoco.MjModel:
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scene = mujoco.MjSpec.from_file(
        os.path.join(repo, "assets", "tracks", "straight_corridor.xml")
    )
    car = mujoco.MjSpec.from_file(os.path.join(repo, "assets", "neoracer.xml"))
    scene.worldbody.add_frame().attach_body(car.body("car"), "", "")
    return scene.compile()


def _load_ramp_model() -> mujoco.MjModel:
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scene = mujoco.MjSpec.from_file(
        os.path.join(repo, "assets", "tracks", "ramp_course.xml")
    )
    car = mujoco.MjSpec.from_file(os.path.join(repo, "assets", "neoracer.xml"))
    scene.worldbody.add_frame().attach_body(car.body("car"), "", "")
    return scene.compile()


def _settle(model: mujoco.MjModel) -> mujoco.MjData:
    data = mujoco.MjData(model)
    data.ctrl[:] = 0
    for _ in range(400):
        mujoco.mj_step(model, data)
    return data


# --- wall_geom_ids -----------------------------------------------------------


def test_wall_geom_ids_finds_corridor_walls():
    model = _load_corridor_model()
    names = {model.geom(i).name for i in wall_geom_ids(model)}
    assert names == {"corridor_wall_left", "corridor_wall_right"}


def test_wall_geom_ids_excludes_ramp_and_floor():
    """ramp_course.xml's ramp is drivable scenery, not a wall -- it uses a
    different material and must not be treated as one."""
    model = _load_ramp_model()
    names = {model.geom(i).name for i in wall_geom_ids(model)}
    assert names == {"turn1", "turn2"}


def test_wall_geom_ids_empty_when_no_walls():
    """The bare car on its own plane (no track attached) has no wall
    material at all, so this must return empty rather than error."""
    model = mujoco.MjModel.from_xml_path(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "assets",
            "neoracer.xml",
        )
    )
    assert wall_geom_ids(model) == set()


# --- car_hit_wall / reset_if_wall_hit -----------------------------------------


def test_no_reset_at_rest_in_corridor_center():
    model = _load_corridor_model()
    data = _settle(model)
    assert not car_hit_wall(model, data)
    assert not reset_if_wall_hit(model, data)


def test_reset_triggers_on_wall_penetration():
    model = _load_corridor_model()
    data = _settle(model)

    # Corridor half-width is 0.5 m; teleport the car into the left wall.
    data.qpos[1] = 0.6
    mujoco.mj_forward(model, data)

    assert car_hit_wall(model, data)
    was_reset = reset_if_wall_hit(model, data)
    assert was_reset
    assert data.qpos[1] == 0.0, "reset must restore the car's spawn position"
    assert not car_hit_wall(model, data), (
        "freshly reset state must not itself count as a hit"
    )


def test_no_reset_when_car_never_touches_wall():
    """Driving straight down the middle of the corridor for a while must
    never trigger a reset -- otherwise the driver's test would be useless."""
    model = _load_corridor_model()
    data = _settle(model)
    for _ in range(500):
        data.ctrl[0:4] = 0.1
        mujoco.mj_step(model, data)
        assert not reset_if_wall_hit(model, data)


def test_bare_car_never_resets():
    """No wall material exists on the bare car model, so nothing should
    ever be flagged as a collision regardless of physics state."""
    model = mujoco.MjModel.from_xml_path(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "assets",
            "neoracer.xml",
        )
    )
    data = _settle(model)
    assert not reset_if_wall_hit(model, data)
