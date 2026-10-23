"""
Laws for truth.py: each truth sensor gives the value that the compiled model
already has elsewhere, so code can read it from sensordata alone.
"""

import os

import mujoco
import numpy as np
import pytest

from neoracer_mujoco import Scene, assets, sim, truth
from neoracer_mujoco.control.steering import ControlCommand
from neoracer_mujoco.scene import compile_scene
from neoracer_mujoco.truth import WALLS_BODY


def _loop_scene() -> Scene:
    track = os.path.join(assets.TRACK_DIR, "loop_corridor.xml")
    return assets.compose(mujoco.MjSpec.from_file(track), pos=(0, -2.5, 0))


def _wall_contacts(scene: Scene, data: mujoco.MjData) -> int:
    """The contacts between the car's body tree and the walls body, counted
    from data.contact."""
    model = scene.model
    walls = scene.spec.body(WALLS_BODY)
    assert walls is not None
    car_root = model.bind_scalar(scene.car.body).id
    walls_id = model.bind_scalar(walls).id
    count = 0
    for contact in data.contact[: data.ncon]:
        bodies = model.geom_bodyid[[contact.geom1, contact.geom2]]
        on_car = model.body_rootid[bodies] == car_root
        on_walls = bodies == walls_id
        count += bool((on_car[0] and on_walls[1]) or (on_walls[0] and on_car[1]))
    return count


def _drive_into_the_walls(scene: Scene, steps: int):
    """Yield data after each step of a drive that turns into the walls."""
    data = mujoco.MjData(scene.model)
    for k in range(steps):
        steer = 0.4 if k > 400 else 0.0
        sim.apply_command(scene.car, data, ControlCommand(throttle=0.5, steer=steer))
        mujoco.mj_step(scene.model, data)
        yield data


def test_pos_and_zaxis_are_the_car_body_frame():
    """After every step, truth.pos is data.xpos of the car body and
    truth.zaxis is the third column of its data.xmat."""
    scene = _loop_scene()
    truth, body = scene.car.truth, scene.car.body
    for data in _drive_into_the_walls(scene, 600):
        frame = data.bind_scalar(body)
        assert np.array_equal(data.bind_scalar(truth.pos).data, frame.xpos)
        zaxis = frame.xmat.reshape(3, 3)[:, 2]
        assert np.array_equal(data.bind_scalar(truth.zaxis).data, zaxis)


def test_wall_contact_counts_the_contacts_with_the_walls():
    """On a drive that hits the walls, truth.wall_contact equals the count of
    car-wall contacts in data.contact after every step."""
    scene = _loop_scene()
    sensor = scene.car.truth.wall_contact
    hits = 0
    for data in _drive_into_the_walls(scene, 1500):
        expected = _wall_contacts(scene, data)
        assert data.bind_scalar(sensor).data[0] == expected
        hits += expected > 0
    assert hits > 0, "the drive never touched a wall"


def test_a_scene_spec_compiles_again():
    """compile_scene keeps the truth sensors that a spec already has, so the
    spec of a Scene, for example with a changed friction, compiles again."""
    scene = assets.load()
    again = compile_scene(scene.spec)
    assert again.model.nsensor == scene.model.nsensor


def test_a_sensor_with_a_truth_name_but_another_definition_is_rejected():
    """A car XML that already has a sensor named truth_zaxis, but on another
    object, in another frame, with a cutoff or with noise, does not compile:
    car_upright_cos would read the wrong value."""
    for other in [
        {"objtype": mujoco.mjtObj.mjOBJ_SITE, "objname": "imu"},
        {"reftype": mujoco.mjtObj.mjOBJ_SITE, "refname": "imu"},
        {"cutoff": 0.5},
        {"noise": 0.01},
    ]:
        spec = assets.from_file()
        sensor = spec.add_sensor(
            name=truth.ZAXIS,
            type=mujoco.mjtSensor.mjSENS_FRAMEZAXIS,
            objtype=mujoco.mjtObj.mjOBJ_XBODY,
            objname="car",
        )
        for field, setting in other.items():
            setattr(sensor, field, setting)
        with pytest.raises(ValueError, match=truth.ZAXIS):
            compile_scene(spec)
