"""Scenes and scene laws that several test files use."""

import os
import warnings
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from neoracer_mujoco import Scene, assets, sim
from neoracer_mujoco.scene import compile_scene


def corridor_scene() -> Scene:
    """The default car on assets/tracks/straight_corridor.xml. The corridor
    runs along +X and has a half-width of 0.5 m."""
    return assets.compose(
        mujoco.MjSpec.from_file(os.path.join(assets.TRACK_DIR, "straight_corridor.xml"))
    )


def reversed_car() -> Scene:
    """The default car with its actuators and its sensors in reverse order."""
    root = ET.parse(assets.DEFAULT_CAR).getroot()
    for tag in ("actuator", "sensor"):
        section = root.find(tag)
        assert section is not None, tag
        children = list(section)
        for child in children:
            section.remove(child)
        section.extend(reversed(children))
    spec = mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))
    # A spec made from a string has no file directory, so give it the one of
    # the car XML. Mesh paths in neoracer.xml are relative to it.
    spec.meshdir = assets.ASSET_DIR
    return compile_scene(spec)


def assert_keeps_car_solver_options(
    track: mujoco.MjSpec, car_path: str = assets.DEFAULT_CAR
) -> None:
    """Attach the car to track and check the solver options of the result.

    When MuJoCo attaches a car to a track, it keeps the <option> of the track.
    So the track must have the friction cone and impratio of the car, and the
    attach must give no warning."""
    car = assets.load(car_path).model
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        composed = assets.compose(track, car_path).model
    assert [str(w.message) for w in caught] == []
    assert composed.opt.cone == car.opt.cone
    assert composed.opt.impratio == car.opt.impratio


def assert_car_rests(scene: Scene) -> None:
    """Check that the car settles on the floor of the scene with zero
    control: it does not fall through and it does not move."""
    data = sim.settle(scene.model)
    z = float(data.bind_scalar(scene.car.body).xpos[2])
    assert 0.05 < z < 0.2, f"car z = {z:.3f} m"
    assert (
        float(np.linalg.norm(data.bind_scalar(scene.car.joint).qvel[:3])) < 0.05
    )  # m/s
