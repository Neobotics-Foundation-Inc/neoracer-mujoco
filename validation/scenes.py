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


def reversed_car(path: str = assets.DEFAULT_CAR) -> Scene:
    """The car of the XML at path, the default car's by default, with its
    actuators and its sensors in reverse order."""
    root = ET.fromstring(assets.read_xml(path))
    for tag in ("actuator", "sensor"):
        section = root.find(tag)
        assert section is not None, tag
        children = list(section)
        for child in children:
            section.remove(child)
        section.extend(reversed(children))
    spec = mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))
    # A spec made from a string has no file directory, so give it the one of
    # the car XML. Its mesh paths are relative to it.
    spec.meshdir = os.path.dirname(path)
    return compile_scene(spec)


def assert_takes_car_solver_options(
    track: mujoco.MjSpec, car_path: str = assets.DEFAULT_CAR
) -> None:
    """Attach the car to track and check the solver options of the result:
    the scene runs with the car's options, and the attach gives no
    warning."""
    car = assets.load(car_path).model
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        composed = assets.compose(track, car_path).model
    assert [str(w.message) for w in caught] == []
    for name in ("timestep", "integrator", "cone", "impratio"):
        assert getattr(composed.opt, name) == getattr(car.opt, name), name


def assert_car_rests(scene: Scene) -> None:
    """Check that the car settles on the floor of the scene with zero
    control: it does not fall through and it does not move."""
    data = sim.settle(scene.model)
    z = float(data.bind_scalar(scene.car.body).xpos[2])
    # The body frame is on the axle line, which rests near the tyre radius.
    assert 0.03 < z < 0.1, f"car z = {z:.3f} m"
    assert (
        float(np.linalg.norm(data.bind_scalar(scene.car.joint).qvel[:3])) < 0.05
    )  # m/s
