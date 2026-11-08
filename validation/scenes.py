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


def two_cars(behind: float = 1.5) -> Scene:
    """Two default cars without visuals on assets/tracks/loop_corridor.xml:
    "a/" at the start, and "b/" behind it by behind (m)."""
    return assets.compose_cars(
        mujoco.MjSpec.from_file(os.path.join(assets.TRACK_DIR, "loop_corridor.xml")),
        [assets.Entry("a/"), assets.Entry("b/", (-behind, 0.0))],
        visuals=False,
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


def assert_same_model(actual: mujoco.MjModel, expected: mujoco.MjModel) -> None:
    """Check that actual is expected, every field equal, those of opt, stat
    and vis too, but for the paths of the files that each was compiled from:
    MuJoCo keeps each asset's path as it read it, so a model compiled from
    another folder or an archive differs in paths, their length (npaths),
    where each asset's path starts in them (*_pathadr) and the size of the
    model's memory (nbuffer) alone."""
    _assert_same(actual, expected, "model")


def _assert_same(actual: object, expected: object, where: str) -> None:
    """Check that the public fields of actual, a struct of MuJoCo's, are
    those of expected, the struct-valued ones field by field."""
    for name in dir(expected):
        value = getattr(expected, name)
        skip = name in ("paths", "npaths", "nbuffer") or name.endswith("_pathadr")
        if name.startswith("_") or skip or callable(value):
            continue
        here = f"{where}.{name}"
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(getattr(actual, name), value, err_msg=here)
        elif isinstance(value, (bool, int, float, bytes)):
            assert getattr(actual, name) == value, here
        else:
            _assert_same(getattr(actual, name), value, here)


def assert_car_rests(scene: Scene) -> None:
    """Check that the car settles on the floor of the scene with zero
    control: it does not fall through and it does not move."""
    data = sim.settle(scene.model)
    z = float(data.bind_scalar(scene.car.body).xpos[2])
    # The body frame is on the axle line, which rests near the tire radius.
    assert 0.03 < z < 0.1, f"car z = {z:.3f} m"
    assert (
        float(np.linalg.norm(data.bind_scalar(scene.car.joint).qvel[:3])) < 0.05
    )  # m/s
