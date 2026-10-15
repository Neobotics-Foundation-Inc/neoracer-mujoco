"""
Laws for the car and track XML files in assets/.
"""

import os
import tomllib
from pathlib import Path

import mujoco
import numpy as np
import pytest
from scenes import assert_car_rests, assert_keeps_car_solver_options

from neoracer_mujoco import assets, sim

TRACKS = assets.tracks()


@pytest.mark.parametrize("track", TRACKS, ids=os.path.basename)
@pytest.mark.parametrize("car_path", assets.cars(), ids=os.path.basename)
def test_car_on_track_keeps_the_car_solver_options(car_path, track):
    assert_keeps_car_solver_options(mujoco.MjSpec.from_file(track), car_path)


@pytest.mark.parametrize("track", TRACKS, ids=os.path.basename)
def test_car_comes_to_rest_on_every_track(track):
    """Each track has its own floor, and the car rests on it. compose
    attaches only the car body, so the floor of neoracer.xml is not in the
    scene."""
    assert_car_rests(assets.compose(mujoco.MjSpec.from_file(track)))


def test_the_wheel_puts_assets_where_assets_py_looks():
    """pyproject.toml copies assets/ into the wheel, and assets.py finds it
    there by name. If one of them changes the name, an installed package
    cannot find its car."""
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    wheel = tomllib.loads(pyproject.read_text())["tool"]["hatch"]["build"]["targets"][
        "wheel"
    ]
    assert wheel["force-include"] == {
        "assets": f"neoracer_mujoco/{assets.PACKAGED_DIR}"
    }


@pytest.mark.parametrize("track", TRACKS, ids=os.path.basename)
def test_a_scene_without_visuals_senses_the_same(track):
    """visuals=False deletes only meshes that collide with nothing, so every
    sensor reads the same, bit for bit, while the car drives and steers."""
    spec = mujoco.MjSpec.from_file(track)
    full, bare = assets.compose(spec), assets.compose(spec, visuals=False)
    assert bare.model.nmesh < full.model.nmesh
    readings = []
    for scene in (full, bare):
        data = sim.settle(scene.model)
        data.bind_scalar(scene.car.drive).ctrl = 1.0  # full throttle
        data.bind_scalar(scene.car.steer).ctrl = 0.3  # rad
        steps = []
        for _ in range(300):  # 0.6 s
            mujoco.mj_step(scene.model, data)
            steps.append(data.sensordata.copy())
        readings.append(np.array(steps))
    np.testing.assert_array_equal(readings[0], readings[1])


@pytest.mark.parametrize("track", TRACKS, ids=os.path.basename)
def test_a_scene_reloads_from_its_xml(track):
    """The XML of a composed scene compiles to the same model, up to the six
    significant digits that MuJoCo's XML writer keeps. Without visuals, the
    scene names no mesh file that from_xml_string would have to find."""
    scene = assets.compose(mujoco.MjSpec.from_file(track), visuals=False)
    reloaded = mujoco.MjModel.from_xml_string(scene.spec.to_xml())
    for name in dir(scene.model):
        value = getattr(scene.model, name)
        if isinstance(value, np.ndarray):
            np.testing.assert_allclose(
                getattr(reloaded, name), value, rtol=1e-5, atol=1e-6, err_msg=name
            )


def _mesh_geom(spec: mujoco.MjSpec, body: mujoco.MjsBody, name: str, mass: float):
    """Add to spec a tetrahedron mesh, and a geom of it on body that collides
    with nothing and has mass (kg)."""
    mesh = spec.add_mesh(name=name)
    mesh.uservert = [0, 0, 0, 0.1, 0, 0, 0, 0.1, 0, 0, 0, 0.1]
    body.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=name,
        contype=0,
        conaffinity=0,
        mass=mass,
        pos=[5, 5, 1],
    )


def test_delete_visuals_keeps_a_mesh_with_mass():
    """A mesh geom that collides with nothing but has mass moves its body,
    so delete_visuals keeps it."""
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="body")
    body.add_freejoint()
    _mesh_geom(spec, body, "heavy", mass=0.5)
    _mesh_geom(spec, body, "cosmetic", mass=0.0)
    assets.delete_visuals(spec)
    assert [g.name for g in spec.geoms] == ["heavy"]
    assert [m.name for m in spec.meshes] == ["heavy"]


def test_a_scene_without_visuals_keeps_the_track_meshes():
    """visuals=False changes only the car: a mesh of the track stays, even
    one with no mass that collides with nothing, as a LiDAR ray can meet it."""
    track = mujoco.MjSpec.from_file(os.path.join(assets.TRACK_DIR, "loop_corridor.xml"))
    _mesh_geom(track, track.worldbody, "track_sign", mass=0.0)
    scene = assets.compose(track, visuals=False)
    assert scene.spec.geom("track_sign") is not None
    assert scene.model.nmesh == 1
