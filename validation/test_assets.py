"""
Laws for the car and track XML files in assets/.
"""

import os
import tomllib
import zipfile
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
        for motor in scene.car.drive:
            data.bind_scalar(motor).ctrl = 0.3  # N·m
        data.bind_scalar(scene.car.steer).ctrl = 0.3  # rad
        steps = []
        for _ in range(300):  # 0.6 s
            mujoco.mj_step(scene.model, data)
            steps.append(data.sensordata.copy())
        readings.append(np.array(steps))
    np.testing.assert_array_equal(readings[0], readings[1])


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


@pytest.mark.parametrize("path", [*assets.cars(), *TRACKS], ids=os.path.basename)
def test_an_mjz_compiles_to_the_model_of_its_xml(path, tmp_path):
    """write_mjz packs the XML as written and the files it names, so the
    archive compiles to the same model, bit for bit, and writing it again
    gives the same bytes."""
    first, second = tmp_path / "first.mjz", tmp_path / "second.mjz"
    assets.write_mjz(path, first)
    assets.write_mjz(path, second)
    assert first.read_bytes() == second.read_bytes()
    expected = mujoco.MjSpec.from_file(path).compile()
    packed = mujoco.MjSpec.from_zip(str(first)).compile()
    for name in dir(expected):
        value = getattr(expected, name)
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(getattr(packed, name), value, err_msg=name)


def test_the_car_mjz_holds_the_xml_and_its_meshes(tmp_path):
    out = tmp_path / "neoracer.mjz"
    assets.write_mjz(assets.DEFAULT_CAR, out)
    meshes = {m.file for m in mujoco.MjSpec.from_file(assets.DEFAULT_CAR).meshes}
    assert set(zipfile.ZipFile(out).namelist()) == {assets.MJZ_MODEL, *meshes}
