"""
Laws for the car and track XML files in assets/.
"""

import os
import struct
import tomllib
import zipfile
import zlib
from pathlib import Path

import mujoco
import numpy as np
import pytest
from cases import case
from scenes import assert_car_rests, assert_keeps_car_solver_options

from neoracer_mujoco import assets, sim
from neoracer_mujoco.control.steering import ControlCommand

TRACKS = assets.tracks()


def test_the_car_composes_onto_every_track_and_rests_on_its_floor():
    """Every car composes onto every track with the car's solver options and
    no warning. Each track has its own floor, and the car rests on it:
    compose attaches only the car body, so the floor of neoracer.xml is not
    in the scene."""
    for track in TRACKS:
        with case(os.path.basename(track)):
            for car_path in assets.cars():
                assert_keeps_car_solver_options(
                    mujoco.MjSpec.from_file(track), car_path
                )
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


def test_mujoco_is_pinned_to_one_version():
    """pyproject.toml and requirements.txt name the same exact MuJoCo, and it
    is the one installed: the WebAssembly build gives the same physics only
    at the same version."""
    root = Path(__file__).resolve().parent.parent
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    pins = [d for d in project["dependencies"] if d.startswith("mujoco")]
    requirements = (root / "requirements.txt").read_text().splitlines()
    assert pins == [line for line in requirements if line.startswith("mujoco")]
    assert pins == [f"mujoco=={mujoco.__version__}"]


def test_a_scene_without_visuals_senses_the_same():
    """visuals=False deletes only meshes of the car that collide with
    nothing, so every sensor reads the same, bit for bit, while the car
    drives and steers. It leaves the track alone (the next tests), so one
    track shows it."""
    spec = mujoco.MjSpec.from_file(os.path.join(assets.TRACK_DIR, "loop_corridor.xml"))
    full, bare = assets.compose(spec), assets.compose(spec, visuals=False)
    assert bare.model.nmesh < full.model.nmesh
    readings = []
    for scene in (full, bare):
        data = sim.settle(scene.model)
        sim.apply_command(scene.car, data, ControlCommand(throttle=1.0, steer=0.3))
        steps = []
        for _ in range(round(0.6 / scene.model.opt.timestep)):
            mujoco.mj_step(scene.model, data)
            steps.append(data.sensordata.copy())
        readings.append(np.array(steps))
    np.testing.assert_array_equal(readings[0], readings[1])


def test_a_scene_reloads_from_its_xml():
    """The XML of a scene composed on any track compiles to the same model, up
    to the six significant digits that MuJoCo's XML writer keeps. Without
    visuals, the scene names no mesh file that from_xml_string would have to
    find."""
    for track in TRACKS:
        scene = assets.compose(mujoco.MjSpec.from_file(track), visuals=False)
        reloaded = mujoco.MjModel.from_xml_string(scene.spec.to_xml())
        for name in dir(scene.model):
            value = getattr(scene.model, name)
            if isinstance(value, np.ndarray):
                np.testing.assert_allclose(
                    getattr(reloaded, name),
                    value,
                    rtol=1e-5,
                    atol=1e-6,
                    err_msg=f"{os.path.basename(track)}: {name}",
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


def test_an_mjz_compiles_to_the_model_of_its_xml(tmp_path):
    """For every car and track, write_mjz packs its XML (assets.read_xml) and
    the files it names, so the archive compiles to the same model, bit for
    bit, and writing it again gives the same bytes."""
    first, second = tmp_path / "first.mjz", tmp_path / "second.mjz"
    for path in [*assets.cars(), *TRACKS]:
        name = os.path.basename(path)
        assets.write_mjz(path, first)
        assets.write_mjz(path, second)
        assert first.read_bytes() == second.read_bytes(), name
        expected = assets.from_file(path).compile()
        packed = mujoco.MjSpec.from_zip(str(first)).compile()
        for field in dir(expected):
            value = getattr(expected, field)
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(
                    getattr(packed, field), value, err_msg=f"{name}: {field}"
                )


def test_write_mjz_refuses_an_xml_that_includes_another(tmp_path):
    """MuJoCo reads an included XML into the spec, so an archive of the XML
    alone would not load."""
    (tmp_path / "world.xml").write_text("<mujoco><worldbody/></mujoco>")
    model = tmp_path / "model.xml"
    model.write_text('<mujoco><include file="world.xml"/></mujoco>')
    mujoco.MjSpec.from_file(str(model))  # a valid model
    with pytest.raises(ValueError, match="includes"):
        assets.write_mjz(model, tmp_path / "model.mjz")


def _png(path: Path) -> None:
    """Write a 2x2 grey PNG at path."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    rows = b"".join(b"\x00" + b"\x80" * 6 for _ in range(2))  # 2 RGB pixels a row
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


def test_an_mjz_holds_the_six_images_of_a_cube_texture(tmp_path):
    """A cube texture names its images in fileleft, fileright and so on,
    which MjSpec keeps in cubefiles, not in file."""
    sides = ["left", "right", "up", "down", "front", "back"]
    (tmp_path / "sky").mkdir()
    for side in sides:
        _png(tmp_path / "sky" / f"{side}.png")
    files = " ".join(f'file{side}="sky/{side}.png"' for side in sides)
    model = tmp_path / "model.xml"
    model.write_text(
        f'<mujoco><asset><texture name="sky" type="cube" {files}/></asset></mujoco>'
    )
    out = tmp_path / "model.mjz"
    assets.write_mjz(model, out)
    names = set(zipfile.ZipFile(out).namelist())
    assert names == {assets.MJZ_MODEL, *(f"sky/{side}.png" for side in sides)}
    mujoco.MjSpec.from_zip(str(out)).compile()
