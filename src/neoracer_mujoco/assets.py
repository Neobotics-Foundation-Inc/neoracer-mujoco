"""
Locating and loading the car and track XMLs, and putting cars on a track.

The XMLs and meshes live in assets/ at the repo root. A wheel copies assets/
into the package as neoracer_mujoco/data (see pyproject.toml). In a source
checkout or an editable install, the package has no data/ folder, and this
module uses the repo's assets/ instead.

The NeoRacer's and the RACECAR Neo V2's XMLs in assets/ are templates: they
hold none of the numbers that the cars' specs give (neoracer_spec), and
compile only after read_xml adds them. Load a car with load, compose or
from_file, not with mujoco.MjSpec.from_file. python -m
neoracer_mujoco.write_car writes the complete XMLs, for other MuJoCo tools.
"""

import glob
import os
import re
import shutil
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import resources

import mujoco
import numpy as np

from neoracer_spec import derive_v2, mjcf
from neoracer_spec.derive import derive
from neoracer_spec.neoracer import NEORACER
from neoracer_spec.racecar_neo_v2 import RACECAR_NEO_V2
from neoracer_spec.spec import Spec

from . import contract
from .scene import Scene, compile_scene

PACKAGED_DIR = "data"
"""The name of assets/ inside an installed wheel."""

_package = resources.files("neoracer_mujoco")
_packaged = _package / PACKAGED_DIR
_checkout = os.path.normpath(os.path.join(str(_package), "..", "..", "assets"))
"""The repo's assets/ in a source checkout, where the package is
repo/src/neoracer_mujoco."""
ASSET_DIR = str(_packaged) if _packaged.is_dir() else _checkout
TRACK_DIR = os.path.join(ASSET_DIR, "tracks")
DEFAULT_CAR = os.path.join(ASSET_DIR, "neoracer.xml")
MJZ_MODEL = "model.xml"
"""The name of the XML in an archive of write_mjz."""
START_SITE = "start"
"""The site of a track where attach_car puts a car."""


def cars() -> list[str]:
    """Every car XML in assets/, sorted. This function finds a new car XML in
    assets/ automatically."""
    return sorted(glob.glob(os.path.join(ASSET_DIR, "*.xml")))


def tracks() -> list[str]:
    """Every track XML in assets/tracks/, sorted. A track is scenery only: no
    car, so it is not in cars() or the car conformance tests. It puts every
    wall in the body named "walls" (truth.WALLS_BODY)."""
    return sorted(glob.glob(os.path.join(TRACK_DIR, "*.xml")))


def read_xml(path: str | os.PathLike[str]) -> str:
    """The MJCF of the XML at path: the file's text, with the numbers of the
    car's spec added if it is the NeoRacer's or the RACECAR Neo V2's
    template in assets/."""
    completed = _completed(path)
    return _read(path) if completed is None else completed


def from_file(path: str | os.PathLike[str] = DEFAULT_CAR) -> mujoco.MjSpec:
    """The MjSpec of the XML at path. A car's template in assets/ gets the
    numbers of its spec (read_xml); MuJoCo reads any other XML as
    MjSpec.from_file does, with the files that it includes."""
    completed = _completed(path)
    if completed is None:
        return mujoco.MjSpec.from_file(str(path))
    spec = mujoco.MjSpec.from_string(completed)
    spec.modelfiledir = os.path.dirname(os.path.abspath(path))
    return spec


def _completed(path: str | os.PathLike[str]) -> str | None:
    """The car's template at path with the numbers of its spec, or None if
    path is not a template in assets/."""
    if os.path.realpath(os.path.dirname(path)) != os.path.realpath(ASSET_DIR):
        return None
    match os.path.basename(path):
        case "neoracer.xml":
            return neoracer_xml(NEORACER)
        case "racecar_neo_v2.xml":
            p = derive_v2.model(RACECAR_NEO_V2)
            return mjcf.racecar_neo_v2_xml(p, _read(path))
        case _:
            return None


def neoracer_xml(car: Spec) -> str:
    """The MJCF of a NeoRacer: assets/neoracer.xml with the numbers of car."""
    return mjcf.car_xml(derive(car).model, _read(DEFAULT_CAR))


def _read(path: str | os.PathLike[str]) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def load(path: str | os.PathLike[str] = DEFAULT_CAR) -> Scene:
    """Compile a car XML into a Scene. Defaults to assets/neoracer.xml."""
    return compile_scene(from_file(path))


def compose(
    track: mujoco.MjSpec,
    car: str | os.PathLike[str] | mujoco.MjSpec = DEFAULT_CAR,
    visuals: bool = True,
) -> Scene:
    """Compile the track with the car attached at the track's start
    (attach_car), the car's elements keeping their names. The car is a car
    XML's path or a car's MjSpec, such as export.spec(car) for a variant.
    Neither track nor a car's MjSpec is changed: the car is attached from
    copies.

    Only the car body is attached, so the track keeps its own floor, and
    the track XML never names the car file. The scene takes the car's
    <option>, the solver's settings that its model is made for, so a track
    sets none.

    With visuals=False, the car has none of the meshes that only show it
    (see delete_visuals), and the model is much smaller and steps faster.
    The track keeps all of its geoms. Use it where nothing renders the
    scene, as in a batch.

        compose(mujoco.MjSpec.from_file(os.path.join(TRACK_DIR, "f1_track.xml")))
    """
    return compose_cars(track, (Entry("", car=car),), visuals)


@dataclass(frozen=True)
class Entry:
    """A car of a scene of several (compose_cars)."""

    prefix: str
    """What the names of the car's elements start with in the scene: its
    motor is prefix + "motor"."""
    offset: tuple[float, float] = (0.0, 0.0)
    """Where the car starts (m), from the track's start, in its frame: +X
    forward, +Y left."""
    heading: float = 0.0
    """Which way the car faces (rad), turned about +Z from the start's +X,
    + to the left."""
    car: str | os.PathLike[str] | mujoco.MjSpec = DEFAULT_CAR
    """A car XML's path or a car's MjSpec, as compose takes."""


def compose_cars(
    track: mujoco.MjSpec, entries: Sequence[Entry], visuals: bool = True
) -> Scene:
    """Compile the track with each entry's car attached (attach_car), in
    order: scene.cars[k] is the car of entries[k]. Neither track nor a car's
    MjSpec is changed, and visuals is as for compose. Raise ValueError if an
    entry's prefix starts another's (scene.compile_scene).

        grid = [Entry("a/"), Entry("b/", offset=(-0.6, 0.0), heading=0.1)]
        scene = compose_cars(track, grid, visuals=False)
    """
    scene = track.copy()
    for entry in entries:
        attach_car(scene, entry.car, entry.prefix, entry.offset, entry.heading, visuals)
    return compile_scene(scene, [entry.prefix for entry in entries])


def attach_car(
    spec: mujoco.MjSpec,
    car: str | os.PathLike[str] | mujoco.MjSpec = DEFAULT_CAR,
    prefix: str = "",
    offset: tuple[float, float] = (0.0, 0.0),
    heading: float = 0.0,
    visuals: bool = True,
) -> None:
    """Attach the car's root body (contract.CAR_BODY) to spec, with the names
    of its elements behind prefix, at the track's start: its site
    START_SITE, facing the site's +X, or the origin, facing +X, if it has
    none, moved by offset (m) in that frame, +X forward and +Y left, and
    turned by heading (rad) about +Z, + to the left. The site must be on the
    world body, as the car's free joint must. A car's MjSpec is not changed:
    the car is attached from a copy. spec takes the car's <option>, and
    visuals is as for compose."""
    car = car.copy() if isinstance(car, mujoco.MjSpec) else from_file(car)
    spec.option = car.option
    start = spec.site(START_SITE)
    if start is not None and start.parent != spec.worldbody:
        raise ValueError(
            f"the track's {START_SITE} site is on the body {start.parent.name},"
            " not on the world body, where the car's free joint must be"
        )
    mount = start if start is not None else spec.worldbody.add_frame()
    if not visuals:
        delete_visuals(car)
    # The car's root default class is "main", as the track's is. Attached
    # with no prefix, it keeps that name, and to_xml writes it as a nested
    # <default> with no class, which the XML parser rejects. A name of its
    # own makes the scene's XML load again.
    car.default.name = "car"
    body = mount.attach_body(car.body(contract.CAR_BODY), prefix, "")
    # attach_body puts the body in a frame at the mount, so its pos and quat
    # are in the start's frame.
    body.pos = np.asarray(body.pos) + [*offset, 0.0]
    turn = np.array([np.cos(heading / 2), 0.0, 0.0, np.sin(heading / 2)])
    quat = np.empty(4)
    mujoco.mju_mulQuat(quat, turn, np.asarray(body.quat, dtype=np.float64))
    body.quat = quat


def delete_visuals(spec: mujoco.MjSpec) -> None:
    """Delete from spec every mesh geom that has no mass and collides with
    nothing, and the meshes that no geom uses after that. In neoracer.xml
    these geoms only show the car, and their collision trees are most of the
    compiled model: 33 of its 35 MB on loop_corridor.xml.

    Limit: a rangefinder sees a geom whatever its collision masks. No LiDAR
    ray of neoracer.xml meets these meshes, which a test checks on every
    track. For another car, check its sensors the same way."""
    for geom in list(spec.geoms):
        cosmetic = geom.contype == 0 and geom.conaffinity == 0 and geom.mass == 0
        if geom.type == mujoco.mjtGeom.mjGEOM_MESH and cosmetic:
            spec.delete(geom)
    used = {geom.meshname for geom in spec.geoms}
    for mesh in list(spec.meshes):
        if mesh.name not in used:
            spec.delete(mesh)


def named_files(xml_path: str | os.PathLike[str]) -> list[str]:
    """Every mesh and texture file that the MJCF file at xml_path names,
    sorted, at the path that it names them by, relative to its folder.
    Raise ValueError for an XML that includes another XML, or names a
    height field or skin file, or a file outside its folder: none of ours
    does, and a copy of the XML and these files would miss them."""
    # MuJoCo reads an <include>d file into the spec, so only the XML itself
    # shows one. MuJoCo's parser accepts "--" in a comment, which Python's
    # XML parsers refuse.
    xml = re.sub(r"<!--.*?-->", "", read_xml(xml_path), flags=re.DOTALL)
    if re.search(r"<include\b", xml):
        raise ValueError(f"{xml_path} includes another XML file")
    spec = from_file(xml_path)
    if any(f.file for f in (*spec.hfields, *spec.skins)):
        raise ValueError(f"{xml_path} names a height field or skin file")
    named = [os.path.join(spec.compiler.meshdir, m.file) for m in spec.meshes]
    # A cube texture names its six images in cubefiles, not file.
    textures = [f for t in spec.textures for f in (t.file, *t.cubefiles)]
    named += [os.path.join(spec.compiler.texturedir, f) for f in textures]
    files = sorted({os.path.normpath(n) for n in named if os.path.basename(n)})
    for name in files:
        if os.path.isabs(name) or name.startswith(".."):
            raise ValueError(f"{xml_path} names {name}, outside its folder")
    return files


def write_car(xml: str, name: str, folder: str | os.PathLike[str]) -> str:
    """Write the MJCF xml to folder as name, with every file that it names
    (named_files) copied from assets/, and return the XML's path. folder
    must not be assets/, which holds the cars' templates."""
    if os.path.realpath(folder) == os.path.realpath(ASSET_DIR):
        raise ValueError(f"{folder} is assets/, which holds the templates")
    path = os.path.join(folder, name)
    os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(xml)
    for file in named_files(path):
        target = os.path.join(folder, file)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copyfile(os.path.join(ASSET_DIR, file), target)
    return path


def write_mjz(xml_path: str | os.PathLike[str], out: str | os.PathLike[str]) -> None:
    """Write to out an .mjz archive of the MJCF file at xml_path: its text,
    read_xml(xml_path), as model.xml, and its named_files. It is that text,
    not MjSpec.to_xml(), which keeps six significant digits, so the archive
    compiles to the same model, bit for bit.

    MuJoCo reads the archive with MjSpec.from_zip(out) in Python, and as
    "out/model.xml" in its WebAssembly build. The same files always give the
    same bytes."""
    folder = os.path.dirname(os.path.abspath(xml_path))
    contents = [(MJZ_MODEL, read_xml(xml_path).encode("utf-8"))]
    for name in named_files(xml_path):
        with open(os.path.join(folder, name), "rb") as f:
            contents.append((name, f.read()))
    with zipfile.ZipFile(out, "w") as archive:
        for name, data in contents:
            # A fixed date, so that the bytes depend on the files only.
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
