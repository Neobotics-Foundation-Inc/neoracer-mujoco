"""
Locating and loading the car and track XMLs, and putting the car on a track.

The XMLs and meshes live in assets/ at the repo root, so that users can open
them in other MuJoCo tools. A wheel copies assets/ into the package as
neoracer_mujoco/data (see pyproject.toml). In a source checkout or an
editable install, the package has no data/ folder, and this module uses the
repo's assets/ instead.
"""

import glob
import os
import re
import zipfile
from importlib import resources

import mujoco
import numpy as np

from .scene import Scene, compile_scene

PACKAGED_DIR = "data"  # the name of assets/ inside an installed wheel

_package = resources.files("neoracer_mujoco")
_packaged = _package / PACKAGED_DIR
# In a source checkout the package is repo/src/neoracer_mujoco.
_checkout = os.path.normpath(os.path.join(str(_package), "..", "..", "assets"))
ASSET_DIR = str(_packaged) if _packaged.is_dir() else _checkout
TRACK_DIR = os.path.join(ASSET_DIR, "tracks")
DEFAULT_CAR = os.path.join(ASSET_DIR, "neoracer.xml")
MJZ_MODEL = "model.xml"  # the name of the XML in an archive of write_mjz

type Position = tuple[float, float, float]  # m, world frame


def cars() -> list[str]:
    """Every car XML in assets/, sorted. This function finds a new car XML in
    assets/ automatically."""
    return sorted(glob.glob(os.path.join(ASSET_DIR, "*.xml")))


def tracks() -> list[str]:
    """Every track XML in assets/tracks/, sorted. A track is scenery only: no
    car, so it is not in cars() or the car conformance tests. It puts every
    wall in the body named "walls" (truth.WALLS_BODY)."""
    return sorted(glob.glob(os.path.join(TRACK_DIR, "*.xml")))


def load(path: str | os.PathLike[str] = DEFAULT_CAR) -> Scene:
    """Compile a car XML into a Scene. Defaults to assets/neoracer.xml."""
    return compile_scene(mujoco.MjSpec.from_file(str(path)))


def compose(
    track: mujoco.MjSpec,
    car_path: str | os.PathLike[str] = DEFAULT_CAR,
    pos: Position = (0.0, 0.0, 0.0),
    visuals: bool = True,
) -> Scene:
    """Compile the track with the car's "car" body attached at pos (m),
    facing +X. track is not changed: the car is attached to a copy.

    Only the car body is attached, so the track keeps its own floor and
    <option>, and the track XML never names the car file.

    With visuals=False, the car has none of the meshes that only show it
    (see delete_visuals), and the model is much smaller and steps faster.
    The track keeps all of its geoms. Use it where nothing renders the
    scene, as in a batch.

        compose(mujoco.MjSpec.from_file(os.path.join(TRACK_DIR, "f1_track.xml")))
    """
    scene = track.copy()
    car = mujoco.MjSpec.from_file(str(car_path))
    frame = scene.worldbody.add_frame()
    frame.pos = np.array(pos)
    if not visuals:
        delete_visuals(car)
    frame.attach_body(car.body("car"), "", "")
    return compile_scene(scene)


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


def write_mjz(xml_path: str | os.PathLike[str], out: str | os.PathLike[str]) -> None:
    """Write to out an .mjz archive of the MJCF file at xml_path: the file as
    written, as model.xml, and every mesh and texture file that it names, at
    the path that it names them by. It is the XML as written, not
    MjSpec.to_xml(), which keeps six significant digits, so the archive
    compiles to the same model, bit for bit.

    MuJoCo reads the archive with MjSpec.from_zip(out) in Python, and as
    "out/model.xml" in its WebAssembly build. The same files always give the
    same bytes. Raise ValueError for an XML that includes another XML, or
    names a height field or skin file, or a file outside its folder: none of
    ours does, and the archive would miss them."""
    # MjSpec.from_file reads an <include>d file into the spec, so only the
    # XML itself shows one, and the archive would not hold it. MuJoCo's
    # parser accepts "--" in a comment, which Python's XML parsers refuse.
    with open(xml_path, encoding="utf-8") as f:
        text = re.sub(r"<!--.*?-->", "", f.read(), flags=re.DOTALL)
    if re.search(r"<include\b", text):
        raise ValueError(f"{xml_path} includes another XML file")
    spec = mujoco.MjSpec.from_file(str(xml_path))
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
    folder = os.path.dirname(os.path.abspath(xml_path))
    sources = [(MJZ_MODEL, str(xml_path))]
    sources += [(name, os.path.join(folder, name)) for name in files]
    with zipfile.ZipFile(out, "w") as archive:
        for name, source in sources:
            # A fixed date, so that the bytes depend on the files only.
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            with open(source, "rb") as f:
                archive.writestr(info, f.read())
