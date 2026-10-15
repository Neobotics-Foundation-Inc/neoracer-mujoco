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
    # The car's root default class is "main", as the track's is. Attached
    # with no prefix, it keeps that name, and to_xml writes it as a nested
    # <default> with no class, which the XML parser rejects. A name of its
    # own makes the scene's XML load again.
    car.default.name = "car"
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
