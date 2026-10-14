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
) -> Scene:
    """Compile the track with the car's "car" body attached at pos (m),
    facing +X. track is not changed: the car is attached to a copy.

    Only the car body is attached, so the track keeps its own floor and
    <option>, and the track XML never names the car file.

        compose(mujoco.MjSpec.from_file(os.path.join(TRACK_DIR, "f1_track.xml")))
    """
    scene = track.copy()
    car = mujoco.MjSpec.from_file(str(car_path))
    frame = scene.worldbody.add_frame()
    frame.pos = np.array(pos)
    frame.attach_body(car.body("car"), "", "")
    return compile_scene(scene)
