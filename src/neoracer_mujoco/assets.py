"""
Locating and loading the car and track XMLs, and putting the car on a track.

assets/ stays at the repo root, so that users can open the XML files in
other MuJoCo tools. assets/ is not package data, so this module cannot load
it with importlib.resources (AGENTS.md rule 10). It finds assets/ from the
path of this file in the source checkout.

Limit: this module finds assets/ three directories above this file
(repo/src/neoracer_mujoco/assets.py -> repo). This works from any working
directory in a source checkout. It does not work if pip installs the package
in a different tree. In that case, make assets/ package data (see
pyproject.toml) and load it with importlib.resources.
"""

import glob
import os

import mujoco
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASSET_DIR = os.path.join(_REPO, "assets")
TRACK_DIR = os.path.join(ASSET_DIR, "tracks")
DEFAULT_CAR = os.path.join(ASSET_DIR, "neoracer.xml")

type Position = tuple[float, float, float]  # m, world frame


def cars() -> list[str]:
    """Every car XML in assets/, sorted. This function finds a new car XML in
    assets/ automatically."""
    return sorted(glob.glob(os.path.join(ASSET_DIR, "*.xml")))


def tracks() -> list[str]:
    """Every track XML in assets/tracks/, sorted. A track is scenery only: no
    car, so it is not in cars() or the car conformance tests."""
    return sorted(glob.glob(os.path.join(TRACK_DIR, "*.xml")))


def load(path: str | os.PathLike[str] = DEFAULT_CAR) -> mujoco.MjModel:
    """Compile a car XML into an MjModel. Defaults to assets/neoracer.xml."""
    return mujoco.MjModel.from_xml_path(str(path))


def compose(
    track: mujoco.MjSpec,
    car_path: str | os.PathLike[str] = DEFAULT_CAR,
    pos: Position = (0.0, 0.0, 0.0),
) -> mujoco.MjModel:
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
    return scene.compile()
