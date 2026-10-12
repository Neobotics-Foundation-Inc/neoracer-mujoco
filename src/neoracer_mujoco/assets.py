"""
Locating and loading the car and track XMLs, and putting the car on a track.

assets/ deliberately stays at the repo root (the XML is the product, meant to
be drag-droppable into other MuJoCo tools), so it is not bundled as package
data. Paths here are resolved relative to the source checkout.

Limit: resolves assets/ as three dirs up from this file (repo/src/
neoracer_mujoco/assets.py -> repo). Works from any cwd in a source checkout;
does NOT work if the package is pip-installed into a different tree. Bundle
assets as package data (see pyproject.toml) if that case ever matters.
"""

import glob
import os

import mujoco
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASSET_DIR = os.path.join(_REPO, "assets")
TRACK_DIR = os.path.join(ASSET_DIR, "tracks")
DEFAULT_CAR = os.path.join(ASSET_DIR, "neoracer.xml")

type Vec3 = tuple[float, float, float]


def cars() -> list[str]:
    """Every car XML in assets/, sorted. New XML in assets/ -> discovered here."""
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
    pos: Vec3 = (0.0, 0.0, 0.0),
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
