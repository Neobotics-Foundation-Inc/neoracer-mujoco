"""
Locating and loading the car XML, and putting the car on a track.

assets/ deliberately stays at the repo root (the XML is the product, meant to
be drag-droppable into other MuJoCo tools), so it is not bundled as package
data. Paths here are resolved relative to the source checkout.

ponytail: resolves assets/ as three dirs up from this file (repo/src/
neoracer_mujoco/assets.py -> repo). Works from any cwd in a source checkout;
does NOT work if the package is pip-installed into a different tree. Bundle
assets as package data (see pyproject.toml) if that case ever matters.
"""

import glob
import os

import mujoco

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASSET_DIR = os.path.join(_REPO, "assets")
DEFAULT_CAR = os.path.join(ASSET_DIR, "neoracer.xml")


def cars():
    """Every car XML in assets/, sorted. New XML in assets/ -> discovered here."""
    return sorted(glob.glob(os.path.join(ASSET_DIR, "*.xml")))


def tracks():
    """Every hand-written track XML in assets/tracks/, sorted. These are scenery
    only (no car), so they are deliberately kept out of cars() and the car
    conformance battery -- they have no actuators or sensors to conform to.
    compose() resolves bare names against this list."""
    return sorted(glob.glob(os.path.join(ASSET_DIR, "tracks", "*.xml")))


def load(path=None):
    """Compile a car XML into an MjModel. Defaults to assets/neoracer.xml."""
    return mujoco.MjModel.from_xml_path(str(path or DEFAULT_CAR))


def _track_spec(track) -> mujoco.MjSpec:
    """MjSpec for a Track, MJCF text, a path, or a name in assets/tracks/."""
    from .track_generation import (  # local: assets/ must not need the track pkg
        Track,
        to_mjcf,
    )

    if isinstance(track, Track):
        return mujoco.MjSpec.from_string(to_mjcf(track))
    if "<mujoco" in track:
        return mujoco.MjSpec.from_string(track)
    if os.path.exists(track):
        return mujoco.MjSpec.from_file(str(track))
    for path in tracks():
        if os.path.splitext(os.path.basename(path))[0] == track:
            return mujoco.MjSpec.from_file(path)
    raise FileNotFoundError(f"no track XML, path, or MJCF text matching {track!r}")


def compose(track, car=None) -> mujoco.MjModel:
    """Compile a scene: a track with the car attached at the origin facing +X.

        model = compose("straight_corridor")            # a hand-written track
        model = compose(generate_track(seed=3))         # a generated one

    track  a Track, MJCF text, a path to a track XML, or the bare name of one
           in assets/tracks/.
    car    path to a car XML. Defaults to assets/neoracer.xml.

    Attaching rather than referencing is what keeps track XMLs independent of
    the car model: only the car's "car" body is grafted in, so the car file
    carries its own meshdir and the track never names it.
    """
    scene = _track_spec(track)
    car_spec = mujoco.MjSpec.from_file(str(car or DEFAULT_CAR))
    scene.worldbody.add_frame().attach_body(car_spec.body("car"), "", "")
    return scene.compile()
