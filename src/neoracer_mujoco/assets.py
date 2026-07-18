"""
Locating and loading the car XML.

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


def load(path=None):
    """Compile a car XML into an MjModel. Defaults to assets/neoracer.xml."""
    return mujoco.MjModel.from_xml_path(str(path or DEFAULT_CAR))
