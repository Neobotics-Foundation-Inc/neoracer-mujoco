"""
A car's model for the simulators: its XML, an MjSpec of it, and a folder of
files that MuJoCo without Python loads, the browser's WebAssembly build
included.

The XML is assets/neoracer.xml with the numbers that the car's parts give
(mjcf.car_xml), so a variant of the NeoRacer is a car of its own:

    car = dataclasses.replace(NEORACER, battery=Battery(7.4))
    scene = compose(track, spec(car))     # CPU, or a batch on the GPU
    write(car, "dist/slow")               # dist/slow/neoracer.mjz for the web
"""

import os
import shutil

import mujoco

from .. import assets
from . import derive, mjcf
from .parts import Car

XML_NAME = "neoracer.xml"
MJZ_NAME = "neoracer.mjz"


def car_xml(car: Car) -> str:
    """The car's MJCF: the NeoRacer's XML with car's numbers."""
    with open(assets.DEFAULT_CAR, encoding="utf-8") as f:
        return mjcf.car_xml(derive.model(car), f.read())


def spec(car: Car) -> mujoco.MjSpec:
    """The car's MjSpec. Its meshes are read from assets/."""
    s = mujoco.MjSpec.from_string(car_xml(car))
    s.modelfiledir = assets.ASSET_DIR
    return s


def write(car: Car, folder: str | os.PathLike[str]) -> None:
    """Write to folder the car's XML (XML_NAME), the mesh files it names, and
    an archive of them (MJZ_NAME, assets.write_mjz) that MjSpec.from_zip and
    MuJoCo's WebAssembly build load."""
    xml = os.path.join(folder, XML_NAME)
    os.makedirs(folder, exist_ok=True)
    with open(xml, "w", encoding="utf-8") as f:
        f.write(car_xml(car))
    for mesh in spec(car).meshes:
        source = os.path.join(assets.ASSET_DIR, mesh.file)
        target = os.path.join(folder, mesh.file)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copyfile(source, target)
    assets.write_mjz(xml, os.path.join(folder, MJZ_NAME))
