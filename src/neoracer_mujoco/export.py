"""
A car's model for the simulators: its XML, an MjSpec of it, and a folder
of files that MuJoCo without Python loads, the browser's WebAssembly build
included. The settings beside the model are derive(car) (neoracer_spec).

The XML is assets/neoracer.xml with the numbers that the car's parts give
(neoracer_spec.mjcf.car_xml), so a variant of the NeoRacer is a car of its
own:

    car = at(NEORACER, lambda c: c.battery.voltage).set(chosen(7.4, "slower"))
    scene = compose(track, spec(car))     # CPU, or a batch on the GPU
    write(car, "dist/slow")               # dist/slow/neoracer.mjz for the web
"""

import os
import shutil

import mujoco

from neoracer_spec import mjcf
from neoracer_spec.derive import derive
from neoracer_spec.spec import Spec

from . import assets

XML_NAME = "neoracer.xml"
MJZ_NAME = "neoracer.mjz"


def write(car: Spec, folder: str | os.PathLike[str]) -> None:
    """Write to folder the car's XML (XML_NAME), the mesh files it names, an
    archive of them (MJZ_NAME, assets.write_mjz) that MjSpec.from_zip and
    MuJoCo's WebAssembly build load. folder must not be assets/, whose XML
    the car's is made from."""
    if os.path.realpath(folder) == os.path.realpath(assets.ASSET_DIR):
        raise ValueError(f"{folder} is assets/, the NeoRacer's own folder")
    text = car_xml(car)
    xml = os.path.join(folder, XML_NAME)
    os.makedirs(folder, exist_ok=True)
    with open(xml, "w", encoding="utf-8") as f:
        f.write(text)
    for mesh in spec(car).meshes:
        source = os.path.join(assets.ASSET_DIR, mesh.file)
        target = os.path.join(folder, mesh.file)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copyfile(source, target)
    assets.write_mjz(xml, os.path.join(folder, MJZ_NAME))


def car_xml(car: Spec) -> str:
    """The car's MJCF: the NeoRacer's XML with car's numbers."""
    with open(assets.DEFAULT_CAR, encoding="utf-8") as f:
        return mjcf.car_xml(derive(car).model, f.read())


def spec(car: Spec) -> mujoco.MjSpec:
    """The car's MjSpec. Its meshes are read from assets/."""
    s = mujoco.MjSpec.from_string(car_xml(car))
    s.modelfiledir = assets.ASSET_DIR
    return s
