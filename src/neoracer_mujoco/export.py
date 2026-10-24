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

import mujoco

from neoracer_spec.spec import Spec

from . import assets

XML_NAME = os.path.basename(assets.DEFAULT_CAR)
MJZ_NAME = "neoracer.mjz"


def write(car: Spec, folder: str | os.PathLike[str]) -> None:
    """Write to folder the car's XML (XML_NAME), the files it names, and an
    archive of them (MJZ_NAME, assets.write_mjz) that MjSpec.from_zip and
    MuJoCo's WebAssembly build load. folder must not be assets/, whose
    template the car's XML is made from."""
    xml = assets.write_car(car_xml(car), XML_NAME, folder)
    assets.write_mjz(xml, os.path.join(folder, MJZ_NAME))


def car_xml(car: Spec) -> str:
    """The car's MJCF: the NeoRacer's XML with car's numbers."""
    return assets.neoracer_xml(car)


def spec(car: Spec) -> mujoco.MjSpec:
    """The car's MjSpec. Its meshes are read from assets/."""
    s = mujoco.MjSpec.from_string(car_xml(car))
    s.modelfiledir = assets.ASSET_DIR
    return s
