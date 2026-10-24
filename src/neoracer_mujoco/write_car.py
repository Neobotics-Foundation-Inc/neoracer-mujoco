"""
Write every car of assets/ into a folder, complete: its XML with the numbers
of its spec (assets.read_xml), and the files that it names. MuJoCo tools
other than this package then open the cars from there:

    python -m neoracer_mujoco.write_car cars
    python -m mujoco.viewer --mjcf=cars/neoracer.xml
"""

import os
import sys

from . import assets


def write(folder: str | os.PathLike[str]) -> None:
    """Write each car of assets.cars() into folder, which must not be
    assets/, the folder of the cars' templates."""
    for path in assets.cars():
        assets.write_car(assets.read_xml(path), os.path.basename(path), folder)


if __name__ == "__main__":
    (folder,) = sys.argv[1:]
    write(folder)
