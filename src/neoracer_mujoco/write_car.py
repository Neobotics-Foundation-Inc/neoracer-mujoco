"""
Write every car of assets/ into a folder, complete: its XML with the numbers
of its spec (assets.read_xml), and the files that it names. MuJoCo tools
other than this package then open the cars from there:

    python -m neoracer_mujoco.write_car cars
    python -m mujoco.viewer --mjcf=cars/neoracer.xml
"""

import os
import shutil
import sys

from . import assets


def write(folder: str | os.PathLike[str]) -> None:
    """Write each car of assets.cars() into folder, which must not be
    assets/, the folder of the cars' templates."""
    if os.path.realpath(folder) == os.path.realpath(assets.ASSET_DIR):
        raise ValueError(f"{folder} is assets/, which holds the templates")
    os.makedirs(folder, exist_ok=True)
    for path in assets.cars():
        with open(
            os.path.join(folder, os.path.basename(path)), "w", encoding="utf-8"
        ) as f:
            f.write(assets.read_xml(path))
        for name in assets.named_files(path):
            target = os.path.join(folder, name)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copyfile(os.path.join(assets.ASSET_DIR, name), target)


if __name__ == "__main__":
    (folder,) = sys.argv[1:]
    write(folder)
