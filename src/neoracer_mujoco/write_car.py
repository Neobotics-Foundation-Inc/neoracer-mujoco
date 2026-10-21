"""
Write the cars into a folder of car files: the numbers of the NeoRacer
(neoracer_spec.neoracer.NEORACER) and the RACECAR Neo V2
(neoracer_spec.racecar_neo_v2.RACECAR_NEO_V2) into their XMLs there, in
place. Run it after a change to a car, with the TypeScript generator for the
JS package:

    python -m neoracer_mujoco.write_car assets
    python -m neoracer_spec.typescript js/src/spec.ts
"""

import os
import sys
from collections.abc import Callable

from neoracer_spec import derive_v2, mjcf
from neoracer_spec.derive import derive
from neoracer_spec.neoracer import NEORACER
from neoracer_spec.racecar_neo_v2 import RACECAR_NEO_V2

from . import export

V2_XML_NAME = "racecar_neo_v2.xml"


def _rewrite(path: str, write: Callable[[str], str]) -> None:
    with open(path, encoding="utf-8") as f:
        text = write(f.read())
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


if __name__ == "__main__":
    (folder,) = sys.argv[1:]
    _rewrite(
        os.path.join(folder, export.XML_NAME),
        lambda xml: mjcf.car_xml(derive(NEORACER).model, xml),
    )
    _rewrite(
        os.path.join(folder, V2_XML_NAME),
        lambda xml: mjcf.racecar_neo_v2_xml(derive_v2.model(RACECAR_NEO_V2), xml),
    )
