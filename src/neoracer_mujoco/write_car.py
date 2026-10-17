"""
Write the NeoRacer (neoracer_spec.neoracer.NEORACER) into a folder of car
files: its numbers into the car XML there, in place, and its settings beside
the model as JSON. Run it after a change to the car:

    python -m neoracer_mujoco.write_car assets
"""

import os
import sys

from neoracer_spec import mjcf
from neoracer_spec.derive import derive
from neoracer_spec.neoracer import NEORACER

from . import export

if __name__ == "__main__":
    (folder,) = sys.argv[1:]
    xml = os.path.join(folder, export.XML_NAME)
    with open(xml, encoding="utf-8") as f:
        text = mjcf.car_xml(derive(NEORACER).model, f.read())
    with open(xml, "w", encoding="utf-8") as f:
        f.write(text)
    with open(os.path.join(folder, export.SETTINGS_NAME), "w", encoding="utf-8") as f:
        f.write(export.settings_json(NEORACER))
