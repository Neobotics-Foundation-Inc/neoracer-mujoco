"""
Write the NeoRacer's model numbers (twin.derive.model of twin.neoracer's
NEORACER) into a car XML, in place:

    python -m neoracer_mujoco.write_car_xml assets/neoracer.xml
"""

import sys

from .twin import derive, mjcf
from .twin.neoracer import NEORACER

if __name__ == "__main__":
    (path,) = sys.argv[1:]
    with open(path) as f:
        text = mjcf.car_xml(derive.model(NEORACER), f.read())
    with open(path, "w") as f:
        f.write(text)
