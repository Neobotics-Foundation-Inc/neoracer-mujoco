"""
Write the numbers that hardware.py derives from the car's hardware into a car
XML, in place:

    python -m neoracer_mujoco.write_car_xml assets/neoracer.xml
"""

import sys

from . import hardware

if __name__ == "__main__":
    (path,) = sys.argv[1:]
    with open(path) as f:
        text = hardware.car_xml(hardware.Hardware(), f.read())
    with open(path, "w") as f:
        f.write(text)
