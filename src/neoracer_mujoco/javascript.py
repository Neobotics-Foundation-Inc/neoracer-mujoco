"""
The JS package's generated sources, js/src/generated/, written from this
package so that the JS package holds no copy of it:

    spec.ts     the spec's types, NEORACER and NEORACER_PARAMS
                (neoracer_spec.typescript)
    shared.ts   the names and constants that both packages use
    car.ts      the NeoRacer's MJCF, as CAR_XML

They are not checked in. Write them before building the JS package, from
the repository root:

    python -m neoracer_mujoco.javascript js/src/generated
"""

import json
import os
import sys
from collections.abc import Mapping
from types import MappingProxyType

from neoracer_spec import typescript
from neoracer_spec.neoracer import NEORACER

from . import contract, encoder, export, lidar, sensors, sim

HEADER = "// Written by python -m neoracer_mujoco.javascript. Do not edit.\n"

# The constants of shared.ts: the parts of the car that code finds by name,
# and the numbers that both packages need to agree on.
SHARED: Mapping[str, object] = MappingProxyType(
    {
        "CAR_BODY": contract.CAR_BODY,
        "DRIVE_ACTUATOR": contract.DRIVE_ACTUATOR,
        "STEER_ACTUATOR": contract.STEER_ACTUATOR,
        "LIDAR_SITE": contract.LIDAR_SITE,
        "DRIVE_POS_SENSOR": contract.DRIVE_POS_SENSOR,
        "IMU_ACCEL_SENSOR": contract.IMU_ACCEL_SENSOR,
        "IMU_GYRO_SENSOR": contract.IMU_GYRO_SENSOR,
        "WHEEL_GEOMS": contract.WHEEL_GEOMS,
        "NO_RETURN": sensors.NO_RETURN,
        "SLOT_EPS": lidar.SLOT_EPS,
        "FIELD_EPS": contract.FIELD_EPS,
        "WINDOW_EPS": encoder.WINDOW_EPS,
        "SETTLE_TIME": sim.SETTLE_TIME,
    }
)


def shared() -> str:
    """shared.ts: each constant of SHARED as a TypeScript constant."""
    lines = []
    for name, value in SHARED.items():
        if isinstance(value, tuple):
            lines.append(f"export const {name} = {json.dumps(list(value))} as const;")
        else:
            lines.append(f"export const {name} = {json.dumps(value)};")
    return HEADER + "\n".join(lines) + "\n"


def car() -> str:
    """car.ts: the NeoRacer's MJCF, so a browser bundle needs no file access
    to load the car."""
    xml = json.dumps(export.car_xml(NEORACER))
    return f"{HEADER}export const CAR_XML: string = {xml};\n"


def sources() -> dict[str, str]:
    """Each generated file's name and text."""
    return {
        "spec.ts": HEADER + "\n" + typescript.module(),
        "shared.ts": shared(),
        "car.ts": car(),
    }


if __name__ == "__main__":
    (folder,) = sys.argv[1:]
    os.makedirs(folder, exist_ok=True)
    for name, text in sources().items():
        with open(os.path.join(folder, name), "w", encoding="utf-8") as f:
            f.write(text)
