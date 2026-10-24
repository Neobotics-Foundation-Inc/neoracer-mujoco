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
from types import MappingProxyType, ModuleType

from neoracer_spec import typescript
from neoracer_spec.neoracer import NEORACER

from . import assets, contract, encoder, export, lidar, sensors, sim

HEADER = "// Written by python -m neoracer_mujoco.javascript. Do not edit.\n"

SHARED: Mapping[str, ModuleType] = MappingProxyType(
    {
        "CAR_BODY": contract,
        "START_SITE": assets,
        "DRIVE_ACTUATOR": contract,
        "STEER_ACTUATOR": contract,
        "LIDAR_SITE": contract,
        "DRIVE_POS_SENSOR": contract,
        "IMU_ACCEL_SENSOR": contract,
        "IMU_GYRO_SENSOR": contract,
        "WHEEL_GEOMS": contract,
        "NO_RETURN": sensors,
        "SLOT_EPS": lidar,
        "FIELD_EPS": contract,
        "WINDOW_EPS": encoder,
        "SETTLE_TIME": sim,
    }
)
"""The constants of shared.ts, each by the module that defines it: the parts
of the car that code finds by name, and the numbers that both packages need
to agree on."""


def shared() -> str:
    """shared.ts: each constant of SHARED as a TypeScript constant, under the
    TSDoc of its docstring."""
    lines = []
    for name, module in SHARED.items():
        value = getattr(module, name)
        lines += typescript.doc_comment(module, name)
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
