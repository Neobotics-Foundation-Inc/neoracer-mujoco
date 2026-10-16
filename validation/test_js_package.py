"""
Laws for js/, the JS package that runs the car in MuJoCo's WebAssembly
build. Its own tests (npm test) compare it with this package in Node. These
check, without Node, that the values it copies from this package agree.
"""

import json
import math
import re
import tomllib
from pathlib import Path

from neoracer_mujoco import contract, encoder, lidar, sensors, sim

_REPO = Path(__file__).resolve().parent.parent
_JS = _REPO / "js"


def _constants(name: str) -> dict[str, str]:
    """The `export const NAME = value;` lines of js/src/<name>.ts."""
    text = (_JS / "src" / f"{name}.ts").read_text()
    return dict(
        re.findall(r"^export const (\w+)(?:: \w+)? = (.+?);", text, flags=re.MULTILINE)
    )


def _number(text: str) -> float:
    """A TS number literal, or one times Math.PI: `12`, `1.5 * Math.PI`."""
    factor, times_pi, _ = text.strip().partition(" * Math.PI")
    return float(factor) * (math.pi if times_pi else 1.0)


def _fields(literal: str) -> dict[str, float]:
    """The fields of a TS object literal of numbers, `{ a: 1, b: 2 * Math.PI }`."""
    return {k: _number(v) for k, v in re.findall(r"(\w+): ([^,}]+)", literal)}


def test_the_part_names_are_those_of_contract_py():
    names = _constants("contract")
    assert (
        json.loads(names["DRIVE_ACTUATOR"].replace("'", '"')) == contract.DRIVE_ACTUATOR
    )
    assert (
        json.loads(names["STEER_ACTUATOR"].replace("'", '"')) == contract.STEER_ACTUATOR
    )
    assert json.loads(names["LIDAR_SITE"].replace("'", '"')) == contract.LIDAR_SITE
    assert (
        json.loads(names["DRIVE_POS_SENSOR"].replace("'", '"'))
        in contract.REQUIRED_SENSORS
    )
    assert json.loads(names["CAR_BODY"].replace("'", '"')) == "car"


def test_the_numbers_are_those_of_the_python_package():
    enc = _constants("encoder")
    assert float(enc["WHEEL_RADIUS"]) == sensors.WHEEL_RADIUS
    assert _fields(enc["ENCODER"]) == {"window": 0.02, "alpha": 0.95}
    assert _fields(enc["ENCODER"])["window"] == encoder.EncoderParams().window
    assert _fields(enc["ENCODER"])["alpha"] == encoder.EncoderParams().alpha
    params = lidar.LidarParams()
    assert _fields(_constants("lidar")["LIDAR"]) == {
        "slots": params.slots,
        "scanRate": params.scan_rate,
        "maxRange": params.max_range,
        "field": params.field,
    }
    assert float(_constants("lidar")["NO_RETURN"]) == lidar.NO_RETURN
    assert int(_constants("scene")["SETTLE_STEPS"]) == sim.SETTLE_STEPS


def test_it_has_the_version_and_the_mujoco_of_the_python_package():
    package = json.loads((_JS / "package.json").read_text())
    pyproject = tomllib.loads((_REPO / "pyproject.toml").read_text())
    assert package["version"] == pyproject["project"]["version"]
    pinned = [
        d for d in pyproject["project"]["dependencies"] if d.startswith("mujoco==")
    ]
    assert pinned == [f"mujoco=={package['peerDependencies']['@mujoco/mujoco']}"]
    assert (
        package["devDependencies"]["@mujoco/mujoco"]
        == package["peerDependencies"]["@mujoco/mujoco"]
    )
