"""
Laws for js/, the JS package that runs the car in MuJoCo's WebAssembly
build. Its own tests (npm test) compare it with this package in Node. These
check, without Node, the sources that this package writes for it
(neoracer_mujoco.javascript) and its manifest.
"""

import json
import re
import tomllib
from pathlib import Path

from cases import case

from neoracer_mujoco import javascript
from neoracer_spec import typescript

_REPO = Path(__file__).resolve().parent.parent
_JS = _REPO / "js"


def test_the_shared_constants_read_back_as_the_python_packages():
    """shared.ts gives each name and number that both packages use its value
    here, under its docstring, so the JS package holds no copy of them."""
    text = javascript.shared()
    written = dict(
        re.findall(r"^export const (\w+) = (.+?)(?: as const)?;$", text, re.MULTILINE)
    )
    assert written.keys() == javascript.SHARED.keys()
    documented = 0
    for name, module in javascript.SHARED.items():
        value = getattr(module, name)
        expected = list(value) if isinstance(value, tuple) else value
        with case(name):
            assert json.loads(written[name]) == expected
            for line in typescript.doc_comment(module, name):
                assert f"{line}\nexport const {name} =" in text
                documented += 1
    assert documented > 0


def test_it_needs_the_mujoco_of_the_python_package():
    """pip and npm each read their own list of dependencies, so MuJoCo's
    version is in pyproject.toml and in package.json. The JS package names
    it once, as a peer dependency, which npm installs with it."""
    package = json.loads((_JS / "package.json").read_text())
    pyproject = tomllib.loads((_REPO / "pyproject.toml").read_text())
    pinned = [
        d for d in pyproject["project"]["dependencies"] if d.startswith("mujoco==")
    ]
    assert pinned == [f"mujoco=={package['peerDependencies']['@mujoco/mujoco']}"]
    assert "@mujoco/mujoco" not in package["devDependencies"]
