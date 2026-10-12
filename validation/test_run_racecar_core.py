"""
Tests for examples/run_racecar_core.py.

These tests call run_controller() and run the command line as a subprocess,
with the example wall follower in examples/ultimate_wall_follower/.
No real racecar_core package is installed, so the tests show that none is needed.
test_wall_follower_example.py tests the controller callbacks on the adapter directly.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from examples import run_racecar_core

_REPO = Path(__file__).resolve().parent.parent
_EXAMPLES_DIR = _REPO / "examples"
_CANONICAL_DIR = _EXAMPLES_DIR / "ultimate_wall_follower"
_TESTDATA_DIR = _REPO / "validation" / "testdata" / "ultimate_wall_follower"


@pytest.mark.parametrize(
    "relpath",
    ["wall_follower.py", "control.py", "config/wall_follower.yaml"],
)
def test_canonical_example_matches_testdata_fixture_byte_for_byte(relpath):
    """examples/ultimate_wall_follower/ is a byte-for-byte copy of the fixture
    in validation/testdata/. This test fails if one copy changes."""
    canonical = (_CANONICAL_DIR / relpath).read_bytes()
    fixture = (_TESTDATA_DIR / relpath).read_bytes()
    assert canonical == fixture


def test_run_controller_reaches_wall_follower_start(capsys, monkeypatch):
    """run_controller() runs wall_follower.py until its start() prints its ready
    line. Afterwards, the fake module, the controller's own control module and
    the sys.path entry are gone."""
    monkeypatch.delitem(sys.modules, "racecar_core", raising=False)
    monkeypatch.delitem(sys.modules, "control", raising=False)

    run_racecar_core.run_controller(str(_CANONICAL_DIR / "wall_follower.py"), steps=10)

    out = capsys.readouterr().out
    assert ">> Ultimate Wall Follower ready." in out
    assert "racecar_core" not in sys.modules
    assert "control" not in sys.modules
    assert str(_CANONICAL_DIR) not in sys.path


def test_run_controller_drives_the_car_forward():
    """The go() loop steps the physics with the drive commands of update(),
    so the car moves forward along +X."""
    dx = run_racecar_core.run_controller(
        str(_CANONICAL_DIR / "wall_follower.py"), steps=300
    )
    assert dx > 0.1, f"expected forward progress, got dx={dx:.3f} m"


def test_run_controller_rejects_missing_path():
    with pytest.raises(FileNotFoundError):
        run_racecar_core.run_controller(str(_EXAMPLES_DIR / "does_not_exist.py"))


def test_run_controller_rejects_non_python_file():
    with pytest.raises(ValueError):
        run_racecar_core.run_controller(str(_REPO / "README.md"))


def test_cli_runs_canonical_wall_follower_end_to_end():
    """The command line as a subprocess: argparse, the error handling and the
    exit code, not only run_controller()."""
    result = subprocess.run(
        [
            sys.executable,
            str(_EXAMPLES_DIR / "run_racecar_core.py"),
            str(_CANONICAL_DIR / "wall_follower.py"),
            "--steps",
            "50",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert ">> Ultimate Wall Follower ready." in result.stdout
    assert "steps done" in result.stdout


def test_cli_fails_clearly_on_missing_controller():
    result = subprocess.run(
        [
            sys.executable,
            str(_EXAMPLES_DIR / "run_racecar_core.py"),
            str(_EXAMPLES_DIR / "does_not_exist.py"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode != 0
    assert "error:" in result.stderr
