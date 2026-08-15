"""
Tests for the production runtime bootstrap, examples/run_racecar_core.py.

test_wall_follower_example.py already proves the racecar_core_adapter
contract inside a pytest fixture (a hand-rolled sys.modules injection local
to that test file). This file instead exercises the actual runner a user
invokes with `python3 examples/run_racecar_core.py <controller.py>`:
run_controller()/main() themselves, plus the real CLI subprocess for the
canonical vendored example. No real racecar_core package is installed
anywhere in this environment -- these tests are the proof that none is
needed.
"""

import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
_EXAMPLES_DIR = _REPO / "examples"
_CANONICAL_DIR = _EXAMPLES_DIR / "ultimate_wall_follower"
_TESTDATA_DIR = _REPO / "validation" / "testdata" / "ultimate_wall_follower"

sys.path.insert(0, str(_EXAMPLES_DIR))
import run_racecar_core  # noqa: E402  (needs the sys.path.insert above)


@pytest.mark.parametrize(
    "relpath",
    ["wall_follower.py", "control.py", "config/wall_follower.yaml"],
)
def test_canonical_example_matches_testdata_fixture_byte_for_byte(relpath):
    """examples/ultimate_wall_follower/ must stay a byte-for-byte copy of the
    vendored fixture under validation/testdata/ -- see both SOURCE.md files.
    Catches accidental edits to either copy going out of sync."""
    canonical = (_CANONICAL_DIR / relpath).read_bytes()
    fixture = (_TESTDATA_DIR / relpath).read_bytes()
    assert canonical == fixture


def test_run_controller_reaches_wall_follower_start(capsys):
    """run_controller() -- the function main() calls -- must run the
    unmodified wall_follower.py far enough to hit its start() lifecycle
    (its own ">> Ultimate Wall Follower ready." print), with no real
    racecar_core package installed."""
    assert "racecar_core" not in sys.modules

    run_racecar_core.run_controller(
        str(_CANONICAL_DIR / "wall_follower.py"), steps=10
    )

    out = capsys.readouterr().out
    assert ">> Ultimate Wall Follower ready." in out
    # the shim must not leak into the rest of the test session
    assert "racecar_core" not in sys.modules
    assert str(_CANONICAL_DIR) not in sys.path


def test_run_controller_drives_the_car(capsys):
    """End-to-end: after run_controller() returns, the car must have
    actually moved -- proof the go() loop stepped physics and update()
    fed real drive commands, not just that start() printed."""
    run_racecar_core.run_controller(
        str(_CANONICAL_DIR / "wall_follower.py"), steps=300
    )
    out = capsys.readouterr().out
    assert "steps done" in out


def test_run_controller_rejects_missing_path():
    with pytest.raises(FileNotFoundError):
        run_racecar_core.run_controller(str(_EXAMPLES_DIR / "does_not_exist.py"))


def test_run_controller_rejects_non_python_file():
    with pytest.raises(ValueError):
        run_racecar_core.run_controller(str(_REPO / "README.md"))


def test_cli_runs_canonical_wall_follower_end_to_end():
    """The exact user-facing command, as a subprocess -- proves the CLI
    entry point (argparse, error handling, exit code) works, not just the
    library function it wraps."""
    result = subprocess.run(
        [sys.executable, str(_EXAMPLES_DIR / "run_racecar_core.py"),
         str(_CANONICAL_DIR / "wall_follower.py"), "--steps", "50"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert ">> Ultimate Wall Follower ready." in result.stdout
    assert "steps done" in result.stdout


def test_cli_fails_clearly_on_missing_controller():
    result = subprocess.run(
        [sys.executable, str(_EXAMPLES_DIR / "run_racecar_core.py"),
         str(_EXAMPLES_DIR / "does_not_exist.py")],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert "error:" in result.stderr
