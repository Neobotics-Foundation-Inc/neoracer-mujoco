"""
Tests for examples/run_racecar_core.py.

These tests call run_controller() and run the command line as a subprocess,
with the example wall follower in examples/ultimate_wall_follower/.
No real racecar_core package is installed, so the tests show that none is needed.
test_wall_follower_example.py tests the controller callbacks on the adapter directly.
"""

import importlib
import subprocess
import sys
import time
import types
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


def test_a_controller_reads_the_simulated_time(tmp_path):
    """In controller_imports, time.time(), time.monotonic() and
    time.perf_counter() read the simulated time, so a controller that
    measures its frame time sees the physics step on any machine. After the
    block, the real clock is back."""
    scene = run_racecar_core.load_scene()
    data, rc = run_racecar_core.build_racecar(scene, steps=10)
    (tmp_path / "clock_probe.py").write_text(
        "import time\n\n\ndef now():\n"
        "    return time.time(), time.monotonic(), time.perf_counter()\n"
    )
    with run_racecar_core.controller_imports(rc, tmp_path):
        probe = importlib.import_module("clock_probe")
        data.time = 1.25
        assert probe.now() == (1.25, 1.25, 1.25)
    assert time.time() > 1e9  # s since 1970


def test_a_star_import_gets_the_simulated_clocks(tmp_path):
    """`from time import *` in a controller gets the simulated clocks and
    the rest of the time module."""
    data, rc = run_racecar_core.build_racecar(run_racecar_core.load_scene(), 1)
    (tmp_path / "star_probe.py").write_text(
        "from time import *\n\n\ndef now():\n"
        "    return time(), monotonic(), perf_counter(), monotonic_ns(), sleep\n"
    )
    with run_racecar_core.controller_imports(rc, tmp_path):
        probe = importlib.import_module("star_probe")
        data.time = 1.25
        assert probe.now() == (1.25, 1.25, 1.25, 1_250_000_000, time.sleep)


def test_a_library_imported_by_a_controller_keeps_the_real_clock(tmp_path):
    """Only the files of the controller read the simulated time. A module
    outside its folder that the controller imports first in the block, as a
    library, gets the real time module, during the block and after it."""
    controller, library = tmp_path / "controller", tmp_path / "library"
    controller.mkdir()
    library.mkdir()
    (library / "clock_library.py").write_text(
        "import time\nfrom time import monotonic\n"
    )
    (controller / "clock_user.py").write_text("import clock_library\nimport time\n")
    _data, rc = run_racecar_core.build_racecar(run_racecar_core.load_scene(), 1)
    sys.path.insert(0, str(library))
    try:
        with run_racecar_core.controller_imports(rc, controller):
            user = importlib.import_module("clock_user")
            assert user.time is not time
            assert user.clock_library.time is time
            assert user.clock_library.monotonic is time.monotonic
        assert sys.modules["time"] is time
    finally:
        sys.path.remove(str(library))
        sys.modules.pop("clock_library", None)


def test_controller_imports_restores_the_import_state(tmp_path, monkeypatch):
    """Inside the block, the controller's own files and the fake racecar_core
    win over cached modules with the same names. After the block, the
    caller's modules are back, and nothing from the controller stays."""
    (tmp_path / "control.py").write_text("WHO = 'controller'\n")
    cached_control = types.ModuleType("control")
    cached_racecar_core = types.ModuleType("racecar_core")
    monkeypatch.setitem(sys.modules, "control", cached_control)
    monkeypatch.setitem(sys.modules, "racecar_core", cached_racecar_core)
    _data, rc = run_racecar_core.build_racecar(run_racecar_core.load_scene(), 1)

    with run_racecar_core.controller_imports(rc, tmp_path):
        control = importlib.import_module("control")
        racecar_core = importlib.import_module("racecar_core")
        assert control.WHO == "controller"
        assert racecar_core.create_racecar() is rc

    assert sys.modules["control"] is cached_control
    assert sys.modules["racecar_core"] is cached_racecar_core
    assert str(tmp_path) not in sys.path
