"""Parametrize every conformance test over each car XML found in assets/.

New XML in assets/ -> discovered by neoracer_mujoco.cars() -> tested with no
edits here."""

import os

import pytest

from neoracer_mujoco import cars, load


@pytest.fixture(params=cars(), ids=lambda p: os.path.basename(p))
def car(request):
    """A freshly compiled MjModel for each car."""
    return load(request.param)
