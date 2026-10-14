"""The `scene` fixture runs a test once for each car XML in assets/.

neoracer_mujoco.cars() finds every car XML in assets/. So a new car XML gets
the conformance tests with no change to this file."""

import os

import pytest

from neoracer_mujoco import cars, load


@pytest.fixture(params=cars(), ids=os.path.basename)
def scene(request):
    """A freshly compiled Scene for each car."""
    return load(request.param)
