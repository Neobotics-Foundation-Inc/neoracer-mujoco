"""Put scripts/ on the path so tests can import sensor_logger, and parametrize
every conformance test over each car XML found in assets/."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts"))

from _sim import CARS, compile  # noqa: E402


@pytest.fixture(params=CARS, ids=lambda p: os.path.basename(p))
def car(request):
    """A freshly compiled MjModel for each car. New XML in assets/ -> tested automatically."""
    return compile(request.param)
