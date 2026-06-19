"""
Track integration tests — load track XML files and verify they compile and settle cleanly.

Separate from the car conformance suite: a track test loads its own XML (which
includes the car via <include>), so it does not use the parametrized car fixture.
Add one test per track file in assets/tracks/.
"""

import os

import mujoco
import numpy as np
import pytest

import _sim


def test_ramp_course_loads():
    """ramp_course.xml must compile (resolving the <include> path to neoracer.xml),
    and the car must settle on it without NaN. Catches: broken <include> paths from
    moving neoracer.xml, spawn-penetration blowups from ramp geometry, and any
    mesh path regression introduced by the <compiler meshdir=".."/> override."""
    ramp_path = os.path.join(_sim.ASSET_DIR, "tracks", "ramp_course.xml")
    if not os.path.exists(ramp_path):
        pytest.skip("ramp_course.xml not found")
    model = mujoco.MjModel.from_xml_path(ramp_path)
    data = _sim.settle(model)
    assert np.all(np.isfinite(data.qpos)), "NaN in qpos after settling on ramp course"
    z = float(data.body("car").xpos[2])
    assert z > 0.05, f"car z = {z:.3f} m — sank through the ramp course floor"
