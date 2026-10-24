"""
Laws for neoracer_mujoco.control.track_centering.

The tests call step() and compute_signals() with fake SensorReadings and
do not use the simulator. test_steering.py has the laws of the steer and
throttle shaping, and test_track_centering_demo.py runs the controller on
the car.

Steer sign: + = left (contract.py).
"""

import numpy as np
import pytest
from fake_sensors import beams, mirror, random_beams

from neoracer_mujoco.control import PDState, steering
from neoracer_mujoco.control.track_centering import (
    TrackCenteringConfig,
    compute_signals,
    step,
)
from neoracer_mujoco.sensors import LIDAR_BEAM_ORDER

DT = 0.02
"""One 50 Hz control step (s)."""
CFG = TrackCenteringConfig()


def _random_state(rng: np.random.Generator) -> PDState:
    limit = CFG.shaping.steer_limit
    return PDState(
        prev_error=float(rng.uniform(-3.0, 3.0)),
        prev_steer=float(rng.uniform(-limit, limit)),
    )


def test_mirrored_scene_gives_mirrored_command():
    """If the scene and the state are mirrored left to right, the steer
    changes sign and the throttle stays the same. lateral_error changes sign
    in the mirror, so the mirrored state negates prev_error too."""
    rng = np.random.default_rng(0)
    for _ in range(500):
        readings, state = random_beams(rng), _random_state(rng)
        mirrored_state = PDState(-state.prev_error, -state.prev_steer)
        next_state, control = step(CFG, state, readings, DT)
        next_mirrored, mirrored = step(CFG, mirrored_state, mirror(readings), DT)
        assert mirrored.steer == pytest.approx(-control.steer, abs=1e-12)
        assert mirrored.throttle == pytest.approx(control.throttle, abs=1e-12)
        assert next_mirrored.prev_steer == pytest.approx(-next_state.prev_steer)
        assert next_mirrored.prev_error == pytest.approx(-next_state.prev_error)


def test_closer_left_wall_steers_right():
    """With the car parallel to the walls and closer to the left wall, the
    steer is negative (right). The mirror law gives the right wall case."""
    rng = np.random.default_rng(1)
    for near, far in np.sort(rng.uniform(0.1, 3.0, (200, 2)), axis=1):
        if far - near < 1e-3:
            continue
        readings = beams(
            lidar_045=near,
            lidar_090=near,
            lidar_135=near,
            lidar_225=far,
            lidar_270=far,
            lidar_315=far,
        )
        _, control = step(CFG, PDState(), readings, DT)
        assert control.steer < 0.0, (near, far)


def test_nose_toward_left_wall_steers_right():
    """If the car is centered but the front-left beam is shorter than the
    rear-left beam, the nose points at the left wall. Then heading_error is
    negative and the steer is negative (right)."""
    rng = np.random.default_rng(2)
    for side, skew in rng.uniform([0.5, 0.01], [2.5, 0.4], (200, 2)):
        # The mean of the three left beams is `side`, the same as the right
        # mean, so lateral_error is 0 and only the heading term steers.
        readings = beams(
            lidar_045=side - skew,
            lidar_090=side,
            lidar_135=side + skew,
            lidar_225=side,
            lidar_270=side,
            lidar_315=side,
        )
        signals = compute_signals(readings, CFG)
        assert signals.lateral_error == pytest.approx(0.0, abs=1e-12)
        assert signals.heading_error < 0.0
        _, control = step(CFG, PDState(), readings, DT)
        assert control.steer < 0.0


def test_too_many_corrupt_beams_give_the_fallback():
    """If every beam is NaN, or every beam infinite, the corrupt count is
    above CFG.max_corrupt_beams, and step() returns steering.fallback."""
    infinities = (float("inf"), float("-inf"))
    state = PDState(prev_error=0.4, prev_steer=0.2)
    for name, readings in [
        ("nan", beams(**{n: float("nan") for n in LIDAR_BEAM_ORDER})),
        (
            "inf",
            beams(**{n: infinities[i % 2] for i, n in enumerate(LIDAR_BEAM_ORDER)}),
        ),
    ]:
        assert compute_signals(readings, CFG).n_corrupt > CFG.max_corrupt_beams, name
        assert step(CFG, state, readings, DT) == steering.fallback(
            CFG.shaping, state, DT
        ), name


def test_no_hit_reading_is_not_treated_as_corrupt():
    """A -1 (no hit) is valid data that means "clear". It is not a sensor
    fault, so it does not count as corrupt."""
    readings = beams(**{name: -1.0 for name in LIDAR_BEAM_ORDER})
    signals = compute_signals(readings, CFG)
    assert signals.n_corrupt == 0
    assert signals.lateral_error == pytest.approx(0.0)
