"""
Laws for neoracer_mujoco.control.track_centering.

The pure-logic tests call step() and compute_signals() with fake
SensorReadings and do not use the simulator. The last test runs the
controller on the car in straight_corridor.xml.

Steer sign: + = left (contract.py).
"""

import mujoco
import numpy as np
import pytest
from fake_sensors import beams, mirror, random_beams
from scenes import corridor_scene

from neoracer_mujoco import sensors, sim
from neoracer_mujoco.control import PDState
from neoracer_mujoco.control.track_centering import (
    TrackCenteringConfig,
    compute_signals,
    step,
)
from neoracer_mujoco.sensors import LIDAR_BEAM_ORDER

DT = 0.02  # s, one 50 Hz control step
CFG = TrackCenteringConfig()


def _random_state(rng: np.random.Generator) -> PDState:
    limit = CFG.shaping.steer_limit
    return PDState(
        prev_error=float(rng.uniform(-3.0, 3.0)),
        prev_steer=float(rng.uniform(-limit, limit)),
    )


def test_equal_left_right_is_neutral():
    _, control = step(CFG, PDState(), beams(), DT)
    assert control.steer == pytest.approx(0.0, abs=1e-9)


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


def test_steer_never_exceeds_the_steer_limit():
    rng = np.random.default_rng(3)
    for _ in range(500):
        _, control = step(CFG, _random_state(rng), random_beams(rng), DT)
        assert abs(control.steer) <= CFG.shaping.steer_limit


def test_steer_saturates_at_the_steer_limit():
    readings = beams(
        lidar_045=10.0,
        lidar_090=10.0,
        lidar_135=10.0,
        lidar_225=0.05,
        lidar_270=0.05,
        lidar_315=0.05,
    )
    state = PDState()
    for _ in range(200):  # 4 s: the steer rate limit lets the steer reach its limit
        state, control = step(CFG, state, readings, DT)
    assert control.steer == pytest.approx(CFG.shaping.steer_limit, abs=1e-6)


def test_step_is_a_pure_function():
    """step() keeps no hidden state: the same (state, readings) gives the
    same result, also after calls with other inputs."""
    rng = np.random.default_rng(4)
    cases = [(_random_state(rng), random_beams(rng)) for _ in range(200)]
    first = [step(CFG, state, readings, DT) for state, readings in cases]
    again = [step(CFG, state, readings, DT) for state, readings in reversed(cases)]
    assert first == again[::-1]


def test_invalid_readings_trigger_safe_fallback():
    """If every beam is NaN, the corrupt count is above
    CFG.max_corrupt_beams. step() then returns the fallback command."""
    readings = beams(**{name: float("nan") for name in LIDAR_BEAM_ORDER})
    assert compute_signals(readings, CFG).n_corrupt > CFG.max_corrupt_beams
    _, control = step(CFG, PDState(), readings, DT)
    assert control.throttle == pytest.approx(
        CFG.shaping.target_throttle * CFG.shaping.safe_fallback_throttle_fraction
    )
    assert abs(control.steer) < 0.05, (
        "fallback must not issue an extreme steering command"
    )


def test_infinite_readings_also_trigger_fallback():
    infinities = (float("inf"), float("-inf"))
    readings = beams(
        **{name: infinities[i % 2] for i, name in enumerate(LIDAR_BEAM_ORDER)}
    )
    _, control = step(CFG, PDState(), readings, DT)
    assert control.throttle == pytest.approx(
        CFG.shaping.target_throttle * CFG.shaping.safe_fallback_throttle_fraction
    )


def test_no_hit_reading_is_not_treated_as_corrupt():
    """A -1 (no hit) is valid data that means "clear". It is not a sensor
    fault, so it does not count as corrupt."""
    readings = beams(**{name: -1.0 for name in LIDAR_BEAM_ORDER})
    signals = compute_signals(readings, CFG)
    assert signals.n_corrupt == 0
    assert signals.lateral_error == pytest.approx(0.0)


def test_throttle_decreases_for_sharp_steering_correction():
    _, straight = step(CFG, PDState(), beams(), DT)
    state = PDState()
    readings = beams(lidar_045=0.2, lidar_090=0.2, lidar_135=0.2)
    for _ in range(200):  # 4 s
        state, sharp = step(CFG, state, readings, DT)
    assert sharp.throttle < straight.throttle


def test_throttle_decreases_for_close_forward_obstacle():
    _, clear = step(CFG, PDState(), beams(lidar_000=CFG.max_beam_range), DT)
    _, blocked = step(CFG, PDState(), beams(lidar_000=0.1), DT)
    assert blocked.throttle < clear.throttle
    assert blocked.throttle == pytest.approx(0.0, abs=1e-9)


# --- physics integration: the compiled car in the corridor --------------------


def test_controller_stable_on_corridor():
    """Run the controller on the car in the corridor for 3 s. The values stay
    finite, the steer stays in the actuator range, the car stays upright and
    it does not drive into a wall."""
    scene = corridor_scene()
    model, car = scene.model, scene.car
    data = sim.settle(model)
    data.bind_scalar(car.joint).qpos[1] += 0.25  # m, toward the left wall
    mujoco.mj_forward(model, data)

    dt = model.opt.timestep
    steer_low, steer_high = model.bind_scalar(car.steer).ctrlrange
    state = PDState()
    min_front = float("inf")
    for _ in range(1500):  # 3 s at the 2 ms timestep
        readings = sensors.read(car.sensors, data)
        state, control = step(CFG, state, readings, dt)
        sim.apply_command(car, data, control)
        mujoco.mj_step(model, data)

        assert steer_low <= control.steer <= steer_high
        assert np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))
        min_front = min(min_front, compute_signals(readings, CFG).front_distance)

    upright_cos = sim.car_upright_cos(data, car)
    assert upright_cos > 0.7, "car should not have flipped during the run"
    assert min_front > 0.05, "controller should not have driven straight into a wall"
