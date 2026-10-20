"""
Laws for neoracer_mujoco.control.wall_following.

The pure-logic tests call step() and compute_signals() with fake
SensorReadings and do not use the simulator; test_steering.py has the laws
of the steer and throttle shaping. The last test runs the controller on the
car in straight_corridor.xml.

Steer sign: + = left (contract.py). With follow_side="right", a steer toward
the wall is negative.
"""

import mujoco
import numpy as np
import pytest
from fake_sensors import beams, mirror, random_beams
from scenes import corridor_scene

from neoracer_mujoco import sensors, sim
from neoracer_mujoco.control import PDState, steering
from neoracer_mujoco.control.wall_following import (
    WallFollowingConfig,
    compute_signals,
    step,
)

DT = 0.02  # s, one 50 Hz control step
RIGHT = WallFollowingConfig(follow_side="right", target_distance=0.3)


def _command(cfg, readings):
    """Command of the first step of an episode."""
    _, command = step(cfg, PDState(), readings, DT)
    return command


def _random_state(rng: np.random.Generator) -> PDState:
    limit = RIGHT.shaping.steer_limit
    return PDState(
        prev_error=float(rng.uniform(-3.0, 3.0)),
        prev_steer=float(rng.uniform(-limit, limit)),
    )


def test_left_and_right_are_mirror_images():
    """Follow the left wall in the mirrored scene, and the steer changes sign
    and the throttle stays the same. distance_error is measured from the
    followed wall, so it is the same in the mirror: only prev_steer changes
    sign in the mirrored state."""
    left = WallFollowingConfig(follow_side="left")
    right = WallFollowingConfig(follow_side="right")
    rng = np.random.default_rng(0)
    for _ in range(500):
        readings, state = random_beams(rng), _random_state(rng)
        mirrored_state = PDState(state.prev_error, -state.prev_steer)
        next_right, right_control = step(right, state, readings, DT)
        next_left, left_control = step(left, mirrored_state, mirror(readings), DT)
        assert left_control.steer == pytest.approx(-right_control.steer, abs=1e-12)
        assert left_control.throttle == pytest.approx(right_control.throttle)
        assert next_left.prev_steer == pytest.approx(-next_right.prev_steer)
        assert next_left.prev_error == pytest.approx(next_right.prev_error)


def test_steer_decreases_as_the_wall_distance_increases():
    """Sweep the right beam distance. The steer does not increase as the
    distance increases. It is negative (toward the wall) above
    target_distance and positive (away from the wall) below it."""
    distances = np.linspace(0.0, RIGHT.max_beam_range + 1.0, 400)  # m
    steers = np.array(
        [_command(RIGHT, beams(lidar_270=float(d))).steer for d in distances]
    )
    assert np.all(np.diff(steers) <= 0.0)
    assert np.all(steers[distances > RIGHT.target_distance + 1e-6] < 0.0)
    assert np.all(steers[distances < RIGHT.target_distance - 1e-6] > 0.0)


def test_nose_angled_toward_wall_corrects_away():
    """At the target distance, the front-right beam is shorter than the
    rear-right beam, so the nose points at the wall. The car steers away
    from the wall (left, positive)."""
    readings = beams(lidar_270=0.3, lidar_315=0.2, lidar_225=0.4)
    signals = compute_signals(readings, RIGHT)
    assert signals.distance_error == pytest.approx(0.0)
    assert signals.wall_skew > 0.0
    assert _command(RIGHT, readings).steer > 0.0


def test_too_many_corrupt_beams_give_the_fallback():
    nan = float("nan")
    readings = beams(lidar_270=nan, lidar_315=nan, lidar_225=nan, lidar_000=1.0)
    assert compute_signals(readings, RIGHT).n_corrupt > RIGHT.max_corrupt_beams
    state = PDState(prev_error=0.4, prev_steer=0.2)
    assert step(RIGHT, state, readings, DT) == steering.fallback(
        RIGHT.shaping, state, DT
    )


def test_no_hit_reading_is_not_treated_as_corrupt():
    """A -1 (no hit) is valid data that means "clear", so it does not count
    as corrupt."""
    readings = beams(lidar_270=-1.0, lidar_315=-1.0, lidar_225=-1.0)
    assert compute_signals(readings, RIGHT).n_corrupt == 0


# --- physics integration: the compiled car in the corridor --------------------


def test_controller_converges_to_target_distance_on_corridor():
    """Run the controller on the car in the corridor for 3 s. In the last
    quarter of the run, the mean right beam distance is near target_distance.
    The car does not flip, the values stay finite and the steer stays in the
    actuator range."""
    scene = corridor_scene()
    model, car = scene.model, scene.car
    data = sim.settle(model)
    dt = model.opt.timestep
    steer_low, steer_high = model.bind_scalar(car.steer).ctrlrange
    cfg = WallFollowingConfig(follow_side="right")
    state = PDState()

    side_distances = []
    for _ in range(round(3.0 / dt)):  # 3 s
        readings = sensors.read(car.sensors, data)
        state, control = step(cfg, state, readings, dt)
        sim.apply_command(car, data, control)
        mujoco.mj_step(model, data)

        assert steer_low <= control.steer <= steer_high
        assert np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))
        side_distances.append(readings.lidar_270)

    upright_cos = sim.car_upright_cos(data, car)
    assert upright_cos > 0.7, "car should not have flipped during the run"

    quarter = len(side_distances) // 4
    last_quarter_mean = sum(side_distances[-quarter:]) / quarter
    assert last_quarter_mean == pytest.approx(cfg.target_distance, abs=0.1)
