"""
Laws for the car's steering: the shaping that the LiDAR controllers share
(control/steering.py), and the servo, which holds the wheels still in a
steady turn so the car does not shimmy.
"""

import mujoco
import numpy as np
import pytest

from neoracer_mujoco import assets, sim
from neoracer_mujoco.control import steering
from neoracer_mujoco.control.steering import ControlCommand, PDState, ShapingConfig

DT = 0.02
"""One 50 Hz control step (s)."""
P = ShapingConfig()
LIMIT = P.steer_limit
"""The steer limit (rad)."""


def test_command_keeps_the_steer_within_its_limit_and_its_rate():
    """Whatever the raw steer, the steer stays within the steer limit and
    moves at most steer_slew_limit * dt in a step; held, it reaches the
    limit."""
    rng = np.random.default_rng(0)
    for _ in range(500):
        state = PDState(prev_steer=float(rng.uniform(-LIMIT, LIMIT)))
        raw = float(rng.uniform(-5 * LIMIT, 5 * LIMIT))
        _, out = steering.command(P, state, 0.0, raw, 10.0, DT)
        assert abs(out.steer) <= LIMIT, (state, raw)
        assert abs(out.steer - state.prev_steer) <= P.steer_slew_limit * DT + 1e-12
    state = PDState()
    for _ in range(round(2 * LIMIT / (P.steer_slew_limit * DT)) + 1):
        state, out = steering.command(P, state, 0.0, 5 * LIMIT, 10.0, DT)
    assert out.steer == LIMIT


def test_command_slows_for_a_large_steer_and_a_close_obstacle():
    """The speed asked for never rises as the steer grows or as the clearance
    ahead shrinks, and it is 0 at min_safe_forward_distance or closer."""

    def speed(steer: float, front: float) -> float:
        # prev_steer = steer, so the slew limit leaves the steer as it is.
        _, out = steering.command(P, PDState(prev_steer=steer), 0.0, steer, front, DT)
        return out.target

    by_steer = [speed(s, 10.0) for s in np.linspace(0.0, LIMIT, 50)]
    assert np.all(np.diff(by_steer) <= 0.0) and by_steer[-1] < by_steer[0]
    fronts = np.linspace(2.0, 0.0, 100)  # m, closing in
    by_front = [speed(0.0, f) for f in fronts]
    assert np.all(np.diff(by_front) <= 0.0)
    for front, v in zip(fronts, by_front, strict=True):
        if front <= P.min_safe_forward_distance:
            assert v == 0.0, front


def test_fallback_slews_the_steer_to_center_at_a_reduced_speed():
    """fallback asks for target_speed * safe_fallback_speed_fraction,
    moves the steer toward 0 at the slew rate, and clears the error history,
    so the bad step gives the next D term no spike."""
    rng = np.random.default_rng(1)
    for _ in range(200):
        state = PDState(
            prev_error=float(rng.uniform(-3.0, 3.0)),
            prev_steer=float(rng.uniform(-LIMIT, LIMIT)),
        )
        after, out = steering.fallback(P, state, DT)
        expected = state.prev_steer - np.clip(
            state.prev_steer, -P.steer_slew_limit * DT, P.steer_slew_limit * DT
        )
        assert out.steer == pytest.approx(expected, abs=1e-12)
        assert out.target == P.target_speed * P.safe_fallback_speed_fraction
        assert after == PDState(prev_error=0.0, prev_steer=out.steer)


def test_a_steady_part_throttle_turn_does_not_shimmy():
    """Held at 0.6 throttle and 0.6 or all of full lock, the part throttle
    where the steering shook, the car's yaw rate (rad/s) spreads by less than
    0.02 over 4 s once it has settled into the turn. A shimmy at 15 to 30 Hz
    spreads it by 0.07."""
    scene = assets.load()  # the car on its own floor, 10 m square
    model, car = scene.model, scene.car
    lock = model.bind_scalar(car.steer).ctrlrange[1]  # rad
    rate = round(1 / model.opt.timestep)  # steps a second
    for lock_share in (0.6, 1.0):
        data = mujoco.MjData(model)
        command = ControlCommand(throttle=0.6, steer=lock_share * lock)
        yaw_rates = []
        for step in range(10 * rate):
            sim.apply_command(car, data, command)
            mujoco.mj_step(model, data)
            if step >= 6 * rate:
                yaw_rates.append(data.bind_scalar(car.sensors.imu_gyro).data[2])
        assert np.std(yaw_rates) < 0.02, lock_share
