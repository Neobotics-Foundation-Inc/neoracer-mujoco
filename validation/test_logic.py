"""
Pure-logic tests: math that has a correct answer independent of the physics sim.
These catch silent numerical bugs (wrong unit conversion, a polyfit that's the
right shape but a few percent off) that the behavioral conformance tests miss.
"""

from dataclasses import fields

import mujoco
import numpy as np
import pytest

import sensor_logger as sl


def test_wheel_speed_conversion():
    """rad/s -> m/s is angular velocity x wheel radius, per wheel."""
    fake = sl.SensorReadings(
        **{
            n: np.array([2.0])
            for n in ("fl_wheel_vel", "fr_wheel_vel", "rl_wheel_vel", "rr_wheel_vel")
        }
    )
    out = sl.wheel_speed_ms(fake)
    assert out["fl_wheel_speed_ms"] == pytest.approx(2.0 * sl.WHEEL_RADIUS)
    assert set(out) == {
        "fl_wheel_speed_ms",
        "fr_wheel_speed_ms",
        "rl_wheel_speed_ms",
        "rr_wheel_speed_ms",
    }


def test_read_returns_every_sensor(car):
    """sensor_logger.read must surface all sensors with correctly-sliced widths."""
    d = mujoco.MjData(car)
    mujoco.mj_step(car, d)
    out = sl.read(car, d)
    assert len(fields(sl.SensorReadings)) == car.nsensor
    assert out.imu_accel.shape == (3,) and out.imu_quat.shape == (4,)


def test_ackermann_polycoef_matches_exact_geometry(car):
    """
    The steering polycoef baked into the XML must reproduce exact Ackermann for
    this car's own wheelbase/track. Validates the model's CALCULATED coefficients
    against the analytic arctan formula.
    """
    try:
        eq = car.eq("ackermann_left")
        fl = car.body("fl_wheel_body")
    except KeyError:
        pytest.skip("car has no Ackermann steering")

    L = 2 * abs(float(fl.pos[0]))  # wheelbase
    W = 2 * abs(float(fl.pos[1]))  # track
    coef = np.array(eq.data[:5])  # a0..a4

    delta = np.linspace(-0.4, 0.4, 41)
    delta = delta[delta != 0]
    exact = np.arctan(L / (L / np.tan(delta) - W / 2))  # inner (left) wheel
    poly = np.polyval(coef[::-1], delta)
    err = float(np.max(np.abs(poly - exact)))
    assert err < 0.01, f"Ackermann polyfit off by {err:.4f} rad vs exact geometry"
