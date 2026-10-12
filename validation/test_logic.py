"""
Pure-logic tests: math that has a correct answer independent of the physics sim.
These catch silent numerical bugs (wrong unit conversion, a polyfit that's the
right shape but a few percent off) that the behavioral conformance tests miss.
"""

from dataclasses import fields

import mujoco
import numpy as np
import pytest
from fake_sensors import fake_readings

from neoracer_mujoco import sensors as sl


def test_read_returns_every_sensor(car):
    """sensor_logger.read must surface all sensors with correctly-sliced widths."""
    d = mujoco.MjData(car)
    mujoco.mj_step(car, d)
    out = sl.read(car, d)
    assert len(fields(sl.SensorReadings)) == car.nsensor
    assert out.imu_accel.shape == (3,) and out.imu_gyro.shape == (3,)
    assert out.imu_quat.shape == (4,)
    assert isinstance(out.lidar_000, float) and isinstance(out.steer_cmd_pos, float)


def test_sensor_layout_rejects_a_model_without_the_sensors():
    """sensor_layout parses the model once and names each missing sensor."""
    bare = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
    with pytest.raises(ValueError, match="imu_accel is missing"):
        sl.sensor_layout(bare)


def test_motor_encoder_is_mean_of_wheel_vel():
    """motor_encoder_rad_s() returns the mean of the four wheel speeds."""
    rng = np.random.default_rng(0)
    for fl, fr, rl, rr in rng.uniform(-100.0, 100.0, (50, 4)):
        readings = fake_readings(
            fl_wheel_vel=fl, fr_wheel_vel=fr, rl_wheel_vel=rl, rr_wheel_vel=rr
        )
        mean = (fl + fr + rl + rr) / 4
        assert sl.motor_encoder_rad_s(readings) == pytest.approx(mean)


def test_estimated_linear_speed_ms():
    """Wheel-radius speed estimate, on known input."""
    assert sl.estimated_linear_speed_ms(2.0) == pytest.approx(2.0 * sl.WHEEL_RADIUS)


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
