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


def test_motor_encoder_reading_is_mean_of_wheel_vel(car):
    """motor_encoder_reading() must return the single-motor mean, not per-wheel values."""
    d = mujoco.MjData(car)
    mujoco.mj_step(car, d)
    sensors = sl.read(car, d)
    encoder = sl.motor_encoder_reading(sensors)
    expected = np.mean(
        [
            sensors.fl_wheel_vel[0],
            sensors.fr_wheel_vel[0],
            sensors.rl_wheel_vel[0],
            sensors.rr_wheel_vel[0],
        ]
    )
    assert isinstance(encoder.angular_velocity, float)
    assert encoder.angular_velocity == pytest.approx(expected)


def test_estimated_linear_speed_ms():
    """Wheel-radius speed estimate, on known input."""
    encoder = sl.MotorEncoderReading(angular_velocity=2.0)
    assert sl.estimated_linear_speed_ms(encoder) == pytest.approx(2.0 * sl.WHEEL_RADIUS)


def test_imu_reading_construction_shapes(car):
    """IMUReading built directly from SensorReadings must have accel/gyro as
    3-vectors and orientation as a quaternion."""
    d = mujoco.MjData(car)
    mujoco.mj_step(car, d)
    sensors = sl.read(car, d)
    imu = sl.IMUReading(
        acceleration=sensors.imu_accel,
        angular_velocity=sensors.imu_gyro,
        orientation=sensors.imu_quat,
    )
    assert imu.acceleration.shape == (3,)
    assert imu.angular_velocity.shape == (3,)
    assert imu.orientation.shape == (4,)


def test_calibrate_imu_preserves_values_and_copies_arrays():
    """
    calibrate_imu() has no hardware bias/misalignment data to apply yet (see its
    TODO in sensor_logger.py), so the returned values must match the raw ones
    exactly. This test is meant to start failing the moment real calibration
    values are added — update it then. It also locks in that calibrate_imu()
    returns a new IMUReading with copied arrays, not the original object, per
    Bassel's review comment removing imu_readings().
    """
    raw = sl.IMUReading(
        acceleration=np.array([0.1, -0.2, 9.81]),
        angular_velocity=np.array([0.01, -0.02, 0.03]),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
    )
    calibrated = sl.calibrate_imu(raw)

    assert np.array_equal(calibrated.acceleration, raw.acceleration)
    assert np.array_equal(calibrated.angular_velocity, raw.angular_velocity)
    assert np.array_equal(calibrated.orientation, raw.orientation)

    assert calibrated is not raw
    assert not np.shares_memory(calibrated.acceleration, raw.acceleration)
    assert not np.shares_memory(calibrated.angular_velocity, raw.angular_velocity)
    assert not np.shares_memory(calibrated.orientation, raw.orientation)


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
