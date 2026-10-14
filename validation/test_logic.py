"""
Pure-logic tests: math with a correct answer that does not depend on the
physics simulation. These tests find numerical errors that the behavior
tests in test_conformance.py do not find, for example a wrong unit
conversion or a polynomial fit that is a few percent wrong.
"""

from dataclasses import fields

import mujoco
import numpy as np
import pytest
from fake_sensors import fake_readings

from neoracer_mujoco import contract, sensors, truth


def test_read_returns_every_sensor(scene):
    """sensors.read returns every sensor of the model but the truth sensors,
    each with its size."""
    d = mujoco.MjData(scene.model)
    mujoco.mj_step(scene.model, d)
    out = sensors.read(scene.car.sensors, d)
    read_count = len(fields(sensors.SensorReadings))
    assert read_count + len(fields(truth.TruthSensors)) == scene.model.nsensor
    assert out.imu_accel.shape == (3,) and out.imu_gyro.shape == (3,)
    assert out.imu_quat.shape == (4,)
    assert isinstance(out.lidar_000, float) and isinstance(out.steer_cmd_pos, float)


def test_find_sensors_names_every_missing_or_wrong_size_sensor():
    """find_sensors parses the model once, and one ValueError names each
    problem: every missing sensor and every sensor with the wrong size."""
    spec = mujoco.MjSpec.from_string(
        """<mujoco><worldbody><body><site name="s"/></body></worldbody>
        <sensor><framepos name="steer_cmd_pos" objtype="site" objname="s"/></sensor>
        </mujoco>"""
    )
    with pytest.raises(ValueError) as error:
        sensors.find_sensors(spec, spec.compile())
    message = str(error.value)
    assert "steer_cmd_pos has 3 values, not 1" in message
    for f in fields(sensors.SensorReadings):
        if f.name != "steer_cmd_pos":
            assert f"{f.name} is missing" in message


def test_motor_encoder_is_mean_of_wheel_vel():
    """motor_encoder_rad_s() returns the mean of the four wheel speeds."""
    rng = np.random.default_rng(0)
    for fl, fr, rl, rr in rng.uniform(-100.0, 100.0, (50, 4)):
        readings = fake_readings(
            fl_wheel_vel=fl, fr_wheel_vel=fr, rl_wheel_vel=rl, rr_wheel_vel=rr
        )
        mean = (fl + fr + rl + rr) / 4
        assert sensors.motor_encoder_rad_s(readings) == pytest.approx(mean)


def test_estimated_linear_speed_is_linear_in_the_motor_speed():
    """The speed estimate (m/s) is the motor speed (rad/s) times
    WHEEL_RADIUS (m), for every motor speed, also a negative one."""
    rng = np.random.default_rng(1)
    for motor_rad_s in rng.uniform(-500.0, 500.0, 200):
        assert sensors.estimated_linear_speed_ms(motor_rad_s) == pytest.approx(
            motor_rad_s * sensors.WHEEL_RADIUS
        )


@pytest.mark.parametrize(
    ("constraint", "side"),
    [("ackermann_left", -1.0), ("ackermann_right", 1.0)],
)
def test_ackermann_polycoef_matches_exact_geometry(scene, constraint, side):
    """The steering polycoef in the car XML gives the exact Ackermann angle
    for the wheelbase and track of the car, to 0.01 rad. A + steer turns
    left, so the left wheel is the inner wheel (side -1) and the right wheel
    is the outer wheel (side +1)."""
    model, spec = scene.model, scene.spec
    eq = model.bind_scalar(spec.equality(constraint))
    fl = model.bind_scalar(spec.body("fl_wheel_body"))
    wheelbase = 2 * abs(float(fl.pos[0]))  # m
    track = 2 * abs(float(fl.pos[1]))  # m
    coef = np.array(eq.data[:5])  # a0..a4

    limit = contract.actuator_limits(model, scene.car).steer_rad  # rad
    # An even count of angles leaves out 0, where 1 / tan is not defined.
    delta = np.linspace(-limit, limit, 40)  # rad
    exact = np.arctan(wheelbase / (wheelbase / np.tan(delta) + side * track / 2))
    poly = np.polyval(coef[::-1], delta)
    err = float(np.max(np.abs(poly - exact)))
    assert err < 0.01, f"Ackermann polyfit off by {err:.4f} rad vs exact geometry"
