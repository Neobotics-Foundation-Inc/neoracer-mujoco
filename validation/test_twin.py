"""
Laws for neoracer_mujoco.twin: the car's XML holds the numbers derived from
the NeoRacer's parts, each derivation keeps what it derives from, and the
settings beside the model are the sensor modules' defaults.
"""

import dataclasses
import math

import numpy as np
import pytest

from neoracer_mujoco import assets, encoder, lidar
from neoracer_mujoco.twin import ackermann, derive, mjcf, parts
from neoracer_mujoco.twin.neoracer import NEORACER

CAR = NEORACER


@pytest.fixture(scope="module")
def scene():
    return assets.load()


def test_the_xml_holds_the_numbers_derived_from_the_parts():
    """The checked-in XML is what car_xml writes: after a change to the car,
    run python -m neoracer_mujoco.write_car_xml assets/neoracer.xml."""
    with open(assets.DEFAULT_CAR) as f:
        xml = f.read()
    assert mjcf.car_xml(derive.model(CAR), xml) == xml


def test_the_sensor_settings_are_the_sensor_modules_defaults():
    s = derive.settings(CAR)
    assert s.lidar == lidar.LidarParams()
    assert s.encoder == encoder.EncoderParams()


def test_the_drive_motor_turns_at_its_free_speed_and_stalls_at_its_torque(scene):
    """MuJoCo's dcmotor reads nominal as voltage, stall torque and free speed:
    its K and R give them back."""
    r, k = scene.model.bind_scalar(scene.car.drive).gainprm[:2]
    volts, stall, free = derive.model(CAR).motor.nominal
    assert volts / k == pytest.approx(free, rel=1e-5)
    assert volts * k / r == pytest.approx(stall, rel=1e-5)
    rating = CAR.motor.rating
    assert free == pytest.approx(math.radians(rating.free_speed_rpm * 6), rel=1e-5)


@pytest.mark.parametrize("factor", [0.5, 2.0])
def test_a_rewound_motor_stalls_in_proportion_to_its_free_speed(factor):
    rating = CAR.motor.rating
    faster = parts.rewound(rating, rating.free_speed_rpm * factor)
    assert faster.stall_torque == pytest.approx(rating.stall_torque * factor)
    assert faster.voltage == rating.voltage


@pytest.mark.parametrize("rating", [CAR.servo.low, CAR.servo.high])
def test_the_servo_has_its_datasheet_rating_at_a_rated_voltage(rating):
    servo = dataclasses.replace(CAR.servo, supply=rating.voltage)
    assert derive.servo_rating(servo) == pytest.approx(rating)


def test_the_servo_turns_its_horn_to_the_drivers_lock_at_full_lock(scene):
    """At full lock, the steering input times the linkage is the horn at
    driver_lock of its half range, and the servo's loop is clamped to its
    supply."""
    servo = scene.model.bind_scalar(scene.car.steer)
    lock = servo.ctrlrange[1]
    horn = CAR.steering.driver_lock * CAR.servo.horn_half_range_degrees
    assert lock * derive.linkage_ratio(CAR) == pytest.approx(math.radians(horn))
    assert servo.gainprm[7] == CAR.servo.supply


def test_the_servo_at_the_steering_input_is_its_rating_through_the_linkage(scene):
    """The compiled servo stalls at the horn's torque times the linkage and
    runs free at the horn's speed over it, the linkage being the horn's angle
    at the driver's lock over full lock."""
    servo = scene.model.bind_scalar(scene.car.steer)
    r, k, *_, vclamp = servo.gainprm[:8]
    horn = CAR.steering.driver_lock * CAR.servo.horn_half_range_degrees
    linkage = math.radians(horn) / servo.ctrlrange[1]
    rating = derive.servo_rating(CAR.servo)
    horn_stall = rating.stall_torque_kgf_cm * 9.80665 / 100  # N·m
    horn_free = math.radians(60) / rating.seconds_per_60_degrees  # rad/s
    assert vclamp * k / r == pytest.approx(horn_stall * linkage, rel=1e-5)
    assert vclamp / k == pytest.approx(horn_free / linkage, rel=1e-5)


def test_full_lock_turns_the_inner_wheel_to_its_lock():
    c = CAR.chassis
    inner = ackermann.left_wheel(c.wheelbase, c.track, derive.steer_lock(CAR))
    assert inner == pytest.approx(
        math.radians(CAR.steering.inner_lock_degrees), rel=1e-5
    )


def test_the_ackermann_polynomials_follow_ideal_ackermann_steering():
    """Each front wheel's polynomial is within 0.5 mrad of ideal Ackermann
    steering over the steering input's range, the right wheel mirroring the
    left."""
    c, r = CAR.chassis, CAR.steering.fit_range
    poly = np.array(derive.ackermann_left(CAR))
    x = np.linspace(-r, r, 101)
    x = x[x != 0]
    left = np.polynomial.polynomial.polyval(x, poly)
    ideal = np.array([ackermann.left_wheel(c.wheelbase, c.track, s) for s in x])
    assert np.max(np.abs(left - ideal)) < 5e-4
    right = np.polynomial.polynomial.polyval(x, poly * [-1, 1, -1, 1, -1])
    np.testing.assert_allclose(right, -left[::-1])


def test_the_front_wheels_turn_as_far_as_the_ackermann_constraints_ask(scene):
    """Over the steering input's whole range, the polynomials keep each front
    wheel inside its steer joint's range."""
    model = scene.model
    lo, hi = model.joint("steer_input").range
    for name, joint in [
        ("ackermann_left", "fl_steer"),
        ("ackermann_right", "fr_steer"),
    ]:
        poly = model.equality(name).data[:5]
        angles = np.polynomial.polynomial.polyval(np.linspace(lo, hi, 101), poly)
        j_lo, j_hi = model.joint(joint).range
        assert j_lo < angles.min() and angles.max() < j_hi


def test_the_springs_hold_the_sprung_weight_at_rest():
    weight = derive.sprung_mass(CAR) * derive.G
    assert -4 * CAR.suspension.stiffness * derive.spring_ref(CAR) == pytest.approx(
        weight
    )


def test_the_car_weighs_its_mass(scene):
    car = scene.model.bind_scalar(scene.car.body)
    assert car.subtreemass[0] == pytest.approx(CAR.chassis.mass)


def test_a_car_whose_full_lock_leaves_the_fit_range_is_refused():
    steering = dataclasses.replace(CAR.steering, inner_lock_degrees=31.0)
    with pytest.raises(ValueError, match="fit_range"):
        dataclasses.replace(CAR, steering=steering)


@pytest.mark.parametrize(
    "x", [0.0, 0.1, -0.2, 3.0, 1151.917, -0.0000136675, 0.0425 + 0.001]
)
def test_a_number_reads_back_at_the_models_precision(x):
    assert float(mjcf.number(x)) == derive.precise(x)
    assert derive.precise(x) == pytest.approx(x, rel=5e-6)
