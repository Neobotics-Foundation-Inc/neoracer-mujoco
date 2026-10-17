"""
Laws for neoracer_mujoco.hardware: the car's XML holds the numbers derived
from the hardware, and each derivation keeps what it derives from.
"""

import dataclasses
import math

import numpy as np
import pytest

from neoracer_mujoco import assets, hardware
from neoracer_mujoco.hardware import Hardware

HW = Hardware()


@pytest.fixture(scope="module")
def scene():
    return assets.load()


def test_the_xml_holds_the_numbers_derived_from_the_hardware():
    """The checked-in XML is what car_xml writes: after a change to the
    hardware, run python -m neoracer_mujoco.write_car_xml assets/neoracer.xml."""
    with open(assets.DEFAULT_CAR) as f:
        xml = f.read()
    assert hardware.car_xml(HW, xml) == xml


def test_the_drive_motor_turns_at_its_free_speed_and_stalls_at_its_torque(scene):
    """MuJoCo's dcmotor reads nominal as voltage, stall torque and free speed:
    its K and R give them back."""
    r, k = scene.model.bind_scalar(scene.car.drive).gainprm[:2]
    volts, stall, free = hardware.motor_nominal(HW.motor)
    assert volts / k == pytest.approx(free, rel=1e-5)
    assert volts * k / r == pytest.approx(stall, rel=1e-5)
    assert free == pytest.approx(math.radians(HW.motor.free_speed_rpm * 6))


@pytest.mark.parametrize("factor", [0.5, 2.0])
def test_the_stall_torque_scales_with_the_free_speed(factor):
    """Rewound in the same frame, a motor at a given voltage stalls in
    proportion to its free speed."""
    faster = dataclasses.replace(
        HW.motor, free_speed_rpm=HW.motor.free_speed_rpm * factor
    )
    stall = hardware.motor_nominal(HW.motor)[1]
    assert hardware.motor_nominal(faster)[1] == pytest.approx(stall * factor)


@pytest.mark.parametrize("rating", [HW.servo.low, HW.servo.high])
def test_the_servo_has_its_datasheet_rating_at_a_rated_voltage(rating):
    servo = dataclasses.replace(HW.servo, supply=rating.voltage)
    assert hardware.servo_rating(servo) == pytest.approx(rating)


def test_the_servo_turns_its_horn_to_the_drivers_lock_at_full_lock(scene):
    """At full lock, steer_input times the linkage is the horn at driver_lock
    of its half range, and the servo's loop is clamped to its supply."""
    servo = scene.model.bind_scalar(scene.car.steer)
    lock = servo.ctrlrange[1]
    horn = HW.steering.driver_lock * HW.servo.horn_half_range_degrees
    assert lock * hardware.linkage_ratio(HW) == pytest.approx(math.radians(horn))
    assert servo.gainprm[7] == HW.servo.supply


def test_the_servo_at_steer_input_is_its_rating_through_the_linkage(scene):
    """The compiled servo stalls at the horn's torque times the linkage and
    runs free at the horn's speed over it, the linkage being the horn's angle
    at the driver's lock over full lock."""
    servo = scene.model.bind_scalar(scene.car.steer)
    r, k, *_, vclamp = servo.gainprm[:8]
    horn = HW.steering.driver_lock * HW.servo.horn_half_range_degrees
    linkage = math.radians(horn) / servo.ctrlrange[1]
    rating = hardware.servo_rating(HW.servo)
    horn_stall = rating.stall_torque_kgf_cm * 9.80665 / 100  # N·m
    horn_free = math.radians(60) / rating.seconds_per_60_degrees  # rad/s
    assert vclamp * k / r == pytest.approx(horn_stall * linkage, rel=1e-5)
    assert vclamp / k == pytest.approx(horn_free / linkage, rel=1e-5)


def test_full_lock_turns_the_inner_wheel_to_the_measured_lock():
    inner = hardware.left_wheel_angle(HW.chassis, hardware.steer_lock(HW))
    assert inner == pytest.approx(
        math.radians(HW.steering.inner_lock_degrees), rel=1e-5
    )


def test_the_ackermann_polynomials_follow_ideal_ackermann_steering():
    """Each front wheel's polynomial is within 0.5 mrad of ideal Ackermann
    steering over steer_input's range, the right wheel mirroring the left."""
    poly = hardware.ackermann(HW)
    x = np.linspace(-HW.steering.fit_range, HW.steering.fit_range, 101)
    x = x[x != 0]
    left = np.polynomial.polynomial.polyval(x, poly)
    ideal = np.array([hardware.left_wheel_angle(HW.chassis, s) for s in x])
    assert np.max(np.abs(left - ideal)) < 5e-4
    right = np.polynomial.polynomial.polyval(x, poly * [-1, 1, -1, 1, -1])
    np.testing.assert_allclose(right, -left[::-1])


def test_the_springs_hold_the_sprung_weight_at_rest():
    weight = hardware.sprung_mass(HW.chassis) * hardware.G
    assert -4 * HW.suspension.stiffness * hardware.springref(HW) == pytest.approx(
        weight
    )


def test_the_car_weighs_the_spec(scene):
    car = scene.model.bind_scalar(scene.car.body)
    assert car.subtreemass[0] == pytest.approx(HW.chassis.mass)


def test_hardware_whose_full_lock_leaves_the_fit_range_is_refused():
    steering = dataclasses.replace(HW.steering, inner_lock_degrees=31.0)
    with pytest.raises(ValueError, match="fit_range"):
        Hardware(steering=steering)


@pytest.mark.parametrize(
    "x", [0.0, 0.1, -0.2, 3.0, 1151.917, -0.0000136675, 0.0425 + 0.001]
)
def test_a_number_reads_back_at_the_models_precision(x):
    assert float(hardware.number(x)) == hardware.precise(x)
    assert hardware.precise(x) == pytest.approx(x, rel=5e-6)
