"""
Laws for neoracer_spec: every value of a car's specification has a source,
the NeoRacer's XML holds the numbers derived from its spec, each derivation
keeps what it derives from, the settings beside the model are the sensor
modules' defaults, and the package runs without MuJoCo in the Python of the
browser's Pyodide.
"""

import ast
import dataclasses
import math
import pathlib
import subprocess
import sys

import numpy as np
import pytest

import neoracer_spec
from neoracer_mujoco import assets, encoder, lidar
from neoracer_spec import ackermann, derive, mjcf, provenance, spec
from neoracer_spec.neoracer import NEORACER
from neoracer_spec.provenance import Sourced
from neoracer_spec.racecar_neo_v2 import RACECAR_NEO_V2

CAR = NEORACER
MODEL = derive.model(CAR)


def value[T](s: Sourced[T], v: T) -> Sourced[T]:
    """s with the value v and the same source."""
    return dataclasses.replace(s, value=v)


@pytest.fixture(scope="module")
def scene():
    return assets.load()


def test_every_value_of_a_car_has_a_source():
    """In every car's spec, each part's every field is a Sourced, walk finds
    them all, and every conflict a value lists disagrees with it."""
    for car in (NEORACER, RACECAR_NEO_V2):
        paths = []
        for part in dataclasses.fields(car):
            p = getattr(car, part.name)
            for f in dataclasses.fields(p):
                path = f"{type(car).__name__}.{part.name}.{f.name}"
                assert isinstance(getattr(p, f.name), Sourced), path
                paths.append(path)
        walked = [f"{type(car).__name__}.{path}" for path, _ in provenance.walk(car)]
        assert walked == paths
        for path, s in provenance.walk(car):
            for alt in s.conflicts:
                assert alt.value != s.value, path


def test_the_xml_holds_the_numbers_derived_from_the_spec():
    """The checked-in XML is what car_xml writes: after a change to the car,
    run python -m neoracer_mujoco.write_car assets."""
    with open(assets.DEFAULT_CAR) as f:
        xml = f.read()
    assert mjcf.car_xml(MODEL, xml) == xml


BOX = CAR.chassis.collision_box.value


def test_a_change_to_the_chassis_changes_the_xml():
    """Every chassis value reaches the model: a car with one of them changed
    gets a different XML."""
    with open(assets.DEFAULT_CAR) as f:
        xml = f.read()
    c = CAR.chassis
    for name, chassis in [
        (
            "collision box center",
            dataclasses.replace(
                c,
                collision_box=value(
                    c.collision_box, spec.Box((0.02, 0.0, 0.07), BOX.half_size)
                ),
            ),
        ),
        (
            "collision box size",
            dataclasses.replace(
                c,
                collision_box=value(
                    c.collision_box, spec.Box(BOX.center, (0.2, 0.1, 0.09))
                ),
            ),
        ),
        ("wheelbase", dataclasses.replace(c, wheelbase=value(c.wheelbase, 0.29))),
        ("com_height", dataclasses.replace(c, com_height=value(c.com_height, 0.02))),
    ]:
        car = dataclasses.replace(CAR, chassis=chassis)
        assert mjcf.car_xml(derive.model(car), xml) != xml, name


def test_the_derived_settings_are_the_sensor_modules_defaults():
    p = derive.derive(CAR)
    assert p.model == MODEL
    assert p.lidar == lidar.LidarParams()
    assert p.encoder == encoder.EncoderParams()


def test_the_drive_motor_turns_at_its_free_speed_and_stalls_at_its_torque(scene):
    """MuJoCo's dcmotor reads nominal as voltage, stall torque and free speed:
    its K and R give them back."""
    r, k = scene.model.bind_scalar(scene.car.drive).gainprm[:2]
    volts, stall, free = MODEL.motor.nominal
    assert volts / k == pytest.approx(free, rel=1e-5)
    assert volts * k / r == pytest.approx(stall, rel=1e-5)
    rating = CAR.motor.rating.value
    assert free == pytest.approx(math.radians(rating.free_speed_rpm * 6), rel=1e-5)


def test_a_rewound_motor_stalls_in_proportion_to_its_free_speed():
    rating = CAR.motor.rating.value
    for factor in (0.5, 2.0):
        faster = spec.rewound(rating, rating.free_speed_rpm * factor)
        assert faster.stall_torque == pytest.approx(rating.stall_torque * factor)
        assert faster.voltage == rating.voltage


def test_the_servo_has_its_datasheet_rating_at_a_rated_voltage():
    for rating in (CAR.servo.low.value, CAR.servo.high.value):
        servo = dataclasses.replace(
            CAR.servo, supply=value(CAR.servo.supply, rating.voltage)
        )
        assert derive.servo_rating(servo) == pytest.approx(rating), rating.voltage


def test_the_servo_at_the_steering_input_is_its_rating_through_the_linkage(scene):
    """The linkage is the horn's angle at the driver's lock of its half range
    over full lock. The compiled servo stalls at the horn's torque times the
    linkage, runs free at the horn's speed over it, and its loop is clamped
    to its supply."""
    servo = scene.model.bind_scalar(scene.car.steer)
    r, k, *_, vclamp = servo.gainprm[:8]
    lock = servo.ctrlrange[1]
    horn = CAR.steering.driver_lock.value * CAR.servo.horn_half_range_degrees.value
    linkage = math.radians(horn) / lock
    assert derive.linkage_ratio(CAR) == pytest.approx(linkage)
    assert vclamp == CAR.servo.supply.value
    rating = derive.servo_rating(CAR.servo)
    horn_stall = rating.stall_torque_kgf_cm * 9.80665 / 100  # N·m
    horn_free = math.radians(60) / rating.seconds_per_60_degrees  # rad/s
    assert vclamp * k / r == pytest.approx(horn_stall * linkage, rel=1e-5)
    assert vclamp / k == pytest.approx(horn_free / linkage, rel=1e-5)


def test_full_lock_turns_the_inner_wheel_to_its_lock():
    c = CAR.chassis
    lock = derive.steer_lock(CAR)
    inner = ackermann.left_wheel(c.wheelbase.value, c.track.value, lock)
    assert inner == pytest.approx(
        math.radians(CAR.steering.inner_lock_degrees.value), rel=1e-5
    )


def test_the_ackermann_polynomials_follow_ideal_ackermann_steering():
    """Each front wheel's polynomial is within 0.5 mrad of ideal Ackermann
    steering over the steering input's range, the right wheel mirroring the
    left."""
    l, w = CAR.chassis.wheelbase.value, CAR.chassis.track.value
    r = CAR.steering.fit_range.value
    poly = np.array(derive.ackermann_left(CAR))
    x = np.linspace(-r, r, 101)
    x = x[x != 0]
    left = np.polynomial.polynomial.polyval(x, poly)
    ideal = np.array([ackermann.left_wheel(l, w, s) for s in x])
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
    stiffness = CAR.suspension.stiffness.value
    assert -4 * stiffness * derive.spring_ref(CAR) == pytest.approx(weight)


def test_the_car_weighs_its_mass(scene):
    car = scene.model.bind_scalar(scene.car.body)
    assert car.subtreemass[0] == pytest.approx(CAR.chassis.mass.value)


def test_a_car_whose_full_lock_leaves_the_fit_range_is_refused():
    s = CAR.steering
    steering = dataclasses.replace(
        s, inner_lock_degrees=value(s.inner_lock_degrees, 31.0)
    )
    with pytest.raises(ValueError, match="fit_range"):
        dataclasses.replace(CAR, steering=steering)


def test_a_number_reads_back_at_the_models_precision():
    for x in (0.0, 0.1, -0.2, 3.0, 1151.917, -0.0000136675, 0.0425 + 0.001):
        assert float(mjcf.number(x)) == derive.precise(x), x
        assert derive.precise(x) == pytest.approx(x, rel=5e-6), x


def test_the_spec_runs_without_mujoco():
    code = (
        "import sys\n"
        "from neoracer_spec import derive, mjcf\n"
        "from neoracer_spec.neoracer import NEORACER\n"
        "derive.derive(NEORACER)\n"
        "assert 'mujoco' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_the_spec_keeps_to_the_python_of_pyodide():
    """The browser's Pyodide runs Python 3.12: the package keeps to its
    syntax."""
    for path in sorted(pathlib.Path(neoracer_spec.__file__).parent.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        ast.parse(source, filename=path.name, feature_version=(3, 12))
