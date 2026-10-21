"""
Laws for neoracer_spec: every value of a car's specification has a source,
each car's XML holds the numbers derived from its spec, each derivation
keeps what it derives from, the settings beside the model are the sensor
modules' defaults, and the package runs without MuJoCo in the Python of the
browser's Pyodide.
"""

import ast
import builtins
import dataclasses
import math
import os
import pathlib
import subprocess
import sys

import mujoco
import numpy as np
import pytest

import neoracer_spec
from neoracer_mujoco import assets, encoder, lidar, sim
from neoracer_spec import (
    ackermann,
    derive,
    derive_v2,
    mjcf,
    provenance,
    racecar_neo_v2,
    spec,
)
from neoracer_spec.neoracer import NEORACER
from neoracer_spec.provenance import Sourced
from neoracer_spec.racecar_neo_v2 import RACECAR_NEO_V2

CAR = NEORACER
MODEL = derive.model(CAR)
V2_XML = os.path.join(assets.ASSET_DIR, "racecar_neo_v2.xml")


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


def test_the_v2_xml_holds_the_numbers_derived_from_its_spec():
    """The checked-in RACECAR Neo V2 XML is what racecar_neo_v2_xml writes:
    after a change to the car, run python -m neoracer_mujoco.write_car
    assets."""
    with open(V2_XML) as f:
        xml = f.read()
    assert mjcf.racecar_neo_v2_xml(derive_v2.model(RACECAR_NEO_V2), xml) == xml


def test_the_v2_model_has_the_cars_mass_centre_of_mass_and_inertia():
    """The chassis, the wheels and the steering input's body together weigh
    the car's mass, with its centre of mass and its principal moments of
    inertia, as MuJoCo computes them from the bodies. The derivation counts
    the steering input's body as a point; its own inertia is 1e-6 kg·m². The
    chassis' principal axes are the car's, so the wheels, below the centre
    of mass and not midway along the car, leave products of inertia of
    under 0.2% of the moments."""
    model = assets.load(V2_XML).model
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)  # at qpos0: the springs neither up nor down
    car = model.body("car").id
    bodies = [b for b in range(model.nbody) if b == car or model.body_rootid[b] == car]
    mass = sum(model.body_mass[b] for b in bodies)
    com = sum(model.body_mass[b] * data.xipos[b] for b in bodies) / mass
    inertia = np.zeros((3, 3))
    for b in bodies:
        rot = data.ximat[b].reshape(3, 3)
        d = data.xipos[b] - com
        inertia += rot @ np.diag(model.body_inertia[b]) @ rot.T
        inertia += model.body_mass[b] * (d @ d * np.eye(3) - np.outer(d, d))
    c = RACECAR_NEO_V2.chassis
    spawn = data.xpos[car]
    assert mass == pytest.approx(c.mass.value)
    np.testing.assert_allclose(
        com - spawn, derive_v2.car_com(RACECAR_NEO_V2), atol=1e-6
    )
    np.testing.assert_allclose(np.diag(inertia), c.inertia.value, rtol=1e-3)
    np.testing.assert_allclose(inertia - np.diag(np.diag(inertia)), 0, atol=3e-5)


def test_the_v2_rests_in_the_middle_of_its_travel():
    """Each axle's springs hold their share of the weight with their joints
    at 0, the middle of their travel: the settled car's suspension is
    within 1 mm of it."""
    scene = assets.load(V2_XML)
    data = sim.settle(scene.model)
    for name in ("fl_susp", "fr_susp", "rl_susp", "rr_susp"):
        assert abs(data.joint(name).qpos[0]) < 0.001, name


def test_the_v2_spec_lists_its_questions_for_racecarsim():
    """racecarsim_questions finds where RacecarSim's model of the V2 is not
    the car: each value with a conflict from that model, and each the spec
    takes from it unchecked or corrects."""
    questions = racecar_neo_v2.racecarsim_questions(RACECAR_NEO_V2)
    assert [path for path, _ in questions] == [
        "chassis.wheelbase",
        "chassis.ride_height",
        "footprint.front",
        "footprint.rear",
        "footprint.right",
        "footprint.lidar_top",
        "footprint.antenna_top",
        "footprint.payload_bottom",
        "shape.payload_drop",
        "lidar.position",
        "imu.position",
        "camera.bottom",
    ]


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


def _type_params(params: list[ast.type_param]) -> set[str]:
    return {
        p.name
        for p in params
        if isinstance(p, ast.TypeVar | ast.ParamSpec | ast.TypeVarTuple)
    }


def _named_before_defined(tree: ast.Module) -> list[str]:
    """The names that the module's annotations use before it defines them,
    when Python 3.12 evaluates them: a class's when it defines the class, a
    function's when it defines the function. None when the module postpones
    its annotations."""
    if any(
        isinstance(node, ast.ImportFrom)
        and node.module == "__future__"
        and any(alias.name == "annotations" for alias in node.names)
        for node in tree.body
    ):
        return []
    defined, early = set(dir(builtins)), []

    def check(annotations: list[ast.expr | None], scope: set[str]) -> None:
        for annotation in annotations:
            for n in ast.walk(annotation) if annotation else ():
                if isinstance(n, ast.Name) and n.id not in scope:
                    early.append(n.id)

    def signature(f: ast.FunctionDef, scope: set[str]) -> None:
        a = f.args
        args = a.posonlyargs + a.args + a.kwonlyargs + [a.vararg, a.kwarg]
        own = scope | _type_params(f.type_params)
        check([x.annotation for x in args if x] + [f.returns], own)

    for node in tree.body:
        match node:
            case ast.Import() | ast.ImportFrom():
                defined |= {(a.asname or a.name).split(".")[0] for a in node.names}
            case ast.ClassDef():
                scope = defined | _type_params(node.type_params)
                for item in node.body:
                    if isinstance(item, ast.AnnAssign):
                        check([item.annotation], scope)
                    elif isinstance(item, ast.FunctionDef):
                        signature(item, scope)
                defined.add(node.name)
            case ast.FunctionDef():
                signature(node, defined)
                defined.add(node.name)
            case ast.AnnAssign(target=ast.Name(id=name)):
                check([node.annotation], defined)
                defined.add(name)
            case ast.Assign():
                defined |= {t.id for t in node.targets if isinstance(t, ast.Name)}
            case ast.TypeAlias(name=ast.Name(id=name)):
                defined.add(name)
    return early


def test_the_spec_imports_on_the_python_of_pyodide():
    """Python 3.12 evaluates annotations as it defines what they annotate,
    so a module whose types name types defined below them postpones its
    annotations (from __future__ import annotations): Pyodide can import
    every module of the package."""
    for path in sorted(pathlib.Path(neoracer_spec.__file__).parent.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.name)
        assert _named_before_defined(tree) == [], path.name
