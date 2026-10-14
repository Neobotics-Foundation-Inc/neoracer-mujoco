"""
Laws for physics.py: apply_physics gives the model that compiling the
changed car would give, Physics() changes nothing, FIELDS lists everything it
changes, and the wheel friction factor changes how much the wheels slip.
"""

import copy
import os

import mujoco
import numpy as np
import pytest

from neoracer_mujoco import Scene, assets, physics, sim
from neoracer_mujoco.physics import Physics, PhysicsRanges, Range
from neoracer_mujoco.scene import find_car

TRACK = os.path.join(assets.TRACK_DIR, "loop_corridor.xml")
CHANGED = Physics(
    wheel_friction=0.4, chassis_mass=1.3, motor_torque=0.7, steer_stiffness=1.6
)


def _scene() -> Scene:
    return assets.compose(mujoco.MjSpec.from_file(TRACK), pos=(0, -2.5, 0))


def _applied(scene: Scene, p: Physics) -> mujoco.MjModel:
    model = copy.deepcopy(scene.model)
    physics.apply_physics(model, scene, p)
    return model


def _drive(model: mujoco.MjModel, scene: Scene) -> np.ndarray:
    """The sensordata of 1 s of driving and steering, (500, nsensordata)."""
    data = sim.settle(model)
    out = []
    for t in range(500):
        for motor in scene.car.drive:
            data.bind_scalar(motor).ctrl = 0.5  # N·m
        data.bind_scalar(scene.car.steer).ctrl = 0.3 * np.sin(t / 60)  # rad
        mujoco.mj_step(model, data)
        out.append(data.sensordata.copy())
    return np.array(out)


def test_apply_physics_equals_compiling_the_changed_car():
    """The same factors written into the MjSpec and compiled give the same
    drive, bit for bit."""
    scene = _scene()
    edited, car = scene.spec.copy(), scene.car  # car names the elements
    for wheel in car.wheels:
        edited.geom(wheel.name).friction[0] *= CHANGED.wheel_friction
    body = edited.body(car.body.name)
    body.mass *= CHANGED.chassis_mass
    body.inertia = np.asarray(body.inertia) * CHANGED.chassis_mass
    for motor in car.drive:
        edited.actuator(motor.name).gainprm[0] *= CHANGED.motor_torque
    servo = edited.actuator(car.steer.name)
    servo.gainprm[0] *= CHANGED.steer_stiffness
    servo.biasprm[1] *= CHANGED.steer_stiffness
    model = edited.compile()  # edited has the truth sensors already
    compiled = Scene(edited, model, find_car(edited, model))
    np.testing.assert_array_equal(
        _drive(_applied(scene, CHANGED), scene), _drive(compiled.model, compiled)
    )


def test_the_nominal_physics_changes_no_field():
    scene = _scene()
    model = _applied(scene, Physics())
    for name in physics.FIELDS:
        np.testing.assert_array_equal(
            physics.field(model, name), physics.field(scene.model, name), err_msg=name
        )


def test_fields_lists_every_field_that_apply_physics_changes():
    """The Warp batch gives each environment its own copy of FIELDS only, so
    a field that apply_physics changes and FIELDS misses would be lost."""
    scene = _scene()
    model = _applied(scene, CHANGED)
    changed = {
        name
        for name in dir(model)
        if isinstance(getattr(model, name), np.ndarray)
        and not np.array_equal(getattr(model, name), getattr(scene.model, name))
    }
    # light_poscom0 is where a light that tracks the center of mass starts; it
    # moves no body.
    if model.stat.meaninertia != scene.model.stat.meaninertia:
        changed.add("stat.meaninertia")
    assert changed - {"light_poscom0"} == set(physics.FIELDS)


def test_less_wheel_friction_means_more_slip():
    """At full motor torque from rest, the rear wheels spin faster than the
    car moves, and more so as the friction drops."""
    scene = _scene()
    slips = []
    for friction in (0.2, 0.5, 1.0):
        model = _applied(scene, Physics(wheel_friction=friction))
        data = sim.settle(model)
        for motor in scene.car.drive:
            data.bind_scalar(motor).ctrl = 1.0  # N·m
        for _ in range(250):  # 0.5 s
            mujoco.mj_step(model, data)
        surface = data.bind_scalar(scene.car.sensors.rl_wheel_vel).data[0] * 0.05
        speed = data.bind_scalar(scene.car.sensors.imu_linvel).data[0]
        slips.append(surface - speed)  # m/s
    assert slips[0] > slips[1] > slips[2] > 0


def test_sample_draws_each_factor_from_its_range():
    ranges = PhysicsRanges(
        wheel_friction=Range(0.3, 0.6),
        chassis_mass=Range(0.9, 1.1),
        motor_torque=Range(0.5, 0.5),
    )
    rng = np.random.default_rng(0)
    for _ in range(200):
        p = physics.sample(ranges, rng)
        assert 0.3 <= p.wheel_friction <= 0.6
        assert 0.9 <= p.chassis_mass <= 1.1
        assert p.motor_torque == 0.5
        assert p.steer_stiffness == 1.0


@pytest.mark.parametrize(("low", "high"), [(0.0, 1.0), (-0.5, 1.0), (1.2, 1.1)])
def test_a_range_needs_0_below_low_below_high(low, high):
    with pytest.raises(ValueError, match="range"):
        Range(low, high)
