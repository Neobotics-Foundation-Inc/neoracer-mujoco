"""
Laws for scene.py: code binds each actuator and sensor by its element in the
MjSpec, so the order of the actuators and sensors in the car XML does not
change what a command writes or what a read returns.
"""

import os
from dataclasses import fields, is_dataclass

import mujoco
import numpy as np
import pytest
from cases import case
from scenes import reversed_car, two_cars

from neoracer_mujoco import Scene, assets, contract, sensors, sim
from neoracer_mujoco.control.steering import ControlCommand
from neoracer_mujoco.scene import compile_scene


def _layout(scene: Scene) -> tuple[list[int], int]:
    """Where the actuators sit in ctrl and where one sensor sits in
    sensordata."""
    model, car = scene.model, scene.car
    ctrl = [model.bind_scalar(a).id for a in (car.drive, car.steer)]
    return ctrl, int(model.bind_scalar(car.sensors.imu_accel).adr[0])


def test_apply_command_writes_each_named_actuator():
    """apply_command writes the throttle, as that fraction of the motor's
    voltage limit, to the drive motor and the steer to the steering servo,
    whatever their order in the XML."""
    cars = (assets.load(), reversed_car())
    assert _layout(cars[0]) != _layout(cars[1])
    for throttle, steer in [(0.7, -0.3), (-1.0, 0.4)]:
        for scene in cars:
            data = mujoco.MjData(scene.model)
            sim.apply_command(
                scene.car, data, ControlCommand(throttle=throttle, steer=steer)
            )
            motor = scene.spec.actuator(contract.DRIVE_ACTUATOR)
            voltage = contract.actuator_limits(scene.model, scene.car).voltage
            assert data.bind_scalar(motor).ctrl[0] == throttle * voltage
            steer_servo = scene.spec.actuator(contract.STEER_ACTUATOR)
            assert data.bind_scalar(steer_servo).ctrl[0] == steer


def test_read_does_not_depend_on_the_sensor_order():
    """The two cars, settled from the same state, read the same values in
    every SensorReadings field."""

    def settled_readings(scene: Scene) -> sensors.SensorReadings:
        data = sim.settle(scene.model)
        return sensors.read(scene.car.sensors, data)

    car, reversed_ = assets.load(), reversed_car()
    assert _layout(car) != _layout(reversed_)
    a, b = settled_readings(car), settled_readings(reversed_)
    for f in fields(sensors.SensorReadings):
        np.testing.assert_allclose(
            getattr(a, f.name), getattr(b, f.name), atol=1e-9, err_msg=f.name
        )


def test_a_wheel_sets_the_friction_of_its_contacts():
    """The wheel geoms have contact priority, so each contact of a wheel with
    the floor takes the friction of the wheel alone, whatever the friction of
    the floor."""
    scene = assets.load()
    for i, wheel in enumerate(scene.car.wheels):
        scene.model.bind_scalar(wheel).friction[0] = 0.2 + 0.1 * i
    data = sim.settle(scene.model)
    ids = [int(scene.model.bind_scalar(w).id) for w in scene.car.wheels]
    touching = set()
    for contact in data.contact:
        for i, wheel_id in enumerate(ids):
            if wheel_id in contact.geom:
                assert contact.friction[0] == pytest.approx(0.2 + 0.1 * i)
                touching.add(i)
    assert touching == {0, 1, 2, 3}


def test_each_car_of_a_scene_has_its_own_parts():
    """Each Car of a scene of two holds the parts of the car of a one-car
    scene, each named behind the car's prefix if it has a name, so no part
    is both cars'."""
    track = os.path.join(assets.TRACK_DIR, "loop_corridor.xml")
    one = assets.compose(mujoco.MjSpec.from_file(track), visuals=False)
    two = two_cars()
    for prefix, car in zip(("a/", "b/"), two.cars, strict=True):
        with case(prefix):
            expected = [prefix + n if n else n for n in _names(one.car)]
            assert _names(car) == expected
            assert car.sensors.blocked == one.car.sensors.blocked


def _names(value: object) -> list[str]:
    """The name of each spec element in value, a Car or a part of one, in
    the order of its fields."""
    match value:
        case (
            mujoco.MjsBody()
            | mujoco.MjsJoint()
            | mujoco.MjsActuator()
            | mujoco.MjsGeom()
            | mujoco.MjsSite()
            | mujoco.MjsSensor()
        ):
            return [value.name]
        case tuple():
            return [name for v in value for name in _names(v)]
        case frozenset():  # sensors.blocked, the names of beams
            return []
        case _ if is_dataclass(value):
            return [n for f in fields(value) for n in _names(getattr(value, f.name))]
        case _:
            raise TypeError(f"{value!r} is no part of a car")


def test_an_actuator_without_a_name_is_refused():
    """attach puts no prefix on an actuator without a name, so in a scene of
    several cars none would own it: find_car refuses one, even on a car
    alone."""
    spec = assets.from_file()
    spec.add_actuator(
        target="steer_input",
        trntype=mujoco.mjtTrn.mjTRN_JOINT,
        userdata=[1.0, 0.0, 0.0],
    )
    with pytest.raises(ValueError, match="no name"):
        compile_scene(spec)


def test_a_mesh_that_is_a_git_lfs_pointer_is_named_as_one(tmp_path):
    """A clone made without Git LFS has pointers in place of the meshes. A
    scene with one fails with an error that names the file and says to pull
    it; a mesh broken another way keeps MuJoCo's own error."""
    pointer = (
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:cab05147aa867a6966441673c125a8fe5b1ecf6673d45cd25dc33b6184f65faf\n"
        "size 5000034\n"
    )
    for text, named in [(pointer, True), ("solid part\nendsolid part\n", False)]:
        with case(text):
            path = tmp_path / "part.STL"
            path.write_text(text)
            spec = assets.from_file()
            spec.add_mesh(name="part", file=str(path))
            spec.body(contract.CAR_BODY).add_geom(
                type=mujoco.mjtGeom.mjGEOM_MESH, meshname="part"
            )
            with pytest.raises(ValueError) as raised:
                compile_scene(spec)
            message = str(raised.value)
            assert (f"{path} is a Git LFS pointer" in message) == named
            assert ("git lfs pull" in message) == named
