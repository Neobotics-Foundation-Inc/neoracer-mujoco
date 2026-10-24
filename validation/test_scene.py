"""
Laws for scene.py: code binds each actuator and sensor by its element in the
MjSpec, so the order of the actuators and sensors in the car XML does not
change what a command writes or what a read returns.
"""

from dataclasses import fields

import mujoco
import numpy as np
import pytest
from scenes import reversed_car

from neoracer_mujoco import Scene, assets, contract, sensors, sim
from neoracer_mujoco.control.steering import ControlCommand


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
