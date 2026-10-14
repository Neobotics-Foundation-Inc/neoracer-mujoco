"""
Laws for scene.py: code binds each actuator and sensor by its element in the
MjSpec, so the order of the actuators and sensors in the car XML does not
change what a command writes or what a read returns.
"""

import xml.etree.ElementTree as ET
from dataclasses import fields

import mujoco
import numpy as np
import pytest

from neoracer_mujoco import Scene, assets, contract, sensors, sim
from neoracer_mujoco.control.steering import ControlCommand
from neoracer_mujoco.scene import compile_scene


def _reversed_car() -> Scene:
    """The default car with its actuators and its sensors in reverse order."""
    root = ET.parse(assets.DEFAULT_CAR).getroot()
    for tag in ("actuator", "sensor"):
        section = root.find(tag)
        assert section is not None, tag
        children = list(section)
        for child in children:
            section.remove(child)
        section.extend(reversed(children))
    spec = mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))
    # A spec made from a string has no file directory, so give it the one of
    # the car XML. Mesh paths in neoracer.xml are relative to it.
    spec.meshdir = assets.ASSET_DIR
    return compile_scene(spec)


def test_reversed_car_has_another_layout():
    """The reordered car really moves the actuators and the sensors in the
    compiled arrays, so the laws below compare two different layouts."""

    def layout(scene: Scene) -> tuple[list[int], int]:
        model, car = scene.model, scene.car
        ctrl = [model.bind_scalar(a).id for a in (*car.drive, car.steer)]
        return ctrl, int(model.bind_scalar(car.sensors.imu_accel).adr[0])

    assert layout(assets.load()) != layout(_reversed_car())


@pytest.mark.parametrize(("throttle", "steer"), [(0.7, -0.3), (-1.0, 0.4)])
def test_apply_command_writes_each_named_actuator(throttle, steer):
    """apply_command writes the throttle to every drive motor and the steer
    to the steering servo, whatever their order in the XML."""
    for scene in (assets.load(), _reversed_car()):
        data = mujoco.MjData(scene.model)
        sim.apply_command(
            scene.car, data, ControlCommand(throttle=throttle, steer=steer)
        )
        for name in contract.DRIVE_ACTUATORS:
            motor = scene.spec.actuator(name)
            assert data.bind_scalar(motor).ctrl[0] == throttle, name
        steer_servo = scene.spec.actuator(contract.STEER_ACTUATOR)
        assert data.bind_scalar(steer_servo).ctrl[0] == steer


def test_read_does_not_depend_on_the_sensor_order():
    """The two cars, settled from the same state, read the same values in
    every SensorReadings field."""

    def settled_readings(scene: Scene) -> sensors.SensorReadings:
        data = sim.settle(scene.model)
        return sensors.read(scene.car.sensors, data)

    a, b = settled_readings(assets.load()), settled_readings(_reversed_car())
    for f in fields(sensors.SensorReadings):
        np.testing.assert_allclose(
            getattr(a, f.name), getattr(b, f.name), atol=1e-9, err_msg=f.name
        )
