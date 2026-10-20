"""
Laws for sensors.read and sensors.find_sensors: the model's sensors are
read by name, each with its size, and a model that lacks one is refused
with every problem named.
"""

from dataclasses import fields

import mujoco
import pytest

from neoracer_mujoco import sensors, truth


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
