"""
Laws for neoracer_mujoco.twin.export: any car, the NeoRacer or a variant of
it, gives an MjSpec that compiles and composes onto a track, and a folder
whose archive loads to the same model.
"""

import dataclasses
import os

import mujoco
import numpy as np
import pytest

from neoracer_mujoco import assets
from neoracer_mujoco.scene import compile_scene
from neoracer_mujoco.twin import export
from neoracer_mujoco.twin.neoracer import NEORACER
from neoracer_mujoco.twin.parts import Battery

SLOW = dataclasses.replace(NEORACER, battery=Battery(voltage=7.4))


def test_the_neoracers_xml_is_the_checked_in_xml():
    with open(assets.DEFAULT_CAR, encoding="utf-8") as f:
        assert export.car_xml(NEORACER) == f.read()


def test_the_neoracers_settings_are_the_checked_in_json():
    """After a change to the car, run python -m neoracer_mujoco.write_car assets."""
    path = os.path.join(assets.ASSET_DIR, export.SETTINGS_NAME)
    with open(path, encoding="utf-8") as f:
        assert export.settings_json(NEORACER) == f.read()


def test_a_variant_compiles_with_its_parts():
    scene = compile_scene(export.spec(SLOW))
    motor = scene.model.bind_scalar(scene.car.drive)
    np.testing.assert_array_equal(motor.ctrlrange, [-7.4, 7.4])


def test_a_variant_composes_onto_a_track():
    track = mujoco.MjSpec.from_file(os.path.join(assets.TRACK_DIR, "loop_corridor.xml"))
    scene = assets.compose(track, export.spec(SLOW), visuals=False)
    motor = scene.model.bind_scalar(scene.car.drive)
    assert motor.ctrlrange[1] == pytest.approx(7.4)


def test_a_written_archive_loads_to_the_cars_model(tmp_path):
    export.write(SLOW, tmp_path)
    archived = mujoco.MjSpec.from_zip(str(tmp_path / export.MJZ_NAME)).compile()
    direct = export.spec(SLOW).compile()
    settings = (tmp_path / export.SETTINGS_NAME).read_text(encoding="utf-8")
    assert settings == export.settings_json(SLOW)
    for field in (
        "body_mass",
        "body_inertia",
        "actuator_gainprm",
        "eq_data",
        "geom_size",
    ):
        np.testing.assert_array_equal(getattr(archived, field), getattr(direct, field))


def test_writing_into_assets_is_refused():
    with pytest.raises(ValueError, match="assets"):
        export.write(SLOW, assets.ASSET_DIR)
    with open(assets.DEFAULT_CAR, encoding="utf-8") as f:
        assert export.car_xml(NEORACER) == f.read()
