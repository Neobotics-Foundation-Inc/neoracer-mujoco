"""
Laws for write_car, which writes the complete cars for other MuJoCo tools.
"""

import os

import mujoco
import pytest
from cases import case
from scenes import assert_same_model

from neoracer_mujoco import assets, write_car


def test_each_written_car_compiles_to_the_model_that_assets_loads(tmp_path):
    """A written car's XML and the files it names are complete: MuJoCo alone
    compiles them to the model of the car in assets/."""
    write_car.write(tmp_path)
    for path in assets.cars():
        name = os.path.basename(path)
        with case(name):
            written = mujoco.MjSpec.from_file(str(tmp_path / name)).compile()
            assert_same_model(written, assets.from_file(path).compile())


def test_write_car_refuses_assets():
    """assets/ holds the templates, which a complete car would replace."""
    with pytest.raises(ValueError, match="templates"):
        write_car.write(assets.ASSET_DIR)
