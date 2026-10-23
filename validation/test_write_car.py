"""
Laws for write_car, which writes the complete cars for other MuJoCo tools.
"""

import os

import mujoco
import numpy as np
import pytest
from cases import case

from neoracer_mujoco import assets, write_car


def _saved(model: mujoco.MjModel) -> bytes:
    """The model as mj_saveModel writes it."""
    buffer = np.zeros(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, None, buffer)
    return buffer.tobytes()


def test_each_written_car_compiles_to_the_model_that_assets_loads(tmp_path):
    """A written car's XML and the files it names are complete: MuJoCo alone
    compiles them to the model of the car in assets/, byte for byte."""
    write_car.write(tmp_path)
    for path in assets.cars():
        name = os.path.basename(path)
        with case(name):
            written = mujoco.MjSpec.from_file(str(tmp_path / name)).compile()
            assert _saved(written) == _saved(assets.from_file(path).compile())


def test_write_car_refuses_assets():
    """assets/ holds the templates, which a complete car would replace."""
    with pytest.raises(ValueError, match="templates"):
        write_car.write(assets.ASSET_DIR)
