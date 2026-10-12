"""Scenes and scene laws that several test files use."""

import os
import warnings

import mujoco
import numpy as np

from neoracer_mujoco import assets, sim


def corridor_model() -> mujoco.MjModel:
    """The default car on assets/tracks/straight_corridor.xml. The corridor
    runs along +X and has a half-width of 0.5 m."""
    return assets.compose(
        mujoco.MjSpec.from_file(os.path.join(assets.TRACK_DIR, "straight_corridor.xml"))
    )


def assert_keeps_car_solver_options(
    track: mujoco.MjSpec, car_path: str = assets.DEFAULT_CAR
) -> None:
    """Attach the car to track and check the solver options of the result.

    When MuJoCo attaches a car to a track, it keeps the <option> of the track.
    So the track must have the friction cone and impratio of the car, and the
    attach must give no warning."""
    car = assets.load(car_path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        composed = assets.compose(track, car_path)
    assert [str(w.message) for w in caught] == []
    assert composed.opt.cone == car.opt.cone
    assert composed.opt.impratio == car.opt.impratio


def assert_car_rests(model: mujoco.MjModel) -> None:
    """Check that the car settles on the floor of the scene with zero
    control: it does not fall through and it does not move."""
    data = sim.settle(model)
    car = data.body("car")
    assert 0.05 < float(car.xpos[2]) < 0.2, f"car z = {car.xpos[2]:.3f} m"
    assert float(np.linalg.norm(data.qvel[:3])) < 0.05  # m/s
