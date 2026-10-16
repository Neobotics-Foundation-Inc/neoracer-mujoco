"""
Laws for neoracer_mujoco.lidar, the spinning LiDAR.

The tests move the car by hand and call mj_forward, without physics steps:
sweep() reads only the pose in the data, so each slot's reading shows the
pose the car had when the head passed it.
"""

import math
import os

import mujoco
import numpy as np
import pytest

from neoracer_mujoco import Scene, assets, lidar, sim


@pytest.fixture(scope="module")
def corridor() -> Scene:
    track = mujoco.MjSpec.from_file(os.path.join(assets.TRACK_DIR, "loop_corridor.xml"))
    return assets.compose(track, visuals=False)


def _shift(scene: Scene, data: mujoco.MjData, dy: float) -> None:
    """Move the car dy (m) to its left, and update the poses in data."""
    data.bind_scalar(scene.car.joint).qpos[1] += dy
    mujoco.mj_forward(scene.model, data)


def test_it_sees_three_quarters_of_a_turn_centered_on_the_front():
    assert lidar.VISIBLE.sum() == 1080
    assert lidar.VISIBLE[0]
    assert not lidar.VISIBLE[lidar.SLOTS // 2]  # straight behind


def test_a_turn_completes_every_twelve_steps_at_360_hz(corridor):
    model, car = corridor.model, corridor.car
    data = sim.settle(model)
    state = lidar.start(model, data, car)
    completed = []
    for i in range(1, 37):
        last = state.scan
        state = lidar.sweep(model, data, car, state, 1 / 360)
        if state.scan is not last:
            completed.append(i)
    assert completed == [12, 24, 36]


def test_each_slot_reads_the_pose_of_the_step_that_passed_it(corridor):
    """Half a turn from one pose and half from another: the scan holds the
    first half of a scan from the first pose and the second half of a scan
    from the second, as the scan of a moving car is skewed."""
    model, car = corridor.model, corridor.car
    data = sim.settle(model)
    state = before = lidar.start(model, data, car)
    for _ in range(6):
        state = lidar.sweep(model, data, car, state, 1 / 360)
    _shift(corridor, data, 0.2)
    after = lidar.start(model, data, car)
    for _ in range(6):
        state = lidar.sweep(model, data, car, state, 1 / 360)
    half = lidar.SLOTS // 2
    assert not np.array_equal(before.scan, after.scan)
    np.testing.assert_array_equal(state.scan[:half], before.scan[:half])
    np.testing.assert_array_equal(state.scan[half:], after.scan[half:])


@pytest.mark.parametrize("hz", [360, 500, 1000, 60])
def test_a_turn_measures_every_slot_whatever_the_step(corridor, hz):
    """Move the car, then sweep one turn at a step that need not divide it:
    the next complete scan has no slot left from before the move."""
    model, car = corridor.model, corridor.car
    data = sim.settle(model)
    state = lidar.start(model, data, car)
    _shift(corridor, data, -0.2)
    first = state.scan
    for _ in range(math.ceil(hz / lidar.SCAN_RATE)):
        state = lidar.sweep(model, data, car, state, 1 / hz)
    assert state.scan is not first
    np.testing.assert_array_equal(state.scan, lidar.start(model, data, car).scan)


def test_a_step_of_a_turn_or_more_is_refused(corridor):
    model, car = corridor.model, corridor.car
    data = sim.settle(model)
    state = lidar.start(model, data, car)
    with pytest.raises(ValueError, match="turn"):
        lidar.sweep(model, data, car, state, 1 / lidar.SCAN_RATE)


def test_the_scan_plane_tilts_with_the_car(corridor):
    """The rays lie in the plane of the LiDAR site, so a car pitched nose
    down sees the floor ahead, nearer than the wall."""
    model, car = corridor.model, corridor.car
    data = sim.settle(model)
    level = lidar.start(model, data, car).scan[0]
    qpos = data.bind_scalar(car.joint).qpos
    pitch = 0.2  # rad, nose down: a rotation about +Y
    qpos[3:7] = [math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0]
    mujoco.mj_forward(model, data)
    site_z = data.bind_scalar(car.lidar).xpos[2]
    tilted = lidar.start(model, data, car).scan[0]
    assert tilted == pytest.approx(site_z / math.sin(pitch), rel=1e-3)
    assert tilted < level
