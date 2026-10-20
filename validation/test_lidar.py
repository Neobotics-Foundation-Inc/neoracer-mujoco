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

LIDAR = lidar.LidarParams()


@pytest.fixture(scope="module")
def corridor() -> Scene:
    track = mujoco.MjSpec.from_file(os.path.join(assets.TRACK_DIR, "loop_corridor.xml"))
    return assets.compose(track, visuals=False)


def _shift(scene: Scene, data: mujoco.MjData, dy: float) -> None:
    """Move the car dy (m) to its left, and update the poses in data."""
    data.bind_scalar(scene.car.joint).qpos[1] += dy
    mujoco.mj_forward(scene.model, data)


def test_it_sees_three_quarters_of_a_turn_centered_on_the_front():
    seen = lidar.visible(LIDAR)
    assert seen.sum() == 1080
    assert seen[0]
    assert not seen[LIDAR.slots // 2]  # straight behind


def _wall_ahead(x: float) -> Scene:
    """The car at the origin facing a wall x (m) ahead, with nothing else in
    the scan plane."""
    track = mujoco.MjSpec.from_string(
        f"""<mujoco>
        <option timestep="0.002777777777777778" integrator="implicitfast"
                cone="elliptic" impratio="10"/>
        <worldbody>
        <geom type="plane" size="0 0 1"/>
        <geom type="box" pos="{x + 0.05} 0 1" size=".05 2 1"/>
        </worldbody></mujoco>"""
    )
    return assets.compose(track, visuals=False)


def test_it_sees_a_wall_up_to_25_m_away():
    """The LakiBeam1 reaches 25 m on a bright target: it sees a wall 20 m
    ahead and not one 30 m ahead."""
    for x in (20.0, 30.0):
        scene = _wall_ahead(x)
        data = mujoco.MjData(scene.model)
        mujoco.mj_forward(scene.model, data)
        site_x = data.bind_scalar(scene.car.lidar).xpos[0]
        (front,) = lidar.cast(LIDAR, scene.model, data, scene.car, np.array([0]))
        if x < 25.0:
            assert front == pytest.approx(x - site_x, abs=1e-6)
        else:
            assert front == lidar.NO_RETURN


def test_a_turn_completes_every_twelve_steps_at_360_hz(corridor):
    model, car = corridor.model, corridor.car
    data = sim.settle(model)
    state = lidar.start(LIDAR, model, data, car)
    completed = []
    for i in range(1, 37):
        last = state.scan
        state = lidar.sweep(LIDAR, model, data, car, state, 1 / 360)
        if state.scan is not last:
            completed.append(i)
    assert completed == [12, 24, 36]


def test_each_slot_reads_the_pose_of_the_step_that_passed_it(corridor):
    """Half a turn from one pose and half from another: the scan holds the
    first half of a scan from the first pose and the second half of a scan
    from the second, as the scan of a moving car is skewed."""
    model, car = corridor.model, corridor.car
    data = sim.settle(model)
    state = before = lidar.start(LIDAR, model, data, car)
    for _ in range(6):
        state = lidar.sweep(LIDAR, model, data, car, state, 1 / 360)
    _shift(corridor, data, 0.2)
    after = lidar.start(LIDAR, model, data, car)
    for _ in range(6):
        state = lidar.sweep(LIDAR, model, data, car, state, 1 / 360)
    half = LIDAR.slots // 2
    assert not np.array_equal(before.scan, after.scan)
    np.testing.assert_array_equal(state.scan[:half], before.scan[:half])
    np.testing.assert_array_equal(state.scan[half:], after.scan[half:])


def test_a_turn_measures_every_slot_whatever_the_step(corridor):
    """Move the car, then sweep one turn at a step that need not divide it:
    the next complete scan has no slot left from before the move."""
    model, car = corridor.model, corridor.car
    for hz in (360, 500, 1000, 60):
        data = sim.settle(model)
        state = lidar.start(LIDAR, model, data, car)
        _shift(corridor, data, -0.2)
        first = state.scan
        for _ in range(math.ceil(hz / LIDAR.scan_rate)):
            state = lidar.sweep(LIDAR, model, data, car, state, 1 / hz)
        assert state.scan is not first, f"{hz} Hz"
        np.testing.assert_array_equal(
            state.scan, lidar.start(LIDAR, model, data, car).scan, err_msg=f"{hz} Hz"
        )


def test_a_step_of_a_turn_or_more_is_refused(corridor):
    model, car = corridor.model, corridor.car
    data = sim.settle(model)
    state = lidar.start(LIDAR, model, data, car)
    with pytest.raises(ValueError, match="turn"):
        lidar.sweep(LIDAR, model, data, car, state, 1 / LIDAR.scan_rate)


def test_the_scan_plane_tilts_with_the_car(corridor):
    """The rays lie in the plane of the LiDAR site, so a car pitched nose
    down sees the floor ahead, nearer than the wall."""
    model, car = corridor.model, corridor.car
    data = sim.settle(model)
    level = lidar.start(LIDAR, model, data, car).scan[0]
    qpos = data.bind_scalar(car.joint).qpos
    pitch = 0.2  # rad, nose down: a rotation about +Y
    qpos[3:7] = [math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0]
    mujoco.mj_forward(model, data)
    site_z = data.bind_scalar(car.lidar).xpos[2]
    tilted = lidar.start(LIDAR, model, data, car).scan[0]
    assert tilted == pytest.approx(site_z / math.sin(pitch), rel=1e-3)
    assert tilted < level
