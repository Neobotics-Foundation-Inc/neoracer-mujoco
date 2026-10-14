"""
Laws for batch.py: a batch row is one environment stepped on its own, and
command and read address the batch arrays by spec element, whatever the
order of the actuators and the sensors in the car XML.
"""

import os
from dataclasses import fields

import mujoco
import numpy as np
import pytest
from scenes import reversed_car

from neoracer_mujoco import Scene, assets, batch, sim
from neoracer_mujoco.control.steering import ControlCommand
from neoracer_mujoco.scene import compile_scene

NBATCH, NSTEP, CALLS = 6, 17, 8


def _loop_scene() -> Scene:
    track = os.path.join(assets.TRACK_DIR, "loop_corridor.xml")
    return assets.compose(mujoco.MjSpec.from_file(track), pos=(0, -2.5, 0))


def _commands(seed: int) -> np.ndarray:
    """(CALLS, NBATCH, 2) random (throttle N·m, steer rad) pairs."""
    rng = np.random.default_rng(seed)
    return rng.uniform((-0.2, -0.4), (0.4, 0.4), (CALLS, NBATCH, 2))


def _batch_run(scene: Scene, nthread: int) -> np.ndarray:
    """The sensordata of every call of a CpuBatch, (CALLS, NBATCH, NSTEP, n)."""
    sim_batch = batch.CpuBatch(scene.model, NBATCH, nthread)
    state = sim_batch.initial(sim.settle(scene.model))
    out = []
    for c in _commands(0):
        ctrl = batch.command(scene.model, scene.car, c[:, 0], c[:, 1])
        state, sensordata = sim_batch.step(state, ctrl, NSTEP)
        out.append(sensordata)
    sim_batch.close()
    return np.array(out)


def test_a_row_is_one_data_stepped_the_same_way():
    """Each row of a CpuBatch equals, bit for bit, one MjData that starts
    from the same state and starts the solver cold at each call."""
    scene = _loop_scene()
    rows = [sim.settle(scene.model) for _ in range(NBATCH)]
    expected = []
    for c in _commands(0):
        call = []
        for data, (throttle, steer) in zip(rows, c, strict=True):
            data.qacc_warmstart[:] = 0
            command = ControlCommand(throttle=throttle, steer=steer)
            sim.apply_command(scene.car, data, command)
            steps = []
            for _ in range(NSTEP):
                mujoco.mj_step(scene.model, data)
                steps.append(data.sensordata.copy())
            call.append(steps)
        expected.append(call)
    assert np.array_equal(_batch_run(scene, nthread=2), np.array(expected))


def test_the_thread_count_does_not_change_the_result():
    scene = _loop_scene()
    assert np.array_equal(_batch_run(scene, 1), _batch_run(scene, 3))


@pytest.mark.parametrize(
    "scene", [assets.load(), reversed_car()], ids=["car", "reversed"]
)
def test_command_writes_what_apply_command_writes(scene):
    """Each row of command is the ctrl that apply_command writes, whatever
    the order of the actuators in the XML."""
    throttle, steer = np.array([0.1, -0.2, 0.3]), np.array([0.4, 0.0, -0.1])
    rows = batch.command(scene.model, scene.car, throttle, steer)
    for row, t, s in zip(rows, throttle, steer, strict=True):
        data = mujoco.MjData(scene.model)
        sim.apply_command(scene.car, data, ControlCommand(throttle=t, steer=s))
        assert np.array_equal(row, data.ctrl)


@pytest.mark.parametrize(
    "scene", [assets.load(), reversed_car()], ids=["car", "reversed"]
)
def test_read_gives_the_bound_sensor_values(scene):
    """read on sensordata gives what binding the sensor to the data gives,
    for every sensor of the car and every truth sensor."""
    data = sim.settle(scene.model)
    car = scene.car
    sensors = [getattr(car.sensors, f.name) for f in fields(car.sensors)]
    sensors += [getattr(car.truth, f.name) for f in fields(car.truth)]
    for sensor in sensors:
        value = batch.read(scene.model, data.sensordata, sensor)
        assert np.array_equal(value, data.bind_scalar(sensor).data), sensor.name


def test_reset_replaces_only_the_rows_that_are_done():
    scene = assets.load()
    sim_batch = batch.CpuBatch(scene.model, 4, nthread=0)
    initial = sim_batch.initial(sim.settle(scene.model))
    ctrl = batch.command(scene.model, scene.car, np.full(4, 0.3), np.zeros(4))
    state, _ = sim_batch.step(initial, ctrl, NSTEP)
    done = np.array([True, False, False, True])
    reset = sim_batch.reset(state, done, initial)
    assert np.array_equal(reset[done], initial[done])
    assert np.array_equal(reset[~done], state[~done])
    sim_batch.close()


def test_command_rejects_a_model_with_other_actuators():
    spec = mujoco.MjSpec.from_file(assets.DEFAULT_CAR)
    spec.add_actuator(
        name="extra", target="steer_input", trntype=mujoco.mjtTrn.mjTRN_JOINT
    )
    scene = compile_scene(spec)
    with pytest.raises(ValueError, match="actuators"):
        batch.command(scene.model, scene.car, np.zeros(1), np.zeros(1))


def _warp_modules():
    """jax.numpy and batch_warp, or skip the test where the jax extra is not
    installed."""
    for module in ("jax", "mujoco.mjx", "mujoco_warp"):
        pytest.importorskip(module)
    import jax.numpy as jnp

    from neoracer_mujoco import batch_warp

    return jnp, batch_warp


def test_the_warp_batch_follows_the_cpu_batch():
    """On MuJoCo Warp, a short drive from rest ends within 1 mm of the CPU
    batch. Warp has its own collision and solver code, so the rows are close,
    not equal. This test runs only where the jax extra is installed."""
    jnp, batch_warp = _warp_modules()
    scene = _loop_scene()
    data = sim.settle(scene.model)
    cpu, warp = batch.CpuBatch(scene.model, 4, 2), batch_warp.WarpBatch(scene.model, 4)
    cpu_state, warp_state = cpu.initial(data), warp.initial(data)
    throttle, steer = np.array([0.0, 0.1, 0.2, 0.3]), np.array([0.0, 0.2, -0.2, 0.1])
    cpu_ctrl = batch.command(scene.model, scene.car, throttle, steer)
    warp_ctrl = batch.command(
        scene.model, scene.car, jnp.asarray(throttle), jnp.asarray(steer)
    )
    for _ in range(10):
        cpu_state, cpu_data = cpu.step(cpu_state, cpu_ctrl, NSTEP)
        warp_state, warp_data = warp.step(warp_state, warp_ctrl, NSTEP)
    pos = scene.car.truth.pos
    cpu_pos = batch.read(scene.model, cpu_data[:, -1], pos)
    warp_pos = np.asarray(batch.read(scene.model, warp_data[:, -1], pos))
    np.testing.assert_allclose(warp_pos, cpu_pos, atol=1e-3)
    cpu.close()


def test_the_warp_batch_resets_only_the_rows_that_are_done():
    """Warp keeps one contact buffer for all environments, so reset merges
    only the per-environment fields. After a reset with a mixed done mask,
    the done rows are back at the initial state, the others are unchanged,
    and a reset row steps like a fresh one."""
    jnp, batch_warp = _warp_modules()
    scene = _loop_scene()
    warp = batch_warp.WarpBatch(scene.model, 4)
    initial = warp.initial(sim.settle(scene.model))
    ctrl = batch.command(scene.model, scene.car, jnp.full(4, 0.3), jnp.zeros(4))
    state, _ = warp.step(initial, ctrl, NSTEP)
    done = jnp.array([True, False, True, False])
    reset = warp.reset(state, done, initial)
    for field in ("qpos", "qvel"):
        after, before, start = (getattr(d, field) for d in (reset, state, initial))
        np.testing.assert_array_equal(after[done], start[done])
        np.testing.assert_array_equal(after[~done], before[~done])
    _, reset_data = warp.step(reset, ctrl, NSTEP)
    _, fresh_data = warp.step(initial, ctrl, NSTEP)
    np.testing.assert_allclose(reset_data[done], fresh_data[done], atol=1e-6)
