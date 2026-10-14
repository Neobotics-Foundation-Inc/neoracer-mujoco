"""
Laws for batch.py: a batch row is one environment stepped on its own, and
command and read address the batch arrays by spec element, whatever the
order of the actuators and the sensors in the car XML.
"""

import copy
import os
from dataclasses import fields

import mujoco
import numpy as np
import pytest
from scenes import reversed_car

from neoracer_mujoco import Scene, assets, batch, physics, sim
from neoracer_mujoco.control.steering import ControlCommand
from neoracer_mujoco.physics import Physics
from neoracer_mujoco.scene import compile_scene

NBATCH, NSTEP, CALLS = 6, 17, 8
NOMINAL = [Physics()]  # times nbatch: the car of the XML in every row


def _loop_scene() -> Scene:
    track = os.path.join(assets.TRACK_DIR, "loop_corridor.xml")
    return assets.compose(mujoco.MjSpec.from_file(track), pos=(0, -2.5, 0))


def _commands(seed: int) -> np.ndarray:
    """(CALLS, NBATCH, 2) random (throttle N·m, steer rad) pairs."""
    rng = np.random.default_rng(seed)
    return rng.uniform((-0.2, -0.4), (0.4, 0.4), (CALLS, NBATCH, 2))


def _batch_run(scene: Scene, nthread: int) -> np.ndarray:
    """The sensordata of every call of a CpuBatch, (CALLS, NBATCH, NSTEP, n)."""
    sim_batch = batch.CpuBatch(scene, NBATCH, nthread)
    state = sim_batch.initial([sim.settle(scene.model)] * NBATCH, NOMINAL * NBATCH)
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
    sim_batch = batch.CpuBatch(scene, 4, nthread=0)
    initial = sim_batch.initial([sim.settle(scene.model)] * 4, NOMINAL * 4)
    ctrl = batch.command(scene.model, scene.car, np.full(4, 0.3), np.zeros(4))
    state, _ = sim_batch.step(initial, ctrl, NSTEP)
    done = np.array([True, False, False, True])
    reset = sim_batch.reset(state, done, initial)
    assert np.array_equal(reset.rows[done], initial.rows[done])
    assert np.array_equal(reset.rows[~done], state.rows[~done])
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
    cpu, warp = batch.CpuBatch(scene, 4, 2), batch_warp.WarpBatch(scene, 4)
    cpu_state = cpu.initial([data] * 4, NOMINAL * 4)
    warp_state = warp.initial([data] * 4, NOMINAL * 4)
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
    warp = batch_warp.WarpBatch(scene, 4)
    initial = warp.initial([sim.settle(scene.model)] * 4, NOMINAL * 4)
    ctrl = batch.command(scene.model, scene.car, jnp.full(4, 0.3), jnp.zeros(4))
    state, _ = warp.step(initial, ctrl, NSTEP)
    done = jnp.array([True, False, True, False])
    reset = warp.reset(state, done, initial)
    for field in ("qpos", "qvel"):
        after, before, start = (getattr(d.data, field) for d in (reset, state, initial))
        np.testing.assert_array_equal(after[done], start[done])
        np.testing.assert_array_equal(after[~done], before[~done])
    _, reset_data = warp.step(reset, ctrl, NSTEP)
    _, fresh_data = warp.step(initial, ctrl, NSTEP)
    np.testing.assert_allclose(reset_data[done], fresh_data[done], atol=1e-6)


def test_step_rejects_zero_steps():
    """mujoco.rollout reads nstep=0 as "not given" and steps once, so step
    asks for at least one step."""
    scene = assets.load()
    sim_batch = batch.CpuBatch(scene, 2, nthread=0)
    state = sim_batch.initial([mujoco.MjData(scene.model)] * 2, NOMINAL * 2)
    ctrl = batch.command(scene.model, scene.car, np.zeros(2), np.zeros(2))
    with pytest.raises(ValueError, match="nstep"):
        sim_batch.step(state, ctrl, 0)
    sim_batch.close()


def _cpu_batch(scene: Scene, nbatch: int) -> batch.CpuBatch:
    return batch.CpuBatch(scene, nbatch, nthread=0)


def _warp_batch(scene: Scene, nbatch: int):
    _, batch_warp = _warp_modules()
    return batch_warp.WarpBatch(scene, nbatch)


@pytest.mark.parametrize("make", [_cpu_batch, _warp_batch], ids=["cpu", "warp"])
def test_each_row_starts_from_its_own_data(make):
    """initial takes one MjData per row, so the rows can start apart, in
    their order, and rejects a list of another length. The warp case runs
    only where the jax extra is installed."""
    scene = assets.load()
    datas = [sim.settle(scene.model) for _ in range(3)]
    for i, data in enumerate(datas):
        data.bind_scalar(scene.car.joint).qpos[0] += i  # m, along +X
        mujoco.mj_forward(scene.model, data)
    sim_batch = make(scene, 3)
    zeros = np.zeros(3)
    ctrl = batch.command(scene.model, scene.car, zeros, zeros)
    _, sensordata = sim_batch.step(sim_batch.initial(datas, NOMINAL * 3), ctrl, 1)
    x = np.asarray(batch.read(scene.model, sensordata[:, -1], scene.car.truth.pos))
    np.testing.assert_allclose(np.diff(x[:, 0]), 1.0, atol=1e-6)
    with pytest.raises(ValueError, match="one MjData per row"):
        sim_batch.initial(datas[:2], NOMINAL * 2)
    if isinstance(sim_batch, batch.CpuBatch):
        sim_batch.close()


def test_the_warp_batch_compiles_initial_once(monkeypatch):
    """A second call of initial reuses the compiled function, so JAX traces
    mjx.make_data only once, and the second state holds the new datas."""
    _, batch_warp = _warp_modules()
    make_data = batch_warp.mjx.make_data
    traces = []

    def counted(*args, **kwargs):
        traces.append(1)
        return make_data(*args, **kwargs)

    monkeypatch.setattr(batch_warp.mjx, "make_data", counted)
    scene = assets.load()
    warp = batch_warp.WarpBatch(scene, 2)
    datas = [sim.settle(scene.model) for _ in range(2)]
    warp.initial(datas, NOMINAL * 2)
    for i, data in enumerate(datas):
        data.bind_scalar(scene.car.joint).qpos[0] += 1.0 + i  # m, along +X
    second = warp.initial(datas, NOMINAL * 2)
    assert len(traces) == 1
    qpos = second.data.qpos
    expected = np.stack([d.qpos for d in datas]).astype(qpos.dtype)
    np.testing.assert_array_equal(np.asarray(qpos), expected)


def _random_physics(nbatch: int, seed: int) -> list[Physics]:
    """nbatch draws from wide ranges of every factor."""
    ranges = physics.PhysicsRanges(
        wheel_friction=physics.Range(0.2, 1.2),
        chassis_mass=physics.Range(0.7, 1.5),
        motor_torque=physics.Range(0.5, 1.5),
        steer_stiffness=physics.Range(0.5, 2.0),
    )
    rng = np.random.default_rng(seed)
    return [physics.sample(ranges, rng) for _ in range(nbatch)]


def test_a_row_steps_with_its_own_physics():
    """Row i of a CpuBatch equals, bit for bit, one MjData stepped on a copy
    of the model with physics[i] applied."""
    scene = _loop_scene()
    params = _random_physics(4, seed=0)
    sim_batch = batch.CpuBatch(scene, 4, nthread=2)
    state = sim_batch.initial([sim.settle(scene.model)] * 4, params)
    throttle, steer = np.full(4, 0.4), np.array([0.3, -0.3, 0.0, 0.2])
    ctrl = batch.command(scene.model, scene.car, throttle, steer)
    for _ in range(5):
        state, sensordata = sim_batch.step(state, ctrl, NSTEP)
    sim_batch.close()
    for i, p in enumerate(params):
        model = copy.deepcopy(scene.model)
        physics.apply_physics(model, scene, p)
        data = sim.settle(scene.model)
        data.ctrl[:] = ctrl[i]
        for _ in range(5):
            data.qacc_warmstart[:] = 0
            for _ in range(NSTEP):
                mujoco.mj_step(model, data)
        np.testing.assert_array_equal(sensordata[i, -1], data.sensordata)


def test_reset_gives_a_row_the_physics_of_initial():
    """A row that starts over steps with the physics of initial, not with
    the one it had before."""
    scene = _loop_scene()
    sim_batch = batch.CpuBatch(scene, 2, nthread=0)
    datas = [sim.settle(scene.model)] * 2
    slippery = Physics(wheel_friction=0.2)
    before = sim_batch.initial(datas, [slippery, slippery])
    initial = sim_batch.initial(datas, NOMINAL * 2)
    reset = sim_batch.reset(before, np.array([True, False]), initial)
    assert reset.models[0] is initial.models[0]
    assert reset.models[1] is before.models[1]
    sim_batch.close()


def test_the_warp_batch_follows_the_cpu_batch_with_its_physics():
    """With a different physics in each row, a short drive on MuJoCo Warp
    ends within 1 mm of the CPU batch. This test runs only where the jax
    extra is installed."""
    jnp, batch_warp = _warp_modules()
    scene = _loop_scene()
    params = _random_physics(4, seed=1)
    datas = [sim.settle(scene.model)] * 4
    cpu, warp = batch.CpuBatch(scene, 4, 2), batch_warp.WarpBatch(scene, 4)
    cpu_state, warp_state = cpu.initial(datas, params), warp.initial(datas, params)
    throttle, steer = np.full(4, 0.5), np.array([0.3, -0.3, 0.0, 0.2])
    cpu_ctrl = batch.command(scene.model, scene.car, throttle, steer)
    warp_ctrl = jnp.asarray(cpu_ctrl)
    for _ in range(10):
        cpu_state, cpu_data = cpu.step(cpu_state, cpu_ctrl, NSTEP)
        warp_state, warp_data = warp.step(warp_state, warp_ctrl, NSTEP)
    cpu.close()
    pos = scene.car.truth.pos
    cpu_pos = batch.read(scene.model, cpu_data[:, -1], pos)
    warp_pos = np.asarray(batch.read(scene.model, warp_data[:, -1], pos))
    np.testing.assert_allclose(warp_pos, cpu_pos, atol=1e-3)
