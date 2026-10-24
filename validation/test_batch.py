"""
Laws for batch.py: a batch row is one environment stepped on its own, and
command and read address the batch arrays by spec element, whatever the
order of the actuators and the sensors in the car XML. The RACECAR Neo V2's
actuators act on a command 0.06 s late, longer than a call of NSTEP steps,
so its rows carry their command history from call to call.
"""

import copy
import os
from collections.abc import Sequence
from dataclasses import fields

import mujoco
import numpy as np
import pytest
from scenes import corridor_scene, reversed_car

from neoracer_mujoco import Scene, assets, batch, physics, sim, speed_loop
from neoracer_mujoco.control.steering import ControlCommand
from neoracer_mujoco.physics import Physics
from neoracer_mujoco.scene import compile_scene
from neoracer_mujoco.sensors import SensorReadings
from neoracer_mujoco.track_generation import (
    Track,
    WallParams,
    generate_track,
    settings_for_difficulty,
    to_spec,
)
from neoracer_spec.derive import derive
from neoracer_spec.neoracer import NEORACER

NBATCH, NSTEP, CALLS = 6, 17, 8
NOMINAL = [Physics()]
"""A list to multiply by nbatch: the car of the XML in every row."""
V2 = os.path.join(assets.ASSET_DIR, "racecar_neo_v2.xml")


def _loop_scene(car: str = assets.DEFAULT_CAR) -> Scene:
    track = os.path.join(assets.TRACK_DIR, "loop_corridor.xml")
    return assets.compose(mujoco.MjSpec.from_file(track), car)


def _commands(seed: int, nbatch: int = NBATCH) -> np.ndarray:
    """Random (throttle, steer) pairs, throttle in N·m and steer in rad, of
    shape (CALLS, nbatch, 2)."""
    rng = np.random.default_rng(seed)
    return rng.uniform((-0.2, -0.4), (0.4, 0.4), (CALLS, nbatch, 2))


def _batch_run(scene: Scene, nthread: int) -> np.ndarray:
    """The sensordata of every call of a CpuBatch, (CALLS, NBATCH, NSTEP, n)."""
    sim_batch = batch.CpuBatch(scene, NBATCH, nthread)
    state = sim_batch.initial(
        [sim.settle(scene.model)] * NBATCH, [scene] * NBATCH, NOMINAL * NBATCH
    )
    out = []
    for c in _commands(0):
        ctrl = batch.command(scene.model, scene.car, c[:, 0], c[:, 1])
        state, sensordata = sim_batch.step(state, ctrl, NSTEP)
        out.append(sensordata)
    sim_batch.close()
    return np.array(out)


def _stepped(
    scene: Scene, row_physics: Physics, start: mujoco.MjData, commands: np.ndarray
) -> np.ndarray:
    """The sensordata of one MjData that starts from start and steps on scene
    with row_physics, NSTEP steps for each (throttle, steer) of commands,
    starting the solver cold at each: (len(commands), NSTEP, nsensordata)."""
    model = copy.deepcopy(scene.model)
    physics.apply_physics(model, scene, row_physics)
    data = copy.deepcopy(start)
    out = []
    for throttle, steer in commands:
        data.qacc_warmstart[:] = 0
        sim.apply_command(
            scene.car, data, ControlCommand(throttle=throttle, steer=steer)
        )
        steps = []
        for _ in range(NSTEP):
            mujoco.mj_step(model, data)
            steps.append(data.sensordata.copy())
        out.append(steps)
    return np.array(out)


def _random_physics(nbatch: int, seed: int) -> list[Physics]:
    """nbatch draws from wide ranges of every factor."""
    ranges = physics.PhysicsRanges(
        wheel_friction=physics.Range(0.2, 1.2),
        chassis_mass=physics.Range(0.7, 1.5),
        motor_torque=physics.Range(0.5, 1.5),
        steer_stiffness=physics.Range(0.5, 2.0),
        lidar_rate=physics.Range(0.8, 1.25),
    )
    rng = np.random.default_rng(seed)
    return [physics.sample(ranges, rng) for _ in range(nbatch)]


def _tracks(seeds: Sequence[int]) -> list[Scene]:
    """Generated tracks of the same size, the car on each, without visuals."""
    walls = WallParams(boxes_per_wall=192)
    scenes = []
    for seed in seeds:
        track = generate_track(seed, settings_for_difficulty(1))
        assert isinstance(track, Track), seed
        scenes.append(assets.compose(to_spec(track, walls), visuals=False))
    return scenes


def test_a_row_is_one_data_stepped_on_its_scene_with_its_physics():
    """Row i of a CpuBatch equals, bit for bit, one MjData that starts from
    datas[i], starts the solver cold at each call, and steps on scenes[i]
    with physics[i] and its own commands. Rows 0 to 2 get the same commands:
    rows 0 and 1 differ only in the track, rows 0 and 2 only in the physics,
    and both differences show. Row 3 is row 0 with its own start and its own
    commands, so a batch that gave every row row 0's data or commands fails.
    Seeds 9 and 10 give model names of different lengths, which do not
    matter. initial rejects a list that is not one MjData per row."""
    nine, ten = _tracks((9, 10))
    scenes = [nine, ten, nine, nine]
    p, q = _random_physics(2, seed=0)
    params = [p, p, q, p]
    datas = [sim.settle(s.model) for s in scenes]
    datas[3].bind_scalar(nine.car.joint).qpos[0] += 0.05  # m along x
    assert not np.array_equal(datas[3].qpos, datas[0].qpos)
    shared = _commands(0, nbatch=1)
    commands = np.concatenate([shared, shared, shared, _commands(1, nbatch=1)], axis=1)
    sim_batch = batch.CpuBatch(nine, 4, nthread=2)
    state = sim_batch.initial(datas, scenes, params)
    with pytest.raises(ValueError, match="one MjData per row"):
        sim_batch.initial(datas[:2], scenes[:2], params[:2])
    got = []
    for c in commands:
        ctrl = batch.command(nine.model, nine.car, c[:, 0], c[:, 1])
        state, sensordata = sim_batch.step(state, ctrl, NSTEP)
        got.append(sensordata)
    sim_batch.close()
    got = np.array(got)  # (CALLS, 4, NSTEP, n)
    for i, (scene, row_physics) in enumerate(zip(scenes, params, strict=True)):
        expected = _stepped(scene, row_physics, datas[i], commands[:, i])
        np.testing.assert_array_equal(got[:, i], expected, err_msg=f"row {i}")
    assert not np.array_equal(got[-1, 0], got[-1, 1])
    assert not np.array_equal(got[-1, 0], got[-1, 2])


def test_a_v2_row_keeps_its_command_history_across_calls_and_resets():
    """On the V2 a row equals, bit for bit, one MjData stepped with its
    physics and its commands, which change at every call and take every
    sign, so a batch that lost the command history between calls fails. A
    row reset halfway starts over, history and all, as a fresh MjData does."""
    scene = _loop_scene(V2)
    params = _random_physics(2, seed=4)
    datas = [sim.settle(scene.model)] * 2
    commands = _commands(2, nbatch=2)
    commands[2, :, 0] = 0.0  # a coast
    half = CALLS // 2
    sim_batch = batch.CpuBatch(scene, 2, nthread=0)
    initial = sim_batch.initial(datas, [scene] * 2, params)
    state, got = initial, []
    for k, c in enumerate(commands):
        if k == half:
            state = sim_batch.reset(state, np.array([True, False]), initial)
        ctrl = batch.command(scene.model, scene.car, c[:, 0], c[:, 1])
        state, sensordata = sim_batch.step(state, ctrl, NSTEP)
        got.append(sensordata)
    sim_batch.close()
    got = np.array(got)  # (CALLS, 2, NSTEP, n)
    p, q = params
    np.testing.assert_array_equal(
        got[:, 1], _stepped(scene, q, datas[1], commands[:, 1])
    )
    before = _stepped(scene, p, datas[0], commands[:half, 0])
    after = _stepped(scene, p, datas[0], commands[half:, 0])
    np.testing.assert_array_equal(got[:half, 0], before)
    np.testing.assert_array_equal(got[half:, 0], after)


def test_the_thread_count_does_not_change_the_result():
    scene = _loop_scene()
    assert np.array_equal(_batch_run(scene, 1), _batch_run(scene, 3))


def test_command_and_read_address_the_arrays_by_spec_element():
    """Whatever the order of the actuators and the sensors in the XML, each
    row of command is the ctrl that apply_command writes, and read on
    sensordata gives what binding the sensor to the data gives, for every
    sensor of the car and every truth sensor."""
    throttle = np.array([0.1, -0.2, 0.0, 0.3])
    steer = np.array([0.4, 0.0, 0.2, -0.1])
    cases = [
        ("car", assets.load()),
        ("reversed", reversed_car()),
        ("v2", assets.load(V2)),
        ("reversed v2", reversed_car(V2)),
    ]
    for name, scene in cases:
        rows = batch.command(scene.model, scene.car, throttle, steer)
        for row, t, s in zip(rows, throttle, steer, strict=True):
            data = mujoco.MjData(scene.model)
            sim.apply_command(scene.car, data, ControlCommand(throttle=t, steer=s))
            assert np.array_equal(row, data.ctrl), name
        data = sim.settle(scene.model)
        car = scene.car
        # The car's sensors are those of the SensorReadings fields.
        sensors = [getattr(car.sensors, f.name) for f in fields(SensorReadings)]
        sensors += [getattr(car.truth, f.name) for f in fields(car.truth)]
        for sensor in sensors:
            value = batch.read(scene.model, data.sensordata, sensor)
            assert np.array_equal(value, data.bind_scalar(sensor).data), (
                f"{name}: {sensor.name}"
            )


def test_reset_gives_the_done_rows_the_state_and_physics_of_initial():
    """A done row starts over with the state and the physics of initial; the
    other rows keep the state and the physics they had."""
    scene = assets.load()
    sim_batch = batch.CpuBatch(scene, 4, nthread=0)
    datas = [sim.settle(scene.model)] * 4
    slippery = [Physics(wheel_friction=0.2)] * 4
    start = sim_batch.initial(datas, [scene] * 4, slippery)
    ctrl = batch.command(scene.model, scene.car, np.full(4, 0.3), np.zeros(4))
    state, _ = sim_batch.step(start, ctrl, NSTEP)
    initial = sim_batch.initial(datas, [scene] * 4, NOMINAL * 4)
    done = np.array([True, False, False, True])
    reset = sim_batch.reset(state, done, initial)
    assert np.array_equal(reset.rows[done], initial.rows[done])
    assert np.array_equal(reset.rows[~done], state.rows[~done])
    for i, row_done in enumerate(done):
        assert reset.models[i] is (initial if row_done else state).models[i], i
    sim_batch.close()


def test_drive_runs_every_rows_speed_loop_as_one_car_runs_its_own():
    """batch.drive gives each row what one MjData gives that steps with its
    own speed loop (speed_loop.step after each step, on the drivetrain's
    angle at the step's end, which drive_pos reads once mj_forward refreshes
    it) holding its own commanded speed and steer: the throttle changes at
    the loop's ticks, and the solver starts cold at each tick and each call,
    where the batch's pieces start. The two rows ask for speeds of opposite
    signs, so a batch that ran one loop for both fails."""
    scene = _loop_scene()
    model, car = scene.model, scene.car
    params = derive(NEORACER).speed_loop
    dt = float(model.opt.timestep)
    commanded, steer = np.array([1.5, -0.8]), np.array([0.1, -0.2])
    start = sim.settle(model)
    mujoco.mj_forward(model, start)
    angle = float(start.bind_scalar(car.sensors.drive_pos).data[0])
    sim_batch = batch.CpuBatch(scene, 2, nthread=0)
    state = sim_batch.initial([start] * 2, [scene] * 2, NOMINAL * 2)
    loop = speed_loop.start(params, np.full(2, angle))
    got = []
    for _ in range(CALLS):
        state, sensordata, loop = batch.drive(
            sim_batch, model, car, params, state, loop, commanded, steer, NSTEP
        )
        got.append(sensordata)
    sim_batch.close()
    got = np.array(got)  # (CALLS, 2, NSTEP, n)
    for i in range(2):
        data = copy.deepcopy(start)
        row = speed_loop.start(params, angle)
        expected = []
        for _ in range(CALLS):
            fresh = True
            for _ in range(NSTEP):
                if fresh:
                    data.qacc_warmstart[:] = 0
                    sim.apply_command(
                        car, data, ControlCommand(float(row.throttle), steer[i])
                    )
                mujoco.mj_step(model, data)
                expected.append(data.sensordata.copy())
                mujoco.mj_forward(model, data)
                drive_pos = float(data.bind_scalar(car.sensors.drive_pos).data[0])
                row = speed_loop.step(params, row, commanded[i], drive_pos, dt)
                fresh = row.measure.span == 0.0
        np.testing.assert_array_equal(
            got[:, i].reshape(CALLS * NSTEP, -1), np.array(expected), err_msg=f"row {i}"
        )
    throttle = np.asarray(loop.throttle)
    assert throttle[0] > 0.0 > throttle[1]


def test_command_rejects_a_model_with_other_actuators():
    """command writes every ctrl of the car's: an actuator that the car does
    not drive, with no throttle map, is not the car's."""
    spec = assets.from_file()
    spec.add_actuator(
        name="extra", target="steer_input", trntype=mujoco.mjtTrn.mjTRN_JOINT
    )
    with pytest.raises(ValueError, match="extra needs 3 user numbers"):
        compile_scene(spec)
    scene = assets.load()
    model = spec.compile()
    with pytest.raises(ValueError, match="actuators"):
        batch.command(model, scene.car, np.zeros(1), np.zeros(1))


def test_step_rejects_zero_steps():
    """mujoco.rollout reads nstep=0 as "not given" and steps once, so step
    asks for at least one step."""
    scene = assets.load()
    sim_batch = batch.CpuBatch(scene, 2, nthread=0)
    state = sim_batch.initial(
        [mujoco.MjData(scene.model)] * 2, [scene] * 2, NOMINAL * 2
    )
    ctrl = batch.command(scene.model, scene.car, np.zeros(2), np.zeros(2))
    with pytest.raises(ValueError, match="nstep"):
        sim_batch.step(state, ctrl, 0)
    sim_batch.close()


def test_a_batch_refuses_a_scene_of_another_size():
    loop, corridor = _loop_scene(), corridor_scene()
    sim_batch = batch.CpuBatch(loop, 2, nthread=0)
    datas = [sim.settle(loop.model)] * 2
    with pytest.raises(ValueError, match="same sizes"):
        sim_batch.initial(datas, [loop, corridor], NOMINAL * 2)
    sim_batch.close()


def _warp_modules():
    """jax.numpy and batch_warp, or skip the test where the jax extra is not
    installed."""
    for module in ("jax", "mujoco.mjx", "mujoco_warp"):
        pytest.importorskip(module)
    import jax.numpy as jnp

    from neoracer_mujoco import batch_warp

    return jnp, batch_warp


@pytest.fixture(scope="module")
def warp():
    """jax.numpy, the loop scene and a WarpBatch of 4 rows on it, made once
    for the module so that its step compiles once. The tests that use it run
    only where the jax extra is installed."""
    jnp, batch_warp = _warp_modules()
    scene = _loop_scene()
    return jnp, scene, batch_warp.WarpBatch(scene, 4)


def test_the_warp_batch_follows_the_cpu_batch(warp):
    """With rows that start apart and a different physics in each row, a
    short drive on MuJoCo Warp ends within 1 mm of the CPU batch. Warp has
    its own collision and solver code, so the rows are close, not equal.
    initial rejects a list that is not one MjData per row."""
    jnp, scene, warp_batch = warp
    params = _random_physics(4, seed=1)
    datas = [sim.settle(scene.model) for _ in range(4)]
    for i, data in enumerate(datas):
        data.bind_scalar(scene.car.joint).qpos[0] += 0.1 * i  # m, along +X
        mujoco.mj_forward(scene.model, data)
    cpu = batch.CpuBatch(scene, 4, 2)
    cpu_state = cpu.initial(datas, [scene] * 4, params)
    warp_state = warp_batch.initial(datas, [scene] * 4, params)
    with pytest.raises(ValueError, match="one MjData per row"):
        warp_batch.initial(datas[:2], [scene] * 2, params[:2])
    throttle, steer = np.array([0.0, 0.2, 0.4, 0.5]), np.array([0.3, -0.3, 0.0, 0.2])
    cpu_ctrl = batch.command(scene.model, scene.car, throttle, steer)
    warp_ctrl = jnp.asarray(cpu_ctrl)
    for _ in range(10):
        cpu_state, cpu_data = cpu.step(cpu_state, cpu_ctrl, NSTEP)
        warp_state, warp_data = warp_batch.step(warp_state, warp_ctrl, NSTEP)
    cpu.close()
    pos = scene.car.truth.pos
    cpu_pos = batch.read(scene.model, cpu_data[:, -1], pos)
    warp_pos = np.asarray(batch.read(scene.model, warp_data[:, -1], pos))
    np.testing.assert_allclose(warp_pos, cpu_pos, atol=1e-3)


def test_drive_on_warp_follows_drive_on_the_cpu(warp):
    """batch.drive on MuJoCo Warp, with rows that ask for speeds either way,
    one of them standing, ends within 1 mm of the CPU batch's drive, and its
    loops' measured speeds (m/s) and throttles end within 0.01 of the CPU's:
    Warp's physics is close to the CPU's, not equal, and the loop reads it.
    The steer may be a JAX array, as Warp's ctrl is, and the sensor values
    stay JAX arrays, on the GPU where there is one."""
    jnp, scene, warp_batch = warp
    params = derive(NEORACER).speed_loop
    datas = [sim.settle(scene.model)] * 4
    angle = float(datas[0].bind_scalar(scene.car.sensors.drive_pos).data[0])
    commanded = np.array([1.5, -0.8, 0.0, 3.0])
    steer = np.array([0.1, -0.2, 0.3, 0.0])
    cpu = batch.CpuBatch(scene, 4, 2)
    cpu_state = cpu.initial(datas, [scene] * 4, NOMINAL * 4)
    warp_state = warp_batch.initial(datas, [scene] * 4, NOMINAL * 4)
    cpu_loop = warp_loop = speed_loop.start(params, np.full(4, angle))
    for _ in range(CALLS):
        cpu_state, cpu_data, cpu_loop = batch.drive(
            cpu,
            scene.model,
            scene.car,
            params,
            cpu_state,
            cpu_loop,
            commanded,
            steer,
            NSTEP,
        )
        warp_state, warp_data, warp_loop = batch.drive(
            warp_batch,
            scene.model,
            scene.car,
            params,
            warp_state,
            warp_loop,
            commanded,
            jnp.asarray(steer),
            NSTEP,
        )
    cpu.close()
    assert isinstance(warp_data, jnp.ndarray)
    pos = scene.car.truth.pos
    cpu_pos = batch.read(scene.model, cpu_data[:, -1], pos)
    warp_pos = batch.read(scene.model, warp_data[:, -1], pos)
    np.testing.assert_allclose(warp_pos, cpu_pos, atol=1e-3)
    np.testing.assert_allclose(
        warp_loop.measure.speed, cpu_loop.measure.speed, atol=1e-2
    )
    np.testing.assert_allclose(warp_loop.throttle, cpu_loop.throttle, atol=1e-2)


@pytest.fixture(scope="module")
def warp_v2():
    """jax.numpy, the V2 on the loop scene and a WarpBatch of 2 rows on it."""
    jnp, batch_warp = _warp_modules()
    scene = _loop_scene(V2)
    return jnp, scene, batch_warp.WarpBatch(scene, 2)


def test_the_warp_batch_follows_the_cpu_batch_on_the_v2(warp_v2):
    """On the V2, with commands that change at every call, a different
    physics in each row and a row reset halfway, the Warp batch carries the
    command history as the CPU batch does: its rows end within 1 mm of the
    CPU batch's. Its LiDAR beams keep their history too: at every step, over
    the three turns of the head that the calls span, they read the CPU
    batch's ranges. command gives JAX arrays the ctrl it gives NumPy ones."""
    jnp, scene, warp_batch = warp_v2
    params = _random_physics(2, seed=5)
    datas = [sim.settle(scene.model)] * 2
    cpu = batch.CpuBatch(scene, 2, nthread=0)
    cpu_initial = cpu.initial(datas, [scene] * 2, params)
    warp_initial = warp_batch.initial(datas, [scene] * 2, params)
    cpu_state, warp_state = cpu_initial, warp_initial
    beams = [scene.spec.sensor(f"lidar_{b:03d}") for b in range(0, 360, 45)]
    done = np.array([True, False])
    for k, c in enumerate(_commands(3, nbatch=2)):
        if k == CALLS // 2:
            cpu_state = cpu.reset(cpu_state, done, cpu_initial)
            warp_state = warp_batch.reset(warp_state, jnp.asarray(done), warp_initial)
        cpu_ctrl = batch.command(scene.model, scene.car, c[:, 0], c[:, 1])
        warp_ctrl = batch.command(
            scene.model, scene.car, jnp.asarray(c[:, 0]), jnp.asarray(c[:, 1])
        )
        np.testing.assert_allclose(np.asarray(warp_ctrl), cpu_ctrl, rtol=1e-6)
        cpu_state, cpu_data = cpu.step(cpu_state, cpu_ctrl, NSTEP)
        warp_state, warp_data = warp_batch.step(warp_state, warp_ctrl, NSTEP)
        for beam in beams:
            cpu_range = batch.read(scene.model, cpu_data, beam)
            warp_range = np.asarray(batch.read(scene.model, warp_data, beam))
            np.testing.assert_allclose(warp_range, cpu_range, atol=1e-3)
    cpu.close()
    pos = scene.car.truth.pos
    cpu_pos = batch.read(scene.model, cpu_data[:, -1], pos)
    warp_pos = np.asarray(batch.read(scene.model, warp_data[:, -1], pos))
    np.testing.assert_allclose(warp_pos, cpu_pos, atol=1e-3)


def test_the_warp_batch_reset_gives_the_done_rows_initial(warp):
    """Warp keeps one contact buffer for all environments, so reset merges
    only the per-environment fields. After a reset with a mixed done mask,
    the state and every physics field of a done row are those of initial,
    the other rows keep theirs, and a done row steps like a fresh one."""
    jnp, scene, warp_batch = warp
    datas = [sim.settle(scene.model)] * 4
    ctrl = batch.command(scene.model, scene.car, jnp.full(4, 0.3), jnp.zeros(4))
    start = warp_batch.initial(datas, [scene] * 4, _random_physics(4, seed=2))
    before, _ = warp_batch.step(start, ctrl, NSTEP)
    initial = warp_batch.initial(datas, [scene] * 4, _random_physics(4, seed=3))
    done = np.array([True, False, True, False])
    reset = warp_batch.reset(before, jnp.asarray(done), initial)
    for name in ("qpos", "qvel"):
        after, old, new = (
            np.asarray(getattr(s.data, name)) for s in (reset, before, initial)
        )
        np.testing.assert_array_equal(after[done], new[done], err_msg=name)
        np.testing.assert_array_equal(after[~done], old[~done], err_msg=name)
    for name in physics.FIELDS:
        after, old, new = (np.asarray(s.fields[name]) for s in (reset, before, initial))
        assert not np.array_equal(old, new), name
        np.testing.assert_array_equal(after[done], new[done], err_msg=name)
        np.testing.assert_array_equal(after[~done], old[~done], err_msg=name)
    _, reset_data = warp_batch.step(reset, ctrl, NSTEP)
    _, fresh_data = warp_batch.step(initial, ctrl, NSTEP)
    np.testing.assert_allclose(reset_data[done], fresh_data[done], atol=1e-6)


def test_the_warp_batch_refuses_another_scene(warp):
    """MuJoCo Warp keeps the geometry of the model it was made from, so a
    WarpBatch steps every row on its own scene and refuses another, even
    one of the same track."""
    _, scene, warp_batch = warp
    datas = [sim.settle(scene.model)] * 4
    scenes = [scene, _loop_scene(), scene, scene]
    with pytest.raises(ValueError, match="batch's scene"):
        warp_batch.initial(datas, scenes, NOMINAL * 4)


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
    warp_batch = batch_warp.WarpBatch(scene, 2)
    datas = [sim.settle(scene.model) for _ in range(2)]
    warp_batch.initial(datas, [scene] * 2, NOMINAL * 2)
    for i, data in enumerate(datas):
        data.bind_scalar(scene.car.joint).qpos[0] += 1.0 + i  # m, along +X
    second = warp_batch.initial(datas, [scene] * 2, NOMINAL * 2)
    assert len(traces) == 1
    qpos = second.data.qpos
    expected = np.stack([d.qpos for d in datas]).astype(qpos.dtype)
    np.testing.assert_array_equal(np.asarray(qpos), expected)
