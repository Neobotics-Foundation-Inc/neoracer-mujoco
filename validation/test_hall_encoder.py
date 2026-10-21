"""
Laws for neoracer_mujoco.hall_encoder, the speed that the RACECAR Neo V2's
drive board reports from its motor's hall sensors, and for
rc.physics.get_encoder_speed() on the V2.
"""

import math
import os
from dataclasses import replace

import pytest

from neoracer_mujoco import assets, hall_encoder, sim
from neoracer_mujoco.racecar_core_adapter import MujocoRacecar, racecar_neo_v2_params
from neoracer_spec import derive_v2
from neoracer_spec.racecar_neo_v2 import RACECAR_NEO_V2 as V2

PARAMS = derive_v2.encoder_params(V2)
DT = 1 / 360  # s, the car's physics step


def _readings(omega: float, seconds: float) -> list[hall_encoder.HallEncoderState]:
    """The encoder after each step of a drivetrain that turns at omega
    (rad/s) from rest, for seconds."""
    state, states = hall_encoder.start(PARAMS, 0.0), []
    for i in range(1, round(seconds / DT) + 1):
        state = hall_encoder.step(PARAMS, state, omega * i * DT, DT)
        states.append(state)
    return states


def test_an_edge_comes_every_2_25_mm():
    """12 edges a motor turn, through the gears of 18/61 and 15/37 to the
    72 mm wheels: an edge every 2.25 mm that they roll."""
    distance = math.pi * 0.072 * (18 / 61) * (15 / 37) / 12  # m
    assert PARAMS.edge_angle * PARAMS.radius == pytest.approx(distance)


def test_the_filter_converges_to_the_tyres_surface_speed():
    """At a steady speed every step that passes edges measures omega * r,
    since it splits its time in proportion to the angle. From rest, the
    filter, which keeps 1 - alpha of its last value, then reports
    (1 - (1 - alpha)^k) omega r at the k-th of those steps, forward and
    backward."""
    for omega in (20.0, -7.0):  # rad/s
        speeds = [s.speed for s in _readings(omega, 0.3) if s.since < DT]
        assert len(speeds) > 10
        for k, speed in enumerate(speeds, start=1):
            expected = (1 - (1 - PARAMS.alpha) ** k) * omega * PARAMS.radius
            assert speed == pytest.approx(expected), f"{omega} rad/s, edge {k}"


def test_a_creeping_car_reads_zero():
    """Without an edge for stop_timeout, the speed reads 0: at a speed of an
    edge every two timeouts, it reads 0 from a timeout after each edge to the
    next one. The worksheet measures the car reading 0 below 0.07 m/s; the
    board's edges and timeout give 0.075."""
    slowest = PARAMS.edge_angle * PARAMS.radius / PARAMS.stop_timeout  # m/s
    assert slowest == pytest.approx(V2.odometry.min_speed.value, rel=0.1)
    omega = slowest / 2 / PARAMS.radius
    for state in _readings(omega, 1.0):
        if state.since > PARAMS.stop_timeout:
            assert state.speed == 0.0
        else:
            assert state.speed >= 0.0


def test_a_drivetrain_that_turns_back_before_the_next_edge_passes_none():
    """The edges are fixed angles: a drivetrain that rocks between 0.8 and 0.4
    of the way to the next edge, as a car at rest may, crosses none, so for a
    second it reads 0 and its last edge ages."""
    state = hall_encoder.start(PARAMS, 0.0)
    for i in range(360):
        angle = (0.8 if i % 2 == 0 else 0.4) * PARAMS.edge_angle
        state = hall_encoder.step(PARAMS, state, angle, DT)
        assert state.speed == 0.0
    assert state.since == pytest.approx(1.0)


def test_a_car_that_starts_again_reads_until_a_timeout_after_its_edge():
    """After a second at rest, a step that turns 1.5 edges crosses one: its
    last edge is the third of the step old, not a share of the second. The
    reading then holds until stop_timeout after that edge, and is 0 after."""
    state = hall_encoder.start(PARAMS, 0.0)
    for _ in range(360):
        state = hall_encoder.step(PARAMS, state, 0.0, DT)
    state = hall_encoder.step(PARAMS, state, 1.5 * PARAMS.edge_angle, DT)
    assert state.since == pytest.approx(DT / 3)
    assert state.speed > 0.0
    while state.since + DT <= PARAMS.stop_timeout:
        state = hall_encoder.step(PARAMS, state, 1.5 * PARAMS.edge_angle, DT)
        assert state.speed > 0.0
    state = hall_encoder.step(PARAMS, state, 1.5 * PARAMS.edge_angle, DT)
    assert state.speed == 0.0


def test_an_edge_after_a_timeout_starts_the_filter_over():
    """An edge crossed, ten steps at rest, and the next edge half a step into
    the step after: the wait, 30.1 ms, passes stop_timeout within that step,
    so the board read 0 before the edge, and it reads alpha of the edge's
    speed, an edge over the wait, either way."""
    e = PARAMS.edge_angle
    for way, name in ((1.0, "forward"), (-1.0, "backward")):
        state = hall_encoder.start(PARAMS, 0.0)
        state = hall_encoder.step(PARAMS, state, way * 1.5 * e, DT)
        for _ in range(10):
            state = hall_encoder.step(PARAMS, state, way * 1.5 * e, DT)
        assert state.since < PARAMS.stop_timeout, name
        assert state.speed != 0.0, name
        wait = state.since + DT / 2
        assert wait > PARAMS.stop_timeout, name
        state = hall_encoder.step(PARAMS, state, way * 2.5 * e, DT)
        expected = way * PARAMS.alpha * e * PARAMS.radius / wait
        assert state.speed == pytest.approx(expected), name


def test_a_step_longer_than_the_timeout_reads_0_after_an_old_edge():
    """A step of 0.1 s from rest that turns 1.5 edges crosses one a third of
    the step, 33 ms, before its end: more than stop_timeout, so it reads 0."""
    state = hall_encoder.start(PARAMS, 0.0)
    state = hall_encoder.step(PARAMS, state, 1.5 * PARAMS.edge_angle, 0.1)
    assert state.since == pytest.approx(0.1 / 3)
    assert state.since > PARAMS.stop_timeout
    assert state.speed == 0.0


def test_the_params_refuse_settings_it_cannot_step_with():
    bad = [
        ("edge_angle", 0.0),
        ("radius", -0.036),
        ("alpha", 0.0),
        ("alpha", 1.5),
        ("stop_timeout", 0.0),
    ]
    for field, value in bad:
        with pytest.raises(ValueError, match=field):
            replace(PARAMS, **{field: value})


def test_the_adapter_reads_the_v2s_encoder_and_lidar():
    """On the V2, MujocoRacecar with racecar_neo_v2_params scans 1080 slots a
    turn, and a program's speed 1, at the start max speed of 0.25, drives the
    car at its steady speed at command 0.25, which the encoder reads within
    3%."""
    scene = assets.load(os.path.join(assets.ASSET_DIR, "racecar_neo_v2.xml"))
    data = sim.settle(scene.model)
    rc = MujocoRacecar(
        scene,
        data,
        max_steps=round(3.0 / scene.model.opt.timestep),
        car=racecar_neo_v2_params(),
    )
    rc.set_start_update(lambda: rc.drive.set_speed_angle(1.0, 0.0), lambda: None)
    rc.go()
    assert rc.lidar.get_num_samples() == len(rc.lidar.get_samples()) == 1080
    steady = next(a.value for a in V2.drive.steady_speed.value if a.command == 0.25)
    assert rc.physics.get_encoder_speed() == pytest.approx(steady, rel=0.03)
    assert math.isclose(V2.drive.start_max_speed.value, 0.25)
