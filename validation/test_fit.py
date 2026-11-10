"""
Laws of fit.py: the fits recover what made a run, and the specs hold the
values that the fits give on the cars' recorded runs.
"""

import os
from dataclasses import replace

import mujoco
import numpy as np
import pytest

from neoracer_mujoco import assets, fit, sim
from neoracer_mujoco.control.steering import ControlCommand
from neoracer_mujoco.scene import compile_scene
from neoracer_spec import derive_v2
from neoracer_spec.racecar_neo_v2 import RACECAR_NEO_V2 as V2
from neoracer_spec.racecar_neo_v2 import RacecarNeoV2

STEP = 1 / 360
"""The V2's physics step (s)."""


def with_latency(latency: float) -> RacecarNeoV2:
    """The V2 with a steering latency of latency (s)."""
    s = V2.steering
    return replace(V2, steering=replace(s, latency=replace(s.latency, value=latency)))


def model_run(car: RacecarNeoV2, steering: float) -> fit.Run:
    """A run of car's model as the car's recorder would write it: a speed
    command of 0.25, a steering command after 0.8 s, and the gyro's reading
    at every physics step."""
    spec = mujoco.MjSpec.from_string(assets.racecar_neo_v2_xml(car))
    spec.modelfiledir = assets.ASSET_DIR
    assets.delete_visuals(spec)
    scene = compile_scene(spec)
    model, data = scene.model, sim.settle(scene.model)
    assert model.opt.timestep == pytest.approx(STEP)
    start = data.time
    lead = sim.steps_in(0.8, STEP)
    lock = derive_v2.steer_lock(car)
    imu = []
    for steps, command in ((lead, 0.0), (sim.steps_in(1.2, STEP), steering)):
        sim.apply_command(scene.car, data, ControlCommand(0.25, command * lock))
        for _ in range(steps):
            mujoco.mj_step(model, data)
            gyro = data.bind_scalar(scene.car.sensors.imu_gyro).data
            imu.append((data.time - STEP - start, 0.0, 0.0, 0.0, *gyro))
    commands = [(-1.0, 0.0, 0.0), (0.0, 0.25, 0.0), (lead * STEP, 0.25, steering)]
    return fit.Run(np.array(commands), np.zeros((0, 3)), np.array(imu))


def test_a_run_loads_from_its_folder():
    """Each of the V2's runs has its three tables, its commands begin at
    rest a second before the car moves, and its steering step is a
    full-lock command at speed 0.25, one run each way."""
    runs = fit.runs()
    assert len(runs) == len(os.listdir(fit.V2_RUNS)) == 2
    sides = []
    for run in runs:
        assert run.commands.shape[1] == 3
        assert run.odometry.shape[1] == 3
        assert run.imu.shape[1] == 7
        assert tuple(run.commands[0]) == (-1.0, 0.0, 0.0)
        step = fit.steering_step(run)
        assert step.speed == 0.25
        assert step.lead == pytest.approx(step.t)
        sides.append(step.steering)
    assert sorted(sides) == [-1.0, 1.0]


def test_a_run_without_a_steering_step_is_an_error():
    """steering_step refuses a run that never steers, and one whose first
    steering command comes with the car at rest."""
    empty = np.zeros((0, 3))
    straight = np.array([(-1.0, 0.0, 0.0), (0.0, 0.25, 0.0)])
    at_rest = np.array([(-1.0, 0.0, 0.0), (0.0, 0.25, 1.0)])
    for commands in (straight, at_rest):
        with pytest.raises(ValueError, match="steering command"):
            fit.steering_step(fit.Run(commands, empty, empty))


def test_the_fit_recovers_a_models_latency():
    """Fitted to runs of a model with a known steering latency, the fit
    returns that latency to a physics step, whatever latency the car that
    it starts from has. The model's delay is whole steps, rounded up."""
    for latency in (0.03, 0.12):
        runs = [model_run(with_latency(latency), side) for side in (1.0, -1.0)]
        for car in (V2, with_latency(0.2)):
            fitted = fit.racecar_neo_v2_steering_latency(car, runs)
            assert latency - 1e-4 < fitted.latency < latency + STEP + 1e-4
            assert fitted.rms < 0.01


def test_the_v2s_steering_latency_is_the_fits():
    """The spec's steering latency is the fit's on the car's full-lock
    circles, within 1 ms, and its model then follows the car's gyro through
    the turn-in to 0.11 of the settled yaw rate, the gyro's own noise."""
    fitted = fit.racecar_neo_v2_steering_latency()
    assert V2.steering.latency.value == pytest.approx(fitted.latency, abs=1e-3)
    assert fitted.rms < 0.11
