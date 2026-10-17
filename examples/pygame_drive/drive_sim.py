"""
Physics for the pygame driving demo: slewed throttle and steering on top of
a compiled NeoRacer scene. The motor itself, its torque falling with speed
and its braking when the throttle is released, is the car's (neoracer.xml).
drive.py owns the window, the input and the camera.

Frame: +X forward, +Y left, +Z up. The throttle and steer inputs are in
[-1, 1]; + throttle = forward, + steer = left (contract.py).

    params, state = DriveParams(), DriveState()
    state = next_state(params, state, throttle, steer, dt)
    apply_and_step(scene, data, params, state, dt)
"""

from dataclasses import dataclass

import mujoco

from neoracer_mujoco.scene import Scene
from neoracer_mujoco.twin import derive
from neoracer_mujoco.twin.neoracer import NEORACER


@dataclass(frozen=True)
class DriveParams:
    """Input shaping for the demo."""

    # rad, the model's Ackermann command limit, full lock
    max_steer: float = derive.steer_lock(NEORACER)
    throttle_slew: float = 2.5  # 1/s; how fast the applied throttle follows the input
    steer_slew: float = 4.0  # 1/s; smooths digital WASD input, sticks barely notice
    max_dt: float = 0.1  # s; a longer frame is cut, so physics never has to catch up


@dataclass(frozen=True)
class DriveState:
    """The commands that are applied now. They follow the inputs at the slew
    rates of DriveParams."""

    throttle: float = 0.0  # in [-1, 1]
    steer: float = 0.0  # in [-1, 1]


def frame_dt(p: DriveParams, dt: float) -> float:
    """The frame time (s) that physics uses: never negative, at most max_dt."""
    return max(0.0, min(dt, p.max_dt))


def slew(current: float, target: float, max_delta: float) -> float:
    """Move current toward target by at most max_delta, without overshoot."""
    return current + max(-max_delta, min(max_delta, target - current))


def next_state(
    p: DriveParams, state: DriveState, throttle: float, steer: float, dt: float
) -> DriveState:
    """The applied commands after dt (s), for inputs throttle and steer."""
    dt = frame_dt(p, dt)
    throttle = max(-1.0, min(1.0, throttle))
    steer = max(-1.0, min(1.0, steer))
    return DriveState(
        throttle=slew(state.throttle, throttle, p.throttle_slew * dt),
        steer=slew(state.steer, steer, p.steer_slew * dt),
    )


def apply_and_step(
    scene: Scene,
    data: mujoco.MjData,
    p: DriveParams,
    state: DriveState,
    dt: float,
) -> None:
    """Write the commands of state to data.ctrl and step the physics dt (s)."""
    car = scene.car
    data.bind_scalar(car.steer).ctrl = state.steer * p.max_steer
    voltage = scene.model.bind_scalar(car.drive).ctrlrange[1]  # V, full throttle
    data.bind_scalar(car.drive).ctrl = state.throttle * voltage
    end = data.time + frame_dt(p, dt)
    while data.time < end:
        mujoco.mj_step(scene.model, data)
