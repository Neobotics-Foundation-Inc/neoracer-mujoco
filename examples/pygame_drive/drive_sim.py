"""
Physics for the pygame driving demo: a DC-motor throttle model and slewed
steering on top of a compiled NeoRacer scene. drive.py owns the window, the
input and the camera.

Frame: +X forward, +Y left, +Z up. The throttle and steer inputs are in
[-1, 1]; + throttle = forward, + steer = left (contract.py).

    params, state = DriveParams(), DriveState()
    wheels = drive_wheel_dofs(model)  # once per model
    state = next_state(params, state, throttle, steer, dt)
    apply_and_step(model, data, params, wheels, state, dt)
"""

from dataclasses import dataclass

import mujoco
import numpy as np
import numpy.typing as npt

from neoracer_mujoco import contract

type WheelDofs = npt.NDArray[np.intp]  # shape (4,): qvel index of each drive wheel


@dataclass(frozen=True)
class DriveParams:
    """Motor and steering tuning for the demo."""

    # N·m per wheel at full throttle. 0.35 launches like a rocket; below about
    # 0.15 the car cannot carry enough speed over the ramp in ramp_course.xml.
    max_torque: float = 0.15
    # N·m per rad/s of wheel spin: the back-EMF of a DC motor. It sets the top
    # speed (about max_torque / motor_damping rad/s, about 1.9 m/s with rolling
    # friction) and brakes the car when the throttle is released.
    motor_damping: float = 0.0025
    # N·m of rolling resistance per wheel. Back-EMF alone fades near zero speed
    # and leaves the car creeping.
    roll_friction: float = 0.02
    max_steer: float = 0.40  # rad; the model's Ackermann command limit
    throttle_slew: float = 2.5  # 1/s; how fast the applied throttle follows the input
    steer_slew: float = 4.0  # 1/s; smooths digital WASD input, sticks barely notice
    max_dt: float = 0.1  # s; a longer frame is cut, so physics never has to catch up


@dataclass(frozen=True)
class DriveState:
    """The commands that are applied now. They follow the inputs at the slew
    rates of DriveParams."""

    throttle: float = 0.0  # in [-1, 1]
    steer: float = 0.0  # in [-1, 1]


def drive_wheel_dofs(model: mujoco.MjModel) -> WheelDofs:
    """qvel index of the wheel spin joint of each drive actuator (contract.DRIVE)."""
    actuators = range(contract.DRIVE.start, contract.DRIVE.stop)
    return np.array(
        [model.jnt_dofadr[model.actuator_trnid[i, 0]] for i in actuators], dtype=np.intp
    )


def slew(current: float, target: float, max_delta: float) -> float:
    """Move current toward target by at most max_delta, without overshoot."""
    return current + max(-max_delta, min(max_delta, target - current))


def next_state(
    p: DriveParams, state: DriveState, throttle: float, steer: float, dt: float
) -> DriveState:
    """The applied commands after dt (s), for inputs throttle and steer."""
    dt = min(dt, p.max_dt)
    throttle = max(-1.0, min(1.0, throttle))
    steer = max(-1.0, min(1.0, steer))
    return DriveState(
        throttle=slew(state.throttle, throttle, p.throttle_slew * dt),
        steer=slew(state.steer, steer, p.steer_slew * dt),
    )


def wheel_torque(
    p: DriveParams, throttle: float, wheel_spin: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Torque (N·m) for each drive wheel: the commanded torque minus back-EMF
    drag and a smoothed rolling friction. At zero throttle the drag brakes the
    car; at full throttle drive and drag balance at the top speed."""
    torque = throttle * p.max_torque - p.motor_damping * wheel_spin
    torque -= np.clip(wheel_spin * 0.05, -p.roll_friction, p.roll_friction)
    return np.clip(torque, -p.max_torque, p.max_torque)


def apply_and_step(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    p: DriveParams,
    wheels: WheelDofs,
    state: DriveState,
    dt: float,
) -> None:
    """Write the commands of state to data.ctrl and step the physics dt (s)."""
    data.ctrl[contract.STEER] = state.steer * p.max_steer
    data.ctrl[contract.DRIVE] = wheel_torque(p, state.throttle, data.qvel[wheels])
    end = data.time + min(dt, p.max_dt)
    while data.time < end:
        mujoco.mj_step(model, data)
