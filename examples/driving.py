"""
What the two driving demos share: manual_drive.py, with glfw and the
keyboard, and pygame_drive/, with pygame and the keyboard or a gamepad.
Both put a car on a track, follow it with a chase camera, and move its
command toward the driver's input at a rate of their own.

Frame: +X forward, +Y left, +Z up. An input and a Drive are in [-1, 1]:
+ throttle = forward, + steer = left, and 1 is full throttle or full lock,
the car's steer limit (contract.actuator_limits).

    shaping, drive = Shaping(...), Drive()
    drive = next_drive(shaping, drive, inputs, dt)
    touched = apply_and_step(scene, data, drive, frame_dt(shaping, dt))
"""

import math
from dataclasses import dataclass

import mujoco
import numpy as np
import numpy.typing as npt

from neoracer_mujoco import assets, collision, contract, sim
from neoracer_mujoco.control.steering import ControlCommand
from neoracer_mujoco.scene import Car, Scene


@dataclass(frozen=True)
class Drive:
    """A throttle and a steer, each in [-1, 1]: the driver's input, or the
    command that follows it."""

    throttle: float = 0.0
    steer: float = 0.0


@dataclass(frozen=True)
class Rates:
    """How fast a command follows its input (1/s): toward a held input, and
    back to 0 when the input is 0."""

    toward: float
    back: float


@dataclass(frozen=True)
class Shaping:
    """How a demo's command follows the driver's input."""

    throttle: Rates
    steer: Rates
    max_dt: float = 0.1  # s; a longer frame is cut, so physics never has to catch up


def frame_dt(s: Shaping, dt: float) -> float:
    """The frame time (s) that the command and the physics use: never
    negative, at most max_dt."""
    return max(0.0, min(dt, s.max_dt))


def follow(rates: Rates, command: float, target: float, dt: float) -> float:
    """command moved toward target, clipped to [-1, 1], by at most its rate
    times dt (s), without overshoot."""
    target = max(-1.0, min(1.0, target))
    step = (rates.back if target == 0.0 else rates.toward) * dt
    return command + max(-step, min(step, target - command))


def next_drive(s: Shaping, drive: Drive, inputs: Drive, dt: float) -> Drive:
    """The command after a frame of dt (s) with inputs held."""
    dt = frame_dt(s, dt)
    return Drive(
        throttle=follow(s.throttle, drive.throttle, inputs.throttle, dt),
        steer=follow(s.steer, drive.steer, inputs.steer, dt),
    )


def apply_and_step(scene: Scene, data: mujoco.MjData, drive: Drive, dt: float) -> bool:
    """Send drive to the car, a steer of 1 at its steer limit, and step the
    physics for dt (s). Return whether the car touched a wall in a step."""
    limits = contract.actuator_limits(scene.model, scene.car)
    command = ControlCommand(
        throttle=drive.throttle, steer=drive.steer * limits.steer_rad
    )
    sim.apply_command(scene.car, data, command)
    touched = False
    end = data.time + dt
    while data.time < end:
        mujoco.mj_step(scene.model, data)
        touched |= collision.car_touches_wall(data, scene.car)
    return touched


def load_scene(track: str | None) -> Scene:
    """The car on the track XML at the path track, or alone on its own plane
    if track is None."""
    if track is None:
        return assets.load()
    return assets.compose(mujoco.MjSpec.from_file(track))


@dataclass(frozen=True)
class ChaseCam:
    distance: float = 1.6  # m from the car
    elevation: float = -35.0  # deg; a more negative value looks more from the top
    # deg, added to the car's yaw. 0 puts the camera behind the car (at -X,
    # looking at +X), 180 in front of it, and ±90 at its side.
    azimuth: float = 0.0


def chase_camera(p: ChaseCam, model: mujoco.MjModel, car: Car) -> mujoco.MjvCamera:
    """A camera that tracks the car's body. follow_car turns it with the car."""
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = model.bind_scalar(car.body).id
    cam.distance = p.distance
    cam.elevation = p.elevation
    cam.azimuth = p.azimuth
    return cam


def follow_car(
    p: ChaseCam, cam: mujoco.MjvCamera, data: mujoco.MjData, car: Car
) -> None:
    """Turn cam to the car's heading in data."""
    cam.azimuth = p.azimuth + yaw_deg(data.bind_scalar(car.body).xquat)


def yaw_deg(quat: npt.NDArray[np.float64]) -> float:
    """The heading (deg) of a MuJoCo (w, x, y, z) quaternion."""
    w, x, y, z = quat
    return math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
