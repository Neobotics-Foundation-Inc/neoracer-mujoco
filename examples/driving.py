"""
What the two driving demos share: manual_drive.py, with glfw and the
keyboard, and pygame_drive/, with pygame and the keyboard or a gamepad.
Both put a car on a track, follow it with a chase camera, and move its
command toward the driver's input at a rate of their own. The command asks
the car's speed loop for a speed, as racecar_core's set_speed_angle does on
the real car.

Frame: +X forward, +Y left, +Z up. An input and a Drive are in [-1, 1]:
+ speed = forward, + steer = left, and 1 is the speed loop's full speed
(speed_loop.target) or full lock, the car's steer limit
(contract.actuator_limits).

    shaping, drive = Shaping(...), Drive()
    car = racecar.start(racecar.neoracer_params(), scene, data)
    drive = next_drive(shaping, drive, inputs, dt)
    car, touched = apply_and_step(scene, data, car, drive, frame_dt(shaping, dt))
"""

import math
from dataclasses import dataclass
from typing import assert_never

import mujoco
import numpy as np
import numpy.typing as npt

from neoracer_mujoco import assets, collision, contract, racecar, speed_loop
from neoracer_mujoco.control import SpeedAngle
from neoracer_mujoco.scene import Car, Scene


@dataclass(frozen=True)
class Drive:
    """A speed and a steer, each in [-1, 1]: the driver's input, or the
    command that follows it."""

    speed: float = 0.0
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

    speed: Rates
    steer: Rates
    max_dt: float = 0.1
    """The longest frame (s). A longer frame is cut, so physics never has to
    catch up."""


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
        speed=follow(s.speed, drive.speed, inputs.speed, dt),
        steer=follow(s.steer, drive.steer, inputs.steer, dt),
    )


def apply_and_step(
    scene: Scene, data: mujoco.MjData, car: racecar.CarState, drive: Drive, dt: float
) -> tuple[racecar.CarState, bool]:
    """Ask car for drive, a speed of 1 the speed loop's full speed and a
    steer of 1 the car's steer limit, and step the physics for dt (s).
    Return the car after the steps and whether it touched a wall in one."""
    match car.drive:
        case racecar.SpeedLoop(params=params):
            target = float(speed_loop.target(params, drive.speed))
        case racecar.Throttle():
            raise ValueError("the driving demos drive a car with a speed loop")
        case _:
            assert_never(car.drive)
    limits = contract.actuator_limits(scene.model, scene.car)
    car = racecar.ask(car, SpeedAngle(target, drive.steer * limits.steer_rad))
    touched = False
    end = data.time + dt
    while data.time < end:
        car = racecar.step(scene, data, car)
        touched |= collision.car_touches_wall(data, scene.car)
    return car, touched


def load_scene(track: str | None) -> Scene:
    """The car on the track XML at the path track, or alone on its own plane
    if track is None."""
    if track is None:
        return assets.load()
    return assets.compose(mujoco.MjSpec.from_file(track))


@dataclass(frozen=True)
class ChaseCam:
    distance: float = 1.6
    """The camera's distance from the car (m)."""
    elevation: float = -35.0
    """The camera's elevation (deg). A more negative value looks more from the
    top."""
    azimuth: float = 0.0
    """The camera's azimuth (deg), added to the car's yaw. 0 puts the camera
    behind the car (at -X, looking at +X), 180 in front of it, and ±90 at its
    side."""


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
