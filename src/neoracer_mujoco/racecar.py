"""
A car's firmware and sensors around each physics step: the speed loop that
holds the speed the car's program asks for (speed_loop.py), the motor
encoder that measures the speed (encoder.py, or hall_encoder.py on the
RACECAR Neo V2) and the spinning LiDAR (lidar.py). They run whatever the
program does. The racecar_core adapter (racecar_core_adapter.py) runs a
program on them, and the JS package's racecar.ts does the same in the
browser.

start() readies a car at rest, ask() gives it the speed and the steering
angle that a program asks for (SpeedAngle), and step() runs one physics
step with the car's command. step() runs the step in MuJoCo's two halves:
mj_step2 integrates it with the command, and mj_step1 brings the
positions, velocities and sensors up to data.time, so the encoder, the
LiDAR and the speed loop count the step from where it left the car.
Between two steps, data's sensors read the car at data.time, where after a
whole mj_step they read it a step earlier.

    car = racecar.start(racecar.neoracer_params(), scene, data)
    car = racecar.ask(car, SpeedAngle(target=1.5, steer=0.1))
    for _ in range(steps):
        car = racecar.step(scene, data, car)

Frame: +X forward, +Y left, +Z up.
"""

from copy import replace
from dataclasses import dataclass
from typing import assert_never

import mujoco

from neoracer_spec import derive, derive_v2
from neoracer_spec.derive import EncoderParams, LidarParams, SpeedLoopParams
from neoracer_spec.derive_v2 import HallEncoderParams
from neoracer_spec.neoracer import NEORACER
from neoracer_spec.racecar_neo_v2 import RACECAR_NEO_V2, RacecarNeoV2
from neoracer_spec.spec import Spec

from . import encoder, hall_encoder, lidar, sim, speed_loop
from .control.steering import ControlCommand, SpeedAngle
from .scene import Scene


@dataclass(frozen=True)
class CarState:
    """A car between two physics steps. Start it with start()."""

    drive: SpeedLoop | Throttle
    steer: float
    """The steer servo's command (rad), + = left, which it takes at once."""
    encoder: Windowed | Hall
    lidar: Lidar


@dataclass(frozen=True)
class SpeedLoop:
    """A drive whose firmware holds a speed, as the NeoRacer's does."""

    params: SpeedLoopParams
    state: speed_loop.SpeedLoopState
    target: float
    """The speed (m/s) that the program asks it to hold, + = forward."""


@dataclass(frozen=True)
class Throttle:
    """A drive whose model holds a speed itself, as the RACECAR Neo V2's
    does: the program's throttle goes to the motor at once."""

    throttle: float
    """In [-1, 1], + = forward, through the motor's throttle map
    (contract.throttle_map)."""


@dataclass(frozen=True)
class Windowed:
    """The NeoRacer's motor encoder (encoder.py)."""

    params: EncoderParams
    state: encoder.EncoderState


@dataclass(frozen=True)
class Hall:
    """The RACECAR Neo V2's motor encoder (hall_encoder.py)."""

    params: HallEncoderParams
    state: hall_encoder.HallEncoderState


@dataclass(frozen=True)
class Lidar:
    """The spinning LiDAR (lidar.py). Its state's scan is the last complete
    turn."""

    params: LidarParams
    state: lidar.LidarState


@dataclass(frozen=True)
class CarParams:
    """What a car's racecar_core gives a program beside its model: the
    LiDAR's scan, the encoder's speed, the share of full speed that speed 1
    drives until the program calls set_max_speed, and the firmware's speed
    loop (speed_loop.py), if the car's model does not hold its speed itself,
    as the V2's does."""

    lidar: LidarParams
    encoder: EncoderParams | HallEncoderParams
    start_max_speed: float = 1.0
    speed_loop: SpeedLoopParams | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.start_max_speed <= 1.0:
            raise ValueError(
                f"start_max_speed must be in [0, 1], not {self.start_max_speed}"
            )


def neoracer_params(car: Spec = NEORACER) -> CarParams:
    """A NeoRacer's LiDAR and its firmware's encoder and speed loop."""
    p = derive.derive(car)
    return CarParams(lidar=p.lidar, encoder=p.encoder, speed_loop=p.speed_loop)


def racecar_neo_v2_params(car: RacecarNeoV2 = RACECAR_NEO_V2) -> CarParams:
    """A RACECAR Neo V2's LiDAR, its drive board's encoder and its
    racecar_core's start max speed."""
    p = derive_v2.derive(car)
    return CarParams(
        lidar=p.lidar, encoder=p.encoder, start_max_speed=p.start_max_speed
    )


def start(params: CarParams, scene: Scene, data: mujoco.MjData) -> CarState:
    """A car at rest at the pose of data, asked for nothing. It runs
    mj_forward on data first, so that every part starts from the car as it
    is."""
    mujoco.mj_forward(scene.model, data)
    angle = _drive_angle(scene, data)
    drive: SpeedLoop | Throttle
    if params.speed_loop is None:
        drive = Throttle(0.0)
    else:
        loop = speed_loop.start(params.speed_loop, angle)
        drive = SpeedLoop(params.speed_loop, loop, 0.0)
    measure: Windowed | Hall
    match params.encoder:
        case EncoderParams():
            measure = Windowed(params.encoder, encoder.start(params.encoder, angle))
        case HallEncoderParams():
            measure = Hall(params.encoder, hall_encoder.start(params.encoder, angle))
        case _:
            assert_never(params.encoder)
    scan = lidar.start(params.lidar, scene.model, data, scene.car)
    return CarState(drive, 0.0, measure, Lidar(params.lidar, scan))


def ask(car: CarState, command: SpeedAngle) -> CarState:
    """car asked for command: its speed loop holds command.target, and its
    steering servo takes command.steer. Raise ValueError for a car without a
    speed loop, as the V2, whose program sets a throttle instead."""
    match car.drive:
        case SpeedLoop():
            drive = replace(car.drive, target=command.target)
            return replace(car, drive=drive, steer=command.steer)
        case Throttle():
            raise ValueError("this car has no speed loop to hold a speed")
        case _:
            assert_never(car.drive)


def apply(scene: Scene, data: mujoco.MjData, car: CarState) -> None:
    """Write car's command to data.ctrl: its speed loop's throttle, or its
    program's, and its steer."""
    match car.drive:
        case SpeedLoop(state=loop):
            throttle = float(loop.throttle)
        case Throttle(throttle=throttle):
            pass
        case _:
            assert_never(car.drive)
    command = ControlCommand(throttle=throttle, steer=car.steer)
    sim.apply_command(scene.car, data, command)


def step(scene: Scene, data: mujoco.MjData, car: CarState) -> CarState:
    """Step data one physics step with car's command, from where start(),
    step() or mj_step1 left it, to after mj_step1 of the next step, and
    return the car after the encoder, the LiDAR and the speed loop count
    the step. data.ctrl is left with the returned car's command."""
    model = scene.model
    dt = float(model.opt.timestep)
    apply(scene, data, car)
    mujoco.mj_step2(model, data)
    mujoco.mj_step1(model, data)
    angle = _drive_angle(scene, data)
    drive: SpeedLoop | Throttle
    match car.drive:
        case SpeedLoop(params, loop, target):
            loop = speed_loop.step(params, loop, target, angle, dt)
            drive = SpeedLoop(params, loop, target)
        case Throttle():
            drive = car.drive
        case _:
            assert_never(car.drive)
    measure: Windowed | Hall
    match car.encoder:
        case Windowed(params, state):
            measure = Windowed(params, encoder.step(params, state, angle, dt))
        case Hall(params, state):
            measure = Hall(params, hall_encoder.step(params, state, angle, dt))
        case _:
            assert_never(car.encoder)
    p = car.lidar.params
    scan = lidar.sweep(p, model, data, scene.car, car.lidar.state, dt)
    after = CarState(drive, car.steer, measure, Lidar(p, scan))
    apply(scene, data, after)
    return after


def _drive_angle(scene: Scene, data: mujoco.MjData) -> float:
    """The drivetrain's angle (rad) that the drive_pos sensor reads."""
    return float(data.bind_scalar(scene.car.sensors.drive_pos).data[0])
