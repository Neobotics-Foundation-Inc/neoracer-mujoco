"""
MuJoCo backend for the racecar_core controller interface.

racecar_core (Neobotics-Foundation-Inc/racecar-neo-library) is the interface
that NeoRacer controllers use, and the RACECAR Neo V2's racecar_core has the
same methods. This module gives the same method names and signatures over a
car's Scene and its MjData. It does not copy or change racecar_core. So
controller code (`rc.drive`, `rc.lidar`, `rc.set_start_update`, `rc.go`)
runs in this simulator without changes.

This module has Drive, Lidar, Physics, Controller, Display and
MujocoRacecar. MujocoRacecar is the equivalent of create_racecar(): it
connects the other parts and runs the start/update/update_slow loop in go().
CarParams says which car's LiDAR and encoder it has; the default is the
NeoRacer's, and racecar_neo_v2_params() gives the V2's. Physics has only the
encoder's speed. This module has no camera, led, nav, slam, telemetry or
vision, because MuJoCo has no backend for them yet.

Frame: +X forward, +Y left, +Z up.

racecar_core and contract.py use two different conventions. This module
converts between them, so the controllers do not convert them:

  1. Steer sign. In racecar_core, + angle = right. In contract.py, the
     steer_servo + = left. MujocoDrive negates the angle.
  2. LiDAR units. racecar_core and lidar.py both scan clockwise from the
     front, but racecar_core reads cm and 0.0 for no return, where lidar.py
     reads m and -1. MujocoLidar changes the units.

Some parts of this module do not follow AGENTS.md, because racecar_core
code needs them as they are: the Button IntEnum, get_samples_async, the
one-method MujocoDisplay, the mutable state of MujocoDrive and
MujocoRacecar, and set_max_speed(), whose default is the car's start max
speed.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import IntEnum

import mujoco
import numpy as np

from neoracer_spec import derive_v2
from neoracer_spec.derive import EncoderParams, LidarParams
from neoracer_spec.derive_v2 import HallEncoderParams
from neoracer_spec.racecar_neo_v2 import RACECAR_NEO_V2

from . import contract, encoder, hall_encoder, lidar, sim
from .control.steering import ControlCommand
from .scene import Scene


@dataclass(frozen=True)
class CarParams:
    """What a car's racecar_core gives a program beside its model: the
    LiDAR's scan, the encoder's speed, and the share of full throttle that
    speed 1 drives until the program calls set_max_speed. The defaults are
    the NeoRacer's."""

    lidar: LidarParams = LidarParams()
    encoder: EncoderParams | HallEncoderParams = EncoderParams()
    start_max_speed: float = 1.0


def racecar_neo_v2_params() -> CarParams:
    """The RACECAR Neo V2's LiDAR, its drive board's encoder and its
    racecar_core's start max speed."""
    p = derive_v2.derive(RACECAR_NEO_V2)
    return CarParams(
        lidar=p.lidar, encoder=p.encoder, start_max_speed=p.start_max_speed
    )


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


class MujocoDrive:
    """racecar_core.Drive backend over a car's Scene and its MjData.

    set_speed_angle(speed, angle): both unitless ratios in [-1.0, 1.0].
    speed: -1.0 full backward .. +1.0 full forward (racecar_core convention).
    angle: -1.0 full left .. +1.0 full right (racecar_core convention).

    set_speed_angle clamps an input that is out of range, and does not
    reject it. So a bad value from a policy or a noisy controller cannot stop
    a step.
    """

    def __init__(self, scene: Scene, data: mujoco.MjData, start_max_speed: float = 1.0):
        self._car = scene.car
        self._data = data
        self._start_max_speed = start_max_speed
        self._max_speed = start_max_speed  # fraction of the full throttle, 0..1
        self._limits = contract.actuator_limits(scene.model, scene.car)

    def set_speed_angle(self, speed: float, angle: float) -> None:
        speed = _clamp(speed, -1.0, 1.0)
        angle = _clamp(angle, -1.0, 1.0)
        command = ControlCommand(
            throttle=speed * self._max_speed,
            # Negate: in racecar_core + angle = right; for steer_servo + = left.
            steer=-angle * self._limits.steer_rad,
        )
        sim.apply_command(self._car, self._data, command)

    def stop(self) -> None:
        self.set_speed_angle(0.0, 0.0)

    def set_max_speed(self, max_speed: float | None = None) -> None:
        """Set the share of full throttle that speed 1 drives. Without
        max_speed, the car's start max speed, as its racecar_core has it:
        1.0 on the NeoRacer, 0.25 on the V2."""
        limit = self._start_max_speed if max_speed is None else max_speed
        self._max_speed = _clamp(limit, 0.0, 1.0)


class MujocoLidar:
    """racecar_core.Lidar backend over the spinning LiDAR (lidar.py).

    get_samples() returns the last complete turn, as library/lidar.py in
    racecar-neo-library does: LidarParams.slots samples in cm, clockwise, index 0
    straight ahead, and 0.0 for no return, as in the blocked rear.

    MujocoRacecar.go() calls advance() once a step, when the poses are up to
    date, as the LiDAR spins whatever the program does. Before the first step, the scan is a
    whole turn from where the car stands.
    """

    def __init__(
        self, scene: Scene, data: mujoco.MjData, params: LidarParams = LidarParams()
    ):
        self._model, self._car, self._data = scene.model, scene.car, data
        self._dt = float(scene.model.opt.timestep)  # s
        self._params = params
        self._state = lidar.start(self._params, self._model, data, self._car)

    def advance(self) -> None:
        """Sweep the slots that the head passes in one physics step."""
        self._state = lidar.sweep(
            self._params, self._model, self._data, self._car, self._state, self._dt
        )

    def get_samples(self) -> np.ndarray:
        cm = self._state.scan * 100.0
        # In racecar_core, 0.0 means "no return". No real distance is
        # negative, and lidar.NO_RETURN is -1.
        valid = np.isfinite(cm) & (cm > 0.0)
        return np.where(valid, cm, 0.0).astype(np.float32)

    def get_samples_async(self) -> np.ndarray:
        # The name comes from racecar_core, where this method reads the LiDAR
        # when the car is not in "go" mode (Jupyter only on the real car).
        # This backend has no such mode, so the method is not asynchronous
        # and returns the same as get_samples().
        return self.get_samples()

    def get_num_samples(self) -> int:
        return self._params.slots


@dataclass(frozen=True)
class _Windowed:
    """The NeoRacer's encoder (encoder.py)."""

    params: EncoderParams
    state: encoder.EncoderState


@dataclass(frozen=True)
class _Hall:
    """The V2's encoder (hall_encoder.py)."""

    params: HallEncoderParams
    state: hall_encoder.HallEncoderState


class MujocoPhysics:
    """The part of racecar_core.Physics that the simulator has: the speed
    that the car's firmware reports from its motor encoder, the NeoRacer's
    (encoder.py) or the V2's (hall_encoder.py), as params says.

    MujocoRacecar.go() calls advance() once a step, when the sensors are up
    to date, as the firmware samples the encoder whatever the program does.
    """

    def __init__(
        self,
        scene: Scene,
        data: mujoco.MjData,
        params: EncoderParams | HallEncoderParams = EncoderParams(),
    ):
        self._drive_pos = scene.car.sensors.drive_pos
        self._data = data
        self._dt = float(scene.model.opt.timestep)  # s
        angle = self._angle()
        match params:
            case EncoderParams():
                self._encoder = _Windowed(params, encoder.start(params, angle))
            case HallEncoderParams():
                self._encoder = _Hall(params, hall_encoder.start(params, angle))

    def _angle(self) -> float:
        return float(self._data.bind_scalar(self._drive_pos).data[0])  # rad

    def advance(self) -> None:
        """Count one physics step on the encoder."""
        angle = self._angle()
        match self._encoder:
            case _Windowed(params, state):
                state = encoder.step(params, state, angle, self._dt)
                self._encoder = _Windowed(params, state)
            case _Hall(params, state):
                state = hall_encoder.step(params, state, angle, self._dt)
                self._encoder = _Hall(params, state)

    def get_encoder_speed(self) -> float:
        """The car's speed (m/s) from its motor encoder, + = forward."""
        return self._encoder.state.speed


class MujocoController:
    """racecar_core.Controller backend. MuJoCo has no gamepad, so a fixed set
    of buttons is held down for the whole run.

    This backend has only is_down. The wall-following example calls only
    is_down, for its RB deadman button. library/controller.py also has
    was_pressed, was_released, get_trigger and get_joystick, but they have
    no backend here.

    held: the buttons that are held down. The default is (RB,), so a
    controller with an RB deadman drives at once in a headless simulation.
    To simulate RB released, pass held=().
    """

    # racecar_core code reads Button.RB, so this class copies the IntEnum of
    # racecar_core.
    class Button(IntEnum):
        A = 0
        B = 1
        X = 2
        Y = 3
        LB = 4
        RB = 5
        LJOY = 6
        RJOY = 7

    def __init__(self, held: Iterable[Button] = (Button.RB,)):
        self._held = set(held)

    def is_down(self, button: Button) -> bool:
        return button in self._held


class MujocoDisplay:
    """racecar_core.Display backend. A headless MuJoCo run has no display, so
    show_text does nothing, like DisplaySim(headless) in library/display.py.
    racecar_core code calls rc.display.show_text, so this class exists."""

    def show_text(self, text: str) -> None:
        pass


@dataclass(frozen=True)
class Program:
    """The callbacks of a racecar_core controller script. racecar_core gives
    them to set_start_update. The set of programs is open, so they are
    callables (AGENTS.md rule 5)."""

    start: Callable[[], None]
    update: Callable[[], None]
    update_slow: Callable[[], None] | None  # None: the script has no slow update


class MujocoRacecar:
    """racecar_core.Racecar backend over MuJoCo.

    It connects MujocoDrive, MujocoLidar, MujocoPhysics, MujocoController and
    MujocoDisplay to one Scene and MjData, and runs the start/update/update_slow loop in
    go(). So a racecar_core controller script, for example
    neoracer-labs/ultimate-wall-follower/wall_follower.py, runs without
    changes. Only its `import racecar_core` must give this class.

    go() runs each physics step in two halves. mj_step1 brings the
    positions, velocities and sensors up to data.time, and the encoder and
    the LiDAR count the step that led there; update() reads the car as it
    is at that time and sets its command, and mj_step2 integrates the next
    step with that command. go() returns after the first half of a step,
    with every step counted and the sensors up to date. So update() reads
    the sensors after the physics step that its last command caused, as in
    RacecarSim (Unity steps a frame, then calls Python), and no reading
    lags the car by a step, as it does after a whole mj_step.

    max_steps: if None (the default), go() runs until the process stops, as
    on the real car. If it is an int, go() returns after that many steps,
    so a test can run a fixed number of steps. Controller code does not see
    this argument.

    car: the car's LiDAR, encoder and start max speed, the NeoRacer's by
    default. For the V2, pass racecar_neo_v2_params() with its scene.
    """

    def __init__(
        self,
        scene: Scene,
        data: mujoco.MjData,
        max_steps: int | None = None,
        held_buttons: Iterable[MujocoController.Button] = (MujocoController.Button.RB,),
        car: CarParams = CarParams(),
    ):
        # After mj_step, the sensors describe the state before its
        # integration. mj_forward brings them up to data.time, so the
        # encoder starts from the angle the drivetrain has now.
        mujoco.mj_forward(scene.model, data)
        self.drive = MujocoDrive(scene, data, car.start_max_speed)
        self.lidar = MujocoLidar(scene, data, car.lidar)
        self.physics = MujocoPhysics(scene, data, car.encoder)
        self.controller = MujocoController(held_buttons)
        self.display = MujocoDisplay()
        self._model = scene.model
        self._data = data
        self._dt = float(scene.model.opt.timestep)  # s
        self._max_steps = max_steps
        self._update_slow_time = 1.0  # s
        # None until the controller calls set_start_update.
        self._program: Program | None = None

    def set_start_update(
        self,
        start: Callable[[], None],
        update: Callable[[], None],
        update_slow: Callable[[], None] | None = None,
    ) -> None:
        self._program = Program(start, update, update_slow)

    def get_delta_time(self) -> float:
        """The physics step (s)."""
        return self._dt

    def clock(self) -> float:
        """The simulated time (s): what a controller reads as the time."""
        return float(self._data.time)

    def set_update_slow_time(self, time: float = 1.0) -> None:
        """Set the time (s) between two update_slow calls."""
        self._update_slow_time = time

    def go(self) -> None:
        program = self._program
        if program is None:
            raise RuntimeError("call set_start_update() before go()")
        program.start()
        slow_countdown = self._update_slow_time
        step = 0
        mujoco.mj_step1(self._model, self._data)
        while self._max_steps is None or step < self._max_steps:
            program.update()
            if program.update_slow is not None:
                slow_countdown -= self._dt
                if slow_countdown < 0:
                    program.update_slow()
                    slow_countdown = self._update_slow_time
            mujoco.mj_step2(self._model, self._data)
            # The first half of the next step: the sensors at the new time,
            # so the encoder and the LiDAR count the step just finished.
            mujoco.mj_step1(self._model, self._data)
            self.lidar.advance()
            self.physics.advance()
            step += 1
