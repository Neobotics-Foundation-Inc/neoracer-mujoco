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
The car's firmware and sensors run in racecar.py, and the parts read and
command the car there. racecar.CarParams says which car's LiDAR, encoder and
speed loop it has: racecar.neoracer_params() gives a NeoRacer's, the
default, and racecar.racecar_neo_v2_params() a V2's.
Physics has only the encoder's speed. This module has no camera, led, nav,
slam, telemetry or vision, because MuJoCo has no backend for them yet.

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
one-method MujocoDisplay, the mutable car that MujocoRacecar's parts
share, and set_max_speed(), whose default is the car's start max speed.
"""

from collections.abc import Callable, Iterable
from copy import replace
from dataclasses import dataclass
from enum import IntEnum
from typing import assert_never

import mujoco
import numpy as np

from . import contract, racecar, speed_loop
from .scene import Scene


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


class _Car:
    """The car that the parts of a MujocoRacecar share: its scene, its data
    and its state between two physics steps (racecar.CarState), which
    set_speed_angle and go() replace."""

    def __init__(self, scene: Scene, data: mujoco.MjData, state: racecar.CarState):
        self.scene = scene
        self.data = data
        self.state = state


class MujocoDrive:
    """racecar_core.Drive backend over a MujocoRacecar's car.

    set_speed_angle(speed, angle): both unitless ratios in [-1.0, 1.0].
    speed: -1.0 full backward .. +1.0 full forward (racecar_core convention).
    angle: -1.0 full left .. +1.0 full right (racecar_core convention).

    set_speed_angle clamps an input that is out of range, and does not
    reject it. So a bad value from a policy or a noisy controller cannot stop
    a step.

    With a speed loop (speed_loop.py), as on the NeoRacer, speed asks for a
    speed (speed_loop.target), and the loop sets the throttle at its ticks.
    Without one, as on the V2, whose model holds its speed itself, speed
    drives the throttle at once: speed times the max speed.
    """

    def __init__(self, car: _Car, start_max_speed: float):
        self._car = car
        self._start_max_speed = start_max_speed
        self._max_speed = start_max_speed  # share of the full speed
        self._limits = contract.actuator_limits(car.scene.model, car.scene.car)

    def set_speed_angle(self, speed: float, angle: float) -> None:
        speed = _clamp(speed, -1.0, 1.0)
        angle = _clamp(angle, -1.0, 1.0)
        state = self._car.state
        drive: racecar.SpeedLoop | racecar.Throttle
        match state.drive:
            case racecar.SpeedLoop(params=params):
                target = float(speed_loop.target(params, speed, self._max_speed))
                drive = replace(state.drive, target=target)
            case racecar.Throttle():
                drive = racecar.Throttle(speed * self._max_speed)
            case _:
                assert_never(state.drive)
        # Negate: in racecar_core + angle = right; for steer_servo + = left.
        steer = -angle * self._limits.steer_rad
        self._car.state = replace(state, drive=drive, steer=steer)
        racecar.apply(self._car.scene, self._car.data, self._car.state)

    def stop(self) -> None:
        self.set_speed_angle(0.0, 0.0)

    def set_max_speed(self, max_speed: float | None = None) -> None:
        """Set the share of full speed that speed 1 drives. Without
        max_speed, the car's start max speed, as its racecar_core has it
        (racecar.CarParams.start_max_speed)."""
        limit = self._start_max_speed if max_speed is None else max_speed
        self._max_speed = _clamp(limit, 0.0, 1.0)

    @property
    def target(self) -> float:
        """The speed (m/s) that the speed loop holds: 0 without one."""
        match self._car.state.drive:
            case racecar.SpeedLoop(target=target):
                return target
            case racecar.Throttle():
                return 0.0
            case _:
                assert_never(self._car.state.drive)


class MujocoLidar:
    """racecar_core.Lidar backend over the spinning LiDAR (lidar.py).

    get_samples() returns the last complete turn, as library/lidar.py in
    racecar-neo-library does: LidarParams.slots samples in cm, clockwise, index 0
    straight ahead, and 0.0 for no return, as in the blocked rear.

    The LiDAR spins whatever the program does, and MujocoRacecar.go() counts
    each physics step on it. Before the first step, the scan is a whole turn
    from where the car stands.
    """

    def __init__(self, car: _Car):
        self._car = car

    def get_samples(self) -> np.ndarray:
        cm = self._car.state.lidar.state.scan * 100.0
        # In racecar_core, 0.0 means "no return". No real distance is
        # negative, and lidar.NO_RETURN is -1.
        valid = np.isfinite(cm) & (cm > 0.0)
        return np.where(valid, cm, 0.0).astype(np.float32)

    def get_samples_async(self) -> np.ndarray:
        """The name comes from racecar_core, where this method reads the LiDAR
        when the car is not in "go" mode (Jupyter only on the real car). This
        backend has no such mode, so the method is not asynchronous and returns
        the same as get_samples()."""
        return self.get_samples()

    def get_num_samples(self) -> int:
        return self._car.state.lidar.params.slots


class MujocoPhysics:
    """The part of racecar_core.Physics that the simulator has: the speed
    that the car's firmware reports from its motor encoder, the NeoRacer's
    (encoder.py) or the V2's (hall_encoder.py). The firmware samples the
    encoder whatever the program does, and MujocoRacecar.go() counts each
    physics step on it.
    """

    def __init__(self, car: _Car):
        self._car = car

    def get_encoder_speed(self) -> float:
        """The car's speed (m/s) from its motor encoder, + = forward."""
        return float(self._car.state.encoder.state.speed)


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

    class Button(IntEnum):
        """The gamepad's buttons. racecar_core code reads Button.RB, so this
        class copies the IntEnum of racecar_core."""

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
    update_slow: Callable[[], None] | None
    """None if the script has no slow update."""


_NEORACER = racecar.neoracer_params()


class MujocoRacecar:
    """racecar_core.Racecar backend over MuJoCo.

    It connects MujocoDrive, MujocoLidar, MujocoPhysics, MujocoController and
    MujocoDisplay to one Scene and MjData, and runs the start/update/update_slow loop in
    go(). So a racecar_core controller script, for example
    neoracer-labs/ultimate-wall-follower/wall_follower.py, runs without
    changes. Only its `import racecar_core` must give this class.

    go() calls update(), then runs a physics step with racecar.step, and
    again. racecar.step leaves the positions, velocities and sensors at
    data.time, with the encoder, the LiDAR and the speed loop counting the
    step. So update() reads the sensors after the physics step that its last
    command caused, as in RacecarSim (Unity steps a frame, then calls
    Python), and no reading lags the car by a step, as it does after a whole
    mj_step.

    max_steps: if None (the default), go() runs until the process stops, as
    on the real car. If it is an int, go() returns after that many steps,
    so a test can run a fixed number of steps. Controller code does not see
    this argument.

    car: the car's LiDAR, encoder, start max speed and speed loop, the
    NeoRacer's by default. For another car, pass its params with its scene, such as
    racecar_neo_v2_params().
    """

    def __init__(
        self,
        scene: Scene,
        data: mujoco.MjData,
        max_steps: int | None = None,
        held_buttons: Iterable[MujocoController.Button] = (MujocoController.Button.RB,),
        car: racecar.CarParams = _NEORACER,
    ):
        self._car = _Car(scene, data, racecar.start(car, scene, data))
        self.drive = MujocoDrive(self._car, car.start_max_speed)
        self.lidar = MujocoLidar(self._car)
        self.physics = MujocoPhysics(self._car)
        self.controller = MujocoController(held_buttons)
        self.display = MujocoDisplay()
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
        return float(self._car.data.time)

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
        car = self._car
        # Whatever start() changed, the step's first half from data as it is.
        mujoco.mj_step1(car.scene.model, car.data)
        while self._max_steps is None or step < self._max_steps:
            program.update()
            if program.update_slow is not None:
                slow_countdown -= self._dt
                if slow_countdown < 0:
                    program.update_slow()
                    slow_countdown = self._update_slow_time
            car.state = racecar.step(car.scene, car.data, car.state)
            step += 1
