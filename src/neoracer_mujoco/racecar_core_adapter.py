"""
MuJoCo backend for the racecar_core controller interface.

racecar_core (Neobotics-Foundation-Inc/racecar-neo-library) is the interface
that NeoRacer controllers use. This module gives the same method names and
signatures over a NeoRacer Scene and its MjData. It does not copy or change
racecar_core. So controller code (`rc.drive`, `rc.lidar`,
`rc.set_start_update`, `rc.go`) runs in this simulator without changes.

This module has Drive, Lidar, Controller, Display and MujocoRacecar.
MujocoRacecar is the equivalent of create_racecar(): it connects the other
parts and runs the start/update/update_slow loop in go(). This module has no
camera, led, nav, physics, slam, telemetry or vision, because MuJoCo has no
backend for them yet.

Frame: +X forward, +Y left, +Z up.

racecar_core and contract.py use two different conventions. This module
converts between them, so the controllers do not convert them:

  1. Steer sign. In racecar_core, + angle = right. In contract.py, the
     steer_servo + = left. MujocoDrive negates the angle.
  2. LiDAR order and units. racecar_core scans clockwise, in cm, and 0.0
     means no return. sensors.LIDAR_BEAM_ORDER is counterclockwise, in m,
     and -1 means no return. MujocoLidar changes the order and the units.

Some parts of this module do not follow AGENTS.md, because racecar_core
code needs them as they are: the Button IntEnum, get_samples_async, the
one-method MujocoDisplay, and the mutable state of MujocoDrive and
MujocoRacecar.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import IntEnum

import mujoco
import numpy as np

from . import contract, sensors, sim
from .control.steering import ControlCommand
from .scene import Scene

# racecar_core scans clockwise from the front. LIDAR_BEAM_ORDER is
# counterclockwise from the front (see sensors.py). Both orders start at
# lidar_000 (straight ahead), so the reverse of the other seven beams gives
# the clockwise order. An array, not a tuple: numpy reads a tuple index as
# one index per axis.
_CLOCKWISE = np.array([0, 7, 6, 5, 4, 3, 2, 1])  # indices into LIDAR_BEAM_ORDER
RACECAR_CORE_BEAM_ORDER = tuple(sensors.LIDAR_BEAM_ORDER[i] for i in _CLOCKWISE)


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


class MujocoDrive:
    """racecar_core.Drive backend over a NeoRacer Scene and its MjData.

    set_speed_angle(speed, angle): both unitless ratios in [-1.0, 1.0].
    speed: -1.0 full backward .. +1.0 full forward (racecar_core convention).
    angle: -1.0 full left .. +1.0 full right (racecar_core convention).

    set_speed_angle clamps an input that is out of range, and does not
    reject it. So a bad value from a policy or a noisy controller cannot stop
    a step.
    """

    def __init__(self, scene: Scene, data: mujoco.MjData):
        self._car = scene.car
        self._data = data
        self._max_speed = 1.0  # fraction of the full throttle, 0..1
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

    def set_max_speed(self, max_speed: float = 1.0) -> None:
        self._max_speed = _clamp(max_speed, 0.0, 1.0)


class MujocoLidar:
    """racecar_core.Lidar backend over neoracer_mujoco.SensorReadings.

    get_samples() returns the same values as library/lidar.py in
    racecar-neo-library. Units: cm. Order: clockwise. Index 0 points
    straight ahead. 0.0 means no return.

    This backend has 8 beams (assets/neoracer.xml). The other backends have
    about 720 or 1440 samples. racecar_core tells controllers to index with
    len(scan) or get_num_samples(), so get_num_samples() returns 8, and the
    backend does not make up more samples.
    """

    def __init__(self, scene: Scene, data: mujoco.MjData):
        self._sensors = scene.car.sensors
        self._data = data

    def get_samples(self) -> np.ndarray:
        readings = sensors.read(self._sensors, self._data)
        meters = sensors.lidar_ranges(readings)[_CLOCKWISE]
        cm = meters * 100.0
        # In racecar_core, 0.0 means "no return". A -1 (no hit) and a value
        # that is not finite both give 0.0. No real distance is negative.
        valid = np.isfinite(cm) & (cm > 0.0)
        return np.where(valid, cm, 0.0).astype(np.float32)

    def get_samples_async(self) -> np.ndarray:
        # The name comes from racecar_core, where this method reads the LiDAR
        # when the car is not in "go" mode (Jupyter only on the real car).
        # This backend has no such mode and reads the current MjData. So the
        # method is not asynchronous and returns the same as get_samples().
        return self.get_samples()

    def get_num_samples(self) -> int:
        return len(RACECAR_CORE_BEAM_ORDER)


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

    It connects MujocoDrive, MujocoLidar, MujocoController and MujocoDisplay
    to one Scene and MjData, and runs the start/update/update_slow loop in
    go(). So a racecar_core controller script, for example
    neoracer-labs/ultimate-wall-follower/wall_follower.py, runs without
    changes. Only its `import racecar_core` must give this class.

    go() steps the physics and then calls update(), in the same order as
    RacecarSim (Unity steps a frame, then calls Python). So update() reads
    the sensors after the physics step that its last command caused.

    max_steps: if None (the default), go() runs until the process stops, as
    on the real car. If it is an int, go() returns after that many steps,
    so a test can run a fixed number of steps. Controller code does not see
    this argument.
    """

    def __init__(
        self,
        scene: Scene,
        data: mujoco.MjData,
        max_steps: int | None = None,
        held_buttons: Iterable[MujocoController.Button] = (MujocoController.Button.RB,),
    ):
        self.drive = MujocoDrive(scene, data)
        self.lidar = MujocoLidar(scene, data)
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
        while self._max_steps is None or step < self._max_steps:
            mujoco.mj_step(self._model, self._data)
            program.update()
            if program.update_slow is not None:
                slow_countdown -= self._dt
                if slow_countdown < 0:
                    program.update_slow()
                    slow_countdown = self._update_slow_time
            step += 1
