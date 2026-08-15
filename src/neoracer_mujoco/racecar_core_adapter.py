"""
MuJoCo backend for the racecar_core controller interface (issue #8).

racecar_core (Neobotics-Foundation-Inc/racecar-neo-library) is the interface
every NeoRacer controller is meant to be portable against: the physical car
and the browser Playground sim both implement it today
(library/real/*_real.py, library/simulation/*_sim.py). This module is the
third backend, for neoracer_mujoco. It does not vendor or modify
racecar_core itself -- it duck-types the same method names/signatures so
controller code written against `rc.drive` / `rc.lidar` runs unchanged here.

Drive, Lidar, Controller, and Display are implemented, plus MujocoRacecar,
the top-level object (issue #28) that wires them together and replaces
racecar_core.create_racecar()'s start/update/go event loop -- enough for an
unmodified racecar_core controller script (e.g.
neoracer-labs/ultimate-wall-follower/wall_follower.py) to run against
MuJoCo. racecar_core also defines camera, led, nav, physics, slam,
telemetry, and vision modules; none of those have a driving-relevant MuJoCo
equivalent yet, so adding stubs for them now would be an abstraction with no
real backend behind it. Controller and Display are stubs too (no gamepad or
screen in headless MuJoCo) but are implemented because the canonical
wall-following example calls them directly.

Two conventions differ between racecar_core and this simulator's native
contract (see neoracer_mujoco.contract) and must be crossed exactly once,
here, rather than by every controller that uses this adapter:

  1. Steering sign. racecar_core's `angle` is + = right. This sim's
     ctrl[4] (steer_servo) is + = left. MujocoDrive negates angle.
  2. LiDAR winding + units. racecar_core scans clockwise, in cm, 0.0 = no
     return. This sim's 8 named beams (sensors.LIDAR_BEAM_ORDER) are
     ordered counterclockwise, in meters, -1 = no return. MujocoLidar
     re-sequences and rescales.
"""

from enum import IntEnum

import mujoco
import numpy as np

from . import contract
from . import sensors as _sensors

# racecar_core scans clockwise from front; LIDAR_BEAM_ORDER is ordered
# counterclockwise from front (see sensors.py). Both wind through the same
# ring starting at the same beam (000, straight ahead), so reversing every
# beam except the first re-sequences counterclockwise into clockwise without
# changing which physical direction any beam points.
RACECAR_CORE_BEAM_ORDER = (_sensors.LIDAR_BEAM_ORDER[0],) + _sensors.LIDAR_BEAM_ORDER[
    1:
][::-1]


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


class MujocoDrive:
    """racecar_core.Drive backend over a NeoRacer MjModel/MjData.

    set_speed_angle(speed, angle): both unitless ratios in [-1.0, 1.0].
    speed: -1.0 full backward .. +1.0 full forward (racecar_core convention).
    angle: -1.0 full left .. +1.0 full right (racecar_core convention).

    Out-of-range inputs are clamped rather than rejected: a bad value from
    an RL policy or a noisy controller should not be able to crash a step.
    """

    def __init__(self, model, data):
        self._model = model
        self._data = data
        self._max_speed = 1.0
        # ctrl[0:4]'s physical unit is N*m (see contract.py), but the
        # fl_motor actuator's ctrlrange bounds it -- read that bound instead
        # of assuming it happens to equal 1.0, so this still scales
        # correctly if a future car's motors are characterized differently.
        self._torque_limit = float(model.actuator("fl_motor").ctrlrange[1])
        self._steer_limit = float(model.actuator("steer_servo").ctrlrange[1])

    def set_speed_angle(self, speed: float, angle: float) -> None:
        speed = _clamp(speed, -1.0, 1.0)
        angle = _clamp(angle, -1.0, 1.0)
        self._data.ctrl[contract.DRIVE] = speed * self._max_speed * self._torque_limit
        # Negate: racecar_core's angle is +right, ctrl[4] is +left.
        self._data.ctrl[contract.STEER] = -angle * self._steer_limit

    def stop(self) -> None:
        self.set_speed_angle(0.0, 0.0)

    def set_max_speed(self, max_speed: float = 1.0) -> None:
        self._max_speed = _clamp(max_speed, 0.0, 1.0)


class MujocoLidar:
    """racecar_core.Lidar backend over neoracer_mujoco.SensorReadings.

    get_samples() returns distances in cm, clockwise order, index 0 =
    straight ahead, 0.0 = no return -- matching racecar-neo-library's
    library/lidar.py contract exactly. This backend has 8 physical
    rangefinder beams (assets/neoracer.xml), not the ~720/1440 samples of
    the sim/real backends. Per that same interface's own docstring,
    "write length-agnostic code: index with len(scan) or
    get_num_samples(), never a hardcoded count" -- get_num_samples()
    reports 8, truthfully, rather than upsampling to imitate a sample
    count this sensor doesn't have.
    """

    def __init__(self, model, data):
        self._model = model
        self._data = data

    def get_samples(self) -> np.ndarray:
        readings = _sensors.read(self._model, self._data)
        meters = np.array(
            [getattr(readings, name)[0] for name in RACECAR_CORE_BEAM_ORDER]
        )
        cm = meters * 100.0
        # -1 (no hit) and any non-finite reading both mean "no return" (0.0)
        # in racecar_core; a real negative distance is never physical.
        valid = np.isfinite(cm) & (cm > 0.0)
        return np.where(valid, cm, 0.0).astype(np.float32)

    def get_samples_async(self) -> np.ndarray:
        # racecar_core reserves this for reading LIDAR without the car in
        # "go" mode (Jupyter-only on the real car). This backend has no
        # such mode -- every read reflects the current MjData state -- so
        # it is identical to get_samples().
        return self.get_samples()

    def get_num_samples(self) -> int:
        return len(RACECAR_CORE_BEAM_ORDER)


class MujocoController:
    """racecar_core.Controller backend: MuJoCo has no gamepad, so button
    state is a settable set rather than live hardware input.

    Only is_down is implemented -- the only Controller method the canonical
    wall-following example (its RB deadman) actually calls; library/
    controller.py also declares was_pressed/was_released/get_trigger/
    get_joystick, but nothing here has a real backend for those yet.

    held: buttons treated as pressed for the whole run. Defaults to {RB} so
    a deadman-gated controller drives immediately in headless simulation;
    pass held=set() (or set_held(Button.RB, False)) to simulate it released.
    """

    class Button(IntEnum):
        A = 0
        B = 1
        X = 2
        Y = 3
        LB = 4
        RB = 5
        LJOY = 6
        RJOY = 7

    def __init__(self, held=None):
        self._held = set(held) if held is not None else {self.Button.RB}

    def is_down(self, button) -> bool:
        return button in self._held

    def set_held(self, button, is_held: bool) -> None:
        if is_held:
            self._held.add(button)
        else:
            self._held.discard(button)


class MujocoDisplay:
    """racecar_core.Display backend. No graphical display in headless
    MuJoCo, so show_text is a no-op -- callers don't need a headless guard
    around every call, matching library/display.py's DisplaySim(headless)."""

    def show_text(self, text: str) -> None:
        pass


class MujocoRacecar:
    """racecar_core.Racecar backend over MuJoCo (issue #28): wires
    MujocoDrive/MujocoLidar/MujocoController/MujocoDisplay to one
    MjModel/MjData and implements the start/update/update_slow/go event
    loop, so an unmodified racecar_core controller script -- e.g.
    neoracer-labs/ultimate-wall-follower/wall_follower.py -- runs as-is:
    only its `import racecar_core` needs to resolve to this class.

    go() steps physics then calls update(), mirroring RacecarSim's own
    order (Unity advances a frame, then calls back into Python) so a
    controller reading sensors in update() sees the state update() just
    caused, not a stale one.

    max_steps: None (default) matches real racecar_core -- go() blocks
    until the process is stopped externally. Set to an int so a test can
    run a bounded, deterministic number of steps and return instead of
    hanging forever. This is a MujocoRacecar constructor argument, not part
    of start/update/update_slow, so controller code never has to know
    about it.
    """

    def __init__(self, model, data, max_steps=None, held_buttons=None):
        self.drive = MujocoDrive(model, data)
        self.lidar = MujocoLidar(model, data)
        self.controller = MujocoController(held_buttons)
        self.display = MujocoDisplay()
        self._model = model
        self._data = data
        self._dt = float(model.opt.timestep)
        self._max_steps = max_steps
        self._update_slow_time = 1.0
        self._start = None
        self._update = None
        self._update_slow = None

    def set_start_update(self, start, update, update_slow=None) -> None:
        self._start = start
        self._update = update
        self._update_slow = update_slow

    def get_delta_time(self) -> float:
        return self._dt

    def set_update_slow_time(self, time: float = 1.0) -> None:
        self._update_slow_time = time

    def go(self) -> None:
        self._start()
        slow_countdown = self._update_slow_time
        step = 0
        while self._max_steps is None or step < self._max_steps:
            mujoco.mj_step(self._model, self._data)
            self._update()
            if self._update_slow is not None:
                slow_countdown -= self._dt
                if slow_countdown < 0:
                    self._update_slow()
                    slow_countdown = self._update_slow_time
            step += 1
