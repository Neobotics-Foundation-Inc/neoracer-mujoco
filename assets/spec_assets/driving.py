"""Keyboard tele-operation for the car model.

The MuJoCo passive viewer reports key *presses* (not holds or releases), so the
controller keeps a persistent throttle (cruise until you brake) and a steering
command that auto-centers each step. Holding a key repeats the press and sustains
the turn; a single tap gives a brief nudge that fades back to straight.

Controls:
    arrow up    / W : drive forward
    arrow down  / S : drive backward
    arrow left  / A : steer left
    arrow right / D : steer right
    space           : brake (throttle to zero)
"""

import math
import time

import mujoco
import mujoco.viewer

DRIVE_ACTUATORS = ("front left", "front right", "back left", "back right")
STEER_ACTUATOR = "steering"

# GLFW key codes (the viewer hands these straight through to key_callback).
KEY_SPACE = 32
KEY_A, KEY_D, KEY_S, KEY_W = 65, 68, 83, 87
KEY_RIGHT, KEY_LEFT, KEY_DOWN, KEY_UP = 262, 263, 264, 265

_FORWARD_KEYS = {KEY_UP, KEY_W}
_BACKWARD_KEYS = {KEY_DOWN, KEY_S}
_LEFT_KEYS = {KEY_LEFT, KEY_A}
_RIGHT_KEYS = {KEY_RIGHT, KEY_D}


def set_spawn(model, data, position, heading: float = 0.0) -> None:
    """Place the car's free joint at ``position`` (x, y, z) facing ``heading``."""
    adr = model.jnt_qposadr[model.body_jntadr[model.body("car").id]]
    data.qpos[adr : adr + 3] = position
    data.qpos[adr + 3 : adr + 7] = [
        math.cos(heading / 2.0),
        0.0,
        0.0,
        math.sin(heading / 2.0),
    ]
    mujoco.mj_forward(model, data)


class KeyboardController:
    """Maps arrow / WASD key presses to the car's drive and steering actuators."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        drive: float = 0.7,
        max_steer: float = 0.4,
        steer_decay: float = 0.85,
    ):
        self.model = model
        self.data = data
        self.drive = drive
        self.max_steer = max_steer
        self.steer_decay = steer_decay

        self.throttle = 0.0
        self.steer = 0.0

        self._drive_ids = [model.actuator(name).id for name in DRIVE_ACTUATORS]
        self._steer_id = model.actuator(STEER_ACTUATOR).id

    def on_key(self, keycode: int) -> None:
        """Viewer key_callback: update the persistent throttle / steer command."""
        if keycode in _FORWARD_KEYS:
            self.throttle = self.drive
        elif keycode in _BACKWARD_KEYS:
            self.throttle = -self.drive
        elif keycode in _LEFT_KEYS:
            self.steer = self.max_steer
        elif keycode in _RIGHT_KEYS:
            self.steer = -self.max_steer
        elif keycode == KEY_SPACE:
            self.throttle = 0.0
            self.steer = 0.0

    def apply(self) -> None:
        """Write the current command to ctrl, then decay steering toward center."""
        for actuator_id in self._drive_ids:
            self.data.ctrl[actuator_id] = self.throttle
        self.data.ctrl[self._steer_id] = self.steer
        self.steer *= self.steer_decay


def run_keyboard_demo(spec: mujoco.MjSpec, spawn: dict = None, *, drive: float = 0.7) -> None:
    """Compile ``spec``, optionally place the car at ``spawn``, and drive it by keyboard."""
    model = spec.compile()
    data = mujoco.MjData(model)

    if spawn is not None:
        set_spawn(model, data, spawn["position"], spawn.get("heading", 0.0))

    controller = KeyboardController(model, data, drive=drive)

    print(__doc__.split("Controls:")[1].rstrip())

    with mujoco.viewer.launch_passive(
        model, data, key_callback=controller.on_key
    ) as viewer:
        while viewer.is_running():
            controller.apply()
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)
