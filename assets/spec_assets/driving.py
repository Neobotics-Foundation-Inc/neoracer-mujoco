"""Keyboard tele-operation for the car model.

The MuJoCo passive viewer only reports key *presses* (no releases), so throttle is
*persistent and incremental*: each press nudges it and it holds until you change it,
like an RC car's speed control. Steering, by contrast, bleeds back toward center every
step, so tapping left/right turns the car and it straightens out on its own when you
stop -- this gives arrow-key driving the "hold to turn, let go to straighten" feel.

Controls:
    arrow up    / W : accelerate (more forward throttle)
    arrow down  / S : decelerate / reverse
    arrow left  / A : steer left   (hold/tap to turn; auto-centers)
    arrow right / D : steer right  (hold/tap to turn; auto-centers)
    space           : stop and straighten (throttle and steering to zero)
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
        max_throttle: float = 1.0,
        throttle_step: float = 0.25,
        max_steer: float = 0.6,
        steer_step: float = 0.2,
        steer_decay: float = 0.9,  # Target retention rate per ~33ms (keyboard repeat floor)
    ):
        self.model = model
        self.data = data
        self.max_throttle = max_throttle
        self.throttle_step = throttle_step
        self.max_steer = max_steer
        self.steer_step = steer_step
        self.steer_decay = steer_decay

        self.throttle = 0.0
        self.steer = 0.0

        self._drive_ids = [model.actuator(name).id for name in DRIVE_ACTUATORS]
        self._steer_id = model.actuator(STEER_ACTUATOR).id

    def on_key(self, keycode: int) -> None:
        """Viewer key_callback: nudge the persistent throttle / steer command."""
        if keycode in _FORWARD_KEYS:
            self.throttle = min(self.throttle + self.throttle_step, self.max_throttle)
        elif keycode in _BACKWARD_KEYS:
            self.throttle = max(self.throttle - self.throttle_step, -self.max_throttle)
        elif keycode in _LEFT_KEYS:
            self.steer = min(self.steer + self.steer_step, self.max_steer)
        elif keycode in _RIGHT_KEYS:
            self.steer = max(self.steer - self.steer_step, -self.max_steer)
        elif keycode == KEY_SPACE:
            self.throttle = 0.0
            self.steer = 0.0

    def apply(self) -> None:
        """Write the current command to the actuators."""
        dt = self.model.opt.timestep
        
        # Scale the decay factor by time (dt) relative to a standard 30Hz repeat rate.
        time_scaled_decay = self.steer_decay ** (dt / 0.033)
        self.steer *= time_scaled_decay

        if abs(self.steer) < 1e-3:
            self.steer = 0.0
            
        for actuator_id in self._drive_ids:
            self.data.ctrl[actuator_id] = self.throttle
        self.data.ctrl[self._steer_id] = self.steer


def boost_drive(spec: mujoco.MjSpec, gear: float = 4.0, force: float = 12.0) -> None:
    """Scale up the drive motors so the car reaches a fun top speed."""
    for actuator in spec.actuators:
        if actuator.name in DRIVE_ACTUATORS:
            gears = list(actuator.gear)
            gears[0] = gear
            actuator.gear = gears
    for joint in spec.joints:
        if joint.name in DRIVE_ACTUATORS:
            joint.actfrcrange = [-force, force]
            joint.actfrclimited = 1


def run_keyboard_demo(spec: mujoco.MjSpec, spawn: dict = None, *, fast: bool = True) -> None:
    """Compile ``spec``, optionally place the car at ``spawn``, and drive it by keyboard."""
    if fast:
        boost_drive(spec)

    model = spec.compile()
    data = mujoco.MjData(model)

    if spawn is not None:
        set_spawn(model, data, spawn["position"], spawn.get("heading", 0.0))

    controller = KeyboardController(model, data)

    print(__doc__.split("Controls:")[1].rstrip())

    with mujoco.viewer.launch_passive(
        model, data, key_callback=controller.on_key
    ) as viewer:
        while viewer.is_running():
            controller.apply()
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    # Load directly from the external car.xml file
    spec = mujoco.MjSpec.from_file("car.xml")
    run_keyboard_demo(spec, spawn={"position": [0, 0, 0.15], "heading": 0.0})