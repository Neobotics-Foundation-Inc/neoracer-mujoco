"""
Drive the NeoRacer car by hand with the keyboard, like in a video game.

Usage:
    python3 -m examples.manual_drive                         # the ramp course
    python3 -m examples.manual_drive --track path/to/track.xml
    python3 -m examples.manual_drive --bare                  # the car alone on a plane

Use plain python3, not mjpython.
mjpython runs the script on a second thread, but glfw must create its window on the main thread.
With mjpython, the script stops with "NSWindow should only be instantiated on the main thread".

The script opens a glfw window and reads the held keys in each frame.
If you hold a throttle key, the throttle increases. If you release it, the throttle decreases to 0.
If you release the steer keys, the steer returns to the center.

    Up / W   throttle (hold to build speed)     Left  / A   steer left
    Down / S reverse                            Right / D   steer right
    Space    stop (throttle and steer to 0)     Backspace   reset the car
    Esc      quit

If the car touches a wall (see neoracer_mujoco.collision), the script resets the car.
A reset puts the car back at its start position, the same as Backspace.

Frame: +X forward, +Y left, +Z up.
Actuators (see neoracer_mujoco.contract):
  motor             = terminal voltage (V), + = forward; sim.apply_command
                      sends a throttle in [-1, 1] as that fraction of the battery
  steer_servo       = target angle (rad), + = left, up to the car's full lock
                      (contract.actuator_limits)
"""

import argparse
import os
from dataclasses import dataclass

import glfw
import mujoco

from examples.driving import (
    ChaseCam,
    Drive,
    Rates,
    Shaping,
    apply_and_step,
    chase_camera,
    follow_car,
    frame_dt,
    load_scene,
    next_drive,
)
from neoracer_mujoco import assets, collision, contract, sensors
from neoracer_mujoco.scene import Scene


@dataclass(frozen=True)
class ManualDriveParams:
    # Change these values to change how the car responds to the keys. The
    # rates are per second, so the response does not depend on the frame
    # rate: a held throttle key builds the throttle, and a held steer key
    # turns the wheels toward full lock.
    shaping: Shaping = Shaping(
        throttle=Rates(toward=2.3, back=1.7), steer=Rates(toward=3.4, back=5.7)
    )
    cam: ChaseCam = ChaseCam()
    window_px: tuple[int, int] = (900, 600)  # px; width, height of the window
    max_scene_geoms: int = 10000  # the maximum number of geoms in one rendered frame
    hud_period_s: float = 0.1  # s between two sensor print lines


def print_sensors(s: sensors.SensorReadings, wheel_radius: float) -> None:
    """Print a one-line summary of the most useful sensor values; the speed
    is the motor encoder's at the tire of wheel_radius (m)."""
    speed = sensors.motor_encoder_rad_s(s) * wheel_radius  # m/s
    print(
        f"spd={speed:+5.2f}m/s  "
        f"steer={s.steer_cmd_pos:+.3f}({s.fl_steer_pos:+.3f}/{s.fr_steer_pos:+.3f})  "
        f"susp=[{s.fl_susp_pos:+.3f} {s.fr_susp_pos:+.3f} "
        f"{s.rl_susp_pos:+.3f} {s.rr_susp_pos:+.3f}]m  "
        f"az={s.imu_accel[2]:+5.1f}  gz={s.imu_gyro[2]:+.3f}  "
        f"lidar[fwd={s.lidar_000:.2f} L={s.lidar_090:.2f} R={s.lidar_270:.2f}]"
    )


# The glfw keys that the script reads in each frame.
_KEYS = (
    glfw.KEY_UP,
    glfw.KEY_W,
    glfw.KEY_DOWN,
    glfw.KEY_S,
    glfw.KEY_LEFT,
    glfw.KEY_A,
    glfw.KEY_RIGHT,
    glfw.KEY_D,
    glfw.KEY_SPACE,
    glfw.KEY_BACKSPACE,
    glfw.KEY_ESCAPE,
)


def keyboard_inputs(held: frozenset[int]) -> Drive:
    """The driver's input from the set of held glfw keys. Forward wins over
    reverse, and left over right."""
    forward = glfw.KEY_UP in held or glfw.KEY_W in held
    reverse = glfw.KEY_DOWN in held or glfw.KEY_S in held
    left = glfw.KEY_LEFT in held or glfw.KEY_A in held
    right = glfw.KEY_RIGHT in held or glfw.KEY_D in held
    return Drive(
        throttle=1.0 if forward else -1.0 if reverse else 0.0,
        steer=1.0 if left else -1.0 if right else 0.0,
    )


def drive(scene: Scene, params: ManualDriveParams) -> None:
    """Open a window and drive the car in scene with the keyboard until the
    window closes or Esc is pressed."""
    model, car = scene.model, scene.car
    data = mujoco.MjData(model)

    if not glfw.init():
        raise RuntimeError("glfw.init() failed")
    width, height = params.window_px
    window = glfw.create_window(width, height, "NeoRacer — manual drive", None, None)
    glfw.make_context_current(window)
    glfw.swap_interval(1)  # vsync

    cam = chase_camera(params.cam, model, car)
    opt = mujoco.MjvOption()
    mjv_scene = mujoco.MjvScene(model, maxgeom=params.max_scene_geoms)
    ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

    print(__doc__)

    command = Drive()
    last = glfw.get_time()
    hud_accum = 0.0  # s since the last sensor print line

    while not glfw.window_should_close(window):
        held = frozenset(k for k in _KEYS if glfw.get_key(window, k) == glfw.PRESS)
        if glfw.KEY_ESCAPE in held:
            break
        if glfw.KEY_BACKSPACE in held:
            mujoco.mj_resetData(model, data)
            command = Drive()

        now = glfw.get_time()
        dt = frame_dt(params.shaping, now - last)
        last = now

        if glfw.KEY_SPACE in held:
            command = Drive()
        else:
            command = next_drive(params.shaping, command, keyboard_inputs(held), dt)

        # Step the physics as far as the frame took in real time, at most
        # the shaping's max_dt.
        if apply_and_step(scene, data, command, dt):
            collision.reset_car(model, data)
            command = Drive()

        follow_car(params.cam, cam, data, car)
        w, h = glfw.get_framebuffer_size(window)
        viewport = mujoco.MjrRect(0, 0, w, h)
        mujoco.mjv_updateScene(
            model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, mjv_scene
        )
        mujoco.mjr_render(viewport, mjv_scene, ctx)
        glfw.swap_buffers(window)
        glfw.poll_events()

        hud_accum += dt
        if hud_accum >= params.hud_period_s:
            print_sensors(
                sensors.read(car.sensors, data), contract.wheel_radius(model, car)
            )
            hud_accum = 0.0

    glfw.terminate()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--track",
        default=os.path.join(assets.TRACK_DIR, "ramp_course.xml"),
        help="a track XML; the script puts the car on it (default: the ramp course)",
    )
    source.add_argument(
        "--bare", action="store_true", help="the car alone on its own plane"
    )
    args = parser.parse_args()

    drive(load_scene(None if args.bare else args.track), ManualDriveParams())


if __name__ == "__main__":
    main()
