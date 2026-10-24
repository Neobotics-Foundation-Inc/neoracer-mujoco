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
  steer_servo       = target angle (rad), + = left, range ± full lock (neoracer_spec.derive.steer_lock)
"""

import argparse
import math
import os
from dataclasses import dataclass
from typing import Literal, assert_never

import glfw
import mujoco

from neoracer_mujoco import assets, collision, contract, sensors, sim
from neoracer_mujoco.control.steering import ControlCommand
from neoracer_mujoco.scene import Scene
from neoracer_spec import derive
from neoracer_spec.neoracer import NEORACER

type Pedal = Literal["forward", "reverse", "coast"]
type SteerKey = Literal["left", "right", "center"]

# Change these values to change how the car responds to the keys.
# The rates are per second, so the response does not depend on the frame rate.


@dataclass(frozen=True)
class ThrottleParams:
    accel: float = 2.3  # 1/s added while a throttle key is held
    coast: float = 1.7  # 1/s removed while no throttle key is held
    max: float = 1.0  # the throttle stays in [-max, max]


@dataclass(frozen=True)
class SteeringParams:
    rate: float = 1.5  # rad/s toward the limit while a steer key is held
    return_: float = 2.5  # rad/s back to the center while no steer key is held
    # rad, the steer command limit of the model, full lock
    max: float = derive.steer_lock(NEORACER)


@dataclass(frozen=True)
class CamParams:
    distance: float = 1.6  # m from the car
    elevation: float = -35.0  # deg; a more negative value looks more from the top
    # deg, added to the car yaw. 0 puts the camera behind the car (at -X, looking
    # at +X). 180 puts it in front of the car. ±90 puts it at the side.
    azimuth: float = 0.0


@dataclass(frozen=True)
class ManualDriveParams:
    throttle: ThrottleParams = ThrottleParams()
    steering: SteeringParams = SteeringParams()
    cam: CamParams = CamParams()
    window_px: tuple[int, int] = (900, 600)  # px; width, height of the window
    max_scene_geoms: int = 10000  # the maximum number of geoms in one rendered frame
    hud_period_s: float = 0.1  # s between two sensor print lines


def _yaw_deg(quat: sensors.Quat) -> float:
    """Car yaw (heading) in degrees from a MuJoCo (w, x, y, z) quaternion."""
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    return math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _toward_zero(value: float, step: float) -> float:
    """Move value toward zero by at most step. The result does not pass zero."""
    if value > 0:
        return max(0.0, value - step)
    return min(0.0, value + step)


def update(
    params: ManualDriveParams,
    command: ControlCommand,
    dt: float,
    pedal: Pedal,
    steer_key: SteerKey,
) -> ControlCommand:
    """The command for the next frame, from the command of this frame, the
    frame time dt (s) and the held keys. The throttle and the steer (rad)
    stay within the limits in params."""
    tp, sp = params.throttle, params.steering
    match pedal:
        case "forward":
            throttle = command.throttle + tp.accel * dt
        case "reverse":
            throttle = command.throttle - tp.accel * dt
        case "coast":
            throttle = _toward_zero(command.throttle, tp.coast * dt)
        case _:
            assert_never(pedal)

    match steer_key:
        case "left":
            steer = command.steer + sp.rate * dt
        case "right":
            steer = command.steer - sp.rate * dt
        case "center":
            steer = _toward_zero(command.steer, sp.return_ * dt)
        case _:
            assert_never(steer_key)

    return ControlCommand(
        throttle=max(-tp.max, min(tp.max, throttle)),
        steer=max(-sp.max, min(sp.max, steer)),
    )


def print_sensors(s: sensors.SensorReadings, wheel_radius: float) -> None:
    """Print a one-line summary of the most useful sensor values; the speed
    is the motor encoder's at the tyre of wheel_radius (m)."""
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


def parse_pedal(held: frozenset[int]) -> Pedal:
    """The pedal from the set of held glfw keys. Forward wins over reverse."""
    if glfw.KEY_UP in held or glfw.KEY_W in held:
        return "forward"
    if glfw.KEY_DOWN in held or glfw.KEY_S in held:
        return "reverse"
    return "coast"


def parse_steer_key(held: frozenset[int]) -> SteerKey:
    """The steer key from the set of held glfw keys. Left wins over right."""
    if glfw.KEY_LEFT in held or glfw.KEY_A in held:
        return "left"
    if glfw.KEY_RIGHT in held or glfw.KEY_D in held:
        return "right"
    return "center"


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

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = model.bind_scalar(car.body).id
    cam.distance = params.cam.distance
    cam.elevation = params.cam.elevation
    # The loop sets cam.azimuth in each frame from the car yaw.

    opt = mujoco.MjvOption()
    mjv_scene = mujoco.MjvScene(model, maxgeom=params.max_scene_geoms)
    ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

    print(__doc__)

    stopped = ControlCommand(throttle=0.0, steer=0.0)
    command = stopped
    last = glfw.get_time()
    hud_accum = 0.0  # s since the last sensor print line

    while not glfw.window_should_close(window):
        held = frozenset(k for k in _KEYS if glfw.get_key(window, k) == glfw.PRESS)
        if glfw.KEY_ESCAPE in held:
            break
        if glfw.KEY_BACKSPACE in held:
            mujoco.mj_resetData(model, data)
            command = stopped

        now = glfw.get_time()
        dt = now - last
        last = now

        if glfw.KEY_SPACE in held:
            command = stopped
        else:
            command = update(
                params, command, dt, parse_pedal(held), parse_steer_key(held)
            )

        sim.apply_command(car, data, command)

        # Step the physics until the simulation time catches up with real time.
        target = data.time + dt
        while data.time < target:
            mujoco.mj_step(model, data)
            if collision.car_touches_wall(data, car):
                collision.reset_car(model, data)
                command = stopped
                break

        cam.azimuth = params.cam.azimuth + _yaw_deg(data.bind_scalar(car.body).xquat)

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

    if args.bare:
        scene = assets.load()
    else:
        scene = assets.compose(mujoco.MjSpec.from_file(args.track))
    drive(scene, ManualDriveParams())


if __name__ == "__main__":
    main()
