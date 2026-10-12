"""
Drive the NeoRacer with pygame, from the keyboard or a gamepad. The physics
is in drive_sim.py; this file owns the window, the input and the HUD.

Usage (plain python3, NOT mjpython: pygame owns the window):
    python3 -m examples.pygame_drive.drive                          # F1 track
    python3 -m examples.pygame_drive.drive --track ramp_course.xml  # another track
    python3 -m examples.pygame_drive.drive --bare                   # car on its own plane

Keyboard:                          Gamepad (found at start, overrides keys):
    W / S      throttle / reverse      right trigger   throttle
    A / D      steer left / right      left trigger    reverse
    Space      hard stop               left stick X    steer
    Backspace  reset car (un-flip)     Start           reset car
    T          start or stop timer
    R          reset timer
    Esc        quit

Needs pygame-ce: pip install -e ".[demo]".
"""

import argparse
import math
import os
from dataclasses import dataclass
from typing import Literal, assert_never

import mujoco
import numpy as np
import numpy.typing as npt
import pygame

from examples.pygame_drive.drive_sim import (
    DriveParams,
    DriveState,
    apply_and_step,
    drive_wheel_dofs,
    next_state,
)
from neoracer_mujoco import assets, sensors

WIDTH, HEIGHT = 900, 600

# Limit: SDL axis numbers for an Xbox-style pad. Other pads may need other numbers.
STEER_AXIS = 0  # left stick X, +1 = right
ACCEL_AXIS = 5  # right trigger, rests at -1
BRAKE_AXIS = 4  # left trigger, rests at -1
RESET_BUTTON = 7  # Start
DEADZONE = 0.1

CAM_DISTANCE = 1.6  # m behind and above the car
CAM_ELEVATION = -35.0  # deg looking down

LIDAR_MAX = 2.0  # m; a beam past this, or with no hit, is drawn at the rim
RADAR_R = 90  # px radius of the radar disc
RADAR_PAD = 12  # px from the top-right corner

KEY_SZ = 34  # px per key cap
KEY_GAP = 4
KEY_PAD = 12  # px from the bottom-right corner


@dataclass(frozen=True)
class Inputs:
    """One frame of driver input, both in [-1, 1]."""

    throttle: float
    steer: float


type Action = Literal["quit", "reset_car", "toggle_timer", "reset_timer", "none"]

_KEY_ACTIONS: frozendict[int, Action] = frozendict(
    {
        pygame.K_ESCAPE: "quit",
        pygame.K_BACKSPACE: "reset_car",
        pygame.K_t: "toggle_timer",
        pygame.K_r: "reset_timer",
    }
)


def event_action(event: pygame.event.Event) -> Action:
    """What a pygame event asks for. Held keys (WASD, Space) are read each
    frame instead, with pygame.key.get_pressed."""
    if event.type == pygame.QUIT:
        return "quit"
    if event.type == pygame.KEYDOWN:
        return _KEY_ACTIONS.get(event.key, "none")
    if event.type == pygame.JOYBUTTONDOWN and event.button == RESET_BUTTON:
        return "reset_car"
    return "none"


def load_model(track: str, bare: bool) -> mujoco.MjModel:
    """The car on a track from assets/tracks/, or the car alone on the
    plane in neoracer.xml."""
    if bare:
        return assets.load()
    return assets.compose(
        mujoco.MjSpec.from_file(os.path.join(assets.TRACK_DIR, track))
    )


def yaw_deg(quat: npt.NDArray[np.float64]) -> float:
    """Heading (deg) from a MuJoCo (w, x, y, z) quaternion."""
    w, x, y, z = quat
    return math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def keyboard_inputs(keys: pygame.key.ScancodeWrapper) -> Inputs:
    return Inputs(
        throttle=float(keys[pygame.K_w] - keys[pygame.K_s]),
        steer=float(keys[pygame.K_a] - keys[pygame.K_d]),
    )


def gamepad_inputs(pad: pygame.joystick.JoystickType) -> Inputs:
    throttle = (pad.get_axis(ACCEL_AXIS) + 1) / 2 - (pad.get_axis(BRAKE_AXIS) + 1) / 2
    steer = -pad.get_axis(STEER_AXIS)  # stick right = steer right = negative
    return Inputs(throttle=throttle, steer=0.0 if abs(steer) < DEADZONE else steer)


def draw_lidar(screen: pygame.Surface, ranges: sensors.Vec8) -> None:
    """Top-right radar: one dot per beam at its angle and range, yellow when
    far and red when close. Car forward (+X) is up and car left (+Y) is left,
    the same as the chase camera."""
    disc = pygame.Surface((2 * RADAR_R, 2 * RADAR_R), pygame.SRCALPHA)
    pygame.draw.circle(disc, (10, 10, 10, 160), (RADAR_R, RADAR_R), RADAR_R)
    pygame.draw.circle(disc, (0, 255, 0, 90), (RADAR_R, RADAR_R), RADAR_R, 1)
    for i, value in enumerate(ranges):
        angle = math.radians(45 * i)  # LIDAR_BEAM_ORDER is 0deg..315deg in 45deg steps
        dist = (
            sensors.beam_distance(sensors.parse_beam(float(value)), LIDAR_MAX)
            / LIDAR_MAX
        )
        px = RADAR_R - math.sin(angle) * dist * RADAR_R
        py = RADAR_R - math.cos(angle) * dist * RADAR_R
        pygame.draw.circle(disc, (255, int(255 * dist), 0), (int(px), int(py)), 5)
    pygame.draw.circle(disc, (0, 200, 255), (RADAR_R, RADAR_R), 3)  # the car
    screen.blit(disc, (screen.get_width() - 2 * RADAR_R - RADAR_PAD, RADAR_PAD))


def draw_wasd(
    screen: pygame.Surface, keys: pygame.key.ScancodeWrapper, font: pygame.font.Font
) -> None:
    """Bottom-right WASD key caps; a held key is green."""
    caps = (
        ("W", pygame.K_w, 1, 0),
        ("A", pygame.K_a, 0, 1),
        ("S", pygame.K_s, 1, 1),
        ("D", pygame.K_d, 2, 1),
    )
    x0 = screen.get_width() - 3 * (KEY_SZ + KEY_GAP) - KEY_PAD
    y0 = screen.get_height() - 2 * (KEY_SZ + KEY_GAP) - KEY_PAD
    for label, key, col, row in caps:
        rect = pygame.Rect(
            x0 + col * (KEY_SZ + KEY_GAP), y0 + row * (KEY_SZ + KEY_GAP), KEY_SZ, KEY_SZ
        )
        held = keys[key]
        pygame.draw.rect(
            screen, (0, 200, 60) if held else (40, 40, 40), rect, border_radius=6
        )
        pygame.draw.rect(
            screen, (0, 255, 90) if held else (90, 90, 90), rect, 2, border_radius=6
        )
        text = font.render(label, True, (0, 0, 0) if held else (220, 220, 220))
        screen.blit(text, text.get_rect(center=rect.center))


def main(track: str, bare: bool) -> None:
    model = load_model(track, bare)
    data = mujoco.MjData(model)
    params, state = DriveParams(), DriveState()
    wheels = drive_wheel_dofs(model)
    layout = sensors.sensor_layout(model)
    car_id = model.body("car").id

    # The offscreen buffer must be at least the window size before the
    # Renderer is made. The MJCF default is 640x480.
    model.vis.global_.offwidth = max(WIDTH, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(HEIGHT, model.vis.global_.offheight)
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = car_id
    cam.distance = CAM_DISTANCE
    cam.elevation = CAM_ELEVATION

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("NeoRacer: pygame drive")
    clock = pygame.time.Clock()
    key_font = pygame.font.SysFont(None, 26)
    timer_font = pygame.font.SysFont(None, 48)
    pad = pygame.joystick.Joystick(0) if pygame.joystick.get_count() > 0 else None
    if pad is not None:
        print(f"using gamepad: {pad.get_name()}")
    print(__doc__)

    elapsed, timing = 0.0, False
    dt = 0.0
    running = True
    while running:
        for event in pygame.event.get():
            match event_action(event):
                case "quit":
                    running = False
                case "reset_car":
                    mujoco.mj_resetData(model, data)
                    state = DriveState()
                case "toggle_timer":
                    timing = not timing
                case "reset_timer":
                    elapsed, timing = 0.0, False
                case "none":
                    pass
                case other:
                    assert_never(other)

        keys = pygame.key.get_pressed()
        inputs = gamepad_inputs(pad) if pad is not None else keyboard_inputs(keys)
        if keys[pygame.K_SPACE]:
            state, inputs = DriveState(), Inputs(0.0, 0.0)  # hard stop, no slew
        state = next_state(params, state, inputs.throttle, inputs.steer, dt)
        apply_and_step(model, data, params, wheels, state, dt)

        cam.azimuth = yaw_deg(data.xquat[car_id])
        renderer.update_scene(data, camera=cam)
        frame = renderer.render()  # (h, w, 3); a pygame surface is (w, h)
        screen.blit(pygame.surfarray.make_surface(frame.swapaxes(0, 1)), (0, 0))
        if timing:
            elapsed += dt
        draw_lidar(screen, sensors.lidar_ranges(sensors.read_layout(layout, data)))
        draw_wasd(screen, keys, key_font)
        color = (0, 255, 90) if timing else (220, 220, 220)
        label = timer_font.render(
            f"{int(elapsed // 60):02d}:{elapsed % 60:05.2f}", True, color
        )
        screen.blit(label, (14, 12))
        pygame.display.flip()
        dt = clock.tick(60) / 1000

    renderer.close()
    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--track",
        default="f1_track.xml",
        choices=[os.path.basename(t) for t in assets.tracks()],
        help="a track in assets/tracks/",
    )
    parser.add_argument(
        "--bare", action="store_true", help="the car alone on its own plane"
    )
    args = parser.parse_args()
    main(args.track, args.bare)
