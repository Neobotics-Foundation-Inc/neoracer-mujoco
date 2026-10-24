"""
Drive the NeoRacer with pygame, from the keyboard or a gamepad. What it
shares with manual_drive.py, the scene, the chase camera and how the command
follows the input, is in examples/driving.py; this file owns the window, the
input and the HUD.

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
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, assert_never

import mujoco
import pygame

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
from neoracer_mujoco import assets, sensors

# The command follows the input both ways at one rate. The steer's is fast,
# to smooth the keys' all-or-nothing input, which a stick barely notices.
SHAPING = Shaping(
    throttle=Rates(toward=2.5, back=2.5), steer=Rates(toward=4.0, back=4.0)
)


@dataclass(frozen=True)
class ViewParams:
    """The window and the chase camera."""

    width: int = 900  # px
    height: int = 600  # px
    cam: ChaseCam = ChaseCam()


@dataclass(frozen=True)
class GamepadParams:
    """SDL axis and button numbers of the gamepad.

    Limit: the numbers are for an Xbox-style pad. If a pad drives wrong, print
    pad.get_axis(i) for each i in range(pad.get_numaxes()) while you move one
    control at a time. Then change the numbers here.
    """

    steer_axis: int = 0  # left stick X, +1 = right
    accel_axis: int = 5  # right trigger, rests at -1
    brake_axis: int = 4  # left trigger, rests at -1
    reset_button: int = 7  # Start
    deadzone: float = 0.1  # steer input below this is 0


@dataclass(frozen=True)
class HudParams:
    """The LiDAR radar and the WASD key caps."""

    lidar_max: float = 2.0  # m; a beam past this, or with no hit, is drawn at the rim
    radar_radius: int = 90  # px
    radar_pad: int = 12  # px from the top-right corner
    key_size: int = 34  # px per key cap
    key_gap: int = 4  # px
    key_pad: int = 12  # px from the bottom-right corner


type Action = Literal["quit", "reset_car", "toggle_timer", "reset_timer", "none"]

_KEY_ACTIONS: Mapping[int, Action] = MappingProxyType(
    {
        pygame.K_ESCAPE: "quit",
        pygame.K_BACKSPACE: "reset_car",
        pygame.K_t: "toggle_timer",
        pygame.K_r: "reset_timer",
    }
)


def event_action(gamepad: GamepadParams, event: pygame.event.Event) -> Action:
    """What a pygame event asks for. Held keys (WASD, Space) are read each
    frame instead, with pygame.key.get_pressed."""
    if event.type == pygame.QUIT:
        return "quit"
    if event.type == pygame.KEYDOWN:
        return _KEY_ACTIONS.get(event.key, "none")
    if event.type == pygame.JOYBUTTONDOWN and event.button == gamepad.reset_button:
        return "reset_car"
    return "none"


def keyboard_inputs(keys: pygame.key.ScancodeWrapper) -> Drive:
    return Drive(
        throttle=float(keys[pygame.K_w] - keys[pygame.K_s]),
        steer=float(keys[pygame.K_a] - keys[pygame.K_d]),
    )


def gamepad_inputs(p: GamepadParams, pad: pygame.joystick.JoystickType) -> Drive:
    throttle = (pad.get_axis(p.accel_axis) + 1) / 2 - (
        pad.get_axis(p.brake_axis) + 1
    ) / 2
    steer = -pad.get_axis(p.steer_axis)  # stick right = steer right = negative
    return Drive(throttle=throttle, steer=0.0 if abs(steer) < p.deadzone else steer)


def draw_lidar(hud: HudParams, screen: pygame.Surface, ranges: sensors.Vec8) -> None:
    """Top-right radar: one dot per beam at its angle and range, yellow when
    far and red when close. Car forward (+X) is up and car left (+Y) is left,
    the same as the chase camera."""
    r = hud.radar_radius
    disc = pygame.Surface((2 * r, 2 * r), pygame.SRCALPHA)
    pygame.draw.circle(disc, (10, 10, 10, 160), (r, r), r)
    pygame.draw.circle(disc, (0, 255, 0, 90), (r, r), r, 1)
    for i, value in enumerate(ranges):
        angle = math.radians(45 * i)  # LIDAR_BEAM_ORDER is 0deg..315deg in 45deg steps
        beam = sensors.parse_beam(float(value))
        dist = sensors.beam_distance(beam, hud.lidar_max) / hud.lidar_max
        px = r - math.sin(angle) * dist * r
        py = r - math.cos(angle) * dist * r
        pygame.draw.circle(disc, (255, int(255 * dist), 0), (int(px), int(py)), 5)
    pygame.draw.circle(disc, (0, 200, 255), (r, r), 3)  # the car
    screen.blit(disc, (screen.get_width() - 2 * r - hud.radar_pad, hud.radar_pad))


def draw_wasd(
    hud: HudParams,
    screen: pygame.Surface,
    keys: pygame.key.ScancodeWrapper,
    font: pygame.font.Font,
) -> None:
    """Bottom-right WASD key caps; a held key is green."""
    caps = (
        ("W", pygame.K_w, 1, 0),
        ("A", pygame.K_a, 0, 1),
        ("S", pygame.K_s, 1, 1),
        ("D", pygame.K_d, 2, 1),
    )
    size, step = hud.key_size, hud.key_size + hud.key_gap
    x0 = screen.get_width() - 3 * step - hud.key_pad
    y0 = screen.get_height() - 2 * step - hud.key_pad
    for label, key, col, row in caps:
        rect = pygame.Rect(x0 + col * step, y0 + row * step, size, size)
        held = keys[key]
        pygame.draw.rect(
            screen, (0, 200, 60) if held else (40, 40, 40), rect, border_radius=6
        )
        pygame.draw.rect(
            screen, (0, 255, 90) if held else (90, 90, 90), rect, 2, border_radius=6
        )
        text = font.render(label, True, (0, 0, 0) if held else (220, 220, 220))
        screen.blit(text, text.get_rect(center=rect.center))


def main(track: str | None) -> None:
    """Drive the car on the track XML at the path track, or alone on its
    own plane if track is None."""
    view, gamepad, hud = ViewParams(), GamepadParams(), HudParams()
    scene = load_scene(track)
    model, car = scene.model, scene.car
    data = mujoco.MjData(model)
    state = Drive()

    # The offscreen buffer must be at least the window size before the
    # Renderer is made. The MJCF default is 640x480.
    model.vis.global_.offwidth = max(view.width, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(view.height, model.vis.global_.offheight)
    renderer = mujoco.Renderer(model, height=view.height, width=view.width)
    cam = chase_camera(view.cam, model, car)

    pygame.init()
    screen = pygame.display.set_mode((view.width, view.height))
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
            match event_action(gamepad, event):
                case "quit":
                    running = False
                case "reset_car":
                    mujoco.mj_resetData(model, data)
                    state = Drive()
                case "toggle_timer":
                    timing = not timing
                case "reset_timer":
                    elapsed, timing = 0.0, False
                case "none":
                    pass
                case other:
                    assert_never(other)

        keys = pygame.key.get_pressed()
        inputs = (
            gamepad_inputs(gamepad, pad) if pad is not None else keyboard_inputs(keys)
        )
        if keys[pygame.K_SPACE]:
            state, inputs = Drive(), Drive()  # hard stop, no slew
        state = next_drive(SHAPING, state, inputs, dt)
        # The car may touch the walls here: Backspace or Start resets it.
        apply_and_step(scene, data, state, frame_dt(SHAPING, dt))

        follow_car(view.cam, cam, data, car)
        renderer.update_scene(data, camera=cam)
        frame = renderer.render()  # (h, w, 3); a pygame surface is (w, h)
        screen.blit(pygame.surfarray.make_surface(frame.swapaxes(0, 1)), (0, 0))
        if timing:
            elapsed += frame_dt(SHAPING, dt)
        ranges = sensors.lidar_ranges(sensors.read(car.sensors, data))
        draw_lidar(hud, screen, ranges)
        draw_wasd(hud, screen, keys, key_font)
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
    main(None if args.bare else os.path.join(assets.TRACK_DIR, args.track))
