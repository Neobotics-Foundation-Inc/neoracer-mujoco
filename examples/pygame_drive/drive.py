"""
Drive the NeoRacer with pygame — WASD or a gamepad. Input only lives here;
all the MuJoCo work is in sim.py.

Usage (plain python3, NOT mjpython — pygame owns the window):
    python3 drive.py                            # ramp course
    python3 drive.py ../../assets/neoracer.xml  # bare car on a plane

Keyboard:                          Gamepad (auto-detected, overrides keys):
    W / S      throttle / reverse      right trigger   throttle
    A / D      steer left / right      left trigger    reverse
    Space      hard stop               left stick X    steer
    Backspace  reset car (un-flip)     Start           reset car
    Esc        quit
"""

import math
import sys
from pathlib import Path

import pygame

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sim import DriveSim

# ponytail: SDL axis numbers for an Xbox-style pad; other pads may need retuning
STEER_AXIS = 0     # left stick X, +1 = right
ACCEL_AXIS = 5     # right trigger, rests at -1
BRAKE_AXIS = 4     # left trigger, rests at -1
RESET_BUTTON = 7   # Start
DEADZONE = 0.1


LIDAR_MAX = 3.0   # m — beams past this read as "clear" (rangefinder returns -1 too)
RADAR_R = 90      # px radius of the radar disc
RADAR_PAD = 12    # px from the top-right corner


def draw_lidar(screen, beams: dict[str, float]) -> None:
    """Top-right radar: each beam a dot at its angle, distance = range,
    color hot->cold with proximity so nearby clutter reads as a dense blob."""
    cx = screen.get_width() - RADAR_R - RADAR_PAD
    cy = RADAR_R + RADAR_PAD

    disc = pygame.Surface((2 * RADAR_R, 2 * RADAR_R), pygame.SRCALPHA)
    pygame.draw.circle(disc, (10, 10, 10, 160), (RADAR_R, RADAR_R), RADAR_R)
    pygame.draw.circle(disc, (0, 255, 0, 90), (RADAR_R, RADAR_R), RADAR_R, 1)

    for name, rng in beams.items():
        angle = math.radians(int(name.split("_")[1]))
        r = rng if rng >= 0 else LIDAR_MAX          # -1 = no hit -> edge of disc
        r = min(r, LIDAR_MAX)
        near = 1.0 - r / LIDAR_MAX                   # 0 far, 1 point-blank
        # match the chase cam: car forward (+X) = screen up, car left (+Y) = screen left
        px = RADAR_R - math.sin(angle) * near * RADAR_R
        py = RADAR_R - math.cos(angle) * near * RADAR_R
        color = (255, int(255 * (1 - near)), 0)      # yellow far -> red close
        pygame.draw.circle(disc, color, (int(px), int(py)), 5)

    pygame.draw.circle(disc, (0, 200, 255), (RADAR_R, RADAR_R), 3)  # the car
    screen.blit(disc, (cx - RADAR_R, cy - RADAR_R))


def main(xml: str | None = None) -> None:
    pygame.init()
    screen = pygame.display.set_mode((900, 600))
    pygame.display.set_caption("NeoRacer — pygame drive")
    clock = pygame.time.Clock()

    joystick = None
    if pygame.joystick.get_count() > 0:
        joystick = pygame.joystick.Joystick(0)
        print(f"using gamepad: {joystick.get_name()}")

    sim = DriveSim(xml, width=900, height=600)
    print(__doc__)

    running = True
    dt = 0.0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_BACKSPACE:
                sim.reset()
            elif event.type == pygame.JOYBUTTONDOWN and event.button == RESET_BUTTON:
                sim.reset()

        keys = pygame.key.get_pressed()
        if joystick is not None:
            throttle = (joystick.get_axis(ACCEL_AXIS) + 1) / 2 \
                     - (joystick.get_axis(BRAKE_AXIS) + 1) / 2
            steer = -joystick.get_axis(STEER_AXIS)  # stick right -> steer right (negative ctrl)
            if abs(steer) < DEADZONE:
                steer = 0.0
        else:
            throttle = keys[pygame.K_w] - keys[pygame.K_s]
            steer = keys[pygame.K_a] - keys[pygame.K_d]

        if keys[pygame.K_SPACE]:
            sim.stop()
            throttle, steer = 0.0, 0.0

        sim.step(throttle, steer, dt)

        frame = sim.render()  # (h, w, 3); pygame surfaces are (w, h), hence the swap
        screen.blit(pygame.surfarray.make_surface(frame.swapaxes(0, 1)), (0, 0))
        draw_lidar(screen, sim.lidar())
        pygame.display.flip()

        dt = clock.tick(60) / 1000

    pygame.quit()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
