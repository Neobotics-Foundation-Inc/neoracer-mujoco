"""Drive a procedurally-generated track with WASD so you can eyeball it.

Bare-bones cut of my drive.py/sim.py: WASD + chase cam only. No lidar HUD,
no gamepad, no timer — just enough to see the track 01_test.py spat out.

Every fresh track auto-saves a minimap PNG into maps/ next to this file.

Usage (plain python3, NOT mjpython — pygame owns the window):
    python3 test_drive.py                    # fresh 02 walk track (+ minimap)
    python3 test_drive.py --gen 03_random_points.py   # fresh random-points loop
    python3 test_drive.py some_track.xml     # drive an existing track file

    W / S   throttle / reverse      Backspace  reset car
    A / D   steer left / right      Esc        quit
    Space   hard stop
"""

import importlib.util
import math
import random
import sys
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np
import pygame

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parents[2]          # .../neoracer-mujoco
CAR_XML = PROJECT_DIR / "assets" / "neoracer.xml"
MAPS_DIR = HERE / "maps"               # minimaps land here

# physics knobs, same spirit as sim.py (a touch of back-EMF caps the top speed)
MAX_TORQUE = 0.15        # N·m per wheel at full throttle
MOTOR_DAMPING = 0.0025   # N·m per rad/s of wheel spin — sets top speed + braking
MAX_STEER = 0.40         # rad, the model's Ackermann command limit
CAM_DISTANCE = 1.6       # m behind/above the car
CAM_ELEVATION = -35.0    # deg looking down
W, H = 900, 600


def load_generator(filename="02_connect_check.py"):
    """Import a generator module by path (names start with a digit, so no plain
    import). 02_connect_check.py is the sealed-wall random walk; 03_random_points.py
    is the closed-loop random-points generator. Both expose to_xml()."""
    spec = importlib.util.spec_from_file_location("track_gen", HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generate(gen):
    """Ask a loaded generator for a fresh track, whichever API it has. Returns the
    generator's own track object: 02 hands back a points list, 03 a Track tuple."""
    if hasattr(gen, "generate_track"):     # 03: closed-loop random points
        return gen.generate_track(random.Random())
    return gen.random_centerline(random.Random())   # 02: random walk


def save_minimap(gen, track_obj, out_path):
    """Dispatch to the generator's own minimap if it has one (03), else use the
    02-shaped drawer in this file."""
    if hasattr(gen, "create_minimap"):
        gen.create_minimap(track_obj, out_path)
    else:
        create_minimap_image(track_obj, out_path)


def build_model(track_source, gen_file):
    """Return (compiled MjModel, gen, track_obj): the track (a file, or a freshly
    generated one) with the car body grafted on. The track supplies the floor, so
    we attach only the car's body subtree, not its duplicate ground plane. gen and
    track_obj are None for a track loaded from a file (nothing to draw a map from)."""
    gen = track_obj = None
    if track_source is not None:
        scene = mujoco.MjSpec.from_file(str(track_source))
    else:
        gen = load_generator(gen_file)
        track_obj = generate(gen)
        scene = mujoco.MjSpec.from_string(gen.to_xml(track_obj))

    car = mujoco.MjSpec.from_file(str(CAR_XML))
    frame = scene.worldbody.add_frame()
    frame.attach_body(car.body("car"), "", "")
    return scene.compile(), gen, track_obj


MAP_SIZE = 640    # px, square map image
MAP_PAD = 30      # px border around the track


def _box_corners(cx, cy, half_len, yaw, half_t):
    """The four world-space corners of one wall box, matching its MJCF geom
    (center, half-length along its axis, half-thickness across, yaw rotation)."""
    ax, ay = math.cos(yaw), math.sin(yaw)      # along the box's long axis
    nx, ny = -math.sin(yaw), math.cos(yaw)     # across it (thickness)
    return [
        (cx + ax * half_len + nx * half_t, cy + ay * half_len + ny * half_t),
        (cx + ax * half_len - nx * half_t, cy + ay * half_len - ny * half_t),
        (cx - ax * half_len - nx * half_t, cy - ay * half_len - ny * half_t),
        (cx - ax * half_len + nx * half_t, cy - ay * half_len + ny * half_t),
    ]


def create_minimap_image(points, out_path):
    """Save an honest top-down map of the track using pygame (no matplotlib).
    Each wall is drawn as its actual box segment — NOT a smoothed line — so it
    matches the sim geometry exactly. Uses connect_walls() (the sealed walls),
    same as the driven track. Thin gray line = centerline (the true path), blue
    boxes = left wall, orange boxes = right wall, green dot = start, red = finish.
    """
    gen = load_generator()
    left = gen.connect_walls(points, +1)
    right = gen.connect_walls(points, -1)
    half_t = gen.WALL_T

    # world -> pixel: fit everything in the padded square, flip Y so +Y is up.
    # bound on box corners, not just centers, so no wall clips the edge.
    corners = []
    for cx, cy, hl, yaw in left + right:
        corners += _box_corners(cx, cy, hl, yaw, half_t)
    min_x = min(c[0] for c in corners)
    max_x = max(c[0] for c in corners)
    min_y = min(c[1] for c in corners)
    max_y = max(c[1] for c in corners)
    span = max(max_x - min_x, max_y - min_y, 1e-6)
    scale = (MAP_SIZE - 2 * MAP_PAD) / span

    def to_px(pt):
        px = MAP_PAD + (pt[0] - min_x) * scale
        py = MAP_SIZE - MAP_PAD - (pt[1] - min_y) * scale  # flip Y
        return (int(px), int(py))

    if not pygame.get_init():
        pygame.init()
    surf = pygame.Surface((MAP_SIZE, MAP_SIZE))
    surf.fill((17, 17, 17))

    # centerline: the real path the walls were offset from (thin, for reference)
    pygame.draw.lines(surf, (90, 90, 90), False, [to_px(p) for p in points], 1)

    # each wall box as its own filled rectangle — gaps/overlaps stay visible
    for cx, cy, hl, yaw in left:
        quad = [to_px(c) for c in _box_corners(cx, cy, hl, yaw, half_t)]
        pygame.draw.polygon(surf, (91, 155, 213), quad)
    for cx, cy, hl, yaw in right:
        quad = [to_px(c) for c in _box_corners(cx, cy, hl, yaw, half_t)]
        pygame.draw.polygon(surf, (237, 125, 49), quad)

    pygame.draw.circle(surf, (46, 204, 113), to_px(points[0]), 8)
    pygame.draw.circle(surf, (231, 76, 60), to_px(points[-1]), 8)

    pygame.image.save(surf, str(out_path))
    print(f"wrote minimap {out_path}")


def yaw_deg(quat):
    """Car heading in degrees from a MuJoCo [w, x, y, z] quaternion."""
    w, x, y, z = quat
    return math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def main(track_source, gen_file):
    model, gen, track_obj = build_model(track_source, gen_file)
    if track_obj is not None:  # fresh track -> drop a timestamped minimap in maps/
        MAPS_DIR.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_minimap(gen, track_obj, MAPS_DIR / f"track_{stamp}.png")

    data = mujoco.MjData(model)
    car_id = model.body("car").id
    # qvel index of each drive wheel's spin, for the engine-braking term
    wheel_dofs = [model.jnt_dofadr[model.actuator_trnid[i, 0]] for i in range(4)]

    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("NeoRacer — procedural track")
    clock = pygame.time.Clock()

    model.vis.global_.offwidth = max(W, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(H, model.vis.global_.offheight)
    renderer = mujoco.Renderer(model, height=H, width=W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = car_id
    cam.distance = CAM_DISTANCE
    cam.elevation = CAM_ELEVATION

    print(__doc__)
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_BACKSPACE:
                mujoco.mj_resetData(model, data)

        keys = pygame.key.get_pressed()
        throttle = keys[pygame.K_w] - keys[pygame.K_s]
        steer = keys[pygame.K_a] - keys[pygame.K_d]
        if keys[pygame.K_SPACE]:
            throttle, steer = 0.0, 0.0
            data.qvel[:] = 0.0

        # DC-motor model: commanded torque minus back-EMF drag (caps top speed
        # and brakes the car when you let off the throttle)
        spin = data.qvel[wheel_dofs]
        data.ctrl[0:4] = np.clip(throttle * MAX_TORQUE - MOTOR_DAMPING * spin,
                                 -MAX_TORQUE, MAX_TORQUE)
        data.ctrl[4] = steer * MAX_STEER

        # step a wall-clock frame's worth of physics
        target = data.time + clock.get_time() / 1000
        while data.time < target:
            mujoco.mj_step(model, data)

        cam.azimuth = yaw_deg(data.xquat[car_id])
        renderer.update_scene(data, camera=cam)
        frame = renderer.render()   # (H, W, 3); pygame surfaces are (W, H), hence swap
        screen.blit(pygame.surfarray.make_surface(frame.swapaxes(0, 1)), (0, 0))
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    # --gen <file> picks the generator (default 02); a positional .xml drives an
    # existing track instead of generating one.
    args = sys.argv[1:]
    gen_file = "02_connect_check.py"
    if "--gen" in args:
        pos = args.index("--gen")
        gen_file = args[pos + 1]
        args = args[:pos] + args[pos + 2:]
    track_file = args[0] if args else None
    main(track_file, gen_file)
