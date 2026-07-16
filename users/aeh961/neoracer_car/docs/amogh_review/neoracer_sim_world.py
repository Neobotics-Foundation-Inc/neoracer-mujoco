"""
The world. Mix and match these: every experiment calls add_ground; add_oval_track
is optional. Add your own add_* functions here for new courses.
"""

import math
import mujoco
from . import config as c


def add_ground(spec):
    """Gravity, a light, and a flat ground plane."""
    spec.option.gravity = c.GRAVITY
    spec.option.timestep = c.TIMESTEP
    spec.worldbody.add_light(pos=[0, 0, 5], dir=[0, 0, -1])
    spec.worldbody.add_geom(name="ground", type=mujoco.mjtGeom.mjGEOM_PLANE,
                            size=[30, 30, 0.1], friction=c.GROUND_FRICTION)


def add_wall(spec, name, pos, size, yaw=0.0):
    """A static wall box. yaw in radians."""
    spec.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos,
                            size=size, euler=[0, 0, math.degrees(yaw)],
                            rgba=[0.85, 0.55, 0.2, 1])


def add_oval_track(spec):
    """
    An oval: two straights along x joined by two semicircular ends. The bottom
    straight runs through the origin, so a car at the origin starts on the track
    facing +x. Walls are built from short box segments.
    """
    s, r, lane = c.TRACK_STRAIGHT, c.TRACK_RADIUS, c.LANE_HALF
    t, h = 0.05, c.WALL_HEIGHT

    # straight walls (inner + outer, bottom + top)
    add_wall(spec, "wall_bot_out", [0, -lane, h / 2], [s, t, h / 2])
    add_wall(spec, "wall_bot_in",  [0,  lane, h / 2], [s, t, h / 2])
    add_wall(spec, "wall_top_out", [0, 2 * r + lane, h / 2], [s, t, h / 2])
    add_wall(spec, "wall_top_in",  [0, 2 * r - lane, h / 2], [s, t, h / 2])

    # curved walls at each end
    _add_arc(spec, "arc_r_out", s, r, r + lane, -math.pi / 2,  math.pi / 2, h, t)
    _add_arc(spec, "arc_r_in",  s, r, r - lane, -math.pi / 2,  math.pi / 2, h, t)
    _add_arc(spec, "arc_l_out", -s, r, r + lane, math.pi / 2,  3 * math.pi / 2, h, t)
    _add_arc(spec, "arc_l_in",  -s, r, r - lane, math.pi / 2,  3 * math.pi / 2, h, t)


def _add_arc(spec, name, cx, cy, radius, a0, a1, h, t, segments=14):
    """Approximate a wall arc with short tangential box segments."""
    step = (a1 - a0) / segments
    seg_len = radius * step
    for i in range(segments):
        ang = a0 + (i + 0.5) * step
        px, py = cx + radius * math.cos(ang), cy + radius * math.sin(ang)
        add_wall(spec, f"{name}_{i}", [px, py, h / 2],
                 [abs(seg_len) / 2 + t, t, h / 2], yaw=ang + math.pi / 2)
