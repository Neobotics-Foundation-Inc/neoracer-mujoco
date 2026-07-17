"""Brute-force track generator + wall-connectivity check/fix.

Same generator as 01_test.py, but the naive wall_boxes() there leaves gaps on
the outside of every turn (straight boxes offset from the centerline don't tile).
This file adds:

    walls_connected(boxes)   -> is each wall box's end touching the next one's start?
    connect_walls(pts, side) -> the FIX: rebuild the walls so they do touch.

The fix works by offsetting the centerline *vertices* (not segment midpoints) and
spanning one box between consecutive offset vertices, so neighbouring boxes share
a corner exactly. In practice that rotates and resizes each box to meet its
neighbours — which is what closes the gaps.

Run:  python3 02_connect_check.py [out.xml] [--seed N]
      python3 02_connect_check.py --test
"""

import math
import random
import sys

# Track shape knobs (same as 01_test.py).
N_SEGMENTS = 40
SEG_LEN = 1.0          # m per segment
HALF_WIDTH = 0.6       # m, corridor half-width (car track is 0.235 m)
MAX_TURN = math.radians(35)   # max heading change per segment
WALL_H = 0.15          # box half-height
WALL_T = 0.02          # box half-thickness
CONNECT_TOL = 1e-6     # m, how close two box ends must be to count as touching


def segments_cross(a1, a2, b1, b2):
    """True if line segment a1->a2 crosses line segment b1->b2 (orientation test)."""
    def ccw(p, q, r):
        return (r[1] - p[1]) * (q[0] - p[0]) - (q[1] - p[1]) * (r[0] - p[0])
    a1_side = ccw(b1, b2, a1)
    a2_side = ccw(b1, b2, a2)
    b1_side = ccw(a1, a2, b1)
    b2_side = ccw(a1, a2, b2)
    return ((a1_side > 0) != (a2_side > 0)) and ((b1_side > 0) != (b2_side > 0))


def self_intersects(pts):
    for i in range(len(pts) - 1):
        for j in range(i + 2, len(pts) - 1):
            if i == 0 and j == len(pts) - 2:
                continue
            if segments_cross(pts[i], pts[i + 1], pts[j], pts[j + 1]):
                return True
    return False


def random_centerline(rng):
    """Random-walk waypoints, brute-force retry until the path doesn't cross itself."""
    for _ in range(2000):
        pts = [(0.0, 0.0)]
        heading = 0.0
        for _ in range(N_SEGMENTS):
            heading += rng.uniform(-MAX_TURN, MAX_TURN)
            x, y = pts[-1]
            pts.append((x + SEG_LEN * math.cos(heading),
                        y + SEG_LEN * math.sin(heading)))
        if not self_intersects(pts):
            return pts
    raise RuntimeError("couldn't find a non-crossing path; loosen MAX_TURN")


def wall_boxes(pts, side):
    """Naive walls (same as 01_test.py): one box per segment, offset from the
    segment midpoint. Leaves gaps on turns — kept here so the check has something
    broken to catch."""
    boxes = []
    for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
        dir_x, dir_y = x2 - x1, y2 - y1
        length = math.hypot(dir_x, dir_y)
        norm_x, norm_y = -dir_y / length, dir_x / length
        center_x = (x1 + x2) / 2 + side * HALF_WIDTH * norm_x
        center_y = (y1 + y2) / 2 + side * HALF_WIDTH * norm_y
        yaw = math.atan2(dir_y, dir_x)
        boxes.append((center_x, center_y, length / 2, yaw))
    return boxes


def box_ends(box):
    """The two end points of a wall box's centerline (ignoring thickness).
    box = (center_x, center_y, half_len, yaw) -> (start_point, end_point)."""
    cx, cy, half_len, yaw = box
    ax, ay = math.cos(yaw), math.sin(yaw)   # unit vector along the box
    start = (cx - ax * half_len, cy - ay * half_len)
    end = (cx + ax * half_len, cy + ay * half_len)
    return start, end


def walls_connected(boxes, tol=CONNECT_TOL):
    """True if every wall box's end meets the next box's start (within tol).
    A row of walls with a gap on any turn returns False."""
    for a, b in zip(boxes, boxes[1:]):
        _, a_end = box_ends(a)
        b_start, _ = box_ends(b)
        if math.hypot(a_end[0] - b_start[0], a_end[1] - b_start[1]) > tol:
            return False
    return True


def offset_vertices(pts, side):
    """Push every centerline vertex HALF_WIDTH to one side. Interior vertices use
    the average of their two adjoining segment normals so a single offset point is
    shared by the walls on both sides of it — that shared point is what makes the
    boxes connect."""
    def seg_normal(p, q):
        dx, dy = q[0] - p[0], q[1] - p[1]
        length = math.hypot(dx, dy)
        return (-dy / length, dx / length)   # left normal, unit

    out = []
    for i, p in enumerate(pts):
        if i == 0:
            nx, ny = seg_normal(pts[0], pts[1])            # first: only one segment
        elif i == len(pts) - 1:
            nx, ny = seg_normal(pts[-2], pts[-1])          # last: only one segment
        else:
            n1 = seg_normal(pts[i - 1], pts[i])
            n2 = seg_normal(pts[i], pts[i + 1])
            mx, my = n1[0] + n2[0], n1[1] + n2[1]          # average the two normals
            mlen = math.hypot(mx, my) or 1.0
            nx, ny = mx / mlen, my / mlen
        out.append((p[0] + side * HALF_WIDTH * nx, p[1] + side * HALF_WIDTH * ny))
    return out


def connect_walls(pts, side):
    """The FIX: one box spanning each pair of consecutive offset vertices, so a
    box's end and the next box's start are the same point. Returns the same
    (center_x, center_y, half_len, yaw) tuples as wall_boxes(), but connected."""
    verts = offset_vertices(pts, side)
    boxes = []
    for (x1, y1), (x2, y2) in zip(verts, verts[1:]):
        dir_x, dir_y = x2 - x1, y2 - y1
        length = math.hypot(dir_x, dir_y)
        boxes.append(((x1 + x2) / 2, (y1 + y2) / 2, length / 2,
                      math.atan2(dir_y, dir_x)))
    return boxes


def to_xml(pts):
    lines = ['<mujoco model="procedural_track">',
             '    <asset>',
             '        <material name="wall" rgba=".55 .57 .60 1"/>',
             '    </asset>',
             '    <worldbody>',
             '        <geom name="floor" type="plane" size="60 60 .01" pos="0 0 0"'
             ' rgba=".2 .3 .4 1" friction="1.5 0.1 0.1"/>']
    for side, tag in ((+1, "left"), (-1, "right")):
        boxes = connect_walls(pts, side)
        assert walls_connected(boxes), f"{tag} wall still has a gap after fixing"
        for i, (center_x, center_y, half_len, yaw) in enumerate(boxes):
            lines.append(
                f'        <geom name="wall_{tag}_{i}" type="box" material="wall"'
                f' condim="3" friction="1.0 0.1 0.1"'
                f' pos="{center_x:.3f} {center_y:.3f} {WALL_H:.3f}"'
                f' size="{half_len:.3f} {WALL_T} {WALL_H}"'
                f' euler="0 0 {yaw:.4f}"/>')
    lines += ['    </worldbody>', '</mujoco>', '']
    return "\n".join(lines)


def demo():
    pts = random_centerline(random.Random(7))
    # box_ends is correct: a unit box along +x from origin runs (-.5,0)->(.5,0)
    s, e = box_ends((0, 0, 0.5, 0.0))
    assert s == (-0.5, 0.0) and e == (0.5, 0.0)
    # the naive walls are NOT connected on a curvy path...
    assert not walls_connected(wall_boxes(pts, +1)), "expected naive walls to gap"
    # ...and the fix connects them, on both sides
    assert walls_connected(connect_walls(pts, +1))
    assert walls_connected(connect_walls(pts, -1))
    # same box count as before, so nothing was dropped
    assert len(connect_walls(pts, +1)) == len(wall_boxes(pts, +1))
    xml = to_xml(pts)
    assert xml.count("<geom") == 1 + 2 * N_SEGMENTS
    print("ok")


if __name__ == "__main__":
    if "--test" in sys.argv:
        demo()
        sys.exit()
    out = "track.xml"
    seed = None
    args = sys.argv[1:]
    if "--seed" in args:
        seed_pos = args.index("--seed")
        seed = int(args[seed_pos + 1])
        args = args[:seed_pos] + args[seed_pos + 2:]
    if args:
        out = args[0]

    pts = random_centerline(random.Random(seed))
    with open(out, "w") as f:
        f.write(to_xml(pts))
    print(f"wrote {out}  ({N_SEGMENTS} segments, seed={seed}, walls connected)")
