"""Brute-force procedural track generator.

Random-walks a centerline of connected segments, brute-force rejects any path
that crosses itself (naive O(n^2) segment check + coarse grid spacing), then
drops thin wall geoms along both edges and writes an MJCF scene matching the
convention in assets/tracks/*.xml (floor + walls, car attaches at runtime).

Run:  python3 users/amoghmpanhale/01_test.py [out.xml] [--seed N]
"""

import math
import random
import sys

# Track shape knobs. ponytail: hardcoded, wrap in argparse when you actually
# want to sweep these instead of edit-and-rerun.
N_SEGMENTS = 40
SEG_LEN = 1.0          # m per segment
HALF_WIDTH = 0.6       # m, corridor half-width (car track is 0.235 m)
MAX_TURN = math.radians(35)   # max heading change per segment
WALL_H = 0.15          # box half-height
WALL_T = 0.02          # box half-thickness


def segments_cross(a1, a2, b1, b2):
    """True if line segment a1->a2 crosses line segment b1->b2.

    Each arg is an (x, y) point. Uses the standard orientation test: two
    segments cross iff each segment's endpoints straddle the other segment's
    line (one endpoint on each side).
    """
    # ccw returns >0 if p->q->r turns left, <0 if right, 0 if collinear.
    # It's the sign of the cross product of (q-p) and (r-p).
    def ccw(p, q, r):
        return (r[1] - p[1]) * (q[0] - p[0]) - (q[1] - p[1]) * (r[0] - p[0])
    # side of line b for each endpoint of a, and vice versa
    a1_side = ccw(b1, b2, a1)
    a2_side = ccw(b1, b2, a2)
    b1_side = ccw(a1, a2, b1)
    b2_side = ccw(a1, a2, b2)
    # cross only if a's endpoints straddle line b AND b's straddle line a
    a_straddles_b = (a1_side > 0) != (a2_side > 0)
    b_straddles_a = (b1_side > 0) != (b2_side > 0)
    return a_straddles_b and b_straddles_a


def random_centerline(rng):
    """Random-walk waypoints; brute-force retry the whole path until no
    two non-adjacent segments cross. Dumb but works for short tracks."""
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


def self_intersects(pts):
    for i in range(len(pts) - 1):
        for j in range(i + 2, len(pts) - 1):
            if i == 0 and j == len(pts) - 2:
                continue  # skip first/last, they never share a vertex anyway
            if segments_cross(pts[i], pts[i + 1], pts[j], pts[j + 1]):
                return True
    return False


def wall_boxes(pts, side):
    """Build one wall box per centerline segment, pushed out to one edge.

    side = +1 puts the wall on the left of the driving direction, -1 on the
    right. Returns a list of (center_x, center_y, half_length, yaw) tuples,
    one per segment, ready to drop into a <geom type="box">.
    """
    boxes = []
    for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
        # segment direction vector, from point 1 to point 2
        dir_x, dir_y = x2 - x1, y2 - y1
        length = math.hypot(dir_x, dir_y)
        # unit normal pointing left of travel (rotate direction +90 degrees)
        norm_x, norm_y = -dir_y / length, dir_x / length
        # push the wall HALF_WIDTH off the centerline, on the chosen side
        offset_x = side * HALF_WIDTH * norm_x
        offset_y = side * HALF_WIDTH * norm_y
        # box center = segment midpoint + sideways offset
        center_x = (x1 + x2) / 2 + offset_x
        center_y = (y1 + y2) / 2 + offset_y
        # rotate the box so its long axis lines up with the segment
        yaw = math.atan2(dir_y, dir_x)
        boxes.append((center_x, center_y, length / 2, yaw))
    return boxes


def to_xml(pts):
    lines = ['<mujoco model="procedural_track">',
             '    <asset>',
             '        <material name="wall" rgba=".55 .57 .60 1"/>',
             '    </asset>',
             '    <worldbody>',
             '        <geom name="floor" type="plane" size="60 60 .01" pos="0 0 0"'
             ' rgba=".2 .3 .4 1" friction="1.5 0.1 0.1"/>']
    # one row of walls per side of the track
    for side, tag in ((+1, "left"), (-1, "right")):
        for i, (center_x, center_y, half_len, yaw) in enumerate(wall_boxes(pts, side)):
            # box sits at z=WALL_H so it rests on the floor (pos is the center)
            lines.append(
                f'        <geom name="wall_{tag}_{i}" type="box" material="wall"'
                f' condim="3" friction="1.0 0.1 0.1"'
                f' pos="{center_x:.3f} {center_y:.3f} {WALL_H:.3f}"'
                f' size="{half_len:.3f} {WALL_T} {WALL_H}"'
                f' euler="0 0 {yaw:.4f}"/>')
    lines += ['    </worldbody>', '</mujoco>', '']
    return "\n".join(lines)


def demo():
    rng = random.Random(0)
    pts = random_centerline(rng)
    assert len(pts) == N_SEGMENTS + 1
    assert not self_intersects(pts), "generated path crosses itself"
    xml = to_xml(pts)
    assert xml.count("<geom") == 1 + 2 * N_SEGMENTS  # floor + both walls
    # crossing detector actually detects a crossing
    assert segments_cross((0, 0), (2, 2), (0, 2), (2, 0))
    assert not segments_cross((0, 0), (1, 0), (0, 1), (1, 1))
    print("ok")


if __name__ == "__main__":
    if "--test" in sys.argv:
        demo()
        sys.exit()
    # args: an optional output path, plus an optional "--seed N".
    # e.g.  01_test.py mytrack.xml --seed 3
    out = "track.xml"
    seed = None
    args = sys.argv[1:]
    if "--seed" in args:
        seed_pos = args.index("--seed")
        seed = int(args[seed_pos + 1])
        # drop "--seed" and its value so only the output path is left
        args = args[:seed_pos] + args[seed_pos + 2:]
    if args:
        out = args[0]

    pts = random_centerline(random.Random(seed))
    with open(out, "w") as f:
        f.write(to_xml(pts))
    print(f"wrote {out}  ({N_SEGMENTS} segments, seed={seed})")
