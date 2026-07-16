"""Random-points procedural track generator (closed loop + pitlane).

A different approach from 01/02's random walk. Here the centerline is a *closed
loop*, built the way the racing-track-generator crowd does it:

    1. scatter random points
    2. cull the unnatural ones      -> convex hull (drops interior/reflex points)
    3. sort clockwise/anticlockwise -> the hull comes out already ordered (CCW)
    4. enforce sane spacing         -> drop corners that crowd each other
    5. connect through curves/straights -> closed Catmull-Rom spline
    6. find the straights, pick a start/finish (longer = likelier), draw a pit

Using the convex hull for step 2 is the lazy win of the whole file: a convex
loop physically *cannot* cross itself, so none of 01/02's segment-crossing
rejection sampling is needed here. The price is that every track is convex
(rounded n-gons, no chicanes).
# ponytail: convex-only. For chicanes, displace random hull-edge midpoints
# outward before smoothing and re-check self-intersection — much more code.

Run:  python3 03_random_points.py [out.xml] [--seed N]
      python3 03_random_points.py --test
"""

import math
import random
import sys
from collections import namedtuple

import cv2
import numpy as np

# Track shape knobs.
N_POINTS = 22          # random points scattered before hulling
POINT_SPREAD = 25.0    # m, half-size of the box the points land in
MIN_SPACING = 4.0      # m, drop hull corners closer than this to their neighbour
STRAIGHT_TOL = math.radians(4)   # per-vertex turn below this counts as "straight"
MIN_STRAIGHT_PTS = 4   # a straight must span at least this many spline points
SMOOTH_STEPS = 10      # Catmull-Rom samples per hull edge
PIT_OFFSET = 2.5       # m, how far the pit lane sits off its straight

# Corridor / wall knobs (same convention as 01_test.py / 02_connect_check.py).
HALF_WIDTH = 0.2       # m, corridor half-width (car track is 0.235 m)
WALL_H = 0.15          # box half-height
WALL_T = 0.02          # box half-thickness

# What generate_track() hands back: the closed centerline loop, the pit-lane
# polyline (open), and the start/finish point (midpoint of the chosen straight).
Track = namedtuple("Track", "loop pitlane start_finish")


# --- geometry helpers -------------------------------------------------------

def seg_normal(p, q):
    """Unit left normal of the segment p->q (rotate direction +90 degrees)."""
    dx, dy = q[0] - p[0], q[1] - p[1]
    length = math.hypot(dx, dy) or 1.0
    return (-dy / length, dx / length)


def segments_cross(a1, a2, b1, b2):
    """True if segment a1->a2 crosses b1->b2 (orientation test, from 01_test.py)."""
    def ccw(p, q, r):
        return (r[1] - p[1]) * (q[0] - p[0]) - (q[1] - p[1]) * (r[0] - p[0])
    return ((ccw(b1, b2, a1) > 0) != (ccw(b1, b2, a2) > 0)) and \
           ((ccw(a1, a2, b1) > 0) != (ccw(a1, a2, b2) > 0))


def loop_self_intersects(loop):
    """True if the closed polyline crosses itself. Used only as a demo() guard —
    convex hulls never trip it, but the smoother could in principle overshoot."""
    n = len(loop)
    for i in range(n):
        a1, a2 = loop[i], loop[(i + 1) % n]
        for j in range(i + 1, n):
            # skip the two segments that share a vertex with segment i
            if j == i or (j + 1) % n == i or (i + 1) % n == j:
                continue
            b1, b2 = loop[j], loop[(j + 1) % n]
            if segments_cross(a1, a2, b1, b2):
                return True
    return False


# --- step 1+2+3: random points -> convex hull (cull + order) ----------------

def random_points(rng, n=N_POINTS):
    """Scatter n random points in a POINT_SPREAD-sized box centred on origin."""
    return [(rng.uniform(-POINT_SPREAD, POINT_SPREAD),
             rng.uniform(-POINT_SPREAD, POINT_SPREAD)) for _ in range(n)]


def convex_hull(points):
    """cv2.convexHull. Returns the ordered hull vertices with no repeated closing
    point. This is steps 2 (cull the interior/reflex points) and 3 (they come out
    ordered) in one shot. Winding doesn't matter downstream — both walls are drawn
    and 'outward' is measured from the loop centre, not from the winding."""
    hull = cv2.convexHull(np.array(points, dtype=np.float32))  # shape (k, 1, 2)
    return [(float(x), float(y)) for x, y in hull[:, 0, :]]


def enforce_spacing(loop, min_dist=MIN_SPACING):
    """Step 4: walk the loop and drop any corner sitting too close to the one we
    kept last, so the smoother isn't fed a crowded, twitchy set of corners.
    Keeps at least 3 corners so there's still a loop."""
    kept = [loop[0]]
    for p in loop[1:]:
        if math.hypot(p[0] - kept[-1][0], p[1] - kept[-1][1]) >= min_dist:
            kept.append(p)
    # guard the seam: if the last kept corner hugs the first, drop it
    if len(kept) > 3 and math.hypot(kept[-1][0] - kept[0][0],
                                    kept[-1][1] - kept[0][1]) < min_dist:
        kept.pop()
    return kept


# --- step 5: connect the corners with a closed spline -----------------------

def _catmull(p0, p1, p2, p3, t):
    """Uniform Catmull-Rom point at parameter t in [0,1] on the p1->p2 span."""
    t2, t3 = t * t, t * t * t

    def axis(a, b, c, d):
        return 0.5 * (2 * b + (-a + c) * t +
                      (2 * a - 5 * b + 4 * c - d) * t2 +
                      (-a + 3 * b - 3 * c + d) * t3)
    return (axis(p0[0], p1[0], p2[0], p3[0]),
            axis(p0[1], p1[1], p2[1], p3[1]))


def smooth_loop(corners, steps=SMOOTH_STEPS):
    """Closed Catmull-Rom through the hull corners. Long hull edges stay nearly
    straight (that's where the straights come from); corners round off into
    curves. Returns the dense loop with no repeated closing point."""
    n = len(corners)
    out = []
    for i in range(n):
        p0, p1 = corners[(i - 1) % n], corners[i]
        p2, p3 = corners[(i + 1) % n], corners[(i + 2) % n]
        for s in range(steps):
            out.append(_catmull(p0, p1, p2, p3, s / steps))
    return out


# --- step 6: find straights, pick start/finish, draw the pit ----------------

def turn_at(loop, i):
    """Absolute heading change (rad) at vertex i of the closed loop."""
    n = len(loop)
    a, b, c = loop[(i - 1) % n], loop[i], loop[(i + 1) % n]
    h1 = math.atan2(b[1] - a[1], b[0] - a[0])
    h2 = math.atan2(c[1] - b[1], c[0] - b[0])
    d = (h2 - h1 + math.pi) % (2 * math.pi) - math.pi   # wrap to [-pi, pi]
    return abs(d)


def find_straights(loop, tol=STRAIGHT_TOL, min_pts=MIN_STRAIGHT_PTS):
    """Return the straight sections as lists of consecutive loop indices. A vertex
    is 'straight' if it barely turns; a straight is a maximal run of them. Handles
    the seam by merging a run that wraps past index 0."""
    n = len(loop)
    straight = [turn_at(loop, i) < tol for i in range(n)]

    runs, cur = [], []
    for i in range(n):
        if straight[i]:
            cur.append(i)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    # merge a straight that wraps across the seam (touches both index 0 and n-1)
    if len(runs) > 1 and runs[0][0] == 0 and runs[-1][-1] == n - 1:
        runs[0] = runs.pop() + runs[0]
    return [r for r in runs if len(r) >= min_pts]


def straight_length(loop, run):
    """Total centerline length of a straight run, for start/finish weighting."""
    total = 0.0
    for a, b in zip(run, run[1:]):
        total += math.hypot(loop[b][0] - loop[a][0], loop[b][1] - loop[a][1])
    return total


def pick_start_finish(loop, straights, rng):
    """Semi-randomly pick a straight for start/finish, longer straights weighted
    higher. Returns the chosen run (list of indices)."""
    weights = [straight_length(loop, r) for r in straights]
    return rng.choices(straights, weights=weights, k=1)[0]


def centroid(points):
    return (sum(p[0] for p in points) / len(points),
            sum(p[1] for p in points) / len(points))


def build_pitlane(loop, run, offset=PIT_OFFSET):
    """The pit: duplicate the straight's points, shove them ~90 degrees outward
    (away from the loop centre), then join the copy back to the track at both
    ends. Returns the pit-lane polyline (open):
        [straight_start, dup_0 ... dup_k, straight_end]."""
    mid = centroid(loop)
    straight = [loop[i] for i in run]
    # outward direction = away from the loop centre, taken at the straight's middle
    sx, sy = straight[len(straight) // 2]
    ox, oy = sx - mid[0], sy - mid[1]
    olen = math.hypot(ox, oy) or 1.0
    ox, oy = ox / olen, oy / olen
    dup = [(x + offset * ox, y + offset * oy) for x, y in straight]
    return [straight[0]] + dup + [straight[-1]]


def generate_track(rng):
    """Run the whole pipeline and return a Track. Retries the (rare) case where
    hulling + spacing leaves too few corners to smooth."""
    for _ in range(50):
        pts = random_points(rng)
        corners = enforce_spacing(convex_hull(pts))
        if len(corners) < 4:
            continue
        loop = smooth_loop(corners)
        straights = find_straights(loop)
        if not straights:
            continue
        run = pick_start_finish(loop, straights, rng)
        pitlane = build_pitlane(loop, run)
        start_finish = loop[run[len(run) // 2]]
        return Track(loop, pitlane, start_finish)
    raise RuntimeError("couldn't build a track; loosen MIN_SPACING / STRAIGHT_TOL")


# --- walls + MJCF -----------------------------------------------------------

def _boxes_from_vertices(verts, closed):
    """One connected wall box spanning each pair of consecutive offset vertices.
    Same idea as 02_connect_check.py's connect_walls: neighbouring boxes share a
    corner exactly, so there are no gaps on turns."""
    boxes = []
    pairs = zip(verts, verts[1:])
    if closed:
        pairs = zip(verts, verts[1:] + verts[:1])
    for (x1, y1), (x2, y2) in pairs:
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        boxes.append(((x1 + x2) / 2, (y1 + y2) / 2, length / 2, math.atan2(dy, dx)))
    return boxes


def offset_vertices(points, side, closed):
    """Push every vertex HALF_WIDTH to one side, averaging the two adjoining
    segment normals so the offset points stay shared (that's what seals the
    walls). closed=True wraps the ends around the loop."""
    n = len(points)
    out = []
    for i, p in enumerate(points):
        if closed:
            n1 = seg_normal(points[(i - 1) % n], points[i])
            n2 = seg_normal(points[i], points[(i + 1) % n])
        elif i == 0:
            n1 = n2 = seg_normal(points[0], points[1])
        elif i == n - 1:
            n1 = n2 = seg_normal(points[-2], points[-1])
        else:
            n1 = seg_normal(points[i - 1], points[i])
            n2 = seg_normal(points[i], points[i + 1])
        mx, my = n1[0] + n2[0], n1[1] + n2[1]
        mlen = math.hypot(mx, my) or 1.0
        out.append((p[0] + side * HALF_WIDTH * mx / mlen,
                    p[1] + side * HALF_WIDTH * my / mlen))
    return out


def wall_line(points, side, closed):
    """Sealed wall boxes down one side of a polyline (or loop)."""
    return _boxes_from_vertices(offset_vertices(points, side, closed), closed)


def outward_side(points):
    """+1 or -1 for wall_line so its wall lands on the far side from origin —
    used to wall the pit lane's outer edge only (its inner edge is the track's
    own outer wall)."""
    mid = centroid(points)
    nx, ny = seg_normal(points[0], points[-1])
    return 1 if (mid[0] * nx + mid[1] * ny) < 0 else -1


def _geom(name, cx, cy, half_len, yaw, rgba=None):
    colour = f' rgba="{rgba}"' if rgba else ' material="wall"'
    return (f'        <geom name="{name}" type="box"{colour}'
            f' condim="3" friction="1.0 0.1 0.1"'
            f' pos="{cx:.3f} {cy:.3f} {WALL_H:.3f}"'
            f' size="{half_len:.3f} {WALL_T} {WALL_H}"'
            f' euler="0 0 {yaw:.4f}"/>')


def to_xml(track):
    lines = ['<mujoco model="random_points_track">',
             '    <asset>',
             '        <material name="wall" rgba=".55 .57 .60 1"/>',
             '    </asset>',
             '    <worldbody>',
             '        <geom name="floor" type="plane" size="60 60 .01" pos="0 0 0"'
             ' rgba=".2 .3 .4 1" friction="1.5 0.1 0.1"/>']
    # both walls of the closed track loop
    for side, tag in ((+1, "left"), (-1, "right")):
        for i, (cx, cy, hl, yaw) in enumerate(wall_line(track.loop, side, closed=True)):
            lines.append(_geom(f"wall_{tag}_{i}", cx, cy, hl, yaw))
    # pit lane: outer wall only (inner side is the track's own outer wall)
    for i, (cx, cy, hl, yaw) in enumerate(
            wall_line(track.pitlane, outward_side(track.pitlane), closed=False)):
        lines.append(_geom(f"pit_{i}", cx, cy, hl, yaw))
    # start/finish marker: a flat green stripe on the ground at the chosen straight
    sfx, sfy = track.start_finish
    lines.append(f'        <geom name="start_finish" type="box" rgba=".2 .8 .3 1"'
                 f' pos="{sfx:.3f} {sfy:.3f} 0.005"'
                 f' size="{HALF_WIDTH:.3f} 0.05 0.005"/>')
    lines += ['    </worldbody>', '</mujoco>', '']
    return "\n".join(lines)


# --- minimap (top-down PNG, matches sim geometry) ---------------------------

MAP_SIZE = 640    # px, square map image
MAP_PAD = 30      # px border around the track


def _box_corners(cx, cy, half_len, yaw, half_t=WALL_T):
    """Four world-space corners of one wall box (same as test_drive.py)."""
    ax, ay = math.cos(yaw), math.sin(yaw)
    nx, ny = -math.sin(yaw), math.cos(yaw)
    return [(cx + ax * half_len + nx * half_t, cy + ay * half_len + ny * half_t),
            (cx + ax * half_len - nx * half_t, cy + ay * half_len - ny * half_t),
            (cx - ax * half_len - nx * half_t, cy - ay * half_len - ny * half_t),
            (cx - ax * half_len + nx * half_t, cy - ay * half_len + ny * half_t)]


def create_minimap(track, out_path):
    """Honest top-down map: each wall drawn as its real box, so it matches the
    sim exactly. Gray line = centerline loop, blue = left wall, orange = right
    wall, yellow = pit outer wall, green dot = start/finish."""
    import pygame

    # The walls are sealed box chains, but each box is ~0.04 m thick — sub-pixel
    # at map scale, so drawing them as filled boxes looks dotted. Draw the shared
    # offset-vertex chain as one connected polyline instead: honest about
    # connectivity (these are the exact box corners), legible about thickness.
    left = offset_vertices(track.loop, +1, closed=True)
    right = offset_vertices(track.loop, -1, closed=True)
    pit = offset_vertices(track.pitlane, outward_side(track.pitlane), closed=False)

    # world -> pixel: fit everything in the padded square
    allpts = left + right + pit + list(track.loop)
    min_x = min(p[0] for p in allpts)
    max_x = max(p[0] for p in allpts)
    min_y = min(p[1] for p in allpts)
    max_y = max(p[1] for p in allpts)
    span = max(max_x - min_x, max_y - min_y, 1e-6)
    scale = (MAP_SIZE - 2 * MAP_PAD) / span

    def to_px(pt):
        return (int(MAP_PAD + (pt[0] - min_x) * scale),
                int(MAP_SIZE - MAP_PAD - (pt[1] - min_y) * scale))  # flip Y

    if not pygame.get_init():
        pygame.init()
    surf = pygame.Surface((MAP_SIZE, MAP_SIZE))
    surf.fill((17, 17, 17))

    # centerline loop (closed): the true path the walls were offset from
    pygame.draw.lines(surf, (90, 90, 90), True, [to_px(p) for p in track.loop], 1)
    pygame.draw.lines(surf, (91, 155, 213), True, [to_px(p) for p in left], 2)
    pygame.draw.lines(surf, (237, 125, 49), True, [to_px(p) for p in right], 2)
    pygame.draw.lines(surf, (241, 196, 15), False, [to_px(p) for p in pit], 2)
    pygame.draw.circle(surf, (46, 204, 113), to_px(track.start_finish), 8)

    pygame.image.save(surf, str(out_path))
    print(f"wrote minimap {out_path}")


# --- self-check + CLI -------------------------------------------------------

def demo():
    # convex_hull of a filled square returns just the 4 corners (any winding)
    box = [(0, 0), (2, 0), (2, 2), (0, 2), (1, 1), (0.5, 1.5)]
    hull = convex_hull(box)
    assert set(hull) == {(0, 0), (2, 0), (2, 2), (0, 2)}, hull

    track = generate_track(random.Random(3))
    # the loop is closed, simple (convex hull => never crosses itself), non-trivial
    assert len(track.loop) >= 4
    assert not loop_self_intersects(track.loop), "smoothed loop crossed itself"
    # at least one straight was found on the loop
    assert find_straights(track.loop), "no straight detected"
    # the pit lane joins back to the track at both ends (step-6 requirement)
    straights = find_straights(track.loop)
    run = straights[0]
    pit = build_pitlane(track.loop, run)
    assert pit[0] == track.loop[run[0]] and pit[-1] == track.loop[run[-1]]
    # longer straights really are weighted higher: a 10x-longer straight wins ~always
    fake_loop = [(0, 0), (1, 0), (5, 0), (15, 0)]  # run A length 1, run B length 10
    picks = [pick_start_finish(fake_loop, [[0, 1], [2, 3]],
             random.Random(k)) for k in range(200)]
    assert picks.count([0, 1]) < 60, picks.count([0, 1])  # long straight dominates
    # xml is well-formed enough: floor + start/finish + some walls
    xml = to_xml(track)
    assert xml.count("<geom") >= 3
    assert 'name="start_finish"' in xml and 'name="pit_0"' in xml
    print("ok")


if __name__ == "__main__":
    if "--test" in sys.argv:
        demo()
        sys.exit()
    out = "track.xml"
    seed = None
    args = sys.argv[1:]
    if "--seed" in args:
        pos = args.index("--seed")
        seed = int(args[pos + 1])
        args = args[:pos] + args[pos + 2:]
    if args:
        out = args[0]

    track = generate_track(random.Random(seed))
    with open(out, "w") as f:
        f.write(to_xml(track))
    print(f"wrote {out}  (random-points loop, seed={seed}, "
          f"{len(find_straights(track.loop))} straights + pit)")
