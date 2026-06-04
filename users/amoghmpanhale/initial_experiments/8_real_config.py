'''
8_real_config.py — a MuJoCo model of the REAL NeoRacer configuration.

Unlike experiments 1-7 (a 2 m placeholder box pushed via qvel), this models the three
defining features of the actual vehicle, as documented in
`full-vehicle/NEORACER_HARDWARE.md`:

  1. Real scale     — ~0.45 m long, derived from the CAD bounding box (mm -> m).
  2. AWD drivetrain — all 4 wheels are driven by motor actuators (the real car has
                      front + rear + center differentials, i.e. a single motor splits
                      torque to all four wheels; in sim we just torque all four).
  3. Front Ackermann steering — the front two wheels steer via servo-driven knuckles,
                      and the inner wheel turns MORE than the outer (true Ackermann
                      geometry), instead of both front wheels sharing one angle.

Run from the repo root:
    mjpython users/amoghmpanhale/initial_experiments/8_real_config.py
'''

import time
import math

import mujoco
import mujoco.viewer

from utils import DEFAULT_PHYSICS

# --- Real-scale geometry (meters), derived from the CAD bounding box ---------
# STL bbox is 0.268 m wide x 0.191 m tall x 0.446 m long. The chassis is a bit
# smaller than the bbox (which includes wheels/wings sticking out).
WHEELBASE = 0.30          # front-axle to rear-axle distance
TRACK = 0.22              # left wheel to right wheel distance
WHEEL_RADIUS = 0.05       # ~0.10 m diameter wheels
WHEEL_WIDTH = 0.05        # full width (cylinder half-length = WHEEL_WIDTH / 2)
RIDE_HEIGHT = 0.07        # car body center height above ground

BODY_HALF = [0.18, 0.08, 0.03]   # chassis box half-sizes -> 0.36 x 0.16 x 0.06 m
BODY_MASS = 2.0                  # kg (RC-car scale)
WHEEL_MASS = 0.15                # kg per wheel
KNUCKLE_MASS = 0.05              # kg per steering knuckle

STEER_LIMIT = 0.6         # max steer angle of a front wheel (rad, ~34 deg)

# Wheel mounting points relative to the car body origin.
# x = forward/back (+ = front), y = left/right (+ = left), z = down to the axle.
HALF_WB = WHEELBASE / 2
HALF_TRACK = TRACK / 2
WHEEL_Z = -(RIDE_HEIGHT - WHEEL_RADIUS)   # drop axles so wheels touch the ground

# --- Oval ("stadium") track geometry (meters) --------------------------------
# Two straights along x connected by two semicircular ends. The bottom straight's
# centerline runs through the origin, so the car spawns on the track facing +x.
CURVE_R = 3.0             # radius of the curved ends (lane centerline). Kept large so
                          # steering stays gentle — a tight curve makes the driven front
                          # wheels scrub at full lock and the little car understeers wide.
STRAIGHT = 2.5           # half-length of each straight (full straight = 5 m)
LANE_HALF = 0.8          # half lane width (walls sit this far off the centerline)
WALL_H = 0.10            # wall height
WALL_T = 0.02            # wall thickness (half-size)
LAP_LENGTH = 4 * STRAIGHT + 2 * math.pi * CURVE_R

# Ramp (a gentle hump across the far straight, made of two inclined boards).
RAMP_H = 0.05            # peak height
RAMP_RUN = 0.45          # horizontal length of each incline


def add_motor_actuator(spec, name, joint, gear):
    '''Torque actuator on a wheel's spin joint. ctrl value -> torque * gear.'''
    a = spec.add_actuator()
    a.name = name
    a.trntype = mujoco.mjtTrn.mjTRN_JOINT
    a.target = joint.name
    a.gear[0] = gear
    a.gainprm[0] = 1
    return a


def add_position_actuator(spec, name, joint, kp):
    '''PD-style position actuator for a steering knuckle. ctrl value -> target angle.'''
    a = spec.add_actuator()
    a.name = name
    a.trntype = mujoco.mjtTrn.mjTRN_JOINT
    a.target = joint.name
    a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
    a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
    a.gainprm[0] = kp
    a.biasprm[1] = -kp
    return a


def add_wheel(car, name, pos, steerable):
    '''
    Add a wheel to the car body. Returns (spin_joint, steer_joint).

    Non-steerable wheel: a body hung off the chassis with one hinge joint that lets
    it spin (axis = Y). steer_joint is None.

    Steerable wheel: an intermediate "knuckle" body with a vertical hinge (axis = Z)
    that the steering servo turns, and the spinning wheel body hangs off that. This
    is how the real car's knuckle + tie-rod assembly works.
    '''
    if steerable:
        knuckle = car.add_body(name=f"{name}_knuckle", pos=pos)
        steer_joint = knuckle.add_joint(
            name=f"{name}_steer_joint",
            type=mujoco.mjtJoint.mjJNT_HINGE,
            axis=[0, 0, 1],
            range=[-STEER_LIMIT, STEER_LIMIT],
        )
        knuckle.add_geom(
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.02],
            mass=KNUCKLE_MASS,
            rgba=[1, 0, 0, 1],
        )
        wheel = knuckle.add_body(name=f"{name}_wheel")
    else:
        steer_joint = None
        wheel = car.add_body(name=f"{name}_wheel", pos=pos)

    spin_joint = wheel.add_joint(
        name=f"{name}_spin_joint",
        type=mujoco.mjtJoint.mjJNT_HINGE,
        axis=[0, 1, 0],
        # armature = reflected rotor inertia. At this small (RC) scale a light wheel
        # driven by a motor accelerates so fast the integrator blows up; armature
        # stabilizes it. A touch of damping kills residual chatter.
        armature=0.005,
        damping=0.0005,
    )
    wheel.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[WHEEL_RADIUS, WHEEL_WIDTH / 2],
        euler=[90, 0, 0],
        mass=WHEEL_MASS,
        friction=DEFAULT_PHYSICS["wheel_friction"],
        # condim=3 (sliding friction only). condim=6 adds torsional+rolling friction,
        # which chatters on a small cylinder-on-plane contact and destabilizes the sim.
        condim=3,
        rgba=[0, 0, 1, 1] if steerable else [0.2, 0.2, 0.2, 1],
    )
    return spin_joint, steer_joint


def ackermann_angles(steer):
    '''
    True Ackermann geometry. Given a "virtual center" steer angle, return the
    (left, right) front-wheel angles so both wheels trace circles about a common
    center -> the inner wheel turns more than the outer.

    steer > 0 means turning left (center of turn is to the left).
    '''
    if abs(steer) < 1e-4:
        return 0.0, 0.0
    # Turn radius measured to the centerline at the rear axle.
    R = WHEELBASE / math.tan(steer)
    left = math.atan(WHEELBASE / (R - HALF_TRACK))
    right = math.atan(WHEELBASE / (R + HALF_TRACK))
    return left, right


def track_point(s):
    '''
    Point (x, y) on the oval centerline at arc-length s (meters), measured CCW from
    the middle of the bottom straight. Used by the lap controller to know where the
    track goes. The loop is: bottom straight (+x) -> right curve -> top straight
    (-x) -> left curve -> back.
    '''
    s %= LAP_LENGTH
    straight, curve = 2 * STRAIGHT, math.pi * CURVE_R
    if s < straight:                                  # bottom straight, y = 0
        return -STRAIGHT + s, 0.0
    s -= straight
    if s < curve:                                     # right curve, center (STRAIGHT, R)
        ang = -math.pi / 2 + s / CURVE_R
        return STRAIGHT + CURVE_R * math.cos(ang), CURVE_R + CURVE_R * math.sin(ang)
    s -= curve
    if s < straight:                                  # top straight, y = 2R
        return STRAIGHT - s, 2 * CURVE_R
    s -= straight
    ang = math.pi / 2 + s / CURVE_R                   # left curve, center (-STRAIGHT, R)
    return -STRAIGHT + CURVE_R * math.cos(ang), CURVE_R + CURVE_R * math.sin(ang)


def nearest_s(x, y, samples=240):
    '''Arc-length of the centerline point closest to (x, y) — cheap brute-force search.'''
    best_s, best_d = 0.0, float("inf")
    for i in range(samples):
        s = LAP_LENGTH * i / samples
        px, py = track_point(s)
        d = (px - x) ** 2 + (py - y) ** 2
        if d < best_d:
            best_d, best_s = d, s
    return best_s


def add_wall(spec, name, pos, size, angle=0.0):
    '''A static wall box (no body -> fixed in the world). angle is yaw in radians.'''
    spec.worldbody.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=pos,
        size=size,
        euler=[0, 0, math.degrees(angle)],
        rgba=[0.85, 0.55, 0.2, 1],
    )


def add_arc_wall(spec, name, cx, cy, radius, a0, a1, segments=14):
    '''Approximate a circular wall arc (center cx,cy) with short tangential box segments.'''
    step = (a1 - a0) / segments
    seg_len = radius * step                # chord ~ arc for small steps
    for i in range(segments):
        ang = a0 + (i + 0.5) * step
        px, py = cx + radius * math.cos(ang), cy + radius * math.sin(ang)
        add_wall(spec, f"{name}_{i}", [px, py, WALL_H / 2],
                 [abs(seg_len) / 2 + WALL_T, WALL_T, WALL_H / 2], angle=ang + math.pi / 2)


def add_ramp(spec):
    '''A gentle hump across the top straight: two thin inclined boards meeting at a peak.'''
    theta = math.atan2(RAMP_H, RAMP_RUN)              # incline angle
    L = math.hypot(RAMP_RUN, RAMP_H)                  # board length along the slope
    y = 2 * CURVE_R                                   # top-straight centerline
    for sign in (+1, -1):                             # +x board and -x board
        spec.worldbody.add_geom(
            name=f"ramp_{'p' if sign > 0 else 'm'}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[sign * RAMP_RUN / 2, y, RAMP_H / 2],
            size=[L / 2, LANE_HALF, 0.01],
            euler=[0, math.degrees(sign * theta), 0],
            rgba=[0.9, 0.3, 0.3, 1],
        )


def add_track(spec):
    '''Build the oval walls + the ramp around the car.'''
    # Straight walls: inner edge faces the oval interior, outer edge faces away.
    add_wall(spec, "wall_bot_out", [0, -LANE_HALF, WALL_H / 2], [STRAIGHT, WALL_T, WALL_H / 2])
    add_wall(spec, "wall_bot_in", [0,  LANE_HALF, WALL_H / 2], [STRAIGHT, WALL_T, WALL_H / 2])
    add_wall(spec, "wall_top_out", [0, 2 * CURVE_R + LANE_HALF, WALL_H / 2], [STRAIGHT, WALL_T, WALL_H / 2])
    add_wall(spec, "wall_top_in", [0, 2 * CURVE_R - LANE_HALF, WALL_H / 2], [STRAIGHT, WALL_T, WALL_H / 2])
    # Curve walls (outer + inner arcs at each end).
    add_arc_wall(spec, "arc_r_out",  STRAIGHT, CURVE_R, CURVE_R + LANE_HALF, -math.pi / 2,  math.pi / 2)
    add_arc_wall(spec, "arc_r_in",   STRAIGHT, CURVE_R, CURVE_R - LANE_HALF, -math.pi / 2,  math.pi / 2)
    add_arc_wall(spec, "arc_l_out", -STRAIGHT, CURVE_R, CURVE_R + LANE_HALF,  math.pi / 2,  3 * math.pi / 2)
    add_arc_wall(spec, "arc_l_in",  -STRAIGHT, CURVE_R, CURVE_R - LANE_HALF,  math.pi / 2,  3 * math.pi / 2)
    add_ramp(spec)


def build_spec():
    spec = mujoco.MjSpec()
    spec.option.timestep = 0.005
    spec.option.gravity = DEFAULT_PHYSICS["gravity"]

    spec.worldbody.add_light(pos=[0, 0, 5], dir=[0, 0, -1])
    spec.worldbody.add_geom(
        name="ground",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[30, 30, 0.1],
        friction=DEFAULT_PHYSICS["ground_friction"],
    )

    car = spec.worldbody.add_body(name="car", pos=[0, 0, RIDE_HEIGHT])
    car.add_freejoint()
    car.add_geom(
        name="car_body",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=BODY_HALF,
        mass=BODY_MASS,
        rgba=[0.7, 0.7, 0.7, 1],
    )

    # Four corners. Front wheels (+x) steer; all four are driven (AWD).
    fl_spin, fl_steer = add_wheel(car, "fl", [HALF_WB,  HALF_TRACK, WHEEL_Z], steerable=True)
    fr_spin, fr_steer = add_wheel(car, "fr", [HALF_WB, -HALF_TRACK, WHEEL_Z], steerable=True)
    rl_spin, _ = add_wheel(car, "rl", [-HALF_WB,  HALF_TRACK, WHEEL_Z], steerable=False)
    rr_spin, _ = add_wheel(car, "rr", [-HALF_WB, -HALF_TRACK, WHEEL_Z], steerable=False)

    # AWD: a drive motor on every wheel. ctrl[0..3] = FL, FR, RL, RR torque.
    # gear=1 gives brisk-but-stable acceleration at this scale; gear>=2 wheelies/flips.
    add_motor_actuator(spec, "fl_motor", fl_spin, gear=1)
    add_motor_actuator(spec, "fr_motor", fr_spin, gear=1)
    add_motor_actuator(spec, "rl_motor", rl_spin, gear=1)
    add_motor_actuator(spec, "rr_motor", rr_spin, gear=1)

    # Front steering servos. ctrl[4..5] = FL, FR target steer angle.
    add_position_actuator(spec, "fl_steer", fl_steer, kp=20)
    add_position_actuator(spec, "fr_steer", fr_steer, kp=20)

    add_track(spec)

    return spec


def car_yaw(data):
    '''Heading of the car in the ground plane (radians), from the freejoint quaternion.'''
    w, x, y, z = data.qpos[3:7]
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


# --- Lap controller tuning ---------------------------------------------------
TARGET_SPEED = 1.2    # m/s the car tries to hold
LOOKAHEAD = 1.0       # pure-pursuit lookahead distance along the centerline (m)
STEER_GAIN = 0.9      # heading-error -> steering gain
SPEED_GAIN = 0.5      # speed-error -> throttle gain
MAX_THROTTLE = 0.4    # torque clamp (higher wheelies the car)


def control(data):
    '''
    Drive the car around the oval (pure pursuit) at a roughly constant speed.
    Returns (throttle, left_steer, right_steer) for the actuators.
    '''
    x, y = data.qpos[0], data.qpos[1]
    yaw = car_yaw(data)

    # Pure pursuit: aim at a point LOOKAHEAD metres ahead on the centerline.
    s = nearest_s(x, y)
    tx, ty = track_point(s + LOOKAHEAD)
    heading_err = math.atan2(ty - y, tx - x) - yaw
    heading_err = (heading_err + math.pi) % (2 * math.pi) - math.pi   # wrap to [-pi, pi]
    steer = max(-STEER_LIMIT, min(STEER_LIMIT, STEER_GAIN * heading_err))
    left, right = ackermann_angles(steer)

    # Hold target speed using forward velocity (project world velocity onto heading).
    fwd_speed = data.qvel[0] * math.cos(yaw) + data.qvel[1] * math.sin(yaw)
    throttle = max(-MAX_THROTTLE, min(MAX_THROTTLE, SPEED_GAIN * (TARGET_SPEED - fwd_speed)))
    return throttle, left, right


if __name__ == "__main__":
    spec = build_spec()
    model = spec.compile()
    data = mujoco.MjData(model)

    car_id = model.body("car").id

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Make the camera follow the car instead of staying fixed in the world.
        # TRACKING mode keeps the car centered while you orbit/zoom freely.
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = car_id
        viewer.cam.distance = 4.0       # how far back the camera sits (meters)
        viewer.cam.azimuth = 90         # viewing angle around the car (degrees)
        viewer.cam.elevation = -35      # look down at the car (degrees)

        while viewer.is_running():
            throttle, left, right = control(data)
            data.ctrl[0:4] = throttle   # AWD: all four wheels
            data.ctrl[4] = left         # FL knuckle
            data.ctrl[5] = right        # FR knuckle

            mujoco.mj_step(model, data)
            viewer.sync()

            speed = math.hypot(data.qvel[0], data.qvel[1])
            print(f"speed: {speed:5.2f} m/s   throttle: {throttle:+.2f}")

            time.sleep(0.005)
