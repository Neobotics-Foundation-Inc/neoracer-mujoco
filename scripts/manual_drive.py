"""
Drive neoracer.xml by hand, video-game style — a feel-test for the physics.

Usage:
    python3 scripts/manual_drive.py                       # drive the ramp course
    python3 scripts/manual_drive.py assets/neoracer.xml   # bare car on a plane
    python3 scripts/manual_drive.py --selftest            # check math, no window

NOTE: use plain python3, NOT mjpython. mjpython runs the script on a secondary
thread (it keeps the main thread for the passive viewer's Cocoa loop), but a glfw
window must be created on the main thread — mjpython will crash with
"NSWindow should only be instantiated on the main thread".

Unlike the passive viewer (press-only), this owns a glfw window and polls which
keys are *held* every frame, so it behaves like a game: hold to go, release to
coast down, release the wheel to re-center.

    Up / W   throttle (hold to build speed)     Left  / A   steer left
    Down / S reverse                            Right / D   steer right
    Space    hard stop (zero both)              Backspace   reset car (un-flip)
    Esc      quit

Actuator mapping (from assets/neoracer.xml):
  ctrl[0..3] = fl/fr/rl/rr wheel torque (N·m), positive = forward
  ctrl[4]    = steer_servo target angle (rad), positive = left, range ±0.4
"""

import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPTS_DIR.parent
sys.path.insert(0, str(_SCRIPTS_DIR))

# ── tuning knobs (feel is physical — turn these while driving) ─────────────────
# Rates are per second, so the feel is the same regardless of frame rate.
THROTTLE_ACCEL = 0.8     # N·m/s added while a throttle key is held
THROTTLE_COAST = 0.6     # N·m/s bled off when no throttle key is held
THROTTLE_MAX   = 0.35    # N·m cap per wheel (matches run.py FULL throttle)

STEER_RATE     = 1.5     # rad/s toward the limit while a steer key is held
STEER_RETURN   = 2.5     # rad/s back to center when no steer key is held
STEER_MAX      = 0.40    # rad — the model's Ackermann command limit

# ── chase camera (back-top, follows the car's heading) ─────────────────────────
CAM_DISTANCE   = 1.6     # m behind/above the car
CAM_ELEVATION  = -35.0   # deg looking down (more negative = more top-down)
CAM_AZIMUTH    = 0.0   # deg offset added to car yaw. 0 = directly behind
                         # (camera at -X, looking +X). 180 = in front; ±90 = side.


def _yaw_deg(quat) -> float:
    """Car yaw (heading) in degrees from a MuJoCo [w,x,y,z] quaternion."""
    import math
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    return math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _toward(value: float, rate_dt: float) -> float:
    """Move value toward zero by at most rate_dt, without overshooting."""
    if value > 0:
        return max(0.0, value - rate_dt)
    return min(0.0, value + rate_dt)


def update(throttle: float, steer: float, dt: float,
           fwd: bool, rev: bool, left: bool, right: bool) -> tuple[float, float]:
    """
    One frame of game-style control given which keys are currently held.
    Held throttle ramps up; released throttle coasts down. Steer returns to
    center when released. Returns clamped (throttle, steer).
    """
    if fwd:
        throttle += THROTTLE_ACCEL * dt
    elif rev:
        throttle -= THROTTLE_ACCEL * dt
    else:
        throttle = _toward(throttle, THROTTLE_COAST * dt)

    if left:
        steer += STEER_RATE * dt
    elif right:
        steer -= STEER_RATE * dt
    else:
        steer = _toward(steer, STEER_RETURN * dt)

    throttle = max(-THROTTLE_MAX, min(THROTTLE_MAX, throttle))
    steer    = max(-STEER_MAX,    min(STEER_MAX,    steer))
    return throttle, steer


def _selftest() -> None:
    """Smallest runnable check of the control math (no window needed)."""
    dt = 0.01
    # holding forward ramps up and clamps at the cap
    t = 0.0
    for _ in range(1000):
        t, _ = update(t, 0.0, dt, fwd=True, rev=False, left=False, right=False)
    assert t == THROTTLE_MAX, t
    # releasing coasts back to a full stop (not instant, not stuck)
    half = t
    t, _ = update(t, 0.0, dt, fwd=False, rev=False, left=False, right=False)
    assert 0 < t < half, t
    while t > 0:
        t, _ = update(t, 0.0, dt, fwd=False, rev=False, left=False, right=False)
    assert t == 0.0
    # steer holds toward the limit, then returns to exactly center
    s = 0.0
    for _ in range(1000):
        _, s = update(0.0, s, dt, fwd=False, rev=False, left=True, right=False)
    assert s == STEER_MAX, s
    while s > 0:
        _, s = update(0.0, s, dt, fwd=False, rev=False, left=False, right=False)
    assert s == 0.0
    print("selftest OK")


def main(xml: str | None = None) -> None:
    import glfw
    import mujoco
    import sensor_logger as sl

    xml = xml or str(_PROJECT_DIR / "assets" / "tracks" / "ramp_course.xml")
    model = mujoco.MjModel.from_xml_path(xml)
    data  = mujoco.MjData(model)
    car_id = model.body("car").id

    if not glfw.init():
        raise RuntimeError("glfw.init() failed")
    window = glfw.create_window(900, 600, "NeoRacer — manual drive", None, None)
    glfw.make_context_current(window)
    glfw.swap_interval(1)  # vsync

    cam = mujoco.MjvCamera()
    cam.type        = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = car_id
    cam.distance    = CAM_DISTANCE
    cam.elevation   = CAM_ELEVATION
    # azimuth is updated every frame from the car's yaw (see loop below)

    opt   = mujoco.MjvOption()
    scene = mujoco.MjvScene(model, maxgeom=10000)
    ctx   = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

    print(__doc__)

    def held(*keys):
        return any(glfw.get_key(window, k) == glfw.PRESS for k in keys)

    throttle, steer = 0.0, 0.0
    last = glfw.get_time()
    hud_accum = 0.0

    while not glfw.window_should_close(window):
        if held(glfw.KEY_ESCAPE):
            break
        if held(glfw.KEY_BACKSPACE):
            mujoco.mj_resetData(model, data)
            throttle, steer = 0.0, 0.0

        now = glfw.get_time()
        dt = now - last
        last = now

        if held(glfw.KEY_SPACE):
            throttle, steer = 0.0, 0.0
        else:
            throttle, steer = update(
                throttle, steer, dt,
                fwd=held(glfw.KEY_UP, glfw.KEY_W),
                rev=held(glfw.KEY_DOWN, glfw.KEY_S),
                left=held(glfw.KEY_LEFT, glfw.KEY_A),
                right=held(glfw.KEY_RIGHT, glfw.KEY_D),
            )

        data.ctrl[0:4] = throttle
        data.ctrl[4]   = steer

        # step physics to catch up to real (wall-clock) time
        target = data.time + dt
        while data.time < target:
            mujoco.mj_step(model, data)

        cam.azimuth = CAM_AZIMUTH + _yaw_deg(data.xquat[car_id])

        w, h = glfw.get_framebuffer_size(window)
        viewport = mujoco.MjrRect(0, 0, w, h)
        mujoco.mjv_updateScene(model, data, opt, None, cam,
                               mujoco.mjtCatBit.mjCAT_ALL, scene)
        mujoco.mjr_render(viewport, scene, ctx)
        glfw.swap_buffers(window)
        glfw.poll_events()

        hud_accum += dt
        if hud_accum >= 0.1:
            sl.print_sensors(sl.read(model, data))
            hud_accum = 0.0

    glfw.terminate()


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        args = [a for a in sys.argv[1:] if not a.startswith("-")]
        main(args[0] if args else None)
