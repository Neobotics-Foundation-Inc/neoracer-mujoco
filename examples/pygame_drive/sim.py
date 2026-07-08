"""
MuJoCo driving module for the pygame demo — import DriveSim, feed it
normalized commands, blit what render() returns.

    sim = DriveSim()                  # ramp course + car (or DriveSim("path/to/car.xml"))
    sim.step(throttle, steer, dt)     # both in [-1, 1], dt in wall-clock seconds
    frame = sim.render()              # RGB uint8 array (height, width, 3)

Actuator mapping (from assets/neoracer.xml):
  ctrl[0..3] = fl/fr/rl/rr wheel torque (N·m), positive = forward
  ctrl[4]    = steer_servo target angle (rad), positive = left, range ±0.4

Run `python3 sim.py` for a windowless self-check.
"""

import math
from pathlib import Path

import mujoco
import numpy as np

_PROJECT_DIR = Path(__file__).resolve().parents[2]

# ── tuning knobs ────────────────────────────────────────────────────────────────
MAX_TORQUE = 0.17      # N·m per wheel at full throttle (0.35 launches like a rocket;
                        # below ~0.15 the car can't carry enough speed to clear the
                        # ramp jump even with a run-up)
MOTOR_DAMPING = 0.0025  # N·m per rad/s of wheel spin — back-EMF of a DC motor.
                        # Sets the top speed (~MAX_TORQUE/MOTOR_DAMPING in rad/s,
                        # ≈1.9 m/s here after rolling friction) AND brakes the car
                        # when throttle is released, so letting go slows it down.
                        # Below ~1.5 m/s the car can't clear the ramp jump.
ROLL_FRICTION = 0.02    # N·m of rolling resistance per wheel — back-EMF alone
                        # fades near zero speed and leaves the car creeping.
MAX_STEER  = 0.40       # rad — the model's Ackermann command limit
THROTTLE_SLEW = 2.5     # /s — how fast the applied command chases the input.
STEER_SLEW    = 4.0     # /s   Smooths digital WASD input; analog sticks barely notice.

CAM_DISTANCE  = 1.6   # m behind/above the car
CAM_ELEVATION = -35.0 # deg looking down


def _yaw_deg(quat) -> float:
    """Car yaw (heading) in degrees from a MuJoCo [w,x,y,z] quaternion."""
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    return math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _slew(current: float, target: float, rate_dt: float) -> float:
    """Move current toward target by at most rate_dt, without overshooting."""
    if target > current:
        return min(target, current + rate_dt)
    return max(target, current - rate_dt)


def _load_model(xml: str | None):
    """
    No arg -> compose the ramp course + car at runtime (same trick as
    examples/manual_drive.py). A path -> load it as-is.
    """
    if xml:
        return mujoco.MjModel.from_xml_path(xml)
    scene = mujoco.MjSpec.from_file(str(_PROJECT_DIR / "assets" / "tracks" / "f1_track.xml"))
    car   = mujoco.MjSpec.from_file(str(_PROJECT_DIR / "assets" / "neoracer.xml"))
    # attach the whole car spec (not just the car body): the ground plane lives
    # in neoracer.xml's worldbody and must come along, or the scene has no floor
    scene.attach(car, prefix="", frame=scene.worldbody.add_frame())
    return scene.compile()


class DriveSim:
    """Owns the model, physics stepping, and the chase-camera renderer."""

    def __init__(self, xml: str | None = None, width: int = 900, height: int = 600):
        self.model = _load_model(xml)
        self.data = mujoco.MjData(self.model)
        self._car_id = self.model.body("car").id
        # qvel address of each drive wheel's spin, via actuators 0..3 (the
        # repo's ctrl contract), used for engine braking
        self._wheel_dofs = [self.model.jnt_dofadr[self.model.actuator_trnid[i, 0]]
                            for i in range(4)]

        # commands actually applied, chasing the caller's inputs (see step)
        self._throttle = 0.0
        self._steer = 0.0

        # grow the offscreen framebuffer to the requested size before the
        # renderer snapshots it (the MJCF default is only 640x480)
        self.model.vis.global_.offwidth = max(width, self.model.vis.global_.offwidth)
        self.model.vis.global_.offheight = max(height, self.model.vis.global_.offheight)
        self._renderer = mujoco.Renderer(self.model, height=height, width=width)

        self._cam = mujoco.MjvCamera()
        self._cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self._cam.trackbodyid = self._car_id
        self._cam.distance = CAM_DISTANCE
        self._cam.elevation = CAM_ELEVATION

    def step(self, throttle: float, steer: float, dt: float) -> None:
        """Chase the normalized [-1, 1] inputs, then step physics dt seconds."""
        throttle = max(-1.0, min(1.0, throttle))
        steer = max(-1.0, min(1.0, steer))
        dt = min(dt, 0.1)  # a hitched frame shouldn't trigger a physics catch-up spiral

        self._steer = _slew(self._steer, steer, STEER_SLEW * dt)
        self.data.ctrl[4] = self._steer * MAX_STEER

        # DC-motor model: commanded torque minus back-EMF drag. At full throttle
        # the two balance at the top speed; at zero throttle the drag term is
        # engine braking, so the car slows to a stop instead of coasting forever.
        self._throttle = _slew(self._throttle, throttle, THROTTLE_SLEW * dt)
        spin = self.data.qvel[self._wheel_dofs]
        torque = self._throttle * MAX_TORQUE - MOTOR_DAMPING * spin
        torque -= np.clip(spin * 0.05, -ROLL_FRICTION, ROLL_FRICTION)  # smoothed Coulomb
        self.data.ctrl[0:4] = np.clip(torque, -MAX_TORQUE, MAX_TORQUE)

        target = self.data.time + dt
        while self.data.time < target:
            mujoco.mj_step(self.model, self.data)

    def stop(self) -> None:
        """Hard stop: zero both commands immediately (no slew-down)."""
        self._throttle = 0.0
        self._steer = 0.0
        self.data.ctrl[:] = 0.0

    def reset(self) -> None:
        """Put the car back at the start (un-flip), commands zeroed."""
        mujoco.mj_resetData(self.model, self.data)
        self.stop()

    def render(self):
        """RGB frame from a chase camera locked behind the car."""
        self._cam.azimuth = _yaw_deg(self.data.xquat[self._car_id])
        self._renderer.update_scene(self.data, camera=self._cam)
        return self._renderer.render()


def _speed(sim: "DriveSim") -> float:
    return float(np.linalg.norm(sim.data.qvel[0:2]))  # freejoint vx, vy


def _selftest() -> None:
    """Smallest runnable check: drives, tops out, brakes on release, renders."""
    sim = DriveSim(str(_PROJECT_DIR / "assets" / "neoracer.xml"))  # flat plane
    x0 = sim.data.xpos[sim._car_id][0].copy()
    for _ in range(150):  # ~3 s of full throttle
        sim.step(1.0, 0.0, 0.02)
    assert sim._throttle == 1.0, sim._throttle
    moved = sim.data.xpos[sim._car_id][0] - x0
    assert moved > 1.0, f"car only moved {moved:.3f} m under full throttle"
    # top speed must stay near the back-EMF balance point, whatever the knobs say
    top_spin = MAX_TORQUE / MOTOR_DAMPING  # rad/s where drive torque = drag
    spin = float(np.mean(sim.data.qvel[sim._wheel_dofs]))
    assert spin < 1.1 * top_spin, f"wheels spin {spin:.0f} rad/s, model caps at {top_spin:.0f}"
    v_release = _speed(sim)  # release -> drag must brake the car, not coast forever
    for _ in range(200):  # 4 s hands-off
        sim.step(0.0, 0.0, 0.02)
    assert _speed(sim) < 0.1 * v_release, \
        f"only braked {v_release:.2f} -> {_speed(sim):.2f} m/s in 4 s"
    for _ in range(150):
        sim.step(0.0, 1.0, 0.02)
    assert sim._steer == 1.0 and sim.data.ctrl[4] == MAX_STEER
    sim.stop()
    assert sim._throttle == 0.0 and (sim.data.ctrl == 0).all()
    course = DriveSim()  # ramp-course composition compiles and renders
    course.step(1.0, 0.0, 0.02)
    h, w, c = course.render().shape
    assert (h, w, c) == (600, 900, 3), (h, w, c)
    print("selftest OK")


if __name__ == "__main__":
    _selftest()
