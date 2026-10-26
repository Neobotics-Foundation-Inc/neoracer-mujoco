"""
What the envs need to know about a car model, read from the compiled model so
the same code drives either car:

  the GitHub car (upstream neoracer-mujoco, assets/neoracer_github.xml) --
      one DC motor "motor" on the drivetrain tendon, ctrl = the ESC's voltage
      (+-battery), and a DC-motor steering servo "steer_servo", ctrl = the
      target angle (rad); both simulated natively, so a policy's action maps
      straight onto them;
  the legacy car (this repo's assets/neoracer.xml) -- four torque motors
      fl/fr/rl/rr_motor and a position servo, driven through drive_model.py.

Also the wheel radius (for the speed estimate) and, for the 8 LiDAR beams,
how far each beam runs inside the car's own body before leaving it -- the
LiDAR sits at the chassis center on the legacy car but 4 cm behind it, and on
top, on the GitHub car.
"""

import os
from dataclasses import dataclass

import mujoco
import numpy as np

from .. import assets

# The GitHub car: upstream's assets/neoracer.xml, saved under this name next
# to the legacy car (whose file keeps its name, so old runs replay unchanged).
GITHUB_CAR = os.path.join(assets.ASSET_DIR, "neoracer_github.xml")
GITHUB_CAR_HELP = (
    f"the GitHub car file {GITHUB_CAR} is missing. In your neoracer-mujoco folder run:\n"
    "    git fetch origin && git show origin/main:assets/neoracer.xml > assets/neoracer_github.xml\n"
    "(or pass --car with its path, or --vehicle realistic / legacy for the older cars)"
)

LEGACY_MOTORS = ("fl_motor", "fr_motor", "rl_motor", "rr_motor")


def is_native_car(car_path) -> bool:
    """Whether the car file is the GitHub car (a DC motor named "motor")."""
    spec = mujoco.MjSpec.from_file(str(car_path))
    return spec.actuator("motor") is not None


def apply_car_options(model: mujoco.MjModel, car_path) -> None:
    """Give a composed model the GitHub car's own physics settings: its file
    sets the timestep (1/360 s: 12 steps per 30 Hz control step) and the
    implicitfast integrator, which composing onto a track would otherwise
    replace with the track's. The legacy car keeps the track's settings, as
    before, so old runs replay unchanged."""
    spec = mujoco.MjSpec.from_file(str(car_path))
    if spec.actuator("motor") is None:
        return
    model.opt.timestep = spec.option.timestep
    model.opt.integrator = spec.option.integrator
_BEAMS = np.deg2rad(np.arange(0, 360, 45))


@dataclass(frozen=True)
class CarInfo:
    native: bool  # True: the GitHub car (DC motor + DC servo)
    drive_ids: np.ndarray  # actuator ids: [motor] or the four legacy motors
    steer_id: int
    throttle_limit: float  # ctrl at full throttle: battery volts, or N.m per wheel
    steer_limit: float  # ctrl at full steering (rad)
    wheel_radius: float  # m
    beam_extent: np.ndarray  # (8,) m from the LiDAR to the body's edge, per beam


def car_info(model: mujoco.MjModel, prefix: str = "") -> CarInfo:
    names = [model.actuator(i).name for i in range(model.nu)]
    native = prefix + "motor" in names
    if native:
        drive_ids = np.array([model.actuator(prefix + "motor").id])
    else:
        drive_ids = np.array([model.actuator(prefix + n).id for n in LEGACY_MOTORS])
    steer_id = model.actuator(prefix + "steer_servo").id

    car = model.body(prefix + "car").id
    wheel_body = model.body(prefix + "fl_wheel_body").id
    wheel_geom = next(g for g in range(model.ngeom)
                      if model.geom_bodyid[g] == wheel_body and (model.geom_contype[g] or model.geom_conaffinity[g]))
    chassis = model.geom(prefix + "chassis").id  # both cars name their collision box "chassis"
    lidar = model.site(prefix + "lidar_000").id
    return CarInfo(
        native=native,
        drive_ids=drive_ids,
        steer_id=steer_id,
        throttle_limit=float(model.actuator_ctrlrange[drive_ids[0], 1]),
        steer_limit=float(model.actuator_ctrlrange[steer_id, 1]),
        wheel_radius=float(model.geom_size[wheel_geom, 0]),
        beam_extent=_beam_extent(model.site_pos[lidar][:2], model.geom_pos[chassis][:2], model.geom_size[chassis][:2]),
    )


def _beam_extent(origin: np.ndarray, center: np.ndarray, half: np.ndarray) -> np.ndarray:
    """For each beam from origin (inside the box), the distance to the edge of
    the axis-aligned box (center, half) -- in the car body's frame."""
    out = np.zeros(len(_BEAMS))
    for k, a in enumerate(_BEAMS):
        d = np.array([np.cos(a), np.sin(a)])
        ts = [(center[i] + np.sign(d[i]) * half[i] - origin[i]) / d[i] for i in range(2) if abs(d[i]) > 1e-9]
        out[k] = min(ts)
    return out
