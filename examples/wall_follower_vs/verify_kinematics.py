"""
Compare the minimum turn radius of the bicycle model with the turn radius of
the simulated car. The script reads the wheelbase and the steering limit from
assets/neoracer.xml. It then asks the car for a low constant speed, which
its speed loop holds, at full steering lock, and fits a circle to the path
of the car. Run it from the repository root:

    python -m examples.wall_follower_vs.verify_kinematics

Frame: +X forward, +Y left, +Z up.
"""

import math
from dataclasses import dataclass

import numpy as np

from examples.wall_follower_vs.vehicle_space_controller import car_geometry
from neoracer_mujoco import assets, racecar, sim
from neoracer_mujoco.control import SpeedAngle


@dataclass(frozen=True)
class DriveParams:
    speed: float = 0.56
    """The constant speed (m/s)."""
    duration_s: float = 12.0
    """The sim time of the drive."""
    sample_period_s: float = 0.04
    """The time between two path samples."""


def main() -> None:
    params = DriveParams()
    scene = assets.load()
    model = scene.model
    car = car_geometry(scene)
    r_theory = car.wheelbase_m / math.tan(car.steer_max_rad)
    print(
        f"L={car.wheelbase_m}m  delta_max={car.steer_max_rad}rad "
        f"({math.degrees(car.steer_max_rad):.3f}deg)  R_theory={r_theory:.4f}m"
    )

    data = sim.settle(model)
    racer = racecar.start(racecar.neoracer_params(), scene, data)
    racer = racecar.ask(racer, SpeedAngle(params.speed, car.steer_max_rad))
    timestep = float(model.opt.timestep)
    n_steps = round(params.duration_s / timestep)
    sample_every = round(params.sample_period_s / timestep)
    xy = []
    for i in range(n_steps):
        racer = racecar.step(scene, data, racer)
        if i % sample_every == 0:
            xy.append(data.bind_scalar(scene.car.body).xpos[:2].copy())

    path = np.array(xy)
    xs, ys = path[:, 0], path[:, 1]
    # Least-squares circle fit: x² + y² = 2·cx·x + 2·cy·y + c.
    a = np.c_[2 * xs, 2 * ys, np.ones(len(xs))]
    (cx, cy, c), *_ = np.linalg.lstsq(a, xs**2 + ys**2, rcond=None)
    r_measured = math.sqrt(c + cx**2 + cy**2)
    print(
        f"R_measured (circle fit) = {r_measured:.4f}m over the path from "
        f"({xs[0]:.2f},{ys[0]:.2f}) to ({xs[-1]:.2f},{ys[-1]:.2f}), "
        f"n={len(xs)} points"
    )
    print(f"ratio measured/theory = {r_measured / r_theory:.3f}")


if __name__ == "__main__":
    main()
