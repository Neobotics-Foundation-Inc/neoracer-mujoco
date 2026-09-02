"""
Vehicle kinematics verification (Stage: kinematic-trajectory revision).

Computes the theoretical minimum turning radius from assets/neoracer.xml's
actual wheelbase and steer_servo ctrlrange, then empirically measures it by
driving the (untouched) production car model at low constant torque and full
steering lock and least-squares fitting a circle to the resulting path.
This is geometry validation, not tuning -- neoracer.xml is not modified.

Usage:
    python3 examples/wall_follower_vs/verify_kinematics.py
"""

import math

import mujoco
import numpy as np

from neoracer_mujoco import sim as _sim

WHEELBASE_M = 0.2878  # see vehicle_space_controller.py's own derivation
STEER_ANGLE_MAX_RAD = 0.4


def main() -> None:
    r_theory = WHEELBASE_M / math.tan(STEER_ANGLE_MAX_RAD)
    print(
        f"L={WHEELBASE_M}m  delta_max={STEER_ANGLE_MAX_RAD}rad "
        f"({math.degrees(STEER_ANGLE_MAX_RAD):.3f}deg)  R_theory={r_theory:.4f}m"
    )

    model = mujoco.MjModel.from_xml_path("assets/neoracer.xml")
    data = _sim.settle(model)
    data.ctrl[0:4] = 0.03
    data.ctrl[4] = STEER_ANGLE_MAX_RAD  # full lock

    xs, ys = [], []
    for i in range(6000):  # 12s sim time
        mujoco.mj_step(model, data)
        if i % 20 == 0:
            p = data.body("car").xpos
            xs.append(float(p[0]))
            ys.append(float(p[1]))

    xs_arr, ys_arr = np.array(xs), np.array(ys)
    a = np.c_[2 * xs_arr, 2 * ys_arr, np.ones(len(xs_arr))]
    b = xs_arr**2 + ys_arr**2
    sol, _, _, _ = np.linalg.lstsq(a, b, rcond=None)
    cx, cy, c = sol
    r_empirical = math.sqrt(c + cx**2 + cy**2)
    print(
        f"R_empirical (circle fit) = {r_empirical:.4f}m over path from "
        f"({xs_arr[0]:.2f},{ys_arr[0]:.2f}) to ({xs_arr[-1]:.2f},{ys_arr[-1]:.2f}), "
        f"n={len(xs_arr)} pts"
    )
    print(f"ratio empirical/theory = {r_empirical / r_theory:.3f}")


if __name__ == "__main__":
    main()
