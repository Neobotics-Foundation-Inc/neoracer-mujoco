"""
Detects when the car touches a wall, and resets the simulation, so that a
driving demo can recover without a manual reset key.

A wall is a geom of the body named "walls" (truth.WALLS_BODY). The track
XMLs in assets/tracks/ put their walls there; the floor and drivable ramps
are elsewhere. The car is the body tree of its root body ("car", the free
joint body in neoracer.xml), so the chassis and the wheels are both the car.
The truth sensor car.truth.wall_contact counts the contacts between the two.

    if car_touches_wall(data, scene.car):
        reset_car(scene.model, data)
"""

import mujoco

from .scene import Car


def car_touches_wall(data: mujoco.MjData, car: Car) -> bool:
    """True if a geom of the car touches a wall. Like every sensor, the
    count is up to date after mj_forward or mj_step (see truth.py)."""
    return bool(data.bind_scalar(car.truth.wall_contact).data[0] > 0)


def reset_car(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Reset data to the initial state of the model.

    mj_resetData sets qpos, qvel, ctrl and time to their initial values. It
    does not compute the body poses or the sensors, so they stay at zero
    until the next step. This function then calls mj_forward, so the poses
    and the sensors are correct immediately after the reset. A caller with
    its own state, for example a controller state, must reset that too.
    """
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
