"""
Detects when the car touches a wall and resets the simulation, so a driving
demo can recover automatically instead of requiring a manual reset key.

A geom counts as a wall if its MuJoCo material is named "wall" — the
convention assets/tracks/*.xml already use to mark real walls (e.g.
straight_corridor.xml's corridor_wall_left/right, ramp_course.xml's
turn1/turn2), as opposed to the floor plane or ramp_course.xml's drivable
ramp, which use different materials. Tracks with no wall-material geoms
(e.g. the bare car on assets/neoracer.xml) simply never trigger a reset.

The car is identified by its kinematic root body (default name "car", the
freejoint body in neoracer.xml): every geom whose body's root is that body
-- chassis, wheels, everything rigidly/jointly attached to it -- counts as
part of the car for collision purposes.
"""

import mujoco

WALL_MATERIAL = "wall"


def wall_geom_ids(model: mujoco.MjModel) -> set[int]:
    """Geom ids whose material is named WALL_MATERIAL (see module docstring)."""
    ids = set()
    for geom_id in range(model.ngeom):
        mat_id = model.geom_matid[geom_id]
        if mat_id != -1 and model.material(mat_id).name == WALL_MATERIAL:
            ids.add(geom_id)
    return ids


def car_hit_wall(
    model: mujoco.MjModel, data: mujoco.MjData, car_body_name: str = "car"
) -> bool:
    """True if any geom belonging to car_body_name's kinematic tree is
    currently in contact with a wall geom (see module docstring)."""
    walls = wall_geom_ids(model)
    if not walls:
        return False

    car_root = model.body(car_body_name).id
    for i in range(data.ncon):
        contact = data.contact[i]
        g1, g2 = contact.geom1, contact.geom2
        g1_is_car = model.body_rootid[model.geom_bodyid[g1]] == car_root
        g2_is_car = model.body_rootid[model.geom_bodyid[g2]] == car_root
        if (g1_is_car and g2 in walls) or (g2_is_car and g1 in walls):
            return True
    return False


def reset_if_wall_hit(
    model: mujoco.MjModel, data: mujoco.MjData, car_body_name: str = "car"
) -> bool:
    """If the car is touching a wall, reset the simulation to its initial
    state (mujoco.mj_resetData) and return True. Otherwise leave data
    untouched and return False.

    mj_resetData zeroes qpos/qvel/ctrl/time but does NOT recompute forward
    kinematics or sensors -- xpos/xquat/sensordata are left at zero (not
    even the pre-reset values, a degenerate all-zero quaternion) until the
    next mj_step or mj_forward. A caller that reads those right after a
    reset (e.g. a viewer's camera, or a HUD sensor readout) would see
    garbage for one frame, so mj_forward runs here before returning.

    Only resets mujoco state -- any caller-side state (e.g. a demo's own
    throttle/steer accumulators) must be reset separately by the caller.
    """
    if car_hit_wall(model, data, car_body_name):
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        return True
    return False
