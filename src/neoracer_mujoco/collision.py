"""
Detects when the car touches a wall, and resets the simulation, so that a
driving demo can recover without a manual reset key.

A geom is a wall if the name of its MuJoCo material is "wall" or starts with
"wall_". The track XMLs in assets/tracks/ use these materials for real walls.
The floor and drivable ramps use other materials. If a scene has no wall
material, for example the bare car on assets/neoracer.xml, no reset occurs.

The car is the body tree of its root body ("car", the freejoint body in
neoracer.xml). Every geom in that tree, for example the chassis and the
wheels, is part of the car.

Get the wall geoms once, then check each step:

    walls = wall_geom_ids(scene.spec, scene.model)
    if car_touches_wall(scene.model, data, scene.car.body, walls):
        reset_car(scene.model, data)
"""

import mujoco

WALL_MATERIAL = "wall"


def wall_geom_ids(spec: mujoco.MjSpec, model: mujoco.MjModel) -> frozenset[int]:
    """Ids in model, which was compiled from spec, of the geoms whose material
    is a wall material (see module docstring)."""
    return frozenset(
        model.bind_scalar(geom).id
        for geom in spec.geoms
        if geom.material == WALL_MATERIAL
        or geom.material.startswith(WALL_MATERIAL + "_")
    )


def car_touches_wall(
    model: mujoco.MjModel, data: mujoco.MjData, car_root: int, walls: frozenset[int]
) -> bool:
    """True if a geom of the body tree with root body id car_root touches a
    geom in walls. It does no name lookup, so a control loop can call it at
    each step."""
    for i in range(data.ncon):
        contact = data.contact[i]
        g1, g2 = contact.geom1, contact.geom2
        g1_is_car = model.body_rootid[model.geom_bodyid[g1]] == car_root
        g2_is_car = model.body_rootid[model.geom_bodyid[g2]] == car_root
        if (g1_is_car and g2 in walls) or (g2_is_car and g1 in walls):
            return True
    return False


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
