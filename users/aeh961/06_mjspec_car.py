import time

import mujoco
import mujoco.viewer


def build_car_spec():
    spec = mujoco.MjSpec()

    # Basic simulation settings
    spec.option.gravity = [0, 0, -9.81]

    # Add light
    spec.worldbody.add_light(
        name="light",
        pos=[0, 0, 5],
    )

    # Add ground plane
    spec.worldbody.add_geom(
        name="ground",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[20, 20, 0.1],
    )

    # Add car body
    car = spec.worldbody.add_body(
        name="car",
        pos=[0, 0, 1],
    )

    # Freejoint lets the body move freely in 3D
    car.add_freejoint(name="car_freejoint")

    # Main car body
    car.add_geom(
        name="car_body",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[1, 0.5, 0.2],
    )

    # Four visual wheels
    wheel_positions = [
        [0.7, 0.55, -0.2],
        [0.7, -0.55, -0.2],
        [-0.7, 0.55, -0.2],
        [-0.7, -0.55, -0.2],
    ]

    for i, pos in enumerate(wheel_positions):
        car.add_geom(
            name=f"wheel_{i + 1}",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=pos,
            size=[0.18, 0.08],
            euler=[90, 0, 0],
        )

    return spec, car


spec, car = build_car_spec()

# Compile the spec into a MuJoCo model
model = spec.compile()
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # This still uses qvel for simple movement.
        # Later we can replace this with actuator/control logic.
        data.qvel[0] = 0.3

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)