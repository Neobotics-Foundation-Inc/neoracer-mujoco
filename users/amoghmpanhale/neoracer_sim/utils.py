"""
Small helpers used across experiments: compile a spec and run the viewer.
"""

import time
import mujoco
import mujoco.viewer


def compile_spec(spec):
    """Compile a spec into (model, data)."""
    model = spec.compile()
    data = mujoco.MjData(model)
    return model, data


def run_viewer(model, data, control=None, follow=True):
    """
    Open the passive viewer and step the sim until you close the window.

    control: optional function called each step as control(model, data).
             Use it to set data.ctrl or data.qvel. Leave it None to just watch.
    follow:  keep the camera tracking the car.
    """
    with mujoco.viewer.launch_passive(model, data) as viewer:
        if follow:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = model.body("car").id
            viewer.cam.distance = 6.0
            viewer.cam.elevation = -35

        while viewer.is_running():
            if control is not None:
                control(model, data)
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(TIMESTEP_SLEEP)


# small sleep so the viewer runs near real time
TIMESTEP_SLEEP = 0.005


def car_position(data):
    """World (x, y, z) of the car's freejoint."""
    return data.qpos[0:3]


def car_velocity(data):
    """World (vx, vy, vz) of the car's freejoint."""
    return data.qvel[0:3]
