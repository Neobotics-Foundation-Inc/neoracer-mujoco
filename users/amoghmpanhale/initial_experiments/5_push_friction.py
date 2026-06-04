import mujoco
import mujoco.viewer
import time
import users.amoghmpanhale.initial_experiments.utils as utils

model, data = utils.load_car('xml_assets/friction_car.xml')

data.qvel[0] = 3.0

NUM_STEPS = 5
step = 0

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():

        if step < NUM_STEPS:
            data.qvel[0] = 3.0

        mujoco.mj_step(model, data)
        viewer.sync()

        print(f"velocity: {data.qvel[0:3]}")

        time.sleep(0.01)
        step += 1
