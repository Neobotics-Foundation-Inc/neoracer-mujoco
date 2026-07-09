import mujoco
import mujoco.viewer
import time
import users.amoghmpanhale.initial_experiments.utils as utils

model, data = utils.load_car('xml_assets/simple_car.xml')

# push the car forward along x-axis
data.qvel[0] = 3.0

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)
