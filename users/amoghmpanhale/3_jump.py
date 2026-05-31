import mujoco
import mujoco.viewer
import time
import utils

model, data = utils.load_car('xml_assets/simple_car_ramp.xml')

# push the car forward fast enough to jump the ramp
data.qvel[0] = 15.0

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.lookat[:] = [4, 0, 0.5]
    viewer.cam.distance = 12
    viewer.cam.elevation = -20
    viewer.cam.azimuth = 180

    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)
