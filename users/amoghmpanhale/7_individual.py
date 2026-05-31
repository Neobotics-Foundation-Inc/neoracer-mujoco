import mujoco
import mujoco.viewer
import time

from mjspec_assets.plane import Plane
from mjspec_assets.car import Car

spec = mujoco.MjSpec()
Plane(spec)
car = Car(spec, spec.worldbody)

model = spec.compile()
data = mujoco.MjData(model)

pushed = False

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        if not pushed:
            data.qvel[0] = 3.0
            pushed = True

        mujoco.mj_step(model, data)
        viewer.sync()
        print(f"velocity: {car.get_velocity(data)}")
        time.sleep(0.01)
