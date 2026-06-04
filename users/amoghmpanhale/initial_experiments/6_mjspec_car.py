import mujoco
import mujoco.viewer
import time

from users.amoghmpanhale.initial_experiments.mjspec_assets.car_plane import CarPlane

car = CarPlane()
pushed = False

with mujoco.viewer.launch_passive(car.model, car.data) as viewer:
    while viewer.is_running():
        if not pushed:
            car.data.qvel[0] = 3.0
            pushed = True

        car.step()
        viewer.sync()
        print(f"velocity: {car.velocity}")
        time.sleep(0.01)
