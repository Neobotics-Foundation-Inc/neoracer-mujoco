import math
import time

import mujoco
import mujoco.viewer


def run_viewer(spec):
    """
    Compile a spec and run a simple demo controller.
    """

    model = spec.compile()
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:

        t = 0

        while viewer.is_running():

            # rear wheel motors
            data.ctrl[0] = 1.2
            data.ctrl[1] = 1.2

            # steering controller
            steering = 0.35 * math.sin(t)

            data.ctrl[2] = steering
            data.ctrl[3] = steering

            mujoco.mj_step(model, data)

            viewer.sync()

            t += 0.02

            time.sleep(0.005)