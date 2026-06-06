import time

import mujoco
import mujoco.viewer

from users.amoghmpanhale.mjspec_assets.tracks.oval_track import add_oval_track


def main():
    spec = mujoco.MjSpec()

    spec.option.timestep = 0.005
    spec.option.gravity = [0, 0, -9.81]

    spec.worldbody.add_light(pos=[0, 0, 8])

    add_oval_track(spec)

    model = spec.compile()
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.005)


if __name__ == "__main__":
    main()