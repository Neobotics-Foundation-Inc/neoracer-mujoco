import mujoco
import mujoco.viewer
import time
import numpy as np

xml = """
<mujoco>
    <worldbody>

        <light pos="0 0 5"/>
        <geom type="plane" size="20 20 .1"/>

        <body name="car" pos="0 0 1">
            <freejoint/>
            <geom type="box" size="1 .5 .2"/>
        </body>

        <site name="cp1" pos="5 0 .5" size=".3"/>
        <site name="cp2" pos="10 2 .5" size=".3"/>
        <site name="cp3" pos="15 -2 .5" size=".3"/>

    </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

checkpoints = [
    np.array([5,0]),
    np.array([10,2]),
    np.array([15,-2])
]

current=0

with mujoco.viewer.launch_passive(model,data) as viewer:

    while viewer.is_running():

        data.qvel[0]=0.4

        car_pos=np.array(
            [data.qpos[0],data.qpos[1]]
        )

        if current < len(checkpoints):

            dist=np.linalg.norm(
                car_pos-checkpoints[current]
            )

            if dist<1:

                print(
                    f"Checkpoint hit: {current+1}"
                )

                current+=1

        mujoco.mj_step(model,data)

        viewer.sync()

        time.sleep(.01)