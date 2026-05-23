import mujoco
import mujoco.viewer
import time
import math

xml = """
<mujoco>
    <worldbody>
        <light pos="0 0 5"/>
        <geom type="plane" size="20 20 .1"/>

        <body pos="0 0 1">
            <freejoint/>

            <geom type="box" size="1 .5 .2"/>

            <geom type="cylinder" pos=".7 .55 -.2" size=".18 .08" euler="90 0 0"/>
            <geom type="cylinder" pos=".7 -.55 -.2" size=".18 .08" euler="90 0 0"/>
            <geom type="cylinder" pos="-.7 .55 -.2" size=".18 .08" euler="90 0 0"/>
            <geom type="cylinder" pos="-.7 -.55 -.2" size=".18 .08" euler="90 0 0"/>
        </body>
    </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

t = 0

with mujoco.viewer.launch_passive(model,data) as viewer:

    while viewer.is_running():

        data.qvel[0] = .5
        data.qvel[1] = .3*math.sin(t)

        mujoco.mj_step(model,data)

        viewer.sync()

        t += .02

        time.sleep(.01)