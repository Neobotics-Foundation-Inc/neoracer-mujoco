import mujoco
import mujoco.viewer
import time

xml = """
<mujoco>
    <visual>
        <global offwidth="1200" offheight="900"/>
    </visual>

    <worldbody>
        <light pos="0 0 5"/>
        <geom name="ground" type="plane" size="20 20 .1"/>

        <body name="car" pos="0 0 1">
            <freejoint/>
            <geom name="car_body" type="box" size="1 .5 .25"/>
        </body>
    </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        data.qvel[0] = 0.05

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)