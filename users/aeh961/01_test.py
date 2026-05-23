import mujoco
import mujoco.viewer
import time

xml = """
<mujoco>
    <worldbody>
        <body pos="0 0 1">
            <geom type="box" size=".1 .1 .1"/>
            <freejoint/>
        </body>
    </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)