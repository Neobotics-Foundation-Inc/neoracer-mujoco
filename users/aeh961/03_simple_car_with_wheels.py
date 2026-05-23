import mujoco
import mujoco.viewer
import time

xml = """
<mujoco>
    <worldbody>
        <light pos="0 0 5"/>
        <geom name="ground" type="plane" size="20 20 .1"/>

        <body name="car" pos="0 0 1">
            <freejoint/>

            <geom name="car_body" type="box" size="1 .5 .2"/>

            <geom name="front_left_wheel"  type="cylinder" pos=".7 .55 -.2" size=".18 .08" euler="90 0 0"/>
            <geom name="front_right_wheel" type="cylinder" pos=".7 -.55 -.2" size=".18 .08" euler="90 0 0"/>
            <geom name="back_left_wheel"   type="cylinder" pos="-.7 .55 -.2" size=".18 .08" euler="90 0 0"/>
            <geom name="back_right_wheel"  type="cylinder" pos="-.7 -.55 -.2" size=".18 .08" euler="90 0 0"/>
        </body>
    </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        data.qvel[0] = 0.2

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)