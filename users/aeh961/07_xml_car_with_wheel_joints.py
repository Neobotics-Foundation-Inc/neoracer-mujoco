import time

import mujoco
import mujoco.viewer


xml = """
<mujoco>
    <worldbody>
        <light pos="0 0 5"/>
        <geom name="ground" type="plane" size="20 20 0.1"/>

        <body name="car" pos="0 0 1">
            <freejoint name="car_freejoint"/>

            <geom name="car_body" type="box" size="1 0.5 0.2"/>

            <body name="front_left_wheel" pos="0.7 0.55 -0.2">
                <joint name="front_left_wheel_joint" type="hinge" axis="0 1 0"/>
                <geom type="cylinder" size="0.18 0.08" euler="90 0 0"/>
            </body>

            <body name="front_right_wheel" pos="0.7 -0.55 -0.2">
                <joint name="front_right_wheel_joint" type="hinge" axis="0 1 0"/>
                <geom type="cylinder" size="0.18 0.08" euler="90 0 0"/>
            </body>

            <body name="rear_left_wheel" pos="-0.7 0.55 -0.2">
                <joint name="rear_left_wheel_joint" type="hinge" axis="0 1 0"/>
                <geom type="cylinder" size="0.18 0.08" euler="90 0 0"/>
            </body>

            <body name="rear_right_wheel" pos="-0.7 -0.55 -0.2">
                <joint name="rear_right_wheel_joint" type="hinge" axis="0 1 0"/>
                <geom type="cylinder" size="0.18 0.08" euler="90 0 0"/>
            </body>
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