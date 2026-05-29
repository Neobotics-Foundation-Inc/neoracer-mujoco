import time

import mujoco
import mujoco.viewer


xml = """
<mujoco>

    <worldbody>

        <light pos="0 0 5"/>
        <geom type="plane" size="20 20 .1"/>

        <body name="car" pos="0 0 1">

            <freejoint/>

            <geom
                type="box"
                size="1 .5 .2"/>

            <body
                name="rear_left"
                pos="-.7 .55 -.2">

                <joint
                    name="rear_left_joint"
                    type="hinge"
                    axis="0 1 0"/>

                <geom
                    type="cylinder"
                    size=".18 .08"
                    euler="90 0 0"/>

            </body>
            <body name="front_left" pos=".7 .55 -.2">
                <joint name="front_left_joint" type="hinge" axis="0 1 0"/>
                <geom
                    type="cylinder"
                    size=".18 .08"
                    euler="90 0 0"
                    friction="2 0.1 0.1"/>
            </body>

            <body name="front_right" pos=".7 -.55 -.2">
                <joint name="front_right_joint" type="hinge" axis="0 1 0"/>
                <geom
                    type="cylinder"
                    size=".18 .08"
                    euler="90 0 0"
                    friction="2 0.1 0.1"/>
            </body>
            <body
                name="rear_right"
                pos="-.7 -.55 -.2">

                <joint
                    name="rear_right_joint"
                    type="hinge"
                    axis="0 1 0"/>

                <geom
                    type="cylinder"
                    size=".18 .08"
                    euler="90 0 0"/>

            </body>

        </body>

    </worldbody>


    <actuator>

        <motor
            joint="rear_left_joint"
            gear="30"/>

        <motor
            joint="rear_right_joint"
            gear="30"/>

    </actuator>

</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)

data = mujoco.MjData(model)


with mujoco.viewer.launch_passive(model,data) as viewer:

    while viewer.is_running():

        data.ctrl[0]=10
        data.ctrl[1]=10

        mujoco.mj_step(model,data)

        viewer.sync()

        time.sleep(.01)