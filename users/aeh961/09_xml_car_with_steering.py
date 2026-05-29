import time
import math

import mujoco
import mujoco.viewer


xml = """
<mujoco>
    <option timestep="0.005" gravity="0 0 -9.81"/>

    <worldbody>
        <light pos="0 0 5"/>
        <geom name="ground" type="plane" size="30 30 .1" friction="1.5 .1 .1"/>

        <body name="car" pos="0 0 .45">
            <freejoint/>

            <geom name="car_body" type="box" size="1.2 .5 .18" mass="20" rgba=".7 .7 .7 1"/>

            <body name="front_left_steer" pos=".8 .45 -.22">
                <joint name="front_left_steer_joint" type="hinge" axis="0 0 1" range="-.6 .6"/>
                <geom type="sphere" size=".04" mass=".2" rgba="1 0 0 1"/>

                <body name="front_left_wheel">
                    <joint name="front_left_wheel_joint" type="hinge" axis="0 1 0"/>
                    <geom type="cylinder" size=".22 .08" euler="90 0 0" mass="1" friction="1.5 .1 .1" rgba="0 0 1 1"/>
                </body>
            </body>

            <body name="front_right_steer" pos=".8 -.45 -.22">
                <joint name="front_right_steer_joint" type="hinge" axis="0 0 1" range="-.6 .6"/>
                <geom type="sphere" size=".04" mass=".2" rgba="1 0 0 1"/>

                <body name="front_right_wheel">
                    <joint name="front_right_wheel_joint" type="hinge" axis="0 1 0"/>
                    <geom type="cylinder" size=".22 .08" euler="90 0 0" mass="1" friction="1.5 .1 .1" rgba="0 0 1 1"/>
                </body>
            </body>

            <body name="rear_left_wheel" pos="-.8 .45 -.22">
                <joint name="rear_left_wheel_joint" type="hinge" axis="0 1 0"/>
                <geom type="cylinder" size=".22 .08" euler="90 0 0" mass="1" friction="1.5 .1 .1"/>
            </body>

            <body name="rear_right_wheel" pos="-.8 -.45 -.22">
                <joint name="rear_right_wheel_joint" type="hinge" axis="0 1 0"/>
                <geom type="cylinder" size=".22 .08" euler="90 0 0" mass="1" friction="1.5 .1 .1"/>
            </body>
        </body>
    </worldbody>

    <actuator>
        <motor joint="rear_left_wheel_joint" gear="8"/>
        <motor joint="rear_right_wheel_joint" gear="8"/>

        <position joint="front_left_steer_joint" kp="40"/>
        <position joint="front_right_steer_joint" kp="40"/>
    </actuator>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    t = 0

    while viewer.is_running():
        data.ctrl[0] = 1.2
        data.ctrl[1] = 1.2

        steering = 0.35 * math.sin(t)
        data.ctrl[2] = steering
        data.ctrl[3] = steering

        mujoco.mj_step(model, data)
        viewer.sync()

        t += 0.02
        time.sleep(0.005)