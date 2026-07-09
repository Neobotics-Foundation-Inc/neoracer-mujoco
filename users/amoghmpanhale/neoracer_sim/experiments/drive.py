"""
Movement by actuators: drive the rear wheels and steer the front wheels around
the oval track.

    ctrl[0], ctrl[1]  rear left / right motor torque
    ctrl[2], ctrl[3]  front left / right steering angle (rad)

Run from users/amoghmpanhale/:   mjpython -m neoracer_sim.experiments.drive
"""

import math
import mujoco

from neoracer_sim.car import add_car
from neoracer_sim.world import add_ground, add_oval_track
from neoracer_sim.utils import compile_spec, run_viewer

spec = mujoco.MjSpec()
add_ground(spec)
add_oval_track(spec)
add_car(spec)
model, data = compile_spec(spec)


def control(model, data):
    data.ctrl[0] = 1.5                          # rear motors: constant throttle
    data.ctrl[1] = 1.5
    steer = 0.35 * math.sin(data.time)          # sweep the steering back and forth
    data.ctrl[2] = steer
    data.ctrl[3] = steer


run_viewer(model, data, control)
