"""
Movement by cvel push: shove the chassis and watch it coast. No actuators.

Run from users/amoghmpanhale/:   mjpython -m neoracer_sim.experiments.push
"""

import mujoco

from neoracer_sim.car import add_car
from neoracer_sim.world import add_ground
from neoracer_sim.utils import compile_spec, run_viewer, car_velocity

spec = mujoco.MjSpec()
add_ground(spec)
add_car(spec)
model, data = compile_spec(spec)

pushed = False


def control(model, data):
    global pushed
    if not pushed:
        data.qvel[0] = 5.0          # one shove forward (m/s)
        pushed = True
    print(f"velocity: {car_velocity(data)}")


run_viewer(model, data, control)
