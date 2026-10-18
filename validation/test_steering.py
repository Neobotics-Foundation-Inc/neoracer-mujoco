"""
A law for the car's steering: in a steady turn the servo holds the wheels
still, so the car does not shimmy.
"""

import mujoco
import numpy as np
import pytest

from neoracer_mujoco import assets, sim
from neoracer_mujoco.control.steering import ControlCommand


@pytest.mark.parametrize("lock_share", [0.6, 1.0])
def test_a_steady_part_throttle_turn_does_not_shimmy(lock_share):
    """Held at 0.6 throttle and lock_share of full lock, the part throttle
    where the steering shook, the car's yaw rate (rad/s) spreads by less than
    0.02 over 4 s once it has settled into the turn. A shimmy at 15 to 30 Hz
    spreads it by 0.07."""
    scene = assets.load()  # the car on its own floor, 10 m square
    model, car = scene.model, scene.car
    data = mujoco.MjData(model)
    lock = model.bind_scalar(car.steer).ctrlrange[1]  # rad
    command = ControlCommand(throttle=0.6, steer=lock_share * lock)
    rate = round(1 / model.opt.timestep)  # steps a second
    yaw_rates = []
    for step in range(10 * rate):
        sim.apply_command(car, data, command)
        mujoco.mj_step(model, data)
        if step >= 6 * rate:
            yaw_rates.append(data.sensor("imu_gyro").data[2])
    assert np.std(yaw_rates) < 0.02
