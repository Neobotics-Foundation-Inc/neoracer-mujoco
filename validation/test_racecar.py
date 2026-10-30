"""
Laws for racecar.py on a scene of several cars: each car's firmware runs on
its own sensors, and the cars share only the physics step.
"""

import os
from dataclasses import replace

import mujoco
import numpy as np
from scenes import two_cars

from neoracer_mujoco import assets, racecar, sim
from neoracer_mujoco.control import SpeedAngle

LOOP = os.path.join(assets.TRACK_DIR, "loop_corridor.xml")
ASKS = (SpeedAngle(target=1.5, steer=0.05), SpeedAngle(target=0.8, steer=-0.05))
"""What the two cars' programs ask for: speeds and steering of their own."""
PARAMS = racecar.neoracer_params()


def test_each_car_drives_as_it_drives_alone():
    """step_cars runs each car's firmware on its own sensors: two cars 1.5 m
    apart, asked for different speeds and steering, each drive as the only
    car of a scene, started where it starts, to rounding."""
    track = mujoco.MjSpec.from_file(LOOP)
    entries = [assets.Entry("a/"), assets.Entry("b/", (-1.5, 0.0))]
    scene = assets.compose_cars(track, entries, visuals=False)
    data = sim.settle(scene.model)
    started = racecar.start_cars([PARAMS] * 2, scene, data)
    cars = [racecar.ask(car, ask) for car, ask in zip(started, ASKS, strict=True)]
    alone = []
    for entry, ask in zip(entries, ASKS, strict=True):
        one = assets.compose_cars(track, [replace(entry, prefix="")], visuals=False)
        one_data = sim.settle(one.model)
        alone.append(
            (one, one_data, racecar.ask(racecar.start(PARAMS, one, one_data), ask))
        )
    for k in range(500):
        cars = racecar.step_cars(scene, data, cars)
        for i, (one, one_data, car) in enumerate(alone):
            alone[i] = (one, one_data, racecar.step(one, one_data, car))
            np.testing.assert_allclose(
                data.bind_scalar(cars[i].parts.joint).qpos,
                one_data.bind_scalar(one.car.joint).qpos,
                rtol=0,
                atol=1e-9,
                err_msg=f"car {i}, step {k}",
            )


def test_step_cars_does_not_depend_on_the_order_of_the_cars():
    """step_cars of two cars in either order leaves data the same, bit for
    bit, and each car's speed loop and LiDAR the same."""
    scene = two_cars()
    runs = []
    for order in ((0, 1), (1, 0)):
        data = sim.settle(scene.model)
        started = racecar.start_cars([PARAMS] * 2, scene, data)
        cars = [racecar.ask(started[k], ASKS[k]) for k in order]
        for _ in range(500):
            cars = racecar.step_cars(scene, data, cars)
        by_car = [car for _, car in sorted(zip(order, cars, strict=True))]
        runs.append((data.qpos.copy(), data.sensordata.copy(), by_car))
    (qpos, sensordata, cars), (qpos_r, sensordata_r, cars_r) = runs
    np.testing.assert_array_equal(qpos_r, qpos)
    np.testing.assert_array_equal(sensordata_r, sensordata)
    for car, car_r in zip(cars, cars_r, strict=True):
        assert car_r.parts is car.parts
        assert isinstance(car.drive, racecar.SpeedLoop)
        assert isinstance(car_r.drive, racecar.SpeedLoop)
        assert car_r.drive.state.throttle == car.drive.state.throttle
        np.testing.assert_array_equal(car_r.lidar.state.scan, car.lidar.state.scan)
