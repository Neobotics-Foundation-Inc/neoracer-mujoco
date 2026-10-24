"""
Golden runs: the RACECAR Neo V2's model drives the reference car's
calibration runs (calibration_reference_car_2026-10), as the calibration
tools drove the car, and measures what they measured. Each test holds the
model to the values in neoracer_spec.racecar_neo_v2, within the spread of
the car's runs and of what the summaries leave unsaid, such as when a stop
begins.

The model's speed is the encoder's: the drivetrain's speed times the tire's
radius, as /odom/integrated after its calibrated scale. Its pose is
base_link's, on the floor below the middle of the rear axle.
"""

import functools
import math
import os
from dataclasses import dataclass

import mujoco
import numpy as np
import pytest
from cases import case

from neoracer_mujoco import assets, lidar, sim
from neoracer_mujoco.control.steering import ControlCommand
from neoracer_mujoco.scene import Scene
from neoracer_spec import derive_v2
from neoracer_spec.racecar_neo_v2 import RACECAR_NEO_V2 as V2

PARAMS = derive_v2.derive(V2)
L = V2.chassis.wheelbase.value
"""The wheelbase (m)."""
AXLE = V2.wheel.axle_height.value
"""The axle's height (m)."""
R = PARAMS.model.common.wheels.radius
"""The wheels' radius (m)."""
SETTLE = 1.0
"""The time at rest before a run (s)."""


@functools.cache
def v2() -> Scene:
    """The V2's scene, compiled once."""
    return assets.load(os.path.join(assets.ASSET_DIR, "racecar_neo_v2.xml"))


@dataclass(frozen=True)
class Run:
    """A run, sampled every physics step from the first command."""

    t: np.ndarray
    """The time since the first command (s)."""
    xy: np.ndarray
    """The position of base_link (m), of shape (n, 2)."""
    heading: np.ndarray
    """The car's heading (rad)."""
    speed: np.ndarray
    """The encoder's speed (m/s)."""
    yaw_rate: np.ndarray
    """The car's yaw rate (rad/s)."""


def drive(legs: list[tuple[float, float, float]]) -> Run:
    """Settle the car for SETTLE s, then drive legs of (s, throttle,
    steering command in [-1, 1])."""
    scene = v2()
    model, car = scene.model, scene.car
    data = mujoco.MjData(model)
    sim.apply_ctrl_and_step(model, data, 0.0, round(SETTLE / model.opt.timestep))
    drivetrain = scene.spec.tendon("drivetrain")
    rows = []
    t0 = data.time
    for seconds, throttle, steer in legs:
        command = ControlCommand(
            throttle=throttle, steer=steer * PARAMS.model.common.servo.lock
        )
        sim.apply_command(car, data, command)
        for _ in range(round(seconds / model.opt.timestep)):
            mujoco.mj_step(model, data)
            body = data.bind_scalar(car.body)
            frame = body.xmat.reshape(3, 3)
            base = body.xpos + frame @ [-L / 2, 0.0, -AXLE]
            rows.append(
                (
                    data.time - t0,
                    base[0],
                    base[1],
                    math.atan2(frame[1, 0], frame[0, 0]),
                    data.bind_scalar(drivetrain).velocity[0] * R,
                    data.bind_scalar(car.joint).qvel[5],
                )
            )
    a = np.array(rows)
    return Run(
        t=a[:, 0], xy=a[:, 1:3], heading=a[:, 3], speed=a[:, 4], yaw_rate=a[:, 5]
    )


def _at(table, command: float) -> float:
    return next(a.value for a in table if a.command == command)


STRAIGHT = {0.25: 3.0, 0.5: 3.0, 1.0: 2.5}
"""The straight runs as the car drove them: each speed command and its duration (s)."""


def test_the_straight_runs_reach_their_speeds_and_stop():
    """At each speed command the model holds the car's steady speed (the
    mean of the last second) within 2%, reaches 90% of it within 0.05 s of
    the car's time, and after the 0 command comes to rest within 0.06 s of
    the car's time. From where it begins to slow, a latency after the
    command, it stops within 15% of the car's distance. The car's speed
    peaks about 15% above the steady speed a second after the start, which
    the model does not do."""
    d = V2.drive
    for command, seconds in STRAIGHT.items():
        with case(command):
            run = drive([(seconds, command, 0.0), (1.5, 0.0, 0.0)])
            last = (run.t > seconds - 1.0) & (run.t <= seconds)
            steady = run.speed[last].mean()
            assert steady == pytest.approx(_at(d.steady_speed.value, command), rel=0.02)
            rise = run.t[np.argmax(run.speed >= 0.9 * steady)]
            assert rise == pytest.approx(_at(d.rise_time.value, command), abs=0.05)
            after = run.t > seconds
            moving = after & (run.speed > V2.odometry.min_speed.value)
            stop = run.t[moving][-1] - seconds
            assert stop == pytest.approx(_at(d.stop_time.value, command), abs=0.06)
            i = np.argmax(run.t > seconds + V2.drive.latency.value)
            distance = np.linalg.norm(run.xy[-1] - run.xy[i])
            assert distance == pytest.approx(
                _at(d.stop_distance.value, command), rel=0.15
            )


def test_the_coasting_car_slows_as_the_car_does():
    """At throttle 0 the model slows as the law fitted to the car's stops,
    coast[0] + coast[1] * speed, has it slow from the same speed, within
    0.03 m/s: the drag brake and the wheels' damping hold the fitted
    forces. The car pitches as it starts to slow, so the model's
    deceleration rocks about the law's."""
    run = drive([(2.5, 1.0, 0.0), (1.0, 0.0, 0.0)])
    a0, a1 = V2.drive.coast.value
    start = np.argmax(run.t > 2.5 + V2.drive.latency.value)
    law = [run.speed[start]]
    for dt in np.diff(run.t[start:]):
        law.append(max(law[-1] - (a0 + a1 * law[-1]) * dt, 0.0))
    law = np.array(law)
    moving = law > 0.1
    assert np.abs(run.speed[start:][moving] - law[moving]).max() < 0.03


def test_the_car_waits_its_latency_before_it_moves():
    """Nothing moves the car before the drive's latency has passed."""
    run = drive([(0.3, 1.0, 0.0)])
    latency = V2.drive.latency.value
    assert np.abs(run.speed[run.t < latency - 0.005]).max() < 1e-3
    assert run.speed[run.t > latency + 0.05].min() > 0.1


def _arc_curvature(command: float) -> float:
    """The steering tool's arc: the wheels set at rest, 2 s at speed 0.25,
    then a stop. Curvature from the chord and the heading change between
    the two rests, as the tool has it from two LiDAR scans."""
    run = drive([(1.0, 0.0, command), (2.0, 0.25, command), (1.5, 0.0, command)])
    start = np.argmax(run.t >= 1.0)
    heading = math.remainder(run.heading[-1] - run.heading[start], 2 * math.pi)
    chord = float(np.linalg.norm(run.xy[-1] - run.xy[start]))
    return 2 * math.sin(heading / 2) / chord


def test_the_arcs_follow_the_curvature_table():
    """At each steering command of the table, the model's arc is within
    0.04 1/m of the car's measured span, as flat past 0.5 and as different
    by side as the car's."""
    for point in V2.steering.response.value:
        with case(point.command):
            k = _arc_curvature(point.command)
            span = point.value
            assert span.low - 0.04 < k < span.high + 0.04


@dataclass(frozen=True)
class Circle:
    radius: float
    """The radius of base_link's path (m)."""
    overread: float
    """The encoder's path over base_link's, less 1."""
    to_lock: float
    """The time from the command to 95% of the steady yaw rate (s)."""


def _circle(side: float) -> Circle:
    """The dynamics tool's full-lock circle: 0.8 s straight at speed 0.25,
    then 13 s at full lock. A circle fit to base_link's path from 3 s on."""
    run = drive([(0.8, 0.25, 0.0), (13.0, 0.25, side)])
    late = run.t > 3.8
    x, y = run.xy[late, 0], run.xy[late, 1]
    a = np.c_[2 * x, 2 * y, np.ones_like(x)]
    cx, cy, c = np.linalg.lstsq(a, x**2 + y**2, rcond=None)[0]
    path = np.sum(np.linalg.norm(np.diff(run.xy[late], axis=0), axis=1))
    encoder = np.sum(run.speed[late][1:] * np.diff(run.t[late]))
    after = run.t - 0.8
    settled = run.yaw_rate[(after > 0.3) & (after < 1.0)].mean()
    to_lock = after[
        np.argmax((after > 0) & (side * run.yaw_rate >= 0.95 * side * settled))
    ]
    return Circle(
        radius=math.sqrt(c + cx**2 + cy**2),
        overread=encoder / path - 1,
        to_lock=float(to_lock),
    )


def test_the_full_lock_circles():
    """At full lock each way, the model's rear axle turns on the car's
    radius within 3%, its encoder over-reads by the car's share within
    0.015, and its yaw rate reaches 95% within 0.04 s of the car's: the
    car's servo is 13 ms slower to the right than to the left, the model's
    is the same both ways. The over-read comes from the locked center shaft:
    the encoder sees a speed between the rear and the front axles'."""
    s, o = V2.steering, V2.odometry
    for side, radius, overread, to_lock in [
        (
            1.0,
            s.min_turn_radius.value.left,
            o.turn_overread.value.left,
            s.center_to_lock_time.value.left,
        ),
        (
            -1.0,
            s.min_turn_radius.value.right,
            o.turn_overread.value.right,
            s.center_to_lock_time.value.right,
        ),
    ]:
        with case(side):
            c = _circle(side)
            assert c.radius == pytest.approx(radius, rel=0.03)
            assert c.overread == pytest.approx(overread, abs=0.015)
            assert c.to_lock == pytest.approx(to_lock, abs=0.04)


def test_the_lidar_sees_the_antennas_as_the_car_does():
    """Alone on the floor, the model's LiDAR sees its two antennas, about
    0.27 m behind it and 8 to 12 degrees either side of its rear, and
    nothing else: the car's scans see them there."""
    s = v2()
    data = sim.settle(s.model)
    scan = lidar.start(PARAMS.lidar, s.model, data, s.car).scan
    bearings = np.degrees(lidar.bearings(PARAMS.lidar))  # clockwise from the front
    seen = scan != lidar.NO_RETURN
    assert np.all((scan[seen] > 0.26) & (scan[seen] < 0.28))
    for side in (-1, 1):  # left, right of the rear
        with case(side):
            off = side * (bearings[seen] - 180)
            assert np.any((off > 7) & (off < 13))
    assert np.all(np.abs(bearings[seen] - 180) < 13)
