"""
Write, as JSON to the path in argv[1], what the Python package computes for
the cases that the JS package's tests repeat (test/conformance.test.mjs).

Run it from the repository with the package installed:
    python js/test/fixtures.py js/test/fixtures.json
"""

import dataclasses
import json
import math
import os
import sys

import mujoco

from neoracer_mujoco import assets, encoder, lidar, sim
from neoracer_mujoco.control.steering import ControlCommand
from neoracer_mujoco.racecar_core_adapter import MujocoDrive
from neoracer_spec.derive import derive
from neoracer_spec.neoracer import NEORACER

TRACK = os.path.join(assets.TRACK_DIR, "loop_corridor.xml")
DRIVE = ControlCommand(throttle=0.5, steer=0.2)
DRIVE_STEPS = 360  # 1 s at the 1/360 s timestep


def main(out: str) -> None:
    scene = assets.compose(mujoco.MjSpec.from_file(TRACK), visuals=False)
    model, car = scene.model, scene.car
    sizes = {
        n: int(getattr(model, n)) for n in ("nbody", "njnt", "ngeom", "nu", "nq", "nv")
    }

    settled = sim.settle(model)
    data = sim.settle(model)
    params = encoder.EncoderParams()
    drive_pos = data.bind_scalar(car.sensors.drive_pos)
    state = encoder.start(params, float(drive_pos.data[0]))
    sim.apply_command(car, data, DRIVE)
    for _ in range(DRIVE_STEPS):
        mujoco.mj_step(model, data)
        state = encoder.step(
            params, state, float(drive_pos.data[0]), model.opt.timestep
        )

    # A turn swept from 13 poses, each 1 cm to the left of the last.
    lidar_params = lidar.LidarParams()
    poses = []
    sweep_data = sim.settle(model)
    joint = sweep_data.bind_scalar(car.joint)
    sweep = lidar.start(lidar_params, model, sweep_data, car)
    poses.append(joint.qpos.tolist())
    for _ in range(12):
        joint.qpos[1] += 0.01
        mujoco.mj_forward(model, sweep_data)
        poses.append(joint.qpos.tolist())
        sweep = lidar.sweep(
            lidar_params, model, sweep_data, car, sweep, model.opt.timestep
        )

    angles = [0.3 * math.sin(0.05 * i) + 0.02 * i for i in range(200)]
    speeds, enc = [], encoder.start(params, angles[0])
    for a in angles[1:]:
        enc = encoder.step(params, enc, a, 1 / 360)
        speeds.append(enc.speed)

    commands = []
    for speed, angle_cmd, max_speed in [
        (1.0, 0.5, 1.0),
        (-0.3, -1.0, 0.5),
        (2.0, 3.0, 0.25),
    ]:
        d = mujoco.MjData(model)
        drive = MujocoDrive(scene, d)
        drive.set_max_speed(max_speed)
        drive.set_speed_angle(speed, angle_cmd)
        commands.append(
            {
                "speed": speed,
                "angle": angle_cmd,
                "max_speed": max_speed,
                "drive": float(d.bind_scalar(car.drive).ctrl[0]),
                "steer": float(d.bind_scalar(car.steer).ctrl[0]),
            }
        )

    fixtures = {
        "sizes": sizes,
        "timestep": model.opt.timestep,
        "settled_qpos": settled.qpos.tolist(),
        "drive": {
            "throttle": DRIVE.throttle,
            "steer": DRIVE.steer,
            "steps": DRIVE_STEPS,
            "qpos": data.qpos.tolist(),
            "encoder_speed": state.speed,
        },
        "sweep": {"poses": poses, "scan": sweep.scan.tolist(), "head": sweep.head},
        "encoder": {"angles": angles, "speeds": speeds},
        "commands": commands,
        "params": dataclasses.asdict(derive(NEORACER)),
    }
    with open(out, "w") as f:
        json.dump(fixtures, f)


if __name__ == "__main__":
    main(sys.argv[1])
