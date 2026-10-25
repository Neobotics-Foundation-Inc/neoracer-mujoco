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

from neoracer_mujoco import assets, batch, encoder, lidar, sim, speed_loop
from neoracer_mujoco.control.steering import ControlCommand
from neoracer_mujoco.racecar_core_adapter import MujocoDrive, MujocoRacecar
from neoracer_spec.derive import derive
from neoracer_spec.neoracer import NEORACER

TRACK = os.path.join(assets.TRACK_DIR, "loop_corridor.xml")
DRIVE = ControlCommand(throttle=0.5, steer=0.2)
DRIVE_SECONDS = 1.0
RACECAR_SPEED_ANGLE = (0.25, -0.2)


def main(out: str) -> None:
    scene = assets.compose(mujoco.MjSpec.from_file(TRACK), visuals=False)
    model, car = scene.model, scene.car
    sizes = {
        n: int(getattr(model, n)) for n in ("nbody", "njnt", "ngeom", "nu", "nq", "nv")
    }

    settled = sim.settle(model)
    data = sim.settle(model)
    params = derive(NEORACER).encoder
    state = encoder.start(params, batch.drive_angle(model, car, data.qpos))
    sim.apply_command(car, data, DRIVE)
    drive_steps = round(DRIVE_SECONDS / model.opt.timestep)
    for _ in range(drive_steps):
        mujoco.mj_step(model, data)
        angle = batch.drive_angle(model, car, data.qpos)
        state = encoder.step(params, state, angle, model.opt.timestep)

    # A turn swept from 13 poses, each 1 cm to the left of the last.
    lidar_params = derive(NEORACER).lidar
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
        enc = encoder.step(params, enc, a, model.opt.timestep)
        speeds.append(enc.speed)

    # The loop on the same turn of the drivetrain, asked for a speed it
    # reaches, then a stop, then a speed backward.
    loop_params = derive(NEORACER).speed_loop
    commanded = [0.4 if i < 120 else 0.0 if i < 150 else -0.3 for i in range(199)]
    throttles, loop = [], speed_loop.start(loop_params, angles[0])
    for target, a in zip(commanded, angles[1:], strict=True):
        loop = speed_loop.step(loop_params, loop, target, a, model.opt.timestep)
        throttles.append(loop.throttle)

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
                "target": drive.target,
                "steer": float(d.bind_scalar(car.steer).ctrl[0]),
            }
        )

    # A second of set_speed_angle through the adapter, which counts each
    # step after mj_step1.
    rc_data = sim.settle(model)
    rc = MujocoRacecar(scene, rc_data, max_steps=drive_steps)
    rc.set_start_update(
        lambda: rc.drive.set_speed_angle(*RACECAR_SPEED_ANGLE), lambda: None
    )
    rc.go()
    racecar = {
        "speed": RACECAR_SPEED_ANGLE[0],
        "angle": RACECAR_SPEED_ANGLE[1],
        "steps": drive_steps,
        "qpos": rc_data.qpos.tolist(),
        "encoder_speed": rc.physics.get_encoder_speed(),
        "throttle": float(rc_data.bind_scalar(car.drive).ctrl[0])
        / float(model.bind_scalar(car.drive).ctrlrange[1]),
        "scan": rc.lidar._state.scan.tolist(),
        "head": rc.lidar._state.head,
    }

    fixtures = {
        "sizes": sizes,
        "timestep": model.opt.timestep,
        "settled_qpos": settled.qpos.tolist(),
        # Timesteps that divide SETTLE_TIME, and one that leaves half a step.
        "settle_steps": [
            [dt, sim.settle_steps(dt)]
            for dt in (model.opt.timestep, 0.002, sim.SETTLE_TIME / 312.5)
        ],
        "drive": {
            "throttle": DRIVE.throttle,
            "steer": DRIVE.steer,
            "steps": drive_steps,
            "qpos": data.qpos.tolist(),
            "encoder_speed": state.speed,
        },
        "sweep": {"poses": poses, "scan": sweep.scan.tolist(), "head": sweep.head},
        "encoder": {"angles": angles, "speeds": speeds},
        "speed_loop": {"commanded": commanded, "throttles": throttles},
        "commands": commands,
        "racecar": racecar,
        "params": dataclasses.asdict(derive(NEORACER)),
    }
    with open(out, "w") as f:
        json.dump(fixtures, f)


if __name__ == "__main__":
    main(sys.argv[1])
