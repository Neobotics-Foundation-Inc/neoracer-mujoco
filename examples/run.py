"""
Run the NeoRacer car in the MuJoCo viewer with a demo controller.

The controller applies a constant throttle and a sinusoidal steer.
The script prints one diagnostic line every print_every_steps physics steps.

Usage:
    mjpython -m examples.run               # safe mode
    mjpython -m examples.run --mode full   # full mode

Safe mode uses a low throttle and a speed limit.
Full mode uses a high throttle and no speed limit. At high speed, the car rolls over.
In both modes, the script prints a warning when the roll angle is more than roll_warn_deg.

Frame: +X forward, +Y left, +Z up.
Control layout (see neoracer_mujoco.contract):
  ctrl[0..3] = fl/fr/rl/rr wheel torque (N·m), + = forward
  ctrl[4]    = steer_servo target angle (rad), + = left.
               Two Ackermann equality constraints set the fl and fr steer angles from it.
"""

import argparse
import math
import time
from dataclasses import dataclass
from typing import Literal

import mujoco
import mujoco.viewer

from neoracer_mujoco import assets, sensors, sim
from neoracer_mujoco.control.steering import ControlCommand

type Mode = Literal["safe", "full"]


@dataclass(frozen=True)
class DriveParams:
    """Inputs of the demo controller and the log settings. The defaults are safe mode."""

    throttle_nm: float = 0.08  # N·m per wheel; top speed stays near 1.5 m/s
    steer_amp_rad: float = 0.15  # rad; amplitude of the steer sine
    steer_freq_hz: float = 0.30  # Hz; frequency of the steer sine
    # m/s; above this speed the throttle is 0 and the car coasts. None = no limit.
    speed_limit_ms: float | None = 1.5
    print_every_steps: int = 20  # physics steps between two log lines
    roll_warn_deg: float = 25.0  # deg; the log line shows a warning above this roll


MODES: frozendict[Mode, DriveParams] = frozendict(
    {
        "safe": DriveParams(),
        "full": DriveParams(
            throttle_nm=0.35,
            steer_amp_rad=0.35,
            steer_freq_hz=0.40,
            speed_limit_ms=None,
        ),
    }
)


def _roll_pitch_deg(q: sensors.Quat) -> tuple[float, float]:
    """Roll and pitch (deg) from a MuJoCo (w, x, y, z) quaternion."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    roll = math.degrees(math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
    # Clamp the asin argument to [-1, 1], because rounding can move it outside.
    sp = max(-1.0, min(1.0, 2 * (w * y - z * x)))
    pitch = math.degrees(math.asin(sp))
    return roll, pitch


def _command(params: DriveParams, t: float, speed_ms: float) -> ControlCommand:
    """The command at time t (s): a constant throttle (N·m) and a sinusoidal
    steer (rad). If the speed (m/s) is at or above the speed limit, the throttle is 0."""
    steer = params.steer_amp_rad * math.sin(2 * math.pi * params.steer_freq_hz * t)
    limit = params.speed_limit_ms
    throttle = 0.0 if limit is not None and speed_ms >= limit else params.throttle_nm
    return ControlCommand(throttle=throttle, steer=steer)


def _log(
    params: DriveParams,
    data: mujoco.MjData,
    car_id: int,
    command: ControlCommand,
    readings: sensors.SensorReadings,
    speed_ms: float,
) -> None:
    """Print one diagnostic line."""
    car_z = data.xpos[car_id][2]
    roll, pitch = _roll_pitch_deg(readings.imu_quat)
    yaw_rate = math.degrees(readings.imu_gyro[2])  # deg/s

    susp = [
        readings.fl_susp_pos,
        readings.fr_susp_pos,
        readings.rl_susp_pos,
        readings.rr_susp_pos,
    ]

    flag = ""
    if abs(roll) > params.roll_warn_deg:
        flag = f"  *** ROLL WARNING: {roll:+.1f}° ***"

    print(
        f"t={data.time:6.2f}s "
        f"thr={command.throttle:+.2f} steer={command.steer:+.3f}"
        f"(fl={readings.fl_steer_pos:+.3f}/fr={readings.fr_steer_pos:+.3f}) "
        f"spd={speed_ms:+5.2f}m/s "
        f"z={car_z:.3f}m "
        f"roll={roll:+5.1f}° pitch={pitch:+5.1f}° yaw_rate={yaw_rate:+6.1f}°/s "
        f"susp=[{susp[0]:+.3f} {susp[1]:+.3f} {susp[2]:+.3f} {susp[3]:+.3f}]"
        f"{flag}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mode", default="safe", choices=MODES)
    args = parser.parse_args()
    params = MODES[args.mode]

    model = assets.load()
    data = mujoco.MjData(model)
    car_id = model.body("car").id
    layout = sensors.sensor_layout(model)

    print(f"NeoRacer viewer — mode={args.mode}")
    print(
        f"  throttle={params.throttle_nm} N·m  steer_amp={params.steer_amp_rad} rad  "
        f"speed_limit={params.speed_limit_ms} m/s"
    )
    print()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = car_id
        viewer.cam.distance = 1.2  # m from the car
        viewer.cam.azimuth = 90  # deg
        viewer.cam.elevation = -30  # deg; negative = the camera looks down

        step = 0
        while viewer.is_running():
            loop_start = time.time()

            readings = sensors.read_layout(layout, data)
            speed = sensors.estimated_linear_speed_ms(
                sensors.motor_encoder_rad_s(readings)
            )

            command = _command(params, data.time, speed)
            sim.apply_command(data, command)
            mujoco.mj_step(model, data)
            viewer.sync()

            step += 1
            if step % params.print_every_steps == 0:
                _log(params, data, car_id, command, readings, speed)

            # Sleep only for the rest of this timestep, so the simulation
            # keeps pace with real time.
            remaining = model.opt.timestep - (time.time() - loop_start)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
