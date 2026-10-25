"""
Run the NeoRacer car in the MuJoCo viewer with a demo controller.

The controller asks the car's speed loop for a constant speed, and steers
along a sine. The script prints one diagnostic line every print_every_steps
physics steps.

Usage:
    mjpython -m examples.run               # safe mode
    mjpython -m examples.run --mode full   # full mode

Safe mode asks for a low speed and steers gently.
Full mode asks for the loop's full speed, more than the motor reaches, so the
loop sends full throttle, and it steers harder.
In both modes, the script prints a warning when the roll angle is more than roll_warn_deg.

Frame: +X forward, +Y left, +Z up.
Actuators (see neoracer_mujoco.contract):
  motor             = terminal voltage (V), + = forward; the firmware's speed
                      loop (racecar.py) sets it to hold the speed asked for
  steer_servo       = target angle (rad), + = left.
                      Two Ackermann equality constraints set the fl and fr steer angles from it.
"""

import argparse
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

import mujoco
import mujoco.viewer

from neoracer_mujoco import assets, contract, racecar, sensors
from neoracer_mujoco.control import SpeedAngle
from neoracer_mujoco.scene import Car
from neoracer_spec import derive
from neoracer_spec.neoracer import NEORACER

type Mode = Literal["safe", "full"]


@dataclass(frozen=True)
class DriveParams:
    """Inputs of the demo controller and the log settings. The defaults are
    safe mode."""

    speed: float = 1.5
    """The speed (m/s) that the controller asks for."""
    steer_amp_rad: float = 0.15
    """The amplitude of the steer sine."""
    steer_freq_hz: float = 0.30
    """The frequency of the steer sine."""
    print_every_steps: int = 20
    """The physics steps between two log lines."""
    roll_warn_deg: float = 25.0
    """The roll above which the log line shows a warning."""

    def __post_init__(self) -> None:
        # main() logs at every print_every_steps-th step, so it must be >= 1.
        if self.print_every_steps < 1:
            raise ValueError(
                f"print_every_steps must be >= 1, got {self.print_every_steps}"
            )


MODES: Mapping[Mode, DriveParams] = MappingProxyType(
    {
        "safe": DriveParams(),
        "full": DriveParams(
            speed=derive.derive(NEORACER).speed_loop.max_speed,
            steer_amp_rad=0.35,
            steer_freq_hz=0.40,
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


def _command(params: DriveParams, t: float) -> SpeedAngle:
    """The command at time t (s): a constant speed (m/s) and a sinusoidal
    steer (rad)."""
    steer = params.steer_amp_rad * math.sin(2 * math.pi * params.steer_freq_hz * t)
    return SpeedAngle(target=params.speed, steer=steer)


def _log(
    params: DriveParams,
    data: mujoco.MjData,
    car: Car,
    command: SpeedAngle,
    readings: sensors.SensorReadings,
    speed_ms: float,
) -> None:
    """Print one diagnostic line."""
    car_z = data.bind_scalar(car.body).xpos[2]
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
        f"ask={command.target:+.2f}m/s steer={command.steer:+.3f}"
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

    scene = assets.load()
    model, car = scene.model, scene.car
    data = mujoco.MjData(model)
    wheel_radius = contract.wheel_radius(model, car)
    racer = racecar.start(racecar.neoracer_params(), scene, data)

    print(f"NeoRacer viewer — mode={args.mode}")
    print(f"  speed={params.speed} m/s  steer_amp={params.steer_amp_rad} rad")
    print()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = model.bind_scalar(car.body).id
        viewer.cam.distance = 1.2  # m from the car
        viewer.cam.azimuth = 90  # deg
        viewer.cam.elevation = -30  # deg; negative = the camera looks down

        step = 0
        while viewer.is_running():
            loop_start = time.time()

            readings = sensors.read(car.sensors, data)
            speed = sensors.motor_encoder_rad_s(readings) * wheel_radius  # m/s

            command = _command(params, data.time)
            racer = racecar.step(scene, data, racecar.ask(racer, command))
            viewer.sync()

            step += 1
            if step % params.print_every_steps == 0:
                _log(params, data, car, command, readings, speed)

            # Sleep only for the rest of this timestep, so the simulation
            # keeps pace with real time.
            remaining = model.opt.timestep - (time.time() - loop_start)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
