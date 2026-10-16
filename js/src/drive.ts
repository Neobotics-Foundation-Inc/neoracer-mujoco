// Commands to the car's actuators, as sim.apply_command and the
// racecar_core adapter's set_speed_angle write them.
//
// Frame: +X forward, +Y left, +Z up.

import type { MjData, MjModel } from '@mujoco/mujoco';

import type { Car, Scene } from './scene.js';

/** A command to the car. */
export interface Command {
  readonly throttle: number; // in [-1, 1]: that fraction of the battery voltage, + = forward
  readonly steer: number; // rad, + = left
}

/** The upper ends of the car's symmetric ctrlranges, as contract.actuator_limits. */
export interface Limits {
  readonly voltage: number; // V, the battery
  readonly steerRad: number; // rad
}

export function actuatorLimits(model: MjModel, car: Car): Limits {
  return {
    voltage: model.actuator_ctrlrange[2 * car.drive + 1],
    steerRad: model.actuator_ctrlrange[2 * car.steer + 1],
  };
}

/** Write command to data.ctrl: the throttle as that fraction of the battery voltage. */
export function applyCommand(scene: Scene, data: MjData, command: Command): void {
  const { car, model } = scene;
  data.ctrl[car.drive] = command.throttle * actuatorLimits(model, car).voltage;
  data.ctrl[car.steer] = command.steer;
}

const clamp = (x: number, low: number, high: number) => Math.min(Math.max(x, low), high);

/**
 * The command for racecar_core's set_speed_angle(speed, angle): both in
 * [-1, 1] and clamped there, speed + = forward and angle + = right, with the
 * speed scaled by maxSpeed (set_max_speed, in [0, 1]).
 */
export function speedAngle(limits: Limits, speed: number, angle: number, maxSpeed = 1): Command {
  return {
    throttle: clamp(speed, -1, 1) * clamp(maxSpeed, 0, 1),
    // racecar_core turns right for + angle; the steering servo turns left.
    steer: -clamp(angle, -1, 1) * limits.steerRad,
  };
}
