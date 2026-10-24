/**
 * Commands to the car's actuators, as sim.apply_command and the
 * racecar_core adapter's set_speed_angle write them.
 *
 * Frame: +X forward, +Y left, +Z up.
 */

import type { MjData, MjModel } from '@mujoco/mujoco';

import type { Car, Scene } from './scene.js';

/** A command to the car. */
export interface Command {
  /** The fraction of the battery voltage, in [-1, 1], + = forward. */
  readonly throttle: number;
  /** The steer angle (rad), + = left. */
  readonly steer: number;
}

/** The upper ends of the car's symmetric ctrlranges, as contract.actuator_limits. */
export interface Limits {
  /** The battery's voltage (V). */
  readonly voltage: number;
  readonly steerRad: number;
}

/**
 * Throw if a range is not symmetric about 0, as contract.actuator_limits
 * does: speedAngle negates the limit for a negative command.
 */
export function actuatorLimits(model: MjModel, car: Car): Limits {
  const range = (id: number): [number, number] => [model.actuator_ctrlrange[2 * id], model.actuator_ctrlrange[2 * id + 1]];
  const [low, high] = range(car.drive);
  const [steerLow, steerHigh] = range(car.steer);
  const problems = [
    high > 0 && low === -high ? '' : `the drive ctrlrange is not [-a, a] with a > 0: ${low}, ${high}`,
    steerHigh > 0 && steerLow === -steerHigh ? '' : `the steer ctrlrange is not [-a, a] with a > 0: ${steerLow}, ${steerHigh}`,
  ].filter((p) => p !== '');
  if (problems.length > 0) throw new Error(problems.join('; '));
  return { voltage: high, steerRad: steerHigh };
}

/** Write command to data.ctrl: the throttle as that fraction of the battery voltage. */
export function applyCommand(scene: Scene, data: MjData, command: Command): void {
  const { car, model } = scene;
  data.ctrl[car.drive] = command.throttle * model.actuator_ctrlrange[2 * car.drive + 1]; // V
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
