/**
 * Commands to the car's actuators, as sim.apply_command writes them, and
 * what the racecar_core adapter's set_speed_angle asks of the car.
 *
 * Frame: +X forward, +Y left, +Z up.
 */

import type { MjData, MjModel } from '@mujoco/mujoco';

import type { Car, Scene } from './scene.js';
import { type SpeedLoopParams, targetSpeed } from './speedLoop.js';

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

/** What racecar_core's set_speed_angle(speed, angle) asks of the car. */
export interface SpeedAngle {
  /** The speed (m/s) for the speed loop to hold (stepSpeedLoop), + = forward. */
  readonly target: number;
  /** The steer angle (rad), + = left, which the car takes at once. */
  readonly steer: number;
}

/**
 * What set_speed_angle(speed, angle) asks for, as the racecar_core
 * adapter's MujocoDrive asks it: both in [-1, 1] and clamped there, speed
 * + = forward and angle + = right, with the speed scaled by maxSpeed
 * (set_max_speed, in [0, 1]) and the loop's max_speed (targetSpeed).
 */
export function speedAngle(limits: Limits, params: SpeedLoopParams, speed: number, angle: number, maxSpeed = 1): SpeedAngle {
  return {
    target: targetSpeed(params, speed, maxSpeed),
    // racecar_core turns right for + angle; the steering servo turns left.
    steer: -clamp(angle, -1, 1) * limits.steerRad,
  };
}
