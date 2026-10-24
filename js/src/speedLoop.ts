/**
 * How the car's firmware holds the speed that a program commands, as
 * speed_loop.py computes it: every period it measures the speed as the
 * encoder does, with a filter of its own, and sets the ESC's pulse from a
 * feed-forward on the commanded speed and a PID on the error. Under
 * stop_speed it sends neutral and starts over. The throttle is the pulse
 * over the ESC's full_throttle_pulse, held within [-1, 1], and holds
 * between ticks.
 */

import { type EncoderState, startEncoder, stepEncoder } from './encoder.js';
import { NEORACER_PARAMS, type SpeedLoopParams } from './generated/spec.js';

export type { SpeedLoopParams };

/** The NeoRacer's: derive(NEORACER).speed_loop in Python. */
export const SPEED_LOOP: SpeedLoopParams = NEORACER_PARAMS.speed_loop;

/** The loop between two physics steps. */
export interface SpeedLoopState {
  /** The speed it measures, as the encoder does. */
  readonly measure: EncoderState;
  /** The error's integral (m). */
  readonly integral: number;
  /** The error at the last tick (m/s). */
  readonly error: number;
  /** The share of the battery's voltage that the motor gets, in [-1, 1]. */
  readonly throttle: number;
}

const clamp = (x: number, low: number, high: number) => Math.min(Math.max(x, low), high);

/**
 * The loop of a car at rest, its drivetrain at angle (rad): neutral. Throw
 * if stop_speed, pulse_limit, full_throttle_pulse or max_speed is not
 * positive, or deadband is not in [0, feed_forward_limit], as
 * SpeedLoopParams does, or if its measure is not an encoder's (startEncoder).
 */
export function startSpeedLoop(params: SpeedLoopParams, angle: number): SpeedLoopState {
  const positive = ['stop_speed', 'pulse_limit', 'full_throttle_pulse', 'max_speed'] as const;
  const problems = [
    ...positive.map((name) => (params[name] > 0 ? '' : `${name} must be > 0, not ${params[name]}`)),
    params.deadband >= 0 && params.deadband <= params.feed_forward_limit
      ? ''
      : `deadband must be in [0, feed_forward_limit], not ${params.deadband}`,
  ].filter((p) => p !== '');
  if (problems.length > 0) throw new Error(problems.join('; '));
  return { measure: startEncoder(params.measure, angle), integral: 0, error: 0, throttle: 0 };
}

/**
 * The speed (m/s) that racecar_core's set_speed_angle(speed, angle)
 * commands, speed in [-1, 1] and maxSpeed (set_max_speed) in [0, 1], each
 * clamped there.
 */
export function targetSpeed(params: SpeedLoopParams, speed: number, maxSpeed = 1): number {
  return clamp(speed, -1, 1) * clamp(maxSpeed, 0, 1) * params.max_speed;
}

/**
 * The feed-forward's pulse (us) for commanded (m/s), at or over stop_speed:
 * deadband and feed_forward per m/s, rounded to a microsecond and held
 * within [deadband, feed_forward_limit], toward commanded.
 */
export function feedForward(params: SpeedLoopParams, commanded: number): number {
  const pulse = clamp(
    Math.floor(params.deadband + Math.abs(commanded) * params.feed_forward + 0.5),
    params.deadband,
    params.feed_forward_limit,
  );
  return commanded > 0 ? pulse : -pulse;
}

/**
 * The loop after a physics step of dt (s) that left the drivetrain at angle
 * (rad), with commanded (m/s) the speed asked for.
 */
export function stepSpeedLoop(
  params: SpeedLoopParams,
  state: SpeedLoopState,
  commanded: number,
  angle: number,
  dt: number,
): SpeedLoopState {
  const measure = stepEncoder(params.measure, state.measure, angle, dt);
  if (measure.span !== 0) return { ...state, measure }; // the window is open: no tick
  if (Math.abs(commanded) < params.stop_speed) return { measure, integral: 0, error: 0, throttle: 0 };
  const period = params.measure.window;
  let error = commanded - measure.speed;
  if (Math.abs(error) < params.error_deadband) error = 0;
  const integral = clamp(state.integral + error * period, -params.integral_limit, params.integral_limit);
  const derivative = (error - state.error) / period;
  const pid = params.kp * error + params.ki * integral + params.kd * derivative;
  const pulse = clamp(feedForward(params, commanded) + Math.trunc(pid), -params.pulse_limit, params.pulse_limit);
  return { measure, integral, error, throttle: clamp(pulse / params.full_throttle_pulse, -1, 1) };
}
