// The speed that the car's firmware reports from its motor encoder, as
// encoder.py computes it: every window (20 ms) the drivetrain's angle over
// the window becomes the tyre's surface speed, filtered with weight alpha on
// the new window. A window closes at the first step on or after each tick
// and is measured over its own length.

/** How the firmware measures the speed. */
export interface EncoderParams {
  readonly window: number; // s, the firmware's control period
  readonly alpha: number; // the weight of the new window in the filter
}

export const ENCODER: EncoderParams = { window: 0.02, alpha: 0.95 };

/** The tyre's radius (m), as sensors.WHEEL_RADIUS: the 85 mm tyre of the spec. */
export const WHEEL_RADIUS = 0.0425;

/** The encoder between two physics steps. */
export interface EncoderState {
  readonly angle: number; // rad, the drivetrain's angle when the window opened
  readonly span: number; // s since the window opened
  readonly due: number; // s until the firmware's next tick
  readonly speed: number; // m/s, + = forward: the last speed the firmware reported
}

// A window closes when less than this is left of it (s).
const EPS = 1e-9;

/**
 * The encoder of a car at rest, its drivetrain at angle (rad). Throw if the
 * window is not positive or alpha is not in (0, 1], as EncoderParams does.
 */
export function startEncoder(params: EncoderParams, angle: number): EncoderState {
  const problems = [
    params.window > 0 ? '' : `window must be > 0 s, not ${params.window}`,
    params.alpha > 0 && params.alpha <= 1 ? '' : `alpha must be in (0, 1], not ${params.alpha}`,
  ].filter((p) => p !== '');
  if (problems.length > 0) throw new Error(problems.join('; '));
  return { angle, span: 0, due: params.window, speed: 0 };
}

/** The encoder after a physics step of dt (s) that left the drivetrain at angle (rad). */
export function stepEncoder(params: EncoderParams, state: EncoderState, angle: number, dt: number): EncoderState {
  const span = state.span + dt;
  const due = state.due - dt;
  if (due >= EPS) return { angle: state.angle, span, due, speed: state.speed };
  const raw = ((angle - state.angle) * WHEEL_RADIUS) / span; // m/s over the window
  return { angle, span: 0, due: due + params.window, speed: params.alpha * raw + (1 - params.alpha) * state.speed };
}
