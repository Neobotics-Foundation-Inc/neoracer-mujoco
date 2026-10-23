// The spinning LiDAR, as lidar.py models it: slots measured when the head
// passes them, from the pose of that physics step, with one mj_multiRay
// call. Slot i points i * 360 / slots degrees clockwise from the front and
// reads the distance (m), or NO_RETURN where the car blocks the rear or
// nothing lies within range.
//
// Frame: +X forward, +Y left, +Z up.

import type { MjData } from '@mujoco/mujoco';

import type { Mujoco, Scene } from './scene.js';
import { NO_RETURN } from './generated/shared.js';
import { NEORACER_PARAMS } from './generated/spec.js';

/** The LiDAR's scan, as LidarParams in lidar.py. */
export interface LidarParams {
  readonly slots: number; // slots a turn
  readonly scanRate: number; // Hz, turns a second
  readonly maxRange: number; // m, the LakiBeam1's range on a bright target
  readonly field: number; // rad of bearings it sees, centered on the front
}

/** The NeoRacer's LiDAR: lidar.LidarParams() in Python. */
export const LIDAR: LidarParams = {
  slots: NEORACER_PARAMS.lidar.slots,
  scanRate: NEORACER_PARAMS.lidar.scan_rate,
  maxRange: NEORACER_PARAMS.lidar.max_range,
  field: NEORACER_PARAMS.lidar.field,
};

/** Whether it sees slot i: its bearing is in [-field / 2, field / 2). */
export function isVisible(params: LidarParams, i: number): boolean {
  const bearing = i * ((2 * Math.PI) / params.slots);
  return bearing < params.field / 2 || bearing >= 2 * Math.PI - params.field / 2;
}

/** The LiDAR between two physics steps. */
export interface LidarState {
  readonly head: number; // slots of the turn in progress already measured, in [0, slots)
  readonly sweeping: Float64Array; // the turn in progress; slots at or past head are stale
  readonly scan: Float64Array; // the last complete turn: what the LiDAR reports
}

// The part of a slot below which the head has not reached it.
const EPS = 1e-9;

/** The distance (m) of each of slots from the LiDAR at the pose of data, or NO_RETURN. */
function measure(mj: Mujoco, params: LidarParams, scene: Scene, data: MjData, slots: readonly number[]): Float64Array {
  const out = new Float64Array(slots.length).fill(NO_RETURN);
  const seen = slots.flatMap((slot, k) => (isVisible(params, slot) ? [k] : []));
  if (seen.length === 0) return out;
  const { model, car } = scene;
  const xpos = Array.from(data.site_xpos.subarray(3 * car.lidar, 3 * car.lidar + 3) as Float64Array);
  const R = data.site_xmat.subarray(9 * car.lidar, 9 * car.lidar + 9) as Float64Array;
  const rays = new Array<number>(3 * seen.length);
  seen.forEach((k, j) => {
    const bearing = slots[k] * ((2 * Math.PI) / params.slots);
    const x = Math.cos(bearing);
    const y = -Math.sin(bearing); // clockwise from +X is toward -Y
    for (let r = 0; r < 3; r++) rays[3 * j + r] = R[3 * r] * x + R[3 * r + 1] * y;
  });
  const geom = new mj.IntBuffer(seen.length);
  const dist = new mj.DoubleBuffer(seen.length);
  try {
    const body = model.site_bodyid[car.lidar];
    mj.mj_multiRay(model, data, xpos, rays, null as never, true, body, geom, dist, null, seen.length, params.maxRange);
    const g = geom.GetView() as Int32Array;
    const d = dist.GetView() as Float64Array;
    seen.forEach((k, j) => {
      out[k] = g[j] >= 0 && d[j] <= params.maxRange ? d[j] : NO_RETURN;
    });
  } finally {
    geom.delete();
    dist.delete();
  }
  return out;
}

const range = (from: number, to: number) => Array.from({ length: Math.max(to - from, 0) }, (_, i) => from + i);

/** Throw if params is not a scan, as LidarParams in lidar.py checks. */
function checkParams(params: LidarParams): void {
  const problems = [
    Number.isInteger(params.slots) && params.slots >= 1 ? '' : `slots must be an integer >= 1, not ${params.slots}`,
    params.scanRate > 0 ? '' : `scanRate must be > 0 Hz, not ${params.scanRate}`,
    params.maxRange > 0 ? '' : `maxRange must be > 0 m, not ${params.maxRange}`,
    params.field > 0 && params.field <= 2 * Math.PI ? '' : `field must be in (0, 2 pi] rad, not ${params.field}`,
  ].filter((p) => p !== '');
  if (problems.length > 0) throw new Error(problems.join('; '));
}

/**
 * The LiDAR of a car that has stood still at the pose of data for a turn.
 * Throw if params is not a scan.
 */
export function startLidar(mj: Mujoco, params: LidarParams, scene: Scene, data: MjData): LidarState {
  checkParams(params);
  const scan = measure(mj, params, scene, data, range(0, params.slots));
  return { head: 0, sweeping: scan, scan };
}

/**
 * The LiDAR after a physics step of dt (s), less than a turn, that left the
 * car at the pose of data: it measures the slots that its head passed.
 */
export function sweepLidar(
  mj: Mujoco,
  params: LidarParams,
  scene: Scene,
  data: MjData,
  state: LidarState,
  dt: number,
): LidarState {
  const n = params.slots;
  let end = state.head + n * params.scanRate * dt;
  if (!(end - state.head > 0 && end - state.head < n)) throw new Error(`a step of ${dt} s is not between 0 and a turn`);
  const slots = range(Math.ceil(state.head - EPS), Math.ceil(end - EPS));
  const values = measure(mj, params, scene, data, slots.map((s) => s % n));
  let sweeping = state.sweeping.slice();
  let scan = state.scan;
  slots.forEach((s, k) => {
    if (s < n) sweeping[s] = values[k];
  });
  if (end >= n - EPS) {
    scan = sweeping;
    sweeping = new Float64Array(n).fill(NO_RETURN);
    slots.forEach((s, k) => {
      if (s >= n) sweeping[s - n] = values[k];
    });
    end -= n;
  }
  return { head: Math.max(end, 0), sweeping, scan };
}
