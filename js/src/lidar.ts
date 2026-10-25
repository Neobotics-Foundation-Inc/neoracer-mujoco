/**
 * The spinning LiDAR, as lidar.py models it: slots measured when the head
 * passes them, from the pose of that physics step, which lidarPose reads
 * from qpos, so it needs no mj_kinematics after mj_step. Its rays are a
 * RayScene's, which finds the distances that lidar.py's mj_multiRay finds.
 * Slot i points i * 360 / slots degrees clockwise from the front and reads
 * the distance (m), or NO_RETURN where the car blocks the rear or nothing
 * lies within range.
 *
 * Frame: +X forward, +Y left, +Z up.
 */

import type { MjData } from '@mujoco/mujoco';

import { RayScene } from './rays.js';
import { type Car, type Mujoco, type Scene, multiply, rotation } from './scene.js';
import { FIELD_EPS, NO_RETURN, SLOT_EPS } from './generated/shared.js';
import { type LidarParams, NEORACER_PARAMS } from './generated/spec.js';

export type { LidarParams };

/** The NeoRacer's LiDAR: derive(NEORACER).lidar in Python. */
export const LIDAR: LidarParams = NEORACER_PARAMS.lidar;

/** Whether it sees slot i: its bearing in its field, the closed rule of contract.in_field. */
export function isVisible(params: LidarParams, i: number): boolean {
  const bearing = i * ((2 * Math.PI) / params.slots);
  return Math.min(bearing, 2 * Math.PI - bearing) <= params.field / 2 + FIELD_EPS;
}

/** The LiDAR between two physics steps. */
export interface LidarState {
  /** The slots of the turn in progress already measured, in [0, slots). */
  readonly head: number;
  /** The turn in progress. Its slots at or past head are stale. */
  readonly sweeping: Float64Array;
  /** The last complete turn: what the LiDAR reports. */
  readonly scan: Float64Array;
  /** The scene's geoms, as its rays meet them. */
  readonly rays: RayScene;
}

/**
 * The LiDAR site's position and its rotation (world from site, row-major)
 * at qpos, from the car's free joint and the site's mount on the car, as
 * mj_kinematics puts them in site_xpos and site_xmat.
 */
export function lidarPose(car: Car, qpos: ArrayLike<number>): { pos: number[]; mat: number[] } {
  const a = car.free;
  const q = [qpos[a + 3], qpos[a + 4], qpos[a + 5], qpos[a + 6]];
  const norm = Math.hypot(q[0], q[1], q[2], q[3]);
  const body = q.map((x) => x / norm);
  const R = rotation(body), m = car.lidarMount.pos;
  return {
    pos: [0, 1, 2].map((r) => qpos[a + r] + R[3 * r] * m[0] + R[3 * r + 1] * m[1] + R[3 * r + 2] * m[2]),
    mat: rotation(multiply(body, car.lidarMount.quat)),
  };
}

/** The distance (m) of each of slots from the LiDAR at the pose of data, or NO_RETURN. */
function measure(params: LidarParams, scene: Scene, data: MjData, rays: RayScene, slots: readonly number[]): Float64Array {
  const out = new Float64Array(slots.length).fill(NO_RETURN);
  if (!slots.some((slot) => isVisible(params, slot))) return out;
  rays.update(data);
  const { pos: o, mat: R } = lidarPose(scene.car, data.qpos);
  const d = new Float64Array(3);
  slots.forEach((slot, k) => {
    if (!isVisible(params, slot)) return;
    const bearing = slot * ((2 * Math.PI) / params.slots);
    const x = Math.cos(bearing);
    const y = -Math.sin(bearing); // clockwise from +X is toward -Y
    for (let r = 0; r < 3; r++) d[r] = R[3 * r] * x + R[3 * r + 1] * y;
    const t = rays.cast(o, d, params.max_range);
    if (t !== Infinity) out[k] = t;
  });
  return out;
}

const range = (from: number, to: number) => Array.from({ length: Math.max(to - from, 0) }, (_, i) => from + i);

/** Throw if params is not a scan, as LidarParams in neoracer_spec.derive checks. */
function checkParams(params: LidarParams): void {
  const problems = [
    Number.isInteger(params.slots) && params.slots >= 1 ? '' : `slots must be an integer >= 1, not ${params.slots}`,
    params.scan_rate > 0 ? '' : `scan_rate must be > 0 Hz, not ${params.scan_rate}`,
    params.max_range > 0 ? '' : `max_range must be > 0 m, not ${params.max_range}`,
    params.field > 0 && params.field <= 2 * Math.PI ? '' : `field must be in (0, 2 pi] rad, not ${params.field}`,
  ].filter((p) => p !== '');
  if (problems.length > 0) throw new Error(problems.join('; '));
}

/**
 * The LiDAR of a car that has stood still at the pose of data for a turn,
 * its rays met by rays: by default, the scene's geoms but its own body's.
 * Building them costs a pass over every geom, so a caller that starts the
 * LiDAR again on the same scene passes its last state's rays. Throw if
 * params is not a scan.
 */
export function startLidar(
  mj: Mujoco,
  params: LidarParams,
  scene: Scene,
  data: MjData,
  rays: RayScene = new RayScene(mj, scene.model, data, scene.model.site_bodyid[scene.car.lidar]),
): LidarState {
  checkParams(params);
  const scan = measure(params, scene, data, rays, range(0, params.slots));
  return { head: 0, sweeping: scan, scan, rays };
}

/**
 * The LiDAR after a physics step of dt (s), less than a turn, that left the
 * car at the pose of data: it measures the slots that its head passed.
 */
export function sweepLidar(
  params: LidarParams,
  scene: Scene,
  data: MjData,
  state: LidarState,
  dt: number,
): LidarState {
  const n = params.slots;
  let end = state.head + n * params.scan_rate * dt;
  if (!(end - state.head > 0 && end - state.head < n)) throw new Error(`a step of ${dt} s is not between 0 and a turn`);
  const slots = range(Math.ceil(state.head - SLOT_EPS), Math.ceil(end - SLOT_EPS));
  const values = measure(params, scene, data, state.rays, slots.map((s) => s % n));
  let sweeping = state.sweeping.slice();
  let scan = state.scan;
  slots.forEach((s, k) => {
    if (s < n) sweeping[s] = values[k];
  });
  if (end >= n - SLOT_EPS) {
    scan = sweeping;
    sweeping = new Float64Array(n).fill(NO_RETURN);
    slots.forEach((s, k) => {
      if (s >= n) sweeping[s - n] = values[k];
    });
    end -= n;
  }
  return { head: Math.max(end, 0), sweeping, scan, rays: state.rays };
}
