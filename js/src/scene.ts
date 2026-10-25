/**
 * A NeoRacer car on a track, compiled in MuJoCo's WebAssembly build, as
 * assets.compose(track, visuals=False) makes it in Python.
 *
 * Frame: +X forward, +Y left, +Z up.
 */

import type { MainModule, MjData, MjModel, MjSpec, MjsElement, mjtObj } from '@mujoco/mujoco';

import {
  CAR_BODY,
  DRIVE_ACTUATOR,
  DRIVE_POS_SENSOR,
  LIDAR_SITE,
  SETTLE_TIME,
  START_SITE,
  STEER_ACTUATOR,
} from './generated/shared.js';
import { CAR_XML } from './generated/car.js';

export type Mujoco = MainModule;

/** The parts of the car, as indices into the compiled model. */
export interface Car {
  /** The car's root body. */
  readonly body: number;
  /** The drive motor's actuator. */
  readonly drive: number;
  /** The steering servo's actuator. */
  readonly steer: number;
  /** The LiDAR site. */
  readonly lidar: number;
  /** The drivetrain's angle as a sum over qpos: drive_pos's fixed tendon, coef[k] times qpos[qpos[k]] (driveAngle). */
  readonly drivetrain: { readonly qpos: readonly number[]; readonly coef: readonly number[] };
  /** The address in qpos of the root body's free joint: its position, then its quaternion. */
  readonly free: number;
  /** The LiDAR site's place on the root body: its position and its quaternion there. */
  readonly lidarMount: { readonly pos: readonly number[]; readonly quat: readonly number[] };
}

/** A compiled scene: spec is what model was compiled from. */
export interface Scene {
  readonly spec: MjSpec;
  readonly model: MjModel;
  readonly car: Car;
}

function elements(mj: Mujoco, spec: MjSpec, type: mjtObj): MjsElement[] {
  const out: MjsElement[] = [];
  for (let e = mj.mjs_firstElement(spec, type); e; e = mj.mjs_nextElement(spec, e)) out.push(e);
  return out;
}

function find(mj: Mujoco, spec: MjSpec, type: mjtObj, name: string): MjsElement {
  const element = mj.mjs_findElement(spec, type, name);
  if (!element) throw new Error(`the scene has no ${name}`);
  return element;
}

/**
 * Delete from spec every mesh geom that has no mass and collides with
 * nothing, and the meshes no geom uses after that, as assets.delete_visuals.
 * In neoracer.xml these geoms only show the car, and a browser has no mesh
 * files to load.
 */
export function deleteVisuals(mj: Mujoco, spec: MjSpec): void {
  for (const e of elements(mj, spec, mj.mjtObj.mjOBJ_GEOM)) {
    const geom = mj.mjs_asGeom(e)!;
    const cosmetic = geom.contype === 0 && geom.conaffinity === 0 && geom.mass === 0;
    if (geom.type.value === mj.mjtGeom.mjGEOM_MESH.value && cosmetic) mj.mjs_delete(spec, e);
  }
  const used = new Set(elements(mj, spec, mj.mjtObj.mjOBJ_GEOM).map((e) => mj.mjs_asGeom(e)!.meshname));
  for (const e of elements(mj, spec, mj.mjtObj.mjOBJ_MESH)) {
    if (!used.has(mj.mjs_getName(e))) mj.mjs_delete(spec, e);
  }
}

/**
 * Compile the track's MJCF with the car attached (attachCar), as
 * assets.compose does without visuals. The track keeps its own floor.
 * carXml defaults to assets/neoracer.xml.
 */
export function compose(mj: Mujoco, trackXml: string, carXml: string = CAR_XML): Scene {
  const spec = mj.parseXMLString(trackXml);
  attachCar(mj, spec, mj.parseXMLString(carXml));
  const model = mj.mj_compile(spec);
  if (!model) throw new Error(mj.mjs_getError(spec));
  return { spec, model, car: findCar(mj, spec, model) };
}

/**
 * Attach car's root body to spec at the track's start, its site START_SITE,
 * facing the site's +X, or at the origin, facing +X, if it has none, with
 * the names of its elements behind prefix. The site must be on the world
 * body, as the car's free joint must. spec takes car's <option>, the
 * solver's settings that its model is made for, so a track sets none. car
 * loses the meshes that only show it (deleteVisuals). Several cars share a
 * scene, each behind a prefix of its own; findCar finds each one's parts.
 */
export function attachCar(mj: Mujoco, spec: MjSpec, car: MjSpec, prefix = ''): void {
  spec.option = car.option;
  deleteVisuals(mj, car);
  // The car's root default class is "main", as the track's is: a name of
  // its own keeps the two apart, as in assets.compose.
  mj.mjs_setName(mj.mjs_getSpecDefault(car)!.element, 'car');
  const start = mj.mjs_findElement(spec, mj.mjtObj.mjOBJ_SITE, START_SITE);
  const parent = start && mj.mjs_getName(mj.mjs_getParent(start)!.element);
  if (start && parent !== 'world') {
    throw new Error(`the track's ${START_SITE} site is on the body ${parent}, not on the world body, where the car's free joint must be`);
  }
  const mount = start ? mj.mjs_asSite(start)! : origin(mj, spec);
  const body = mj.mjs_findBody(car, CAR_BODY);
  if (!body) throw new Error(`the car has no "${CAR_BODY}" body`);
  if (!mj.mjs_attach(mount.element, body.element, prefix, '')) throw new Error(mj.mjs_getError(spec));
}

/** The parts of the car whose names are behind prefix in model, which was compiled from spec. */
export function findCar(mj: Mujoco, spec: MjSpec, model: MjModel, prefix = ''): Car {
  const O = mj.mjtObj;
  const id = (type: mjtObj, name: string) => mj.mjs_getId(find(mj, spec, type, prefix + name));
  const body = id(O.mjOBJ_BODY, CAR_BODY);
  const lidar = id(O.mjOBJ_SITE, LIDAR_SITE);
  const joint = model.body_jntadr[body];
  if (model.body_jntnum[body] < 1 || model.jnt_type[joint] !== mj.mjtJoint.mjJNT_FREE.value) {
    throw new Error(`the car's ${CAR_BODY} body has no free joint`);
  }
  return {
    body,
    drive: id(O.mjOBJ_ACTUATOR, DRIVE_ACTUATOR),
    steer: id(O.mjOBJ_ACTUATOR, STEER_ACTUATOR),
    lidar,
    drivetrain: drivetrain(mj, model, id(O.mjOBJ_SENSOR, DRIVE_POS_SENSOR)),
    free: model.jnt_qposadr[joint],
    lidarMount: mount(model, body, lidar),
  };
}

/** Site's place on body, composed through the bodies between them: refused if one has a joint. */
function mount(model: MjModel, body: number, site: number): Car['lidarMount'] {
  let pos: number[] = Array.from(model.site_pos.subarray(3 * site, 3 * site + 3) as Float64Array);
  let quat: number[] = Array.from(model.site_quat.subarray(4 * site, 4 * site + 4) as Float64Array);
  for (let b = model.site_bodyid[site]; b !== body; b = model.body_parentid[b]) {
    if (b === 0) throw new Error('the LiDAR site is not on the car');
    if (model.body_jntnum[b] > 0) throw new Error("a joint lies between the LiDAR site and the car's root body");
    const bq: number[] = Array.from(model.body_quat.subarray(4 * b, 4 * b + 4) as Float64Array);
    const r = rotate(bq, pos);
    pos = r.map((x, k) => x + model.body_pos[3 * b + k]);
    quat = multiply(bq, quat);
  }
  return { pos, quat };
}

/** The product of the quaternions a and b (w, x, y, z), as mju_mulQuat. */
export function multiply(a: ArrayLike<number>, b: ArrayLike<number>): number[] {
  return [
    a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
    a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
    a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
    a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
  ];
}

/** v rotated by the unit quaternion q (w, x, y, z), as mju_rotVecQuat. */
export function rotate(q: ArrayLike<number>, v: ArrayLike<number>): number[] {
  const m = rotation(q);
  return [0, 1, 2].map((r) => m[3 * r] * v[0] + m[3 * r + 1] * v[1] + m[3 * r + 2] * v[2]);
}

/** The rotation of the unit quaternion q (w, x, y, z), row-major, as mju_quat2Mat. */
export function rotation(q: ArrayLike<number>): number[] {
  const [w, x, y, z] = [q[0], q[1], q[2], q[3]];
  return [
    1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
    2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
    2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
  ];
}

/** The fixed tendon whose length the drive_pos sensor reads, as batch.drive_angle reads it: refused if it reads anything
 *  else. */
function drivetrain(mj: Mujoco, model: MjModel, sensor: number): Car['drivetrain'] {
  const refused = new Error("the drive_pos sensor does not read a fixed tendon's length");
  if (model.sensor_type[sensor] !== mj.mjtSensor.mjSENS_TENDONPOS.value) throw refused;
  const tendon = model.sensor_objid[sensor];
  const wraps = Array.from({ length: model.tendon_num[tendon] }, (_, k) => model.tendon_adr[tendon] + k);
  if (wraps.some((w) => model.wrap_type[w] !== mj.mjtWrap.mjWRAP_JOINT.value)) throw refused;
  return {
    qpos: wraps.map((w) => model.jnt_qposadr[model.wrap_objid[w]]),
    coef: wraps.map((w) => model.wrap_prm[w]),
  };
}

/**
 * The drivetrain's angle (rad) at qpos, summed as MuJoCo sums the tendon, so
 * it is drive_pos's reading there, as batch.drive_angle. After mj_step the
 * sensors still hold the state at the step's start; this reads it at the
 * step's end.
 */
export function driveAngle(car: Car, qpos: ArrayLike<number>): number {
  const { qpos: adr, coef } = car.drivetrain;
  let angle = 0;
  for (let k = 0; k < adr.length; k++) angle += coef[k] * qpos[adr[k]];
  return angle;
}

/**
 * A new site at the origin, facing +X. A body attaches to a site or a frame,
 * and the bindings cannot add a frame to the world body.
 */
function origin(mj: Mujoco, spec: MjSpec) {
  const world = mj.mjs_findBody(spec, 'world')!;
  const site = mj.mjs_addSite(world, mj.mjs_getSpecDefault(spec)!)!;
  // The site takes the track's site defaults: undo any pose they give it.
  site.pos.set([0, 0, 0]);
  site.quat.set([1, 0, 0, 0]);
  site.alt.type = mj.mjtOrientation.mjORIENTATION_QUAT;
  site.fromto.set([NaN, NaN, NaN, NaN, NaN, NaN]);
  return site;
}

/** New data for model, stepped for SETTLE_TIME with zero control so the car rests. */
export function settle(mj: Mujoco, model: MjModel): MjData {
  const data = new mj.MjData(model);
  const steps = settleSteps(model.opt.timestep);
  for (let i = 0; i < steps; i++) mj.mj_step(model, data);
  return data;
}

/** The steps of timestep (s) in SETTLE_TIME, half a step rounding up: sim.settle_steps in Python. */
export function settleSteps(timestep: number): number {
  return Math.floor(SETTLE_TIME / timestep + 0.5);
}
