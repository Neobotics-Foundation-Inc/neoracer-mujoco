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
  /** The address of drive_pos in sensordata. */
  readonly drivePos: number;
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
  return {
    body: id(O.mjOBJ_BODY, CAR_BODY),
    drive: id(O.mjOBJ_ACTUATOR, DRIVE_ACTUATOR),
    steer: id(O.mjOBJ_ACTUATOR, STEER_ACTUATOR),
    lidar: id(O.mjOBJ_SITE, LIDAR_SITE),
    drivePos: model.sensor_adr[id(O.mjOBJ_SENSOR, DRIVE_POS_SENSOR)],
  };
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
