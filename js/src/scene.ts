// A NeoRacer car on a track, compiled in MuJoCo's WebAssembly build, as
// assets.compose(track, visuals=False) makes it in Python.
//
// Frame: +X forward, +Y left, +Z up.

import type { MainModule, MjData, MjModel, MjSpec, MjsElement, mjtObj } from '@mujoco/mujoco';

import {
  CAR_BODY,
  DRIVE_ACTUATOR,
  DRIVE_POS_SENSOR,
  LIDAR_SITE,
  STEER_ACTUATOR,
} from './contract.js';
import { CAR_XML } from './generated/car.js';

export type Mujoco = MainModule;
export type Position = readonly [number, number, number]; // m, world frame

/** The parts of the car, as indices into the compiled model. */
export interface Car {
  readonly body: number; // the car's root body
  readonly drive: number; // the drive motor's actuator
  readonly steer: number; // the steering servo's actuator
  readonly lidar: number; // the LiDAR site
  readonly drivePos: number; // the address of drive_pos in sensordata
}

/** A compiled scene: spec is what model was compiled from. */
export interface Scene {
  readonly spec: MjSpec;
  readonly model: MjModel;
  readonly car: Car;
}

/** Steps of zero control that bring a new car to rest, as sim.SETTLE_STEPS. */
export const SETTLE_STEPS = 400;

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
 * Compile the track's MJCF with the car attached at pos (m), facing +X, and
 * without the meshes that only show the car. The track keeps its own floor
 * and <option>. carXml defaults to assets/neoracer.xml.
 */
export function compose(
  mj: Mujoco,
  trackXml: string,
  pos: Position = [0, 0, 0],
  carXml: string = CAR_XML,
): Scene {
  const spec = mj.parseXMLString(trackXml);
  const car = mj.parseXMLString(carXml);
  deleteVisuals(mj, car);
  // The car's root default class is "main", as the track's is: a name of
  // its own keeps the two apart, as in assets.compose.
  mj.mjs_setName(mj.mjs_getSpecDefault(car)!.element, 'car');
  // A body attaches to a site or a frame, and the bindings cannot add a
  // frame to the world body, so the car hangs from a site at pos.
  const world = mj.mjs_findBody(spec, 'world')!;
  const mount = mj.mjs_addSite(world, mj.mjs_getSpecDefault(spec)!)!;
  // The site takes the track's site defaults: undo any pose they give it,
  // so the car stands at pos facing +X.
  mount.pos.set(pos);
  mount.quat.set([1, 0, 0, 0]);
  mount.alt.type = mj.mjtOrientation.mjORIENTATION_QUAT;
  mount.fromto.set([NaN, NaN, NaN, NaN, NaN, NaN]);
  const body = mj.mjs_findBody(car, CAR_BODY);
  if (!body) throw new Error(`the car has no "${CAR_BODY}" body`);
  if (!mj.mjs_attach(mount.element, body.element, '', '')) throw new Error(mj.mjs_getError(spec));
  const model = mj.mj_compile(spec);
  if (!model) throw new Error(mj.mjs_getError(spec));
  const O = mj.mjtObj;
  const id = (type: mjtObj, name: string) => mj.mjs_getId(find(mj, spec, type, name));
  return {
    spec,
    model,
    car: {
      body: id(O.mjOBJ_BODY, CAR_BODY),
      drive: id(O.mjOBJ_ACTUATOR, DRIVE_ACTUATOR),
      steer: id(O.mjOBJ_ACTUATOR, STEER_ACTUATOR),
      lidar: id(O.mjOBJ_SITE, LIDAR_SITE),
      drivePos: model.sensor_adr[id(O.mjOBJ_SENSOR, DRIVE_POS_SENSOR)],
    },
  };
}

/** New data for model, stepped SETTLE_STEPS with zero control so the car rests. */
export function settle(mj: Mujoco, model: MjModel): MjData {
  const data = new mj.MjData(model);
  for (let i = 0; i < SETTLE_STEPS; i++) mj.mj_step(model, data);
  return data;
}
