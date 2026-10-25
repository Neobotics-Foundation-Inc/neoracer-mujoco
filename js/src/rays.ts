/**
 * Rays against a compiled scene's geoms, cast in JavaScript: the distances
 * that mj_multiRay finds, without a call into WebAssembly for each ray or a
 * test of every geom. A ray tests only the geoms over the cells of a 1 m grid
 * on the ground plane that it passes, nearest first, and stops at the first
 * cell beyond its nearest hit. The NeoRacer's LiDAR casts 120 rays a physics
 * step: in Node, about 50 us on loop_corridor's 23 geoms, where mj_multiRay
 * takes 70, and about 50 us on a scene of 7300 boxes, where it takes 3500.
 *
 * Like mj_multiRay with every geom group and static geoms included, a ray
 * does not see the geoms of the excluded body or an invisible geom (alpha 0),
 * and it meets a plane only from its front, within its size when that is not
 * zero. It meets a mesh's triangles. The geoms welded to the world, and not
 * to a mocap body, stay where the scene was built; the others are read from
 * data at each update.
 *
 * Frame: +X forward, +Y left, +Z up.
 */

import type { MjData, MjModel } from '@mujoco/mujoco';

import type { Mujoco } from './scene.js';

/** A side (m) of the grid's cells on the ground plane. */
const CELL = 1;

/** Below this, a ray's component along an axis is taken as zero, as MuJoCo's mjMINVAL. */
const MINVAL = 1e-15;

/** One geom, as a ray meets it in the geom's own frame. */
interface Shape {
  readonly geom: number;
  readonly type: number;
  readonly size: Float64Array;
  /** A mesh's vertices (geom frame, 3 a vertex) and triangles (3 vertex indices each). */
  readonly mesh: { readonly vert: Float64Array; readonly face: Int32Array } | null;
  /** The geom's center (world). */
  readonly pos: Float64Array;
  /** World from geom, row-major. */
  readonly mat: Float64Array;
  readonly rbound: number;
  /** The geom's bounding box on the ground plane (world): its center's x and y, and its half sizes. */
  readonly box: readonly [number, number, number, number];
}

/** The geom types a RayScene meets, by their mjtGeom names and values. */
const PLANE = 0, SPHERE = 2, CYLINDER = 5, BOX = 6, MESH = 7;
const SUPPORTED = { mjGEOM_PLANE: PLANE, mjGEOM_SPHERE: SPHERE, mjGEOM_CYLINDER: CYLINDER, mjGEOM_BOX: BOX, mjGEOM_MESH: MESH };

export class RayScene {
  /** Static geoms with a bounding sphere, on the grid. */
  private readonly fixed: Shape[] = [];
  /** Static planes, which no cell bounds. */
  private readonly planes: Shape[] = [];
  /** The geoms that move, posed by update. */
  private readonly moving: Shape[] = [];
  private readonly x0: number;
  private readonly y0: number;
  private readonly w: number;
  private readonly h: number;
  private readonly cells: number[][];
  /** The ray each fixed geom was last tested for: a geom sits in several cells. */
  private readonly tested: Uint32Array;
  private ray = 0;

  /**
   * The geoms of model at the poses of data (after mj_forward or mj_step),
   * but those of body exclude and the invisible ones. Throw for a geom of a
   * type it does not meet: an ellipsoid, capsule, height field or SDF.
   */
  constructor(mj: Mujoco, model: MjModel, data: MjData, exclude: number) {
    for (const [name, value] of Object.entries(SUPPORTED)) {
      const actual = mj.mjtGeom[name as keyof typeof SUPPORTED].value;
      if (actual !== value) throw new Error(`this MuJoCo's ${name} is ${actual}, not ${value}`);
    }
    const supported = new Set(Object.values(SUPPORTED));
    const rgba = model.geom_rgba, matid = model.geom_matid, matRgba = model.mat_rgba;
    const weld = model.body_weldid;
    for (let g = 0; g < model.ngeom; g++) {
      const body = model.geom_bodyid[g];
      if (body === exclude) continue;
      const alpha = matid[g] >= 0 ? matRgba[4 * matid[g] + 3] : rgba[4 * g + 3];
      if (alpha === 0) continue;
      const type = model.geom_type[g];
      if (!supported.has(type)) {
        const name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM.value, g) ?? `geom ${g}`;
        throw new Error(`RayScene meets ${Object.keys(SUPPORTED).join(', ')} geoms; ${name} is of type ${type}`);
      }
      // a mocap body is its own weld root, as a body with a joint is
      const shape = this.shape(model, data, g);
      if (weld[body] !== 0) this.moving.push(shape);
      else if (type === PLANE) this.planes.push(shape);
      else this.fixed.push(shape);
    }
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const { box: [x, y, hx, hy] } of this.fixed) {
      minX = Math.min(minX, x - hx);
      maxX = Math.max(maxX, x + hx);
      minY = Math.min(minY, y - hy);
      maxY = Math.max(maxY, y + hy);
    }
    const none = this.fixed.length === 0;
    this.x0 = none ? 0 : Math.floor(minX);
    this.y0 = none ? 0 : Math.floor(minY);
    this.w = none ? 0 : Math.max(1, Math.ceil((maxX - this.x0) / CELL));
    this.h = none ? 0 : Math.max(1, Math.ceil((maxY - this.y0) / CELL));
    this.cells = Array.from({ length: this.w * this.h }, () => []);
    this.fixed.forEach(({ box: [x, y, hx, hy] }, k) => {
      const i0 = Math.max(0, Math.floor((x - hx - this.x0) / CELL));
      const i1 = Math.min(this.w - 1, Math.floor((x + hx - this.x0) / CELL));
      const j0 = Math.max(0, Math.floor((y - hy - this.y0) / CELL));
      const j1 = Math.min(this.h - 1, Math.floor((y + hy - this.y0) / CELL));
      for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++) this.cells[j * this.w + i].push(k);
    });
    this.tested = new Uint32Array(this.fixed.length);
  }

  private shape(model: MjModel, data: MjData, g: number): Shape {
    let mesh: Shape['mesh'] = null;
    if (model.geom_type[g] === MESH) {
      const m = model.geom_dataid[g];
      const v0 = model.mesh_vertadr[m], nv = model.mesh_vertnum[m];
      const f0 = model.mesh_faceadr[m], nf = model.mesh_facenum[m];
      mesh = {
        vert: Float64Array.from(model.mesh_vert.subarray(3 * v0, 3 * (v0 + nv))),
        face: Int32Array.from(model.mesh_face.subarray(3 * f0, 3 * (f0 + nf))),
      };
    }
    const pos = Float64Array.from(data.geom_xpos.subarray(3 * g, 3 * g + 3));
    const mat = Float64Array.from(data.geom_xmat.subarray(9 * g, 9 * g + 9));
    // geom_aabb: the box's center and half sizes in the geom's frame
    const a = model.geom_aabb.subarray(6 * g, 6 * g + 6);
    const half = (r: number) => Math.abs(mat[3 * r]) * a[3] + Math.abs(mat[3 * r + 1]) * a[4] + Math.abs(mat[3 * r + 2]) * a[5];
    const at = (r: number) => pos[r] + mat[3 * r] * a[0] + mat[3 * r + 1] * a[1] + mat[3 * r + 2] * a[2];
    return {
      geom: g,
      type: model.geom_type[g],
      size: Float64Array.from(model.geom_size.subarray(3 * g, 3 * g + 3)),
      mesh,
      pos,
      mat,
      rbound: model.geom_rbound[g],
      box: [at(0), at(1), half(0), half(1)],
    };
  }

  /** The moving geoms at the poses of data. */
  update(data: MjData): void {
    for (const s of this.moving) {
      s.pos.set(data.geom_xpos.subarray(3 * s.geom, 3 * s.geom + 3));
      s.mat.set(data.geom_xmat.subarray(9 * s.geom, 9 * s.geom + 9));
    }
  }

  /**
   * The distance (m) along the unit ray from origin to its first geom, or
   * Infinity if none lies within maxRange.
   */
  cast(origin: ArrayLike<number>, dir: ArrayLike<number>, maxRange: number): number {
    let best = this.castFixed(origin, dir, maxRange);
    for (const s of this.planes) best = this.hit(s, origin, dir, best);
    for (const s of this.moving) best = this.hit(s, origin, dir, best);
    return best < maxRange ? best : Infinity;
  }

  /** The nearest fixed geom's distance, or best: the cells under the ray in order (Amanatides and Woo's walk). */
  private castFixed(o: ArrayLike<number>, d: ArrayLike<number>, best: number): number {
    if (this.w === 0) return best;
    let tIn = 0, tOut = best;
    const lo = [this.x0, this.y0], hi = [this.x0 + this.w * CELL, this.y0 + this.h * CELL];
    for (let a = 0; a < 2; a++) {
      if (Math.abs(d[a]) < MINVAL) {
        if (o[a] < lo[a] || o[a] > hi[a]) return best;
        continue;
      }
      const t1 = (lo[a] - o[a]) / d[a], t2 = (hi[a] - o[a]) / d[a];
      tIn = Math.max(tIn, Math.min(t1, t2));
      tOut = Math.min(tOut, Math.max(t1, t2));
    }
    if (tIn > tOut) return best;
    if (++this.ray === 0xffffffff) {
      this.tested.fill(0);
      this.ray = 1;
    }
    const clamp = (v: number, n: number) => Math.min(n - 1, Math.max(0, v));
    let i = clamp(Math.floor((o[0] + d[0] * tIn - this.x0) / CELL), this.w);
    let j = clamp(Math.floor((o[1] + d[1] * tIn - this.y0) / CELL), this.h);
    const si = d[0] > 0 ? 1 : -1, sj = d[1] > 0 ? 1 : -1;
    // the ray's t at the next cell edge across x and across y, and from one edge to the next
    let tx = Math.abs(d[0]) < MINVAL ? Infinity : (this.x0 + (i + (si > 0 ? 1 : 0)) * CELL - o[0]) / d[0];
    let ty = Math.abs(d[1]) < MINVAL ? Infinity : (this.y0 + (j + (sj > 0 ? 1 : 0)) * CELL - o[1]) / d[1];
    const dx = Math.abs(d[0]) < MINVAL ? Infinity : CELL / Math.abs(d[0]);
    const dy = Math.abs(d[1]) < MINVAL ? Infinity : CELL / Math.abs(d[1]);
    for (;;) {
      for (const k of this.cells[j * this.w + i]) {
        if (this.tested[k] === this.ray) continue;
        this.tested[k] = this.ray;
        best = this.hit(this.fixed[k], o, d, best);
      }
      if (Math.min(tx, ty) >= best) return best;
      if (tx < ty) {
        i += si;
        tx += dx;
        if (i < 0 || i >= this.w) return best;
      } else {
        j += sj;
        ty += dy;
        if (j < 0 || j >= this.h) return best;
      }
    }
  }

  /** min(best, the ray's distance to shape s). */
  private hit(s: Shape, o: ArrayLike<number>, d: ArrayLike<number>, best: number): number {
    const px = o[0] - s.pos[0], py = o[1] - s.pos[1], pz = o[2] - s.pos[2];
    if (s.type !== PLANE) {
      // the bounding sphere: missed, or no nearer than best
      const tc = -(px * d[0] + py * d[1] + pz * d[2]);
      const miss2 = px * px + py * py + pz * pz - tc * tc;
      if (miss2 > s.rbound * s.rbound || tc + s.rbound < 0 || tc - s.rbound > best) return best;
    }
    // the ray in the geom's frame: mat transposed
    const m = s.mat;
    const p0 = m[0] * px + m[3] * py + m[6] * pz;
    const p1 = m[1] * px + m[4] * py + m[7] * pz;
    const p2 = m[2] * px + m[5] * py + m[8] * pz;
    const v0 = m[0] * d[0] + m[3] * d[1] + m[6] * d[2];
    const v1 = m[1] * d[0] + m[4] * d[1] + m[7] * d[2];
    const v2 = m[2] * d[0] + m[5] * d[1] + m[8] * d[2];
    if (s.type === BOX) {
      // most geoms of a track: the faces inline, nearest first
      const z = s.size;
      let t = best;
      t = boxFaces(t, p0, v0, z[0], p1, v1, z[1], p2, v2, z[2]);
      t = boxFaces(t, p1, v1, z[1], p2, v2, z[2], p0, v0, z[0]);
      return boxFaces(t, p2, v2, z[2], p0, v0, z[0], p1, v1, z[1]);
    }
    const t = this.local(s, p0, p1, p2, v0, v1, v2);
    return t >= 0 && t < best ? t : best;
  }

  /** The ray's first distance to s in s's frame, or -1. */
  private local(s: Shape, p0: number, p1: number, p2: number, v0: number, v1: number, v2: number): number {
    const z = s.size;
    switch (s.type) {
      case PLANE: {
        if (v2 > -MINVAL) return -1;
        const t = -p2 / v2;
        if (t < 0) return -1;
        const x = p0 + t * v0, y = p1 + t * v1;
        return (z[0] <= 0 || Math.abs(x) <= z[0]) && (z[1] <= 0 || Math.abs(y) <= z[1]) ? t : -1;
      }
      case SPHERE: {
        const a = v0 * v0 + v1 * v1 + v2 * v2, b = p0 * v0 + p1 * v1 + p2 * v2;
        const c = p0 * p0 + p1 * p1 + p2 * p2 - z[0] * z[0], det = b * b - a * c;
        if (a < MINVAL || det < 0) return -1;
        const r = Math.sqrt(det), near = (-b - r) / a;
        return near >= 0 ? near : (-b + r) / a;
      }
      case CYLINDER: {
        let best = -1;
        if (Math.abs(v2) > MINVAL) {
          for (let side = -1; side <= 1; side += 2) {
            const t = (side * z[1] - p2) / v2;
            const x = p0 + t * v0, y = p1 + t * v1;
            if (t >= 0 && x * x + y * y <= z[0] * z[0] && (best < 0 || t < best)) best = t;
          }
        }
        const a = v0 * v0 + v1 * v1, b = p0 * v0 + p1 * v1, c = p0 * p0 + p1 * p1 - z[0] * z[0];
        const det = b * b - a * c;
        if (a < MINVAL || det < 0) return best;
        const r = Math.sqrt(det);
        for (let sign = -1; sign <= 1; sign += 2) {
          const t = (-b + sign * r) / a;
          if (t >= 0 && Math.abs(p2 + t * v2) <= z[1] && (best < 0 || t < best)) best = t;
        }
        return best;
      }
      case MESH: {
        const { vert, face } = s.mesh!;
        let best = -1;
        for (let f = 0; f < face.length; f += 3) {
          const t = triangle(vert, face[f], face[f + 1], face[f + 2], p0, p1, p2, v0, v1, v2);
          if (t >= 0 && (best < 0 || t < best)) best = t;
        }
        return best;
      }
      default:
        return -1;
    }
  }
}

/**
 * min(best, the ray's distance to the box's two faces across one axis): p
 * and v are the ray's origin and direction along that axis, s the box's half
 * size along it, and likewise for the two other axes.
 */
function boxFaces(
  best: number,
  p: number,
  v: number,
  s: number,
  pj: number,
  vj: number,
  sj: number,
  pk: number,
  vk: number,
  sk: number,
): number {
  if (Math.abs(v) <= MINVAL) return best;
  const near = (-s - p) / v, far = (s - p) / v;
  if (near >= 0 && near < best && Math.abs(pj + near * vj) <= sj && Math.abs(pk + near * vk) <= sk) best = near;
  if (far >= 0 && far < best && Math.abs(pj + far * vj) <= sj && Math.abs(pk + far * vk) <= sk) best = far;
  return best;
}

/** The ray's distance to the triangle of vertices a, b and c of vert (Moller and Trumbore), or -1. */
function triangle(
  vert: Float64Array,
  a: number,
  b: number,
  c: number,
  p0: number,
  p1: number,
  p2: number,
  v0: number,
  v1: number,
  v2: number,
): number {
  const ax = vert[3 * a], ay = vert[3 * a + 1], az = vert[3 * a + 2];
  const e1x = vert[3 * b] - ax, e1y = vert[3 * b + 1] - ay, e1z = vert[3 * b + 2] - az;
  const e2x = vert[3 * c] - ax, e2y = vert[3 * c + 1] - ay, e2z = vert[3 * c + 2] - az;
  const hx = v1 * e2z - v2 * e2y, hy = v2 * e2x - v0 * e2z, hz = v0 * e2y - v1 * e2x;
  const det = e1x * hx + e1y * hy + e1z * hz;
  if (Math.abs(det) < MINVAL) return -1;
  const sx = p0 - ax, sy = p1 - ay, sz = p2 - az;
  const u = (sx * hx + sy * hy + sz * hz) / det;
  if (u < 0 || u > 1) return -1;
  const qx = sy * e1z - sz * e1y, qy = sz * e1x - sx * e1z, qz = sx * e1y - sy * e1x;
  const w = (v0 * qx + v1 * qy + v2 * qz) / det;
  if (w < 0 || u + w > 1) return -1;
  return (e2x * qx + e2y * qy + e2z * qz) / det;
}
