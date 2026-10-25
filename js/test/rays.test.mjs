/**
 * RayScene against MuJoCo's own rays: on a scene with every geom type it
 * meets, it finds the distances that mj_multiRay finds.
 */

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

import loadMujoco from '@mujoco/mujoco';

import { RayScene, compose } from '../dist/index.js';

const mj = await loadMujoco();
const loop = readFileSync(new URL('../../assets/tracks/loop_corridor.xml', import.meta.url), 'utf8');

/** loop_corridor with one geom of each type a RayScene meets, an invisible box, and a mocap box. */
const SHAPES = `
  <asset>
    <mesh name="cone" vertex="0 0 .5  .2 0 0  .1 .17 0  -.1 .17 0  -.2 0 0  -.1 -.17 0  .1 -.17 0"/>
    <mesh name="deck" vertex="0 0 .1  .8 0 .2  .8 .5 .2  0 .5 .1  0 0 0  .8 0 0  .8 .5 0  0 .5 0"/>
  </asset>
  <worldbody>
    <geom name="cone" type="mesh" mesh="cone" pos="3.5 .5 0"/>
    <geom name="deck" type="mesh" mesh="deck" pos="-3.8 -.5 0" euler="0 0 20"/>
    <geom name="tilted" type="box" size=".2 .1 .15" pos="0 2.5 .2" euler="10 20 30"/>
    <geom name="post" type="cylinder" size=".1 .3" pos="-1 -2.5 .3" euler="0 30 0"/>
    <geom name="ball" type="sphere" size=".15" pos="1 -2.5 .2"/>
    <geom name="ramp" type="plane" size=".5 .4 .05" pos="-3.5 2.5 .1" euler="15 0 0"/>
    <geom name="ghost" type="box" size=".3 .3 .3" pos="0 -2.5 .3" rgba="1 1 1 0"/>
    <body name="other" mocap="true" pos="2 2.5 .1">
      <geom name="other" type="box" size=".24 .14 .1"/>
    </body>
  </worldbody>`;

const xml = loop.replace(/<\/mujoco>\s*$/, `${SHAPES}</mujoco>`);

/** n rays from random points over the track, in random directions, most near the horizontal. */
function rays(n, seed) {
  let s = seed;
  const rand = () => ((s = (s * 1103515245 + 12345) % 2 ** 31) / 2 ** 31);
  return Array.from({ length: n }, () => {
    const o = [8 * rand() - 4, 6 * rand() - 3, 0.02 + 0.5 * rand()];
    const yaw = 2 * Math.PI * rand(), pitch = rand() < 0.8 ? 0.2 * (rand() - 0.5) : Math.PI * (rand() - 0.5);
    return { o, d: [Math.cos(pitch) * Math.cos(yaw), Math.cos(pitch) * Math.sin(yaw), Math.sin(pitch)] };
  });
}

/**
 * mj_multiRay's first geom and distance for each ray, or -1 and Infinity, as
 * the LiDAR casts: every group, static geoms, one body excluded.
 */
function mujocoRays(model, data, all, exclude, maxRange) {
  return all.map(({ o, d }) => {
    const geom = new mj.IntBuffer(1), dist = new mj.DoubleBuffer(1);
    try {
      mj.mj_multiRay(model, data, o, d, null, true, exclude, geom, dist, null, 1, maxRange);
      const g = geom.GetView()[0], t = dist.GetView()[0];
      return g >= 0 && t <= maxRange ? { geom: g, t } : { geom: -1, t: Infinity };
    } finally {
      geom.delete();
      dist.delete();
    }
  });
}

const sameDistance = (t, want) => t === want || Math.abs(t - want) < 1e-9;

test('a RayScene finds the distances that mj_multiRay finds', () => {
  const { model, car } = compose(mj, xml);
  const data = new mj.MjData(model);
  mj.mj_forward(model, data);
  const exclude = model.site_bodyid[car.lidar];
  const scene = new RayScene(mj, model, data, exclude);
  const all = rays(5000, 7);
  const want = mujocoRays(model, data, all, exclude, 10);
  all.forEach(({ o, d }, i) => {
    const t = scene.cast(o, d, 10);
    assert.ok(sameDistance(t, want[i].t), `ray ${i}: ${t} against ${want[i].t}`);
  });
  const hit = new Set(want.map((w) => mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM.value, w.geom)));
  for (const name of ['cone', 'deck', 'tilted', 'post', 'ball', 'ramp', 'other', 'floor', 'outer_top']) {
    assert.ok(hit.has(name), `no ray met ${name}`);
  }
});

test('a RayScene sees a moving geom where data has it after update', () => {
  const { model, car } = compose(mj, xml);
  const data = new mj.MjData(model);
  mj.mj_forward(model, data);
  const exclude = model.site_bodyid[car.lidar];
  const scene = new RayScene(mj, model, data, exclude);
  const mocap = model.body_mocapid[mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY.value, 'other')];
  data.mocap_pos.set([-2, 2.5, 0.1], 3 * mocap);
  mj.mj_forward(model, data);
  scene.update(data);
  const all = rays(2000, 11);
  const want = mujocoRays(model, data, all, exclude, 10);
  all.forEach(({ o, d }, i) => {
    const t = scene.cast(o, d, 10);
    assert.ok(sameDistance(t, want[i].t), `ray ${i}: ${t} against ${want[i].t}`);
  });
});

test('a ray that grazes a sphere meets it where mj_multiRay does', () => {
  const { model, car } = compose(mj, xml);
  const data = new mj.MjData(model);
  mj.mj_forward(model, data);
  const exclude = model.site_bodyid[car.lidar];
  const scene = new RayScene(mj, model, data, exclude);
  const ball = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM.value, 'ball');
  const c = data.geom_xpos.subarray(3 * ball, 3 * ball + 3), r = model.geom_size[3 * ball];
  const dirs = rays(4000, 13).map(({ d }) => d);
  // each ray across the sphere's normal at a point of it, from 1 m back
  const all = Array.from({ length: dirs.length / 2 }, (_, k) => {
    const [n, u] = [dirs[2 * k], dirs[2 * k + 1]];
    const x = [n[1] * u[2] - n[2] * u[1], n[2] * u[0] - n[0] * u[2], n[0] * u[1] - n[1] * u[0]];
    const d = x.map((v) => v / Math.hypot(...x));
    return { o: [0, 1, 2].map((i) => c[i] + r * n[i] - d[i]), d };
  });
  const want = mujocoRays(model, data, all, exclude, 10);
  all.forEach(({ o, d }, i) => {
    const t = scene.cast(o, d, 10);
    assert.ok(sameDistance(t, want[i].t), `ray ${i}: ${t} against ${want[i].t}`);
  });
  assert.ok(want.filter((w) => w.geom === ball).length > all.length / 2);
});

test('a RayScene over a scene kilometers wide finds the distances that mj_multiRay finds', () => {
  const far = loop.replace(
    /<\/mujoco>\s*$/,
    `<worldbody>
      <geom name="east" type="box" size=".5 .5 .5" pos="100000 100000 .5"/>
      <geom name="west" type="box" size=".5 .5 .5" pos="-100000 -100000 .5"/>
    </worldbody></mujoco>`,
  );
  const { model, car } = compose(mj, far);
  const data = new mj.MjData(model);
  mj.mj_forward(model, data);
  const exclude = model.site_bodyid[car.lidar];
  const scene = new RayScene(mj, model, data, exclude);
  // rays over the track, and rays at each far box from a few meters off
  const away = [1, -1].flatMap((s) =>
    rays(200, 19).map(({ o }) => {
      const p = [s * 99995 + o[0], s * 99995 + o[1], o[2]];
      const v = [s * 100000 - p[0], s * 100000 - p[1], 0.5 - p[2]];
      return { o: p, d: v.map((x) => x / Math.hypot(...v)) };
    }),
  );
  const all = [...rays(1000, 17), ...away];
  const want = mujocoRays(model, data, all, exclude, 20);
  all.forEach(({ o, d }, i) => {
    const t = scene.cast(o, d, 20);
    assert.ok(sameDistance(t, want[i].t), `ray ${i}: ${t} against ${want[i].t}`);
  });
  assert.ok(want.slice(-away.length).every((w) => w.geom >= 0));
});

test('a RayScene refuses a geom of a type it does not meet', () => {
  const capsule = loop.replace(/<\/mujoco>\s*$/, '<worldbody><geom type="capsule" size=".1 .2"/></worldbody></mujoco>');
  const { model, car } = compose(mj, capsule);
  const data = new mj.MjData(model);
  mj.mj_forward(model, data);
  assert.throws(() => new RayScene(mj, model, data, model.site_bodyid[car.lidar]), /capsule|type/);
});

/** loop_corridor with a box whose bounds lie on the grid's cell edges, x and y in [1, 2]. */
const edged = loop.replace(/<\/mujoco>\s*$/, '<worldbody><geom name="cell" type="box" size=".5 .5 .5" pos="1.5 1.5 .5"/></worldbody></mujoco>');

test("a ray through a cell's corner meets a geom whose bound lies on the corner", () => {
  const { model, car } = compose(mj, edged);
  const data = new mj.MjData(model);
  mj.mj_forward(model, data);
  const exclude = model.site_bodyid[car.lidar];
  const scene = new RayScene(mj, model, data, exclude);
  const ray = { o: [0, 2, 0.5], d: [Math.SQRT1_2, -Math.SQRT1_2, 0] };
  const [want] = mujocoRays(model, data, [ray], exclude, 10);
  assert.equal(mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM.value, want.geom), 'cell');
  assert.ok(sameDistance(scene.cast(ray.o, ray.d, 10), want.t));
});

test('a hit at exactly maxRange counts, as for mj_multiRay, and a nearer limit misses it', () => {
  const { model, car } = compose(mj, edged);
  const data = new mj.MjData(model);
  mj.mj_forward(model, data);
  const scene = new RayScene(mj, model, data, model.site_bodyid[car.lidar]);
  // the floor plane 1 m below, and the box's face 1 m ahead
  for (const { o, d } of [{ o: [0, 0, 1], d: [0, 0, -1] }, { o: [0, 1.5, 0.5], d: [1, 0, 0] }]) {
    assert.equal(scene.cast(o, d, 1), 1, JSON.stringify(o));
    assert.equal(scene.cast(o, d, 0.999), Infinity, JSON.stringify(o));
  }
});
