// The JS package against the Python package: the same cases, from
// test/fixtures.json, which test/fixtures.py writes.

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

import loadMujoco from '@mujoco/mujoco';

import {
  CAR_BODY,
  CAR_XML,
  DRIVE_POS_SENSOR,
  ENCODER,
  IMU_ACCEL_SENSOR,
  IMU_GYRO_SENSOR,
  LIDAR,
  LIDAR_SITE,
  WHEEL_GEOMS,
  actuatorLimits,
  applyCommand,
  compose,
  settle,
  speedAngle,
  startEncoder,
  startLidar,
  stepEncoder,
  sweepLidar,
} from '../dist/index.js';

const fixtures = JSON.parse(readFileSync(new URL('./fixtures.json', import.meta.url), 'utf8'));
const trackXml = readFileSync(new URL('../../assets/tracks/loop_corridor.xml', import.meta.url), 'utf8');
const mj = await loadMujoco();
const scene = compose(mj, trackXml);
const { model, car } = scene;

const maxDiff = (a, b) => {
  assert.equal(a.length, b.length);
  return Math.max(...Array.from(a, (x, i) => Math.abs(x - b[i])));
};

test('the composed scene has the parts and the step of the Python one', () => {
  for (const [name, value] of Object.entries(fixtures.sizes)) assert.equal(model[name], value, name);
  assert.equal(model.opt.timestep, fixtures.timestep);
});

test("the contract's names are parts of the composed car", () => {
  const O = mj.mjtObj;
  const parts = [
    [O.mjOBJ_BODY, CAR_BODY],
    [O.mjOBJ_SITE, LIDAR_SITE],
    [O.mjOBJ_SENSOR, DRIVE_POS_SENSOR],
    [O.mjOBJ_SENSOR, IMU_ACCEL_SENSOR],
    [O.mjOBJ_SENSOR, IMU_GYRO_SENSOR],
    ...WHEEL_GEOMS.map((name) => [O.mjOBJ_GEOM, name]),
  ];
  for (const [type, name] of parts) assert.ok(mj.mj_name2id(model, type.value, name) >= 0, name);
});

test('the car settles where it settles in Python', () => {
  assert.ok(maxDiff(settle(mj, model).qpos, fixtures.settled_qpos) < 1e-12);
});

test('a second of driving ends where it ends in Python, and the encoder agrees', () => {
  const data = settle(mj, model);
  const angle = () => data.sensordata[car.drivePos];
  let state = startEncoder(ENCODER, angle());
  applyCommand(scene, data, { throttle: fixtures.drive.throttle, steer: fixtures.drive.steer });
  for (let i = 0; i < fixtures.drive.steps; i++) {
    mj.mj_step(model, data);
    state = stepEncoder(ENCODER, state, angle(), model.opt.timestep);
  }
  // WebAssembly and native code may round differently; a second drifts by
  // about 1e-14 m.
  assert.ok(maxDiff(data.qpos, fixtures.drive.qpos) < 1e-9);
  assert.ok(Math.abs(state.speed - fixtures.drive.encoder_speed) < 1e-9);
});

test('a turn swept from 13 poses reads what Python reads', () => {
  const data = new mj.MjData(model);
  const pose = (qpos) => {
    data.qpos.set(qpos);
    mj.mj_forward(model, data);
  };
  pose(fixtures.sweep.poses[0]);
  let state = startLidar(mj, LIDAR, scene, data);
  for (const qpos of fixtures.sweep.poses.slice(1)) {
    pose(qpos);
    state = sweepLidar(mj, LIDAR, scene, data, state, model.opt.timestep);
  }
  assert.ok(maxDiff(state.scan, fixtures.sweep.scan) < 1e-12);
  assert.equal(state.head, fixtures.sweep.head);
});

test('the encoder reports the speeds that Python reports', () => {
  const { angles, speeds } = fixtures.encoder;
  let state = startEncoder(ENCODER, angles[0]);
  const out = angles.slice(1).map((a) => (state = stepEncoder(ENCODER, state, a, 1 / 360)).speed);
  assert.deepEqual(out, speeds);
});

test('set_speed_angle writes the controls that the Python adapter writes', () => {
  const limits = actuatorLimits(model, car);
  for (const c of fixtures.commands) {
    const data = new mj.MjData(model);
    applyCommand(scene, data, speedAngle(limits, c.speed, c.angle, c.max_speed));
    assert.equal(data.ctrl[car.drive], c.drive);
    assert.equal(data.ctrl[car.steer], c.steer);
  }
});

test("the car stands where it is put, facing +X, whatever the track's site defaults", () => {
  const defaults = [
    '<site quat="0 0 0 1"/>',
    '<site euler="0 0 1.2"/>',
    '<site fromto="0 0 0 1 1 1" size=".01"/>',
  ];
  for (const site of defaults) {
    const xml = trackXml.replace('<default>', `<default>${site}`);
    assert.notEqual(xml, trackXml);
    const placed = compose(mj, xml, [1, -2, 0]);
    const data = new mj.MjData(placed.model);
    mj.mj_forward(placed.model, data);
    const b = placed.car.body;
    assert.deepEqual(Array.from(data.xquat.subarray(4 * b, 4 * b + 4), (x) => Math.round(x * 1e9) / 1e9), [1, 0, 0, 0], site);
    const xy = Array.from(data.xpos.subarray(3 * b, 3 * b + 2));
    assert.ok(Math.abs(xy[0] - 1) < 1e-9 && Math.abs(xy[1] + 2) < 1e-9, site);
  }
});

test('a LiDAR that is not a scan is refused at the start', () => {
  const data = settle(mj, model);
  for (const bad of [{ slots: 1.5 }, { slots: 0 }, { scanRate: 0 }, { maxRange: -1 }, { field: 7 }]) {
    assert.throws(() => startLidar(mj, { ...LIDAR, ...bad }, scene, data), Error, JSON.stringify(bad));
  }
});

test('an encoder that the firmware cannot be is refused at the start', () => {
  for (const bad of [{ window: 0 }, { window: -0.02 }, { alpha: 0 }, { alpha: -1 }, { alpha: 1.5 }]) {
    assert.throws(() => startEncoder({ ...ENCODER, ...bad }, 0), Error, JSON.stringify(bad));
  }
});

test('a car whose ranges are not symmetric about 0 has no limits, but takes direct commands', () => {
  for (const [actuator, range] of [['steer_servo', '-.2 .4'], ['motor', '0 7.4']]) {
    const pattern = new RegExp(`(<[a-z]+ name="${actuator}"[^>]*?ctrlrange=")[^"]*(")`, 's');
    const xml = CAR_XML.replace(pattern, `$1${range}$2`);
    assert.notEqual(xml, CAR_XML, actuator);
    const odd = compose(mj, trackXml, [0, 0, 0], xml);
    assert.throws(() => actuatorLimits(odd.model, odd.car), /ctrlrange/, actuator);
    // A direct command needs no symmetric range, as sim.apply_command.
    const data = new mj.MjData(odd.model);
    applyCommand(odd, data, { throttle: 0.5, steer: 0.1 });
    assert.equal(data.ctrl[odd.car.steer], 0.1);
    assert.equal(data.ctrl[odd.car.drive], 0.5 * odd.model.actuator_ctrlrange[2 * odd.car.drive + 1]);
  }
});
