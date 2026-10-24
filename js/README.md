# neoracer-mujoco for JavaScript

This package runs the NeoRacer car in MuJoCo's WebAssembly build (`@mujoco/mujoco`, the same MuJoCo version as the Python package) in a browser or in Node. It carries the car's MJCF, puts the car on a track that you give as MJCF, and drives it and reads its encoder and LiDAR the way the Python package does. Its tests run the same cases in both and compare the results.

Install it with the `npm install` command at the top of the notes of the [latest release](https://github.com/Neobotics-Foundation-Inc/neoracer-mujoco/releases/latest). npm installs the `@mujoco/mujoco` that it names with it.

```js
import loadMujoco from '@mujoco/mujoco';
import {
  ENCODER, LIDAR, actuatorLimits, applyCommand, compose, settle, speedAngle,
  startEncoder, startLidar, stepEncoder, sweepLidar,
} from 'neoracer-mujoco';

const mj = await loadMujoco();
const scene = compose(mj, trackXml);           // the car at the track's start site
const { model } = scene;
const data = settle(mj, model);               // at rest on the track
const dt = model.opt.timestep;                // s
const angle = () => data.sensordata[scene.car.drivePos];

// After mj_step, the sensors and poses are those before its integration.
// So run each step in two halves, as the Python adapter does: mj_step1
// brings them up to data.time, the encoder and LiDAR count the step that led
// there, the program sets its command, and mj_step2 integrates.
mj.mj_step1(model, data);
let encoder = startEncoder(ENCODER, angle());
let lidar = startLidar(mj, LIDAR, scene, data);
for (let i = 0; i < Math.round(1 / dt); i++) { // 1 s
  // racecar_core's rc.drive.set_speed_angle(0.5, -0.2): half throttle, a little left.
  applyCommand(scene, data, speedAngle(actuatorLimits(model, scene.car), 0.5, -0.2));
  mj.mj_step2(model, data);
  mj.mj_step1(model, data);
  encoder = stepEncoder(ENCODER, encoder, angle(), dt);
  lidar = sweepLidar(mj, LIDAR, scene, data, lidar, dt);
}
// encoder.speed is rc.physics.get_encoder_speed() in m/s; lidar.scan holds a
// distance in m a slot, clockwise from the front, -1 for no return.
```

The scene has no meshes that only show the car: render the car with your own model of it.

For several cars in one scene, attach each behind a prefix of its own, compile once, and find each car's parts by its prefix:

```js
import { CAR_XML, attachCar, findCar } from 'neoracer-mujoco';

const spec = mj.parseXMLString(trackXml);
const prefixes = ['', 'car1_', 'car2_'];
for (const prefix of prefixes) attachCar(mj, spec, mj.parseXMLString(CAR_XML), prefix);
const model = mj.mj_compile(spec);
const cars = prefixes.map((prefix) => findCar(mj, spec, model, prefix));
```

`spec` holds the car's specification, typed and written from the Python package's: `spec.NEORACER`, each value with its source and status, and `spec.NEORACER_PARAMS`, the parameters derived from it.

The package's spec, the names it shares with the Python package and the car's MJCF are written from the Python package into `src/generated/`, which is not checked in. To build and test the package, run `npm ci` in `js/`, then, from the repository root with the Python package installed:

```sh
python -m neoracer_mujoco.javascript js/src/generated
python js/test/fixtures.py js/test/fixtures.json
npm test --prefix js
```
