# neoracer-mujoco for JavaScript

This package runs the NeoRacer car in MuJoCo's WebAssembly build (`@mujoco/mujoco`, the same MuJoCo version as the Python package) in a browser or in Node. It carries the car's MJCF, puts the car on a track that you give as MJCF, and drives it and reads its encoder and LiDAR the way the Python package does. Its tests run the same cases in both and compare the results.

```js
import loadMujoco from '@mujoco/mujoco';
import {
  ENCODER, LIDAR, actuatorLimits, applyCommand, compose, settle, speedAngle,
  startEncoder, startLidar, stepEncoder, sweepLidar,
} from 'neoracer-mujoco';

const mj = await loadMujoco();
const scene = compose(mj, trackXml);           // the car at the origin, facing +X
const data = settle(mj, scene.model);         // at rest on the track
const dt = scene.model.opt.timestep;          // 1/360 s
const angle = () => data.sensordata[scene.car.drivePos];
let encoder = startEncoder(ENCODER, angle());
let lidar = startLidar(mj, LIDAR, scene, data);

// racecar_core's rc.drive.set_speed_angle(0.5, -0.2): half throttle, a little left.
applyCommand(scene, data, speedAngle(actuatorLimits(scene.model, scene.car), 0.5, -0.2));
for (let i = 0; i < 360; i++) {
  mj.mj_step(scene.model, data);
  encoder = stepEncoder(ENCODER, encoder, angle(), dt);
  lidar = sweepLidar(mj, LIDAR, scene, data, lidar, dt);
}
// encoder.speed is rc.physics.get_encoder_speed() in m/s; lidar.scan holds
// 1440 distances in m, clockwise from the front, -1 for no return.
```

The scene has no meshes that only show the car: render the car with your own model of it. To build and test it, run `npm ci` and then, from the repository root with the Python package installed, `python js/test/fixtures.py js/test/fixtures.json` and `npm test` in `js/`.
