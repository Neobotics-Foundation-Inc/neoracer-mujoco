/**
 * A car's firmware and sensors around each physics step, as the racecar_core
 * adapter's MujocoRacecar runs them in Python: before a step, the speed
 * loop's throttle and the steering command go to the actuators; after it,
 * the encoder, the speed loop and the LiDAR count the step. They read the car
 * from qpos, so a caller steps with mj_step and runs no mj_kinematics. The
 * geoms that move are where data has them, which after mj_step is the step's
 * start: another car's lags by a step.
 *
 *     let car = startCar(mj, NEORACER_CAR, scene, data);
 *     car = { ...car, command: speedAngle(limits, NEORACER_CAR.speedLoop, 0.25, -0.2) };
 *     for (;;) {
 *       beforeStep(scene, data, car);
 *       mj.mj_step(scene.model, data);
 *       car = afterStep(NEORACER_CAR, scene, data, car, scene.model.opt.timestep);
 *     }
 */

import type { MjData } from '@mujoco/mujoco';

import { type SpeedAngle, applyCommand } from './drive.js';
import { ENCODER, type EncoderParams, type EncoderState, startEncoder, stepEncoder } from './encoder.js';
import { LIDAR, type LidarParams, type LidarState, startLidar, sweepLidar } from './lidar.js';
import type { RayScene } from './rays.js';
import { type Mujoco, type Scene, driveAngle } from './scene.js';
import { SPEED_LOOP, type SpeedLoopParams, type SpeedLoopState, startSpeedLoop, stepSpeedLoop } from './speedLoop.js';

/** What a car runs beside its model: its LiDAR, its encoder and its firmware's speed loop. */
export interface CarParams {
  readonly lidar: LidarParams;
  readonly encoder: EncoderParams;
  readonly speedLoop: SpeedLoopParams;
}

/** The NeoRacer's: Python's neoracer_params(). */
export const NEORACER_CAR: CarParams = { lidar: LIDAR, encoder: ENCODER, speedLoop: SPEED_LOOP };

/** A car between two physics steps. */
export interface CarState {
  /** The program's last command (speedAngle): the speed for the loop to hold and the steering. */
  readonly command: SpeedAngle;
  readonly loop: SpeedLoopState;
  readonly encoder: EncoderState;
  /** Its scan is what rc.lidar.get_samples() reports, in m. */
  readonly lidar: LidarState;
}

/**
 * A car at rest at the pose of data, asked for nothing. The LiDAR's rays are
 * met by rays, or by a new RayScene of the scene, as startLidar's.
 */
export function startCar(mj: Mujoco, params: CarParams, scene: Scene, data: MjData, rays?: RayScene): CarState {
  const angle = driveAngle(scene.car, data.qpos);
  return {
    command: { target: 0, steer: 0 },
    loop: startSpeedLoop(params.speedLoop, angle),
    encoder: startEncoder(params.encoder, angle),
    lidar: startLidar(mj, params.lidar, scene, data, rays),
  };
}

/** Before a physics step: the speed loop's throttle and the steering command into data.ctrl. */
export function beforeStep(scene: Scene, data: MjData, car: CarState): void {
  applyCommand(scene, data, { throttle: car.loop.throttle, steer: car.command.steer });
}

/** After a physics step of dt (s) that left data: the encoder, the speed loop and the LiDAR count it. */
export function afterStep(params: CarParams, scene: Scene, data: MjData, car: CarState, dt: number): CarState {
  const angle = driveAngle(scene.car, data.qpos);
  return {
    command: car.command,
    loop: stepSpeedLoop(params.speedLoop, car.loop, car.command.target, angle, dt),
    encoder: stepEncoder(params.encoder, car.encoder, angle, dt),
    lidar: sweepLidar(params.lidar, scene, data, car.lidar, dt),
  };
}
