// The names of the car's parts, as contract.py gives them. A Python test
// (validation/test_js_package.py) checks that the two agree.
//
// Frame: +X forward, +Y left, +Z up.

export const CAR_BODY = 'car'; // the car's root body, with the free joint
export const DRIVE_ACTUATOR = 'motor'; // voltage at the drive motor (V), + = forward
export const STEER_ACTUATOR = 'steer_servo'; // steering target angle (rad), + = left
export const LIDAR_SITE = 'lidar_mount'; // the centre of the spinning LiDAR
export const DRIVE_POS_SENSOR = 'drive_pos'; // the drivetrain's angle (rad)
export const IMU_ACCEL_SENSOR = 'imu_accel'; // specific force at the IMU (m/s^2), car frame
export const IMU_GYRO_SENSOR = 'imu_gyro'; // angular velocity at the IMU (rad/s), car frame
export const WHEEL_GEOMS = ['fl_wheel', 'fr_wheel', 'rl_wheel', 'rr_wheel'] as const; // the geoms that touch the ground
