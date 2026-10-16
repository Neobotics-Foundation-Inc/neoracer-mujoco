// The names of the car's parts, as contract.py gives them. A Python test
// (validation/test_js_package.py) checks that the two agree.
//
// Frame: +X forward, +Y left, +Z up.

export const CAR_BODY = 'car'; // the car's root body, with the free joint
export const DRIVE_ACTUATOR = 'motor'; // voltage at the drive motor (V), + = forward
export const STEER_ACTUATOR = 'steer_servo'; // steering target angle (rad), + = left
export const LIDAR_SITE = 'lidar_mount'; // the centre of the spinning LiDAR
export const DRIVE_POS_SENSOR = 'drive_pos'; // the drivetrain's angle (rad)
