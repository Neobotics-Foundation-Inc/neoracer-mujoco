# Recorded runs

Runs of the real cars that `neoracer_mujoco.fit` fits the specs' values to, a folder per car and a folder per run.

`racecar_neo_v2/` holds the two full-lock circles of the calibration reference car, from calibration_reference_car_2026-10 (2026-10-05): the car drives straight at a speed command of 0.25, then takes a full-lock steering command, to the right in one run and to the left in the other.

A run is three tables. Their times are in seconds from the first speed command, and each table begins one second before it and ends 2.5 s after the last command.

- `commands.csv`: `t,speed,steering`, a row where the command changes. Both commands are in [-1, 1], the steering positive to the left.
- `odometry.csv`: `t,speed,yaw_rate`, the car's odometry: the encoder's speed (m/s) and the yaw rate (rad/s).
- `imu.csv`: `t,ax,ay,az,gx,gy,gz`, the IMU's acceleration (m/s^2) and angular velocity (rad/s) along the car's x (forward), y (left) and z (up). The recordings have them in the camera's axes; these are turned to the car's.
