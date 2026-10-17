"""
The NeoRacer, part by part: the one place that says what the car is.

Each value says where it comes from. ESTIMATED marks a value that nobody has
measured or read off a datasheet, ASSUMED one that a source implies but does
not state, and INCONSISTENT a value whose sources disagree, with the other
values and the reason for the one chosen. To make a variant of the car,
replace parts: dataclasses.replace(NEORACER, battery=Battery(7.4)).

The sources:
  CAD            the production car's CAD model, the Playground's
                 neoracer-race.glb, measured by its tools/measure-car.mjs
  URDF           osracer-description/urdf/osracer.urdf, an older CAD model
                 (neoracer-hardware-files)
  spec           Koneshka's specification, 2026-09-29: about 3 kg, a custom
                 RS545 brushed motor, a 14T pinion on a 48T spur, 85 mm tyres
  docs           the NeoRacer docs (neoracer-docs), hardware/ pages
  firmware       neoracer-pcb-firmware, firmware/source/main: config.h,
                 Kconfig.projbuild and imu.cpp
  driver         neoracer_ros2_driver, config/*.yaml
  measured       measured on the car, 2026-10-16
"""

from .parts import (
    Battery,
    Camera,
    Car,
    Chassis,
    DcMotorRating,
    DriveMotor,
    Esc,
    Imu,
    InertiaReference,
    Lidar,
    MotorEncoder,
    ServoRating,
    SpeedController,
    Steering,
    SteeringServo,
    Suspension,
    Visuals,
    Wheel,
    rewound,
)

# The chassis' centre of mass above the axles (m): the URDF's, ESTIMATED for
# the production car. The IMU is placed there too.
_COM_HEIGHT = 0.0189

# The RS-545SH-4075's datasheet line (slypro.net's RS-545 table): the RS-545
# winding nearest to the docs' free speed.
_RS545SH_4075 = DcMotorRating(voltage=12.0, stall_torque=0.0793, free_speed_rpm=9_300.0)

NEORACER = Car(
    chassis=Chassis(
        # CAD. INCONSISTENT: the docs' chassis page gives 280 mm and the
        # driver's twist_bridge.yaml 0.285 m; the CAD model is the measured
        # production car.
        wheelbase=0.28764,
        track=0.224,  # CAD
        # spec ("about 3 kg"); the docs say under 3 kg. Nobody has weighed it.
        mass=3.0,
        com_height=_COM_HEIGHT,
        # The URDF's base_link, scaled to the chassis' mass. ESTIMATED: the
        # URDF is an older, lighter car (1.6211 kg).
        inertia=InertiaReference(inertia=(0.002377, 0.0058394, 0.0070057), mass=1.6211),
        # CAD: the body, 481 x 241 x 184 mm from the bottom of the chassis tub
        # to the top of the LiDAR. INCONSISTENT: the docs' chassis page gives
        # 380 x 300 x 220 mm; it is not said what those measure.
        box_center=(0.016865, -0.00287, 0.069),
        box_half_size=(0.240635, 0.12042, 0.09198),
    ),
    wheel=Wheel(
        # spec (85 mm) and the firmware's OSCORE_WHEEL_RADIUS_UM (42500).
        # INCONSISTENT: the docs' drivetrain and chassis pages give 80 mm, and
        # the CAD model's tyre is 90.3 mm. Nobody has measured a tyre.
        radius=0.0425,
        width=0.04825,  # CAD
        mass=0.0587,  # URDF, ESTIMATED for the production car
        friction=1.5,  # ESTIMATED
        # ESTIMATED: stands in for every drivetrain loss, and with the motor
        # sets the top speed.
        drive_damping=0.0005,
    ),
    suspension=Suspension(
        # ESTIMATED, with the damping, for a slightly overdamped spring.
        stiffness=600.0,
        damping=25.0,  # ESTIMATED
        droop=0.010,  # ESTIMATED
        jounce=0.015,  # ESTIMATED
    ),
    # The car's 3S LiPo, nominal (the team, 2026-10-16). The OSCORE board
    # takes 9 to 26 V (docs, oscore-board).
    battery=Battery(voltage=11.1),
    motor=DriveMotor(
        # The docs' drivetrain page: about 11,000 rpm with no load, at a
        # voltage it does not give (ASSUMED 12 V, the RS-545 range's rated
        # voltage), and no torque. No RS-545 datasheet lists that speed, so the
        # nearest winding is rewound to it (parts.rewound): ESTIMATED.
        # INCONSISTENT: the RS-545SH's datasheet at 24,000 rpm and 214 mN·m
        # fits the firmware's speed loop better, whose feed-forward reaches
        # 7.35 m/s at full pulse and whose limit is 6 m/s, both beyond this
        # motor's 4.3 m/s free speed; the docs' encoder page rates the encoder
        # to 20,000 rpm. The spec says only "custom RS545".
        rating=rewound(_RS545SH_4075, free_speed_rpm=11_000.0),
        # firmware: OSCORE_GEAR_RATIO_MILLI = 10550. The spec's 14T pinion on a
        # 48T spur is 3.43:1 of it; the rest, presumably the differentials, is
        # not documented.
        gear_ratio=10.55,
    ),
    # ASSUMED: the model's motor brakes at 0 V. Nobody has said whether the
    # car's ESC brakes or coasts at neutral throttle.
    esc=Esc(neutral="brake"),
    servo=SteeringServo(
        # The SPT5435LV-180W's datasheet (spt-servo.com), the car's servo
        # (2026-10-16). INCONSISTENT: the docs' drivetrain page gives 20 kg·cm
        # stall torque, which fits no line of this datasheet.
        low=ServoRating(
            voltage=4.8, stall_torque_kgf_cm=29.0, seconds_per_60_degrees=0.16
        ),
        high=ServoRating(
            voltage=6.0, stall_torque_kgf_cm=35.0, seconds_per_60_degrees=0.14
        ),
        # docs, oscore-board: the servo header runs on VCC_5V_IO, the 5 V rail.
        supply=5.0,
        # ASSUMED: a 180-degree servo over 500 to 2500 us, so 90 degrees over
        # the firmware's 1000 to 2000 us (STEERING_MIN, STEERING_MAX).
        horn_half_range_degrees=45.0,
        # ESTIMATED: picked so the servo settles without chatter at the
        # model's 1/360 s step. Nobody has timed the real servo's swing.
        kp=50.0,
        kd=1.0,
    ),
    steering=Steering(
        # measured: the front wheels stop at 30 degrees. The firmware's
        # STEERING_MAX_ANGLE_DEG (30) and the driver's max_steering_angle_deg
        # (30) name the full pulse 30 degrees, but the driver commands only
        # max_steering of it (below), where the wheels reach their stop.
        inner_lock_degrees=30.0,
        driver_lock=0.625,  # driver: throttle.yaml, max_steering
        fit_range=0.45,  # the model's choice
        wheel_range_margin=0.015,  # the model's choice
        input_body_mass=0.01,  # the model's choice
    ),
    # firmware, config.h. The simulator does not model this loop yet.
    speed_controller=SpeedController(
        period=0.02,  # CONTROL_INTERVAL
        kp=447.0,  # PID_KP_DEFAULT
        ki=4.7,  # PID_KI_DEFAULT
        kd=4.7,  # PID_KD_DEFAULT
        speed_filter_weight=0.15,  # SPEED_LPF_ALPHA
        feed_forward=68.0,  # THROTTLE_FF_GAIN_US_PER_MPS
        deadband=90.0,  # THROTTLE_FF_DEADBAND_DEFAULT_US
        full_pulse=500.0,  # THROTTLE_FF_MAX_US
        max_speed=6.0,  # MAX_SPEED_FORWARD; the driver's max_speed_mps
    ),
    encoder=MotorEncoder(
        # firmware: OSCORE_ENCODER_PPR. The docs' encoder page: a Hall-effect
        # encoder on the motor shaft, A, B and Z, of 1 to 4096 pulses.
        pulses_per_turn=1024,
        max_speed_rpm=20_000.0,  # docs, sensors/encoders
        report_period=0.02,  # firmware: OSCORE_SPEED_CALC_INTERVAL_MS
        # firmware: ODOM_SPEED_LPF_ALPHA, the speed the car reports. Its speed
        # loop filters more (speed_filter_weight above).
        filter_weight=0.95,
    ),
    imu=Imu(
        # ASSUMED at the centre of mass. The IMU is on the OSCORE board; its
        # position on the car is not measured.
        position=(0.0, 0.0, _COM_HEIGHT),
        # docs, sensors/imu: the QMI8658A accelerometer and gyroscope and the
        # QMC6309 magnetometer, published at 200 Hz. The firmware averages 5
        # samples of the chip's 1000 Hz.
        rate=200.0,
        # firmware, imu.cpp: the QMI8658A set to +-4 g and +-1024 degrees/s.
        # The docs give the chip's ranges, 2 to 16 g and 16 to 2048 degrees/s.
        # The model's IMU does not saturate: a crash in the simulator reads
        # far past 4 g.
        accel_range_g=4.0,
        gyro_range_dps=1024.0,
        magnetometer_range_gauss=30.0,  # docs
    ),
    # docs, sensors/lidar: the LakiBeam L1; the driver's lidar.launch.py
    # runs it at 30 Hz.
    lidar=Lidar(
        position=(-0.0442, 0.0, 0.11995),  # CAD
        scan_rate=30.0,
        resolution_degrees=0.25,
        field_degrees=270.0,
        range_bright=25.0,
        range_dark=15.0,
        accuracy=0.02,
    ),
    # docs, sensors/camera. The simulator has no camera yet.
    camera=Camera(
        position=(0.1692, 0.0, 0.06265),  # CAD
        width=1920,
        height=1200,
        fps=120.0,
        fov_degrees=130.0,
        focal_length=0.0027,
        shutter="global",
        stream_width=640,  # driver, camera.yaml; the docs recommend the same
        stream_height=480,
        stream_fps=60.0,
    ),
    # The cosmetic meshes are the URDF car's, whose wheels are 50 mm.
    visuals=Visuals(mesh_wheel_radius=0.05),
)
