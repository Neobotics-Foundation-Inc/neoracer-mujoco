"""
The NeoRacer, part by part: the one place that says what the car is.

Every value names its source and how we know it (provenance.py). Where
sources disagree, the other values are its conflicts and its note says why
this one was chosen. provenance.walk lists them all. To make a variant of the
car, replace parts:

    dataclasses.replace(NEORACER, battery=Battery(voltage=assumed(7.4, ...)))
"""

from .provenance import (
    Alternative,
    Source,
    assumed,
    calibrated,
    chosen,
    documented,
    estimated,
    measured,
)
from .spec import (
    Battery,
    Box,
    Camera,
    Chassis,
    DcMotorRating,
    DriveMotor,
    Esc,
    Imu,
    InertiaReference,
    Lidar,
    MotorEncoder,
    Resolution,
    ServoRating,
    ServoSteps,
    Spec,
    SpeedController,
    Steering,
    SteeringServo,
    Suspension,
    Visuals,
    Wheel,
    rewound,
)

# The sources of NEORACER's values.
CAD = Source(
    "CAD",
    "the production car's CAD model: the Playground's neoracer-race.glb,"
    " measured by its tools/measure-car.mjs",
)
URDF = Source(
    "URDF",
    "neoracer-hardware-files, osracer-description/urdf/osracer.urdf:"
    " an older, lighter CAD model of the car",
)
SPEC = Source(
    "spec",
    "Koneshka's specification, 2026-09-29: about 3 kg, a custom RS545 brushed"
    " motor, a 14T pinion on a 48T spur, 85 mm tyres",
)
TEAM = Source("team", "measured or stated by the team, 2026-10-16")
TEAM_RATES = Source(
    "team",
    "measured on the car by the team, 2026-07-21: /imu/fused and /odom at 200 Hz",
)
OSCORE_MANUAL = Source(
    "manual", "OSCORE manual, table 4.2.1: VCC_ESC = VCC_IN through, 9 to 26 V"
)
DOCS_CHASSIS = Source("docs", "neoracer-docs, hardware/chassis-and-cad")
DOCS_DRIVETRAIN = Source("docs", "neoracer-docs, hardware/drivetrain")
DOCS_OSCORE = Source("docs", "neoracer-docs, hardware/oscore-board")
DOCS_ENCODERS = Source("docs", "neoracer-docs, hardware/sensors/encoders")
DOCS_IMU = Source("docs", "neoracer-docs, hardware/sensors/imu")
DOCS_LIDAR = Source("docs", "neoracer-docs, hardware/sensors/lidar")
DOCS_CAMERA = Source("docs", "neoracer-docs, hardware/sensors/camera")
FIRMWARE = Source("firmware", "neoracer-pcb-firmware, firmware/source/main/config.h")
FIRMWARE_KCONFIG = Source(
    "firmware", "neoracer-pcb-firmware, firmware/source/main/Kconfig.projbuild"
)
FIRMWARE_IMU = Source("firmware", "neoracer-pcb-firmware, firmware/source/main/imu.cpp")
DRIVER_THROTTLE = Source("driver", "neoracer_ros2_driver, config/throttle.yaml")
DRIVER_TWIST = Source("driver", "neoracer_ros2_driver, config/twist_bridge.yaml")
DRIVER_CAMERA = Source("driver", "neoracer_ros2_driver, config/camera.yaml")
DRIVER_LIDAR = Source("driver", "neoracer_ros2_driver, launch/lidar.launch.py")
RS545_TABLE = Source("datasheet", "slypro.net's RS-545 table: the RS-545SH-4075")
RS545SH = Source("datasheet", "the RS-545SH at 12 V: 24,000 rpm, 214 mN·m")
SERVO_DATASHEET = Source(
    "datasheet",
    "the car's servo's datasheet, a 20 kg waterproof high-torque digital servo"
    " (the team, 2026-10-22)",
)
SERVO_SPEC = Source(
    "team",
    "the team's specification for a replacement servo, 2026-10-22: 500 to"
    " 2500 us for 180 degrees of travel, 1500 us neutral, 50 Hz",
)
TEAM_SERVO = Source("team", "stated by the team, 2026-10-22")
STEPS_AT_REST = Source(
    "steps at rest",
    "the team's steering steps on a NeoRacer at rest on the floor, 2026-10-22:"
    " the driver's full lock written to the OSCORE board at speed 0, three"
    " cycles of centre, left, right, centre, timed from each command by the"
    " IMU at 200 Hz",
)
SERVO_FIT = Source(
    "model fit",
    "fitted in the MuJoCo model, assets/neoracer.xml, to the steps at rest;"
    " validation/test_spec.py holds the model to them",
)

# The RS-545 winding nearest to the docs' free speed.
_RS545SH_4075 = DcMotorRating(voltage=12.0, stall_torque=0.0793, free_speed_rpm=9_300.0)

# The chassis' centre of mass above the axles (m). The IMU is assumed there.
_COM_HEIGHT = 0.0189

NEORACER = Spec(
    chassis=Chassis(
        wheelbase=measured(
            0.28764,
            CAD,
            note="the CAD model is the measured production car",
            conflicts=(
                Alternative(0.280, DOCS_CHASSIS),
                Alternative(0.285, DRIVER_TWIST),
            ),
        ),
        track=measured(0.224, CAD),
        mass=documented(
            3.0,
            SPEC,
            note="about 3 kg, and the docs say under 3 kg; nobody has weighed it",
        ),
        com_height=estimated(_COM_HEIGHT, "the URDF's, an older car's"),
        inertia=estimated(
            InertiaReference(inertia=(0.002377, 0.0058394, 0.0070057), mass=1.6211),
            "the URDF's base_link, scaled to the chassis' mass; the URDF is an"
            " older, lighter car",
        ),
        collision_box=measured(
            Box(
                center=(0.016865, -0.00287, 0.069),
                half_size=(0.240635, 0.12042, 0.09198),
            ),
            CAD,
            note="the body, 481 x 241 x 184 mm, from the bottom of the chassis tub"
            " to the top of the LiDAR; the docs' 380 x 300 x 220 mm do not say"
            " what they measure",
        ),
    ),
    wheel=Wheel(
        radius=documented(
            0.0425,
            SPEC,
            note="the firmware's OSCORE_WHEEL_RADIUS_UM (42500) agrees, and Koneshka"
            " confirmed the 85 mm tyre on 2026-08-22; the docs' 80 mm is out of date",
            conflicts=(
                Alternative(0.040, DOCS_DRIVETRAIN),
                Alternative(0.040, DOCS_CHASSIS),
                Alternative(0.04515, CAD),
            ),
        ),
        width=measured(0.04825, CAD),
        mass=estimated(0.0587, "the URDF's wheels"),
        friction=estimated(1.5, "no tyre has been tested"),
        drive_damping=estimated(
            0.0005,
            "stands in for every drivetrain loss, and with the motor sets the"
            " top speed",
        ),
    ),
    suspension=Suspension(
        stiffness=estimated(600.0, "for a slightly overdamped spring, with damping"),
        damping=estimated(25.0, "for a slightly overdamped spring, with stiffness"),
        droop=estimated(0.010, "not measured"),
        jounce=estimated(0.015, "not measured"),
    ),
    battery=Battery(
        voltage=documented(
            11.1,
            TEAM,
            note="a 3S LiPo, nominal; the OSCORE board takes 9 to 26 V (docs,"
            " oscore-board)",
        )
    ),
    motor=DriveMotor(
        rating=assumed(
            rewound(_RS545SH_4075, free_speed_rpm=11_000.0),
            DOCS_DRIVETRAIN,
            "the docs give about 11,000 rpm with no load but no voltage and no"
            " torque; taken at 12 V, the RS-545 range's rated voltage, and the"
            " nearest RS-545 winding rewound to it (spec.rewound). The firmware's"
            " speed loop reaches 7.35 m/s at full pulse and limits at 6 m/s, both"
            " beyond this motor's 4.3 m/s free speed. The website claims up to 25"
            " km/h (6.9 m/s), which needs about 16,500 rpm at the motor through"
            " the 10.55 reduction and the 42.5 mm tyre radius, and 6 m/s needs 14,200",
            conflicts=(Alternative(DcMotorRating(12.0, 0.214, 24_000.0), RS545SH),),
        ),
        gear_ratio=documented(
            10.55,
            FIRMWARE_KCONFIG,
            note="OSCORE_GEAR_RATIO_MILLI = 10550: the 14T pinion on the 48T spur"
            " (Koneshka confirmed the spur on 2026-08-25) is 3.43:1, and the"
            " differentials the remaining 3.08:1; their tooth counts are not documented",
        ),
    ),
    esc=Esc(
        neutral=estimated(
            "brake",
            "the model's motor brakes at 0 V; nobody has said whether the ESC"
            " brakes or coasts",
        )
    ),
    servo=SteeringServo(
        low=documented(
            ServoRating(
                voltage=5.0, stall_torque_kgf_cm=17.0, seconds_per_60_degrees=0.13
            ),
            SERVO_DATASHEET,
        ),
        high=documented(
            ServoRating(
                voltage=7.4, stall_torque_kgf_cm=20.0, seconds_per_60_degrees=0.10
            ),
            SERVO_DATASHEET,
            note="the docs' 20 kg·cm (hardware/drivetrain) is this line",
        ),
        supply=documented(
            6.0,
            TEAM_SERVO,
            note="the ESC's BEC: on the OSCORE schematic its red wire alone feeds"
            " VCC_ESC, the servo's power pin",
            conflicts=(Alternative(5.0, DOCS_OSCORE), Alternative(11.1, OSCORE_MANUAL)),
        ),
        horn_half_range_degrees=documented(
            45.0,
            SERVO_SPEC,
            note="180 degrees over 500 to 2500 us, so 90 degrees over the"
            " firmware's 1000 to 2000 us",
        ),
        kp=chosen(50.0, "so the servo settles without chatter at the 1/360 s step"),
        kd=chosen(1.0, "so the servo settles without chatter at the 1/360 s step"),
        latency=calibrated(
            0.045,
            SERVO_FIT,
            "the steps at rest: their first motion and the end of each swing,"
            " with the servo at its datasheet rating at 6 V",
            note="from the command written to the board. A racecar_core"
            " program's command also waits for the driver's mux, which resends"
            " it every 0.02 s (mux.yaml), 0.01 s on average, which this leaves out",
        ),
        steps_at_rest=measured(
            ServoSteps(
                first_motion=0.059,
                centre_to_lock=0.175,
                lock_to_lock=0.229,
                lock_to_centre=0.157,
            ),
            STEPS_AT_REST,
            note="first motion: the IMU's yaw rate or sideways acceleration leaves"
            " four times its noise; a swing ends where the yaw rate the scrubbing"
            " tyres give the chassis falls under a quarter of its peak. With the"
            " front lifted off the floor the first motion is the same, so the"
            " delay is not the tyres'",
        ),
    ),
    steering=Steering(
        inner_lock_degrees=measured(
            30.0,
            TEAM,
            note="the front wheels stop there; the firmware's and the driver's"
            " 30 degrees name the full pulse, of which the driver commands only"
            " driver_lock",
        ),
        driver_lock=documented(0.625, DRIVER_THROTTLE, note="max_steering"),
        fit_range=chosen(0.45, "wider than full lock"),
        wheel_range_margin=chosen(0.015, "so the joint limits never bind"),
        input_body_mass=chosen(0.01, "a light virtual body"),
        armature=estimated(
            0.002,
            "tuned in simulation, not measured: the servo's rotor inertia and"
            " gear ratio are not known. Of 0.0002, 0.002 and 0.005, the smallest"
            " that stops the car shimmying at 15 to 30 Hz in part-throttle turns"
            " at the 1/360 s step; 0.0002 left the servo-held steering ringing"
            " near 60 Hz, faster than the step resolves",
        ),
    ),
    speed_controller=SpeedController(
        period=documented(0.02, FIRMWARE, note="CONTROL_INTERVAL"),
        kp=documented(447.0, FIRMWARE, note="PID_KP_DEFAULT"),
        ki=documented(4.7, FIRMWARE, note="PID_KI_DEFAULT"),
        kd=documented(4.7, FIRMWARE, note="PID_KD_DEFAULT"),
        speed_filter_weight=documented(0.15, FIRMWARE, note="SPEED_LPF_ALPHA"),
        feed_forward=documented(68.0, FIRMWARE, note="THROTTLE_FF_GAIN_US_PER_MPS"),
        deadband=documented(90.0, FIRMWARE, note="THROTTLE_FF_DEADBAND_DEFAULT_US"),
        full_pulse=documented(500.0, FIRMWARE, note="THROTTLE_FF_MAX_US"),
        max_speed=documented(
            6.0,
            FIRMWARE,
            note="MAX_SPEED_FORWARD; the driver's max_speed_mps agrees",
        ),
    ),
    encoder=MotorEncoder(
        pulses_per_turn=documented(
            1024,
            FIRMWARE_KCONFIG,
            note="OSCORE_ENCODER_PPR; the docs: a Hall-effect encoder on the motor"
            " shaft, A, B and Z, of 1 to 4096 pulses",
        ),
        max_speed_rpm=documented(20_000.0, DOCS_ENCODERS),
        report_period=documented(
            0.02, FIRMWARE_KCONFIG, note="OSCORE_SPEED_CALC_INTERVAL_MS"
        ),
        filter_weight=documented(
            0.95,
            FIRMWARE,
            note="ODOM_SPEED_LPF_ALPHA, the speed the car reports; the speed loop"
            " filters more (speed_filter_weight)",
        ),
    ),
    imu=Imu(
        position=assumed(
            (0.0, 0.0, _COM_HEIGHT),
            DOCS_IMU,
            "at the centre of mass; the IMU is on the OSCORE board, whose place"
            " on the car is not measured",
        ),
        rate=measured(
            200.0,
            TEAM_RATES,
            note="the QMI8658A and the QMC6309; the firmware averages 5 samples of"
            " the chip's 1000 Hz, and the docs give the same 200 Hz",
        ),
        accel_range_g=documented(
            4.0,
            FIRMWARE_IMU,
            note="the firmware's scale, of the chip's 2 to 16 g (docs); the"
            " model's IMU does not saturate: a crash in the simulator reads far"
            " past 4 g",
        ),
        gyro_range_dps=documented(
            1024.0,
            FIRMWARE_IMU,
            note="the firmware's scale, of the chip's 16 to 2048 degrees/s (docs)",
        ),
        magnetometer_range_gauss=documented(30.0, DOCS_IMU),
    ),
    lidar=Lidar(
        position=measured((-0.0442, 0.0, 0.11995), CAD),
        scan_rate=documented(
            30.0, DOCS_LIDAR, note="the LakiBeam L1; the driver runs it at 30 Hz"
        ),
        resolution_degrees=documented(0.25, DOCS_LIDAR),
        field_degrees=documented(270.0, DOCS_LIDAR),
        range_bright=documented(25.0, DOCS_LIDAR),
        range_dark=documented(15.0, DOCS_LIDAR),
        accuracy=documented(0.02, DOCS_LIDAR),
    ),
    camera=Camera(
        position=measured((0.1692, 0.0, 0.06265), CAD),
        sensor=documented(Resolution(width=1920, height=1200, fps=120.0), DOCS_CAMERA),
        fov_degrees=documented(130.0, DOCS_CAMERA),
        focal_length=documented(0.0027, DOCS_CAMERA),
        shutter=documented("global", DOCS_CAMERA),
        stream=documented(
            Resolution(width=640, height=480, fps=60.0),
            DRIVER_CAMERA,
            note="the docs recommend the same",
        ),
    ),
    visuals=Visuals(
        mesh_wheel_radius=measured(
            0.05, URDF, note="the cosmetic meshes are the URDF car's"
        )
    ),
)
