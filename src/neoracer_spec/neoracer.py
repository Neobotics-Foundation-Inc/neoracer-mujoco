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
    Spec,
    SpeedController,
    Steering,
    SteeringServo,
    Suspension,
    Visuals,
    Wheel,
    rewound,
)

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
SPT5435 = Source("datasheet", "the SPT5435LV-180W's datasheet, spt-servo.com")

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
            note="the firmware's OSCORE_WHEEL_RADIUS_UM (42500) agrees; nobody"
            " has measured a tyre",
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
            " beyond this motor's 4.3 m/s free speed",
            conflicts=(Alternative(DcMotorRating(12.0, 0.214, 24_000.0), RS545SH),),
        ),
        gear_ratio=documented(
            10.55,
            FIRMWARE_KCONFIG,
            note="OSCORE_GEAR_RATIO_MILLI = 10550; the spec's 14T pinion on a 48T"
            " spur is 3.43:1 of it, and the rest, presumably the differentials, is"
            " not documented",
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
                voltage=4.8, stall_torque_kgf_cm=29.0, seconds_per_60_degrees=0.16
            ),
            SPT5435,
            note="the car's servo (the team, 2026-10-16)",
        ),
        high=documented(
            ServoRating(
                voltage=6.0, stall_torque_kgf_cm=35.0, seconds_per_60_degrees=0.14
            ),
            SPT5435,
            note="the docs' 20 kg·cm stall torque fits no line of this datasheet",
            conflicts=(
                Alternative(
                    ServoRating(
                        voltage=6.0,
                        stall_torque_kgf_cm=20.0,
                        seconds_per_60_degrees=0.14,
                    ),
                    DOCS_DRIVETRAIN,
                ),
            ),
        ),
        supply=documented(5.0, DOCS_OSCORE, note="the servo header runs on VCC_5V_IO"),
        horn_half_range_degrees=assumed(
            45.0,
            SPT5435,
            "a 180-degree servo over 500 to 2500 us, so 90 degrees over the"
            " firmware's 1000 to 2000 us",
        ),
        kp=chosen(50.0, "so the servo settles without chatter at the 1/360 s step"),
        kd=chosen(1.0, "so the servo settles without chatter at the 1/360 s step"),
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
        rate=documented(
            200.0,
            DOCS_IMU,
            note="the QMI8658A and the QMC6309; the firmware averages 5 samples of"
            " the chip's 1000 Hz",
        ),
        accel_range_g=documented(
            4.0,
            FIRMWARE_IMU,
            note="the model's IMU does not saturate: a crash in the simulator"
            " reads far past 4 g",
            conflicts=(Alternative(16.0, DOCS_IMU),),
        ),
        gyro_range_dps=documented(
            1024.0, FIRMWARE_IMU, conflicts=(Alternative(2048.0, DOCS_IMU),)
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
