"""
The RACECAR Neo V2, part by part, from its measurement worksheet and its
calibration reference.

The values come from these sources:
- the RACECAR Neo V2 Model Measurement Worksheet: two measurements of the
  car (its C.L and M.Z columns), and the values of a model of the car that
  it checks;
- calibration_reference_car_2026-10: the calibration tools' runs on one
  reference car (2026-10-04 and 05), the parameter files they wrote, and
  the recordings of the runs;
- RacecarSim, the RACECAR Neo simulator in Unity: its model of the car's
  shape, and its estimates of the car's inertia, springs and tyres;
- the payload's CAD model: the shape and the layout of everything on the
  chassis' body posts, which the measured heights of the LiDAR and the
  camera place on the car. RacecarSim's model of the payload is this one,
  6 mm higher. The chassis has no CAD model;
- the MuJoCo model of the car, assets/racecar_neo_v2.xml: what it needs to
  drive as the reference car drove (FIT).

The value is the calibration reference's where it has one, and otherwise
the C.L measurement. Where the M.Z measurement, the model or an earlier
calibration disagree, they are its conflicts. The worksheet also lists
values it computes from these, such as the wheel angles at full lock from
the turning radius, which are left out here.

These sources measure what the car does, not its motor's and servo's
ratings, and the car's firmware holds a speed rather than a voltage. So its
parts are its own types, not those of spec.py, and derive_v2.py turns them
into its MuJoCo model.

For the designer of RacecarSim's model of the car, racecarsim_questions
lists where that model disagrees with the car, as conflicts from
RACECARSIM_MODEL or from MODEL, the worksheet's column that checks it, and
the values the spec takes from it unchecked or corrects.

Frame: the car's base_link, on the floor below the middle of the rear axle
(REP-103): +X forward, +Y left, +Z up. Values are in SI units unless a
field's name says otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .provenance import (
    Alternative,
    Assumed,
    Calibrated,
    Source,
    Sourced,
    assumed,
    calibrated,
    chosen,
    documented,
    estimated,
    measured,
    walk,
)
from .spec import Vec3


@dataclass(frozen=True)
class RacecarNeoV2:
    """The car, part by part. RACECAR_NEO_V2, below, holds its values."""

    chassis: Chassis
    footprint: Footprint
    shape: Shape
    wheel: Wheel
    suspension: Suspension
    steering: Steering
    drive: Drive
    odometry: Odometry
    lidar: Lidar
    imu: Imu
    camera: Camera


@dataclass(frozen=True)
class Chassis:
    wheelbase: Sourced[float]  # m, between the front and rear axles
    track: Sourced[float]  # m, between the rear wheels' centres
    mass: Sourced[float]  # kg, with the battery
    # The share of the weight on the front wheels. The centre of mass is
    # this share of the wheelbase ahead of the rear axle, and laterally centred.
    front_weight_share: Sourced[float]
    com_height: Sourced[float]  # m, of the centre of mass above the floor
    # kg·m², the whole car's principal moments about its centre of mass:
    # roll (about X), pitch (Y), yaw (Z)
    inertia: Sourced[Vec3]
    ride_height: Sourced[float]  # m, of the chassis above the floor, moving


@dataclass(frozen=True)
class Footprint:
    """The car's outline."""

    front: Sourced[float]  # m, x of the front-most point
    rear: Sourced[float]  # m, x of the rear-most point
    left: Sourced[float]  # m, y of the left-most point
    right: Sourced[float]  # m, y of the right-most point
    lidar_top: Sourced[float]  # m, z of the top of the LiDAR
    antenna_top: Sourced[float]  # m, z of the antenna tips, the tallest point
    payload_bottom: Sourced[float]  # m, z of the payload's underside, its floor


@dataclass(frozen=True)
class Shape:
    """What the car's body collides with, and what the LiDAR sees of it.

    The payload is everything on the chassis' body posts. Its values here are
    as RacecarSim's model has it; the car's payload is payload_drop lower."""

    # The chassis less the wheels, and the payload less the antennas, as boxes.
    chassis_boxes: Sourced[tuple[NamedBox, ...]]
    payload_boxes: Sourced[tuple[NamedBox, ...]]
    payload_drop: Sourced[float]  # m
    # Two upright whips at the payload's rear, one each side, within the
    # LiDAR's sweep.
    antenna_x: Sourced[float]  # m, of their axes
    antenna_y: Sourced[float]  # m, of the left one's axis; the right one's is -y
    antenna_bottom: Sourced[float]  # m, z; Footprint.antenna_top is their top
    antenna_diameter: Sourced[float]  # m, at the LiDAR's height


@dataclass(frozen=True)
class NamedBox:
    name: str
    center: Vec3  # m
    half_size: Vec3  # m


@dataclass(frozen=True)
class Wheel:
    diameter: Sourced[float]  # m, the tyre's
    width: Sourced[float]  # m, the tyre's
    axle_height: Sourced[float]  # m, of the rear axle above the floor
    mass: Sourced[float]  # kg, each, with its tyre
    # The model's sliding friction of each axle's tyres on the floor.
    grip: Sourced[Axles[float]]


@dataclass(frozen=True)
class Suspension:
    stiffness: Sourced[Axles[float]]  # N/m, at each wheel
    damping: Sourced[Axles[float]]  # N·s/m, at each wheel
    travel: Sourced[float]  # m, from full droop to full jounce; it rests midway


@dataclass(frozen=True)
class Steering:
    # At full lock, at the middle of the rear axle. The wheels' angles at
    # full lock follow from it and the wheelbase and track.
    min_turn_radius: Sourced[Sides[float]]  # m
    # From the centre to full lock, timed from the yaw rate: an upper bound,
    # since it includes the tyres' lag in following the wheels.
    center_to_lock_time: Sourced[Sides[float]]  # s
    # The curvature of the path that a steering command gives, 1/m, + left.
    response: Sourced[tuple[AtCommand[Span], ...]]
    trim: Sourced[float]  # of command, that the firmware adds to every command
    deadband: Sourced[float]  # of command: smaller changes do nothing
    straight_drift_degrees_per_m: Sourced[Span]  # heading change at command 0, + left
    linkage: Sourced[Linkage]
    # The servo, as a DC motor at the steering input with a position loop.
    servo_stall_torque: Sourced[float]  # N·m
    servo_free_speed: Sourced[float]  # rad/s
    servo_kp: Sourced[float]  # V/rad of the loop, at a supply of servo_supply
    servo_kd: Sourced[float]  # V·s/rad
    servo_supply: Sourced[float]  # V
    latency: Sourced[float]  # s, from a command to the servo


@dataclass(frozen=True)
class Linkage:
    """How a steering command turns the front wheels, as the model has it.
    The command turns a virtual wheel at the middle of the front axle in
    proportion, with a slope of its own each side, until the linkage reaches
    its stop on that side. Ackermann geometry turns the front wheels from
    the virtual one."""

    slope: Sides[float]  # rad per unit of command
    stop: Sides[float]  # rad, the virtual wheel's angle at each stop, > 0


@dataclass(frozen=True)
class Drive:
    """The firmware holds a speed: a drive command is a share of max_speed."""

    max_speed: Sourced[float]  # m/s, the firmware's ceiling, rc.drive.set_max_speed
    speed_floor: Sourced[float]  # m/s, the least speed the firmware sets
    steady_speed: Sourced[tuple[AtCommand[float], ...]]  # m/s, true
    rise_time: Sourced[tuple[AtCommand[float], ...]]  # s, from rest to 90% of it
    stop_distance: Sourced[tuple[AtCommand[float], ...]]  # m, after command 0
    stop_time: Sourced[tuple[AtCommand[float], ...]]  # s, after command 0
    reverse: Sourced[Literal["immediate", "delayed", "double command"]]
    # At command 0 the speed controller lets go and the car slows at
    # coast[0] + coast[1] * speed: its drag brake, and drag that grows with
    # the speed.
    coast: Sourced[tuple[float, float]]  # m/s², 1/s
    # At any other command it pushes the car with gain * (set speed - speed).
    gain: Sourced[float]  # N per m/s
    latency: Sourced[float]  # s, from a command to the speed controller


@dataclass(frozen=True)
class Odometry:
    """The speed and the yaw rate that the firmware reports."""

    min_speed: Sourced[float]  # m/s; below it the reported speed is 0
    speed_scale: Sourced[float]  # true speed / reported speed
    yaw_rate_scale: Sourced[float]  # true yaw rate / reported yaw rate
    # How much more than the true speed the encoder reads at full lock.
    turn_overread: Sourced[Sides[float]]
    telemetry_rate: Sourced[float]  # Hz


@dataclass(frozen=True)
class Lidar:
    position: Sourced[Vec3]  # m, the centre of the scan
    yaw_degrees: Sourced[float]  # of the scan's zero bearing, from +X
    tilt_degrees: Sourced[tuple[float, float]]  # roll, pitch
    samples: Sourced[int]  # a turn
    scan_rate: Sourced[float]  # Hz, turns a second
    min_range: Sourced[float]  # m
    max_range: Sourced[float]  # m


@dataclass(frozen=True)
class Imu:
    """The LSM9DS1 on the drive-controller board. The firmware reports
    (raw - bias) * scale."""

    position: Sourced[Vec3]  # m, the breakout's centre
    rotation_degrees: Sourced[Vec3]  # roll, pitch, yaw of the chip's axes
    accel_bias: Sourced[Vec3]  # m/s²
    accel_scale: Sourced[Vec3]
    gyro_bias: Sourced[Vec3]  # rad/s
    rest_accel: Sourced[Vec3]  # m/s², calibrated, at rest
    rest_gyro: Sourced[Vec3]  # rad/s, calibrated, at rest
    gyro_noise: Sourced[float]  # rad/s, 1 sigma a sample, at rest


@dataclass(frozen=True)
class Camera:
    front: Sourced[float]  # m, x of its front face
    y: Sourced[float]  # m, of its centre
    bottom: Sourced[float]  # m, z of its underside
    pitch_degrees: Sourced[float]


@dataclass(frozen=True)
class Sides[T]:
    """A value that differs between the car's left and right."""

    left: T
    right: T


@dataclass(frozen=True)
class Span:
    """The range of a value over repeated runs."""

    low: float
    high: float


@dataclass(frozen=True)
class Axles[T]:
    """A value that differs between the car's front and rear."""

    front: T
    rear: T


@dataclass(frozen=True)
class AtCommand[T]:
    """A value measured with the car held at a command: a share of full, in
    [-1, 1], + forward or left."""

    command: float
    value: T


# The sources of RACECAR_NEO_V2's values.
C_L = Source("C.L", "RACECAR Neo V2 Model Measurement Worksheet, the C.L column")
M_Z = Source("M.Z", "RACECAR Neo V2 Model Measurement Worksheet, the M.Z column")
MODEL = Source(
    "model", "RACECAR Neo V2 Model Measurement Worksheet, the Model Value column"
)
WORKSHEET_RUNS = Source(
    "worksheet runs",
    "RACECAR Neo V2 Model Measurement Worksheet, the lidar-referenced"
    " calibration runs of 2026-10-05",
)
FRAMES = Source("frames.yaml", "the car's frames.yaml, as the worksheet quotes it")


def _reference(name: str, tool: str) -> Source:
    """A file of calibration_reference_car_2026-10."""
    return Source(name, f"calibration_reference_car_2026-10/{name}, from {tool}")


REFERENCE_CAR = _reference("README.md", "the reference car's table")
ODOM = _reference("parameters/odom.local.yaml", "calibrate_odom.py, 2026-10-05")
DYNAMICS = _reference(
    "parameters/dynamics.local.yaml", "calibrate_dynamics.py, 2026-10-05"
)
STEERING = _reference(
    "parameters/throttle.local.md", "calibrate_steering.py, 2026-10-05"
)
LIDAR_YAW = _reference("parameters/frames.local.yaml", "calibrate_lidar.py, 2026-10-05")
IMU_CAL = _reference(
    "parameters/lsm9ds1_cal.local.yaml", "calibrate_imu.py, 2026-10-04"
)
DYNAMICS_RUNS = _reference(
    "bags/dynamics", "the ten straight runs' /drive and /odom/integrated"
)
SCANS = _reference("bags", "the /scan of every recording")
RACECARSIM = Source(
    "RacecarSim",
    "the RACECAR Neo V2 car of RacecarSim, the RACECAR Neo simulator in Unity:"
    " Assets/Scripts/Editor/RacecarPrefabSetup.cs, 2026-10",
)
RACECARSIM_MODEL = Source(
    "RacecarSim model",
    "RacecarSim's model of the RACECAR Neo V2, Assets/Models/RacecarNeoV2/"
    "RacecarNeoV2.fbx, 2026-10",
)
FIT = Source(
    "model fit",
    "fitted in the MuJoCo model, assets/racecar_neo_v2.xml, to the reference"
    " car's runs, 2026-10-20; validation/test_racecar_neo_v2.py holds the model"
    " to the runs",
)
PAYLOAD_CAD = Source(
    "payload CAD",
    "the payload's CAD model, racecar-neo-v2-body.step, exported from Autodesk"
    " software on 2026-10-21: its y is the car's x less 130 mm, and its floor's"
    " underside is at its z 0",
)


RACECAR_NEO_V2 = RacecarNeoV2(
    chassis=Chassis(
        wheelbase=measured(
            0.230,
            C_L,
            note="the model's 225 mm is the wheelbase that retailers list for the"
            " 1/14-scale LC Racing EMB-SC (one lists 227 mm for its kit, the"
            " EMB-SCH); the car measures 4 to 5 mm longer",
            conflicts=(Alternative(0.229, M_Z), Alternative(0.225, MODEL)),
        ),
        track=measured(0.178, C_L, conflicts=(Alternative(0.174, M_Z),)),
        mass=measured(2.36, C_L),
        front_weight_share=measured(
            0.522,
            C_L,
            note="axle weights on a kitchen scale: front 1160 g, rear 1060 g;"
            " they sum to 140 g short of the total",
        ),
        com_height=measured(0.095, C_L, note="a sideways tip test: 47 degrees a side"),
        inertia=estimated(
            (0.0098, 0.0216, 0.0200),
            "RacecarSim's, which does not say how it was found. The worksheet"
            " estimates the yaw as 0.032, a uniform box of 2.36 kg, 347 x 209 mm:"
            " too much for a car whose battery and motor sit near its middle",
        ),
        ride_height=assumed(0.0205, MODEL, "the model's; not yet measured"),
    ),
    footprint=Footprint(
        front=measured(
            0.2875,
            C_L,
            conflicts=(Alternative(0.286, M_Z), Alternative(0.2933, RACECARSIM_MODEL)),
        ),
        rear=measured(
            -0.0597,
            C_L,
            conflicts=(
                Alternative(-0.059, M_Z),
                Alternative(-0.0613, RACECARSIM_MODEL),
            ),
        ),
        left=measured(0.1045, C_L, conflicts=(Alternative(0.104, M_Z),)),
        right=measured(
            -0.1045,
            C_L,
            note="the model's -0.107 includes the battery connector, which the"
            " outline leaves out",
            conflicts=(Alternative(-0.104, M_Z), Alternative(-0.107, MODEL)),
        ),
        lidar_top=measured(
            0.205, C_L, conflicts=(Alternative(0.204, M_Z), Alternative(0.2105, MODEL))
        ),
        antenna_top=measured(
            0.275,
            C_L,
            note="the payload's CAD has a partial model of the antennas",
            conflicts=(Alternative(0.267, MODEL), Alternative(0.260, PAYLOAD_CAD)),
        ),
        payload_bottom=calibrated(
            0.096,
            PAYLOAD_CAD,
            "the LiDAR's scan plane and top and the camera's underside, measured at"
            " 195, 205 and 100 mm, which put the CAD's floor 95.9, 96.5 and 95.0 mm"
            " up",
            note="the battery chamber hangs 26.5 mm below it, to 69.5 mm",
            conflicts=(Alternative(0.102, RACECARSIM_MODEL),),
        ),
    ),
    shape=Shape(
        chassis_boxes=measured(
            (
                NamedBox("chassis", (0.1286, 0.0, 0.06125), (0.1646, 0.1045, 0.04075)),
                NamedBox(
                    "rear_bumper", (-0.0339, 0.0, 0.0438), (0.0274, 0.0925, 0.0272)
                ),
            ),
            RACECARSIM,
            note="its boxes, fitted to its model",
        ),
        payload_boxes=measured(
            (
                NamedBox("shell_rear", (0.08125, 0.0, 0.1445), (0.10125, 0.1, 0.0425)),
                NamedBox(
                    "shell_front", (0.23125, 0.0, 0.1295), (0.04875, 0.0525, 0.0275)
                ),
                NamedBox("lidar_stack", (0.2385, 0.0, 0.174), (0.035, 0.035, 0.017)),
                NamedBox(
                    "lidar_head", (0.23825, 0.0, 0.20075), (0.035, 0.035, 0.00975)
                ),
            ),
            RACECARSIM,
            note="its boxes, fitted to its model",
        ),
        payload_drop=calibrated(
            0.0066,
            RACECARSIM_MODEL,
            "the car's LiDAR, 195 mm high, which RacecarSim's model has at 201.6",
            note="its model also has the camera 7 mm and the IMU 8.5 mm higher than"
            " they are measured. Its payload is the payload's CAD, 6 mm higher: 107"
            " of its parts match the CAD's across to 1.5 mm, and 96 of them sit 5.7"
            " to 6.3 mm higher, the WiFi adapter and its antennas 7. Lowered by the"
            " drop, its floor is 0.6 mm below where the measured heights put the"
            " CAD's (Footprint.payload_bottom). The MuJoCo model's meshes are lowered"
            " by it",
        ),
        antenna_x=measured(-0.030, RACECARSIM_MODEL),
        antenna_y=measured(
            0.0449,
            RACECARSIM_MODEL,
            note="the reference car's scans see them 0.27 m behind the LiDAR,"
            " 8.5 degrees to its left and 11 to its right",
        ),
        antenna_bottom=measured(0.130, RACECARSIM_MODEL),
        antenna_diameter=measured(
            0.009,
            RACECARSIM_MODEL,
            note="they taper from 12.7 mm at the bottom to 5 mm at the top",
        ),
    ),
    wheel=Wheel(
        diameter=measured(
            0.072,
            C_L,
            note="the firmware reads the speed about 10% high"
            " (Odometry.speed_scale), as if the wheels rolled 65.1 mm across: it"
            " depends on the tyres' diameter and wear",
            conflicts=(Alternative(0.071, M_Z),),
        ),
        width=measured(0.029, C_L, conflicts=(Alternative(0.030, M_Z),)),
        axle_height=measured(0.036, C_L),
        mass=estimated(0.04, "RacecarSim's"),
        grip=calibrated(
            Axles(front=0.987, rear=1.204),
            FIT,
            "the full-lock circles: their radii and the encoder's over-read",
            note="MuJoCo's contacts are not a model of a tyre. The front tyres'"
            " lower grip stands for the slip of tyres steered 25 to 35 degrees,"
            " which the locked centre shaft drags; with the same grip front and"
            " rear the model's encoder over-reads 6 to 10%. RacecarSim estimates"
            " 0.8 for a smooth floor",
        ),
    ),
    suspension=Suspension(
        stiffness=estimated(
            Axles(front=770.0, rear=400.0),
            "RacecarSim's: coilover springs of 1.3 and 1.1 mm wire at an assumed"
            " 0.6 motion ratio",
        ),
        damping=estimated(
            Axles(front=26.0, rear=18.0), "RacecarSim's: 0.6 of critical"
        ),
        travel=estimated(0.020, "RacecarSim's"),
    ),
    steering=Steering(
        min_turn_radius=measured(
            Sides(left=0.420, right=0.466),
            DYNAMICS,
            note="the full-lock circles, from the LiDAR. Left lock turns about 10%"
            " tighter: servo travel or linkage geometry, which no trim corrects."
            " The wheels' angles are 34.8 / 24.3 degrees (inner / outer) to the"
            " left and 31.4 / 22.5 to the right; the model has 22.4 / 18.0",
            conflicts=(Alternative(Sides(left=0.419, right=0.465), WORKSHEET_RUNS),),
        ),
        center_to_lock_time=measured(
            Sides(left=0.203, right=0.216),
            DYNAMICS,
            note="the model swings lock to lock in 0.067 s",
        ),
        response=measured(
            (
                AtCommand(-1.0, Span(-2.140, -2.135)),
                AtCommand(-0.5, Span(-2.098, -2.042)),
                AtCommand(-0.25, Span(-1.043, -1.006)),
                AtCommand(0.0, Span(-0.047, 0.007)),
                AtCommand(0.25, Span(0.884, 0.919)),
                AtCommand(0.5, Span(1.844, 1.858)),
                AtCommand(1.0, Span(2.366, 2.401)),
            ),
            STEERING,
            note="arcs at speed 0.25 with trim 0, from the LiDAR, 3 to 8 runs a"
            " command. It flattens past 0.5 and differs by side: use the table,"
            " not a linear model",
        ),
        trim=calibrated(
            0.0,
            STEERING,
            "the LiDAR: the drift at command 0 over the curvature near the centre",
            note="a first fit of +0.0188 read the difference between the sides as"
            " a centre offset; it moved the drift by under 0.1 degree/m",
        ),
        deadband=measured(
            0.02,
            WORKSHEET_RUNS,
            note="0.01 to 0.02: a trim of +0.019 (9 us) had no measurable effect",
        ),
        straight_drift_degrees_per_m=measured(
            Span(-2.70, 0.42),
            STEERING,
            note="8 runs at trim 0: servo and linkage play, inside the deadband."
            " The worksheet gives -1.7 (-0.8 to -2.6)",
        ),
        linkage=calibrated(
            Linkage(
                slope=Sides(left=0.8678, right=0.9829),
                stop=Sides(left=0.5603, right=0.4890),
            ),
            FIT,
            "the curvature table (response) and the full-lock circles",
            note="the model's tyres slip more than the car's: Ackermann geometry"
            " turns its virtual wheel 0.501 rad left and 0.457 right at the"
            " measured full-lock radii",
        ),
        servo_stall_torque=chosen(3.0, "strong enough that the tyres do not move it"),
        servo_free_speed=calibrated(
            7.92, FIT, "the full-lock circles' yaw rate after the steering command"
        ),
        servo_kp=chosen(50.0, "the NeoRacer's: it settles without chatter"),
        servo_kd=chosen(1.0, "the NeoRacer's: it settles without chatter"),
        servo_supply=chosen(5.0, "the loop's voltage clamp; kp is per this supply"),
        latency=calibrated(
            0.086, FIT, "the full-lock circles' yaw rate after the steering command"
        ),
    ),
    drive=Drive(
        max_speed=documented(
            1.0,
            REFERENCE_CAR,
            note="the setting of every run; the car reaches 1.07 m/s",
        ),
        speed_floor=documented(0.5, REFERENCE_CAR),
        steady_speed=measured(
            (AtCommand(0.25, 0.721), AtCommand(0.5, 0.833), AtCommand(1.0, 1.07)),
            DYNAMICS,
            note="from the odometry, with its lidar-calibrated scale. The worksheet:"
            " the firmware holds about 0.175 m/s above each set speed, in its"
            " own encoder units",
        ),
        rise_time=measured(
            (AtCommand(0.25, 0.33), AtCommand(0.5, 0.33), AtCommand(1.0, 0.36)),
            DYNAMICS,
            note="the model takes 1.3 s",
        ),
        stop_distance=measured(
            (AtCommand(0.25, 0.184), AtCommand(0.5, 0.246), AtCommand(1.0, 0.372)),
            DYNAMICS,
            conflicts=(
                Alternative(
                    (
                        AtCommand(0.25, 0.184),
                        AtCommand(0.5, 0.246),
                        AtCommand(1.0, 0.370),
                    ),
                    C_L,
                ),
            ),
        ),
        stop_time=measured(
            (AtCommand(0.25, 0.56), AtCommand(0.5, 0.59), AtCommand(1.0, 0.72)),
            DYNAMICS,
        ),
        reverse=measured("immediate", C_L),
        coast=measured(
            (0.81, 1.25),
            DYNAMICS_RUNS,
            note="fitted to the speed from the 0 command to rest, with a latency"
            " of 0.05 s. The drag brake alone, 1.52 m/s², fits about as well",
        ),
        gain=calibrated(
            18.0, FIT, "the straight runs: the rise to 90% of the steady speed"
        ),
        latency=calibrated(
            0.06,
            FIT,
            "the straight runs' speed after a command",
            note="a fit to the stops alone gives 0.05 s, to the starts alone 0.095",
        ),
    ),
    odometry=Odometry(
        min_speed=measured(0.07, WORKSHEET_RUNS),
        speed_scale=calibrated(
            0.90406,
            ODOM,
            "the LiDAR, driving straight at a wall",
            note="it depends on the tyres' diameter and wear. Earlier runs against"
            " a tape gave 0.903 and 0.865",
        ),
        yaw_rate_scale=calibrated(
            1.00188, ODOM, "the LiDAR, over three full-lock circles"
        ),
        turn_overread=measured(
            Sides(left=0.0415, right=0.0245),
            DYNAMICS,
            note="the centre shaft is locked, so the encoder reads a speed between"
            " the rear and the front axles'",
        ),
        telemetry_rate=measured(
            170.0, WORKSHEET_RUNS, note="about 1 frame in 5 arrives late, after 10.7 ms"
        ),
    ),
    lidar=Lidar(
        position=measured(
            (0.2383, 0.0, 0.195),
            C_L,
            note="x checked against a wall to about 1 mm",
            conflicts=(
                Alternative((0.240, 0.0, 0.195), M_Z),
                Alternative((0.2383, 0.0, 0.2016), MODEL),
            ),
        ),
        yaw_degrees=calibrated(
            179.619,
            LIDAR_YAW,
            "the odometry's direction of travel, by scan matching",
            note="its motor faces the rear: nominally 180. The mean of six"
            " placements, standard error 0.059 degrees",
            conflicts=(Alternative(179.16, WORKSHEET_RUNS),),
        ),
        tilt_degrees=measured((0.0, 0.0), C_L),
        samples=measured(
            1080,
            C_L,
            note="as RacecarSim's LiDAR has it; the worksheet's model column, from"
            " an earlier RacecarSim, has 720",
        ),
        scan_rate=measured(
            7.4,
            C_L,
            note="as RacecarSim's LiDAR has it; the worksheet's model column, from"
            " an earlier RacecarSim, has 6",
        ),
        min_range=measured(0.05, C_L),
        max_range=measured(12.0, C_L),
    ),
    imu=Imu(
        position=measured(
            (0.190, 0.0, 0.105),
            C_L,
            conflicts=(
                Alternative((0.192, 0.0, 0.105), M_Z),
                Alternative((0.190, 0.0, 0.1135), MODEL),
                Alternative((0.190, 0.0, 0.107), PAYLOAD_CAD),
            ),
        ),
        rotation_degrees=documented(
            (-90.0, 0.0, -90.0), FRAMES, note="the chip is mounted face down"
        ),
        accel_bias=calibrated(
            (-0.2646, -0.0453, 0.0734),
            IMU_CAL,
            "gravity, with each axis up and down",
            note="one board's",
        ),
        accel_scale=calibrated(
            (1.00451, 1.01032, 1.00362), IMU_CAL, "gravity, with each axis up and down"
        ),
        gyro_bias=calibrated(
            (0.02844, 0.00084, -0.06329), IMU_CAL, "the car at rest", note="one board's"
        ),
        rest_accel=measured((-0.15, -0.01, 9.82), C_L),
        rest_gyro=measured((0.0012, -0.0005, 0.0009), C_L),
        gyro_noise=measured(0.039, WORKSHEET_RUNS),
    ),
    camera=Camera(
        front=measured(0.2779, C_L, conflicts=(Alternative(0.275, M_Z),)),
        y=measured(0.0, C_L),
        bottom=measured(
            0.100, C_L, conflicts=(Alternative(0.103, M_Z), Alternative(0.107, MODEL))
        ),
        pitch_degrees=measured(0.0, C_L),
    ),
)


def racecarsim_questions(spec: RacecarNeoV2) -> list[tuple[str, Sourced[object]]]:
    """The values where RacecarSim's model of the car disagrees with the car,
    or that the spec takes from it unchecked or corrects, with their paths:
    what the model's designer could fix, or have measured."""
    models = (MODEL, RACECARSIM_MODEL)
    return [
        (path, s)
        for path, s in walk(spec)
        if any(a.source in models for a in s.conflicts)
        or (isinstance(s.status, Assumed | Calibrated) and s.status.source in models)
    ]
