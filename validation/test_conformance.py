"""
Physics conformance: the tests that every car XML must pass before RL
training uses it. Each test checks one fault that can stop training or that
an RL agent can exploit. The failure message names the fault.

The `scene` fixture (conftest.py) runs each test once for each car XML.
"""

import copy
from dataclasses import fields

import mujoco
import numpy as np
import pytest

from neoracer_mujoco import Scene, contract, sensors, sim
from neoracer_mujoco.control.steering import ControlCommand


def _steps(model: mujoco.MjModel, seconds: float) -> int:
    """The number of physics steps in `seconds` of simulated time."""
    return round(seconds / model.opt.timestep)


def _drive(
    scene: Scene, throttle: float, steer: float, seconds: float
) -> mujoco.MjData:
    """Settle the car, then hold throttle (a fraction of the battery
    voltage) and steer (rad) for `seconds`. Return the MjData."""
    data = sim.settle(scene.model)
    sim.apply_command(scene.car, data, ControlCommand(throttle=throttle, steer=steer))
    for _ in range(_steps(scene.model, seconds)):
        mujoco.mj_step(scene.model, data)
    return data


def _read(scene: Scene, data: mujoco.MjData) -> sensors.SensorReadings:
    return sensors.read(scene.car.sensors, data)


# --- the model has the actuators and sensors of contract.py -------------------


def test_actuator_limits_parse_every_car(scene):
    """Every car has a symmetric drive range and a symmetric steer range."""
    limits = contract.actuator_limits(scene.model, scene.car)
    assert limits.voltage > 0 and limits.steer_rad > 0


@pytest.mark.parametrize(
    ("actuator", "ctrlrange"),
    [("motor", (-0.5, 1.0)), ("motor", (0.0, 1.0)), ("steer_servo", (0.0, 0.4))],
)
def test_actuator_limits_reject_ranges_that_one_limit_cannot_scale(
    scene, actuator, ctrlrange
):
    model = copy.copy(scene.model)  # an independent MjModel; scene is shared
    model.bind_scalar(scene.spec.actuator(actuator)).ctrlrange[:] = ctrlrange
    with pytest.raises(ValueError, match="ctrlrange"):
        contract.actuator_limits(model, scene.car)


def _spin_joint(wheel: mujoco.MjsGeom) -> mujoco.MjsJoint:
    """The joint that turns a wheel: the hinge about the axle (+Y) of the
    wheel's body."""
    hinges = [
        j
        for j in wheel.parent.joints
        if j.type == mujoco.mjtJoint.mjJNT_HINGE and np.allclose(j.axis, [0, 1, 0])
    ]
    assert len(hinges) == 1, wheel.name
    return hinges[0]


def test_the_drivetrain_is_open_differentials(scene):
    """Whatever the speeds of the wheels, the motor turns at their mean and
    each wheel gets a quarter of the motor's force: the laws of open
    differentials."""
    model, car = scene.model, scene.car
    data = sim.settle(model)
    spins = [_spin_joint(w) for w in car.wheels]
    for joint, speed in zip(spins, (10.0, 20.0, 35.0, -5.0), strict=True):
        data.bind_scalar(joint).qvel = speed  # rad/s
    sim.apply_command(car, data, ControlCommand(throttle=1.0, steer=0.0))
    mujoco.mj_forward(model, data)
    motor, motor_model = data.bind_scalar(car.drive), model.bind_scalar(car.drive)
    gear = motor_model.gear[0]
    assert motor.velocity[0] / gear == pytest.approx(np.mean([10, 20, 35, -5]))
    force = motor.force[0] * gear  # on the drivetrain
    assert force != 0.0
    for joint in spins:
        assert data.bind_scalar(joint).qfrc_actuator[0] == pytest.approx(force / 4)


def test_drive_pos_is_the_mean_wheel_angle(scene):
    """drive_pos, which the motor encoder counts, is the mean of the four
    wheels' spin angles: the angle the motor turns with."""
    model, car = scene.model, scene.car
    data = mujoco.MjData(model)
    for wheel, angle in zip(car.wheels, (1.0, 2.0, 4.0, -3.0), strict=True):
        data.bind_scalar(_spin_joint(wheel)).qpos = angle  # rad
    mujoco.mj_forward(model, data)
    assert data.bind_scalar(car.sensors.drive_pos).data[0] == pytest.approx(1.0)


def test_actuator_contract(scene):
    """The car has the actuators of contract.EXPECTED_ACTUATORS and no
    others. Their order in ctrl does not matter: code writes ctrl through
    scene.car."""
    names = sorted(a.name for a in scene.spec.actuators)
    assert names == sorted(contract.EXPECTED_ACTUATORS), f"got {names}"


def test_required_sensors_present(scene):
    missing = [s for s in contract.REQUIRED_SENSORS if scene.spec.sensor(s) is None]
    assert not missing, f"missing sensors: {missing}"


def test_lidar_rays_start_at_the_lidar_inside_the_chassis_box(scene):
    """The LiDAR mount and its 8 ray sites are at one point of the car body,
    the LiDAR's centre, below the top of the chassis box, which reaches the
    top of the LiDAR. A rangefinder ignores the geoms of its own body, so the
    rays leave the car. The ray tests use horizontal distances only, so they
    cannot see this height."""
    model, spec = scene.model, scene.spec
    chassis = model.bind_scalar(spec.geom("chassis"))
    bottom = float(chassis.pos[2] - chassis.size[2])
    top = float(chassis.pos[2] + chassis.size[2])
    mount = model.bind_scalar(spec.site("lidar_mount"))
    assert bottom < float(mount.pos[2]) < top
    for name in sensors.LIDAR_BEAM_ORDER:
        site = model.bind_scalar(spec.site(name))
        assert int(site.bodyid[0]) == model.bind_scalar(scene.car.body).id, name
        np.testing.assert_array_equal(site.pos, mount.pos, err_msg=name)


def test_total_mass_plausible(scene):
    """A geom with no explicit mass gets the default density, and the car
    then weighs tens of kg. The mass band of contract.py finds this error."""
    m = sim.car_mass(scene.model, scene.car)
    assert contract.MASS_MIN < m < contract.MASS_MAX, (
        f"car mass {m:.2f} kg outside RC range"
    )


# --- the car rests without errors ---------------------------------------------


def test_settles_without_nan(scene):
    d = sim.settle(scene.model)
    assert np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel)), (
        "NaN/Inf at rest"
    )


def test_rests_on_ground(scene):
    """The car does not sink through the floor. A bad initial penetration
    does not launch it."""
    body = scene.car.body
    spawn_z = scene.model.bind_scalar(body).pos[2]
    z = sim.settle(scene.model).bind_scalar(body).xpos[2]
    assert 0.02 < z < spawn_z + 0.02, f"settled z={z:.3f} vs spawn {spawn_z:.3f}"


def test_stays_upright_at_rest(scene):
    upright = sim.car_upright_cos(sim.settle(scene.model), scene.car)
    assert upright > 0.95, "car tips over with no input"


def test_motor_encoder_near_zero_at_rest(scene):
    """With no control, the car does not move, so the motor encoder reads
    about 0 rad/s."""
    d = sim.settle(scene.model)
    motor_rad_s = sensors.motor_encoder_rad_s(_read(scene, d))
    assert abs(motor_rad_s) < 0.05, (
        f"stationary motor encoder={motor_rad_s:.4f} rad/s, expected ~0"
    )


def test_imu_stationary_accel_matches_gravity(scene):
    """A stationary accelerometer reads the specific force, not zero. So its
    magnitude is about the gravity of the model. A disconnected or broken
    accelerometer fails this test."""
    readings = _read(scene, sim.settle(scene.model))
    g = float(np.linalg.norm(scene.model.opt.gravity))
    accel_mag = float(np.linalg.norm(readings.imu_accel))
    assert abs(accel_mag - g) < 0.5, (
        f"stationary |accel|={accel_mag:.2f} m/s^2, expected ~{g:.2f}"
    )


def test_imu_stationary_gyro_near_zero(scene):
    """The car does not rotate at rest, so the gyro reads about 0 rad/s."""
    readings = _read(scene, sim.settle(scene.model))
    gyro_mag = float(np.linalg.norm(readings.imu_gyro))
    assert gyro_mag < 0.05, f"stationary |gyro|={gyro_mag:.4f} rad/s, expected ~0"


# --- the car responds to control with the contract.py signs -------------------


def test_throttle_drives_forward(scene):
    body = scene.car.body
    x0 = sim.settle(scene.model).bind_scalar(body).xpos[0]
    d = _drive(scene, throttle=1.0, steer=0.0, seconds=0.8)
    assert d.bind_scalar(body).xpos[0] - x0 > 0.1, (
        "full throttle produced no forward motion"
    )


def test_motor_encoder_positive_under_forward_throttle(scene):
    """Full forward throttle spins the motor forward (+ angular velocity)."""
    d = _drive(scene, throttle=1.0, steer=0.0, seconds=0.8)
    motor_rad_s = sensors.motor_encoder_rad_s(_read(scene, d))
    assert motor_rad_s > 0.1, (
        f"expected motor encoder > 0 under forward throttle, got {motor_rad_s}"
    )


def _steer_angle(scene: Scene, data: mujoco.MjData) -> float:
    return float(data.bind_scalar(scene.car.sensors.steer_cmd_pos).data[0])  # rad


@pytest.mark.parametrize("hz", [500, 360])
def test_the_servo_turns_at_a_servos_speed_and_holds_steady(scene, hz):
    """A servo takes tens of ms to reach full lock and then holds it.

    From rest, a 0.4 rad step reaches 90% in 0.03 to 0.15 s: a position
    actuator that snaps there in a few ms fails. Driving at full lock, the
    steering angle wobbles by less than 0.006 rad (standard deviation): a
    servo loop that chatters, as kd = 1 does under explicit Euler at
    360 Hz (0.01 rad), fails."""
    model = copy.copy(scene.model)  # an independent MjModel; scene is shared
    model.opt.timestep = 1.0 / hz
    data = sim.settle(model)
    sim.apply_command(scene.car, data, ControlCommand(throttle=0.0, steer=0.4))
    steps = 0
    while _steer_angle(scene, data) < 0.9 * 0.4 and steps < _steps(model, 0.5):
        mujoco.mj_step(model, data)
        steps += 1
    assert 0.03 <= steps * model.opt.timestep <= 0.15

    data = sim.settle(model)
    sim.apply_command(scene.car, data, ControlCommand(throttle=0.3, steer=0.4))
    for _ in range(_steps(model, 2.0)):
        mujoco.mj_step(model, data)
    held = []
    for _ in range(_steps(model, 0.5)):
        mujoco.mj_step(model, data)
        held.append(_steer_angle(scene, data))
    assert np.std(held) < 0.006


def test_left_steer_yaws_left(scene):
    """A + steer yaws the car to the left (+Z). A sign error here inverts the
    steering of every agent."""
    steer = contract.actuator_limits(scene.model, scene.car).steer_rad
    d = _drive(scene, throttle=0.5, steer=steer, seconds=0.8)
    yaw_rate = d.bind_scalar(scene.car.joint).qvel[5]  # freejoint yaw rate, rad/s
    assert yaw_rate > 0.1, f"left steer gave yaw rate {yaw_rate:.3f} (expected > 0)"


def test_ackermann_inner_exceeds_outer(scene):
    """In a left turn, the inner (left) front wheel steers more than the
    outer (right) front wheel."""
    steer = contract.actuator_limits(scene.model, scene.car).steer_rad
    s = _read(scene, _drive(scene, throttle=0.0, steer=steer, seconds=0.8))
    assert s.fl_steer_pos > s.fr_steer_pos, "Ackermann differential wrong/absent"


# --- an RL agent cannot exploit the physics -----------------------------------


def test_no_free_energy(scene):
    """With zero control, the car stays at rest. A car that moves by itself
    gives an agent reward for nothing."""
    v = np.linalg.norm(sim.settle(scene.model).qvel)
    assert v < 0.05, f"car drifts at {v:.3f} with no control"


def test_top_speed_bounded(scene):
    """Full throttle for 4 s gives a finite speed below 40 m/s. A larger
    speed is a numerical error, not physics."""
    # Limit: 40 m/s is an error limit, not a realistic top speed. A plain
    # <motor> has no back-EMF, so a light car accelerates toward ~25 m/s. The
    # real car is much slower. If an agent needs a realistic top speed, add a
    # speed limit to the motor in the car XML.
    d = _drive(scene, throttle=1.0, steer=0.0, seconds=4.0)
    v = np.linalg.norm(d.bind_scalar(scene.car.joint).qvel[:3])
    assert np.isfinite(v) and v < 40.0, (
        f"top speed {v:.2f} m/s is a blow-up, not physics"
    )


def test_sensors_finite_under_load(scene):
    """After 1 s of full throttle and full steer, every sensor, the motor
    encoder and the speed estimate are finite."""
    steer = contract.actuator_limits(scene.model, scene.car).steer_rad
    s = _read(scene, _drive(scene, throttle=1.0, steer=steer, seconds=1.0))
    bad = [f.name for f in fields(s) if not np.all(np.isfinite(getattr(s, f.name)))]
    assert not bad, f"non-finite sensors: {bad}"
    motor_rad_s = sensors.motor_encoder_rad_s(s)
    assert np.isfinite(motor_rad_s), f"non-finite motor encoder: {motor_rad_s}"
    assert np.isfinite(sensors.estimated_linear_speed_ms(motor_rad_s))


def test_deterministic(scene):
    """The same start and the same ctrl give the same trajectory. RL fails
    if the simulation has hidden nondeterminism."""

    def rollout():
        return _drive(scene, throttle=1.0, steer=0.2, seconds=0.6).qpos.copy()

    assert np.array_equal(rollout(), rollout()), "trajectory not reproducible"
