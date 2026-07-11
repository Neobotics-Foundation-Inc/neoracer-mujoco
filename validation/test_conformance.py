"""
Physics conformance: the standardized gate every car must clear before it can be
trained on. Each test maps to one distinct way a model can brick training or be
exploited by an RL agent. If one fails, the message tells you which failure mode.

Run:  pytest validation/ -v
"""

import mujoco
import numpy as np

import _sim
import sensor_logger as sl
from _sim import DRIVE, STEER


# --- it exists and is wired the way the contract says ------------------------

def test_actuator_contract(car):
    """5 actuators in the agreed ctrl order; agent code indexes ctrl[0:4]+ctrl[4]."""
    names = [car.actuator(i).name for i in range(car.nu)]
    assert names == _sim.EXPECTED_ACTUATORS, f"got {names}"


def test_required_sensors_present(car):
    names = {car.sensor(i).name for i in range(car.nsensor)}
    missing = [s for s in _sim.REQUIRED_SENSORS if s not in names]
    assert not missing, f"missing sensors: {missing}"


def test_total_mass_plausible(car):
    """Catches the default-density bug: a geom with no explicit mass weighs tens of kg."""
    m = _sim.car_mass(car)
    assert _sim.MASS_MIN < m < _sim.MASS_MAX, f"car mass {m:.2f} kg outside RC range"


# --- it won't explode or fall apart at rest ---------------------------------

def test_settles_without_nan(car):
    d = _sim.settle(car)
    assert np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel)), "NaN/Inf at rest"


def test_rests_on_ground(car):
    """Doesn't sink through the floor or get launched by a bad initial penetration."""
    spawn_z = car.body("car").pos[2]
    z = _sim.settle(car).body("car").xpos[2]
    assert 0.02 < z < spawn_z + 0.02, f"settled z={z:.3f} vs spawn {spawn_z:.3f}"


def test_stays_upright_at_rest(car):
    assert _sim.car_upright_cos(_sim.settle(car)) > 0.95, "car tips over with no input"


def test_motor_encoder_near_zero_at_rest(car):
    """No control, no motion -> wheel angular velocities should read ~0 rad/s."""
    d = _sim.settle(car)
    encoder = sl.motor_encoder_readings(sl.read(car, d))
    mag = float(np.linalg.norm(encoder.angular_velocity))
    assert mag < 0.05, f"stationary |wheel_vel|={mag:.4f} rad/s, expected ~0"


# --- it responds to control correctly ---------------------------------------

def test_throttle_drives_forward(car):
    d = _sim.settle(car)
    x0 = d.body("car").xpos[0].copy()
    ctrl = np.zeros(car.nu); ctrl[DRIVE] = 1.0
    _sim.run(car, d, ctrl, 400)
    assert d.body("car").xpos[0] - x0 > 0.1, "full throttle produced no forward motion"


def test_motor_encoder_positive_under_forward_throttle(car):
    """Full forward throttle must spin every driven wheel forward (+ angular velocity)."""
    d = _sim.settle(car)
    ctrl = np.zeros(car.nu); ctrl[DRIVE] = 1.0
    encoder = sl.motor_encoder_readings(sl.read(car, _sim.run(car, d, ctrl, 400)))
    assert np.all(encoder.angular_velocity > 0.1), (
        f"expected all wheel speeds > 0 under forward throttle, got {encoder.angular_velocity}"
    )


def test_motor_encoder_left_right_consistent_straight(car):
    """Driving straight (no steer) must spin left/right wheel pairs at matching speed."""
    d = _sim.settle(car)
    ctrl = np.zeros(car.nu); ctrl[DRIVE] = 0.5
    encoder = sl.motor_encoder_readings(sl.read(car, _sim.run(car, d, ctrl, 400)))
    fl, fr, rl, rr = encoder.angular_velocity
    assert abs(fl - fr) < 0.5, f"front wheels diverged going straight: fl={fl:.3f} fr={fr:.3f}"
    assert abs(rl - rr) < 0.5, f"rear wheels diverged going straight: rl={rl:.3f} rr={rr:.3f}"


def test_left_steer_yaws_left(car):
    """+steer must yaw +z (left); a sign flip silently inverts the agent's steering."""
    d = _sim.settle(car)
    ctrl = np.zeros(car.nu); ctrl[DRIVE] = 0.5; ctrl[STEER] = 0.4
    _sim.run(car, d, ctrl, 400)
    assert d.qvel[5] > 0.1, f"left steer gave yaw rate {d.qvel[5]:.3f} (expected > 0)"


def test_ackermann_inner_exceeds_outer(car):
    """Turning left, the inner (left) front wheel must steer more than the outer."""
    d = _sim.settle(car)
    ctrl = np.zeros(car.nu); ctrl[STEER] = 0.4
    s = _sim.sensors(car, _sim.run(car, d, ctrl, 400))
    assert s["fl_steer_pos"][0] > s["fr_steer_pos"][0], "Ackermann differential wrong/absent"


# --- it can't be exploited by an RL agent -----------------------------------

def test_no_free_energy(car):
    """Zero control from rest must stay at rest. A self-propelling model is free reward."""
    v = np.linalg.norm(_sim.settle(car).qvel)
    assert v < 0.05, f"car drifts at {v:.3f} with no control"


def test_top_speed_bounded(car):
    """
    Sustained full throttle must stay finite and not blow up (no glitch-accel).
    NOTE: this is a glitch ceiling, NOT a realistic top speed. A plain <motor> has
    no back-EMF / speed cap, so a light car keeps accelerating toward ~25 m/s. The
    real RC car tops out far lower; closing that gap is motor tuning in the XML
    (velocity-limited actuator or drivetrain damping), not a job for this test.
    """
    d = _sim.settle(car)
    ctrl = np.zeros(car.nu); ctrl[DRIVE] = 1.0
    _sim.run(car, d, ctrl, 2000)
    v = np.linalg.norm(d.qvel[:3])
    assert np.isfinite(v) and v < 40.0, f"top speed {v:.2f} m/s is a blow-up, not physics"


def test_sensors_finite_under_load(car):
    """No NaN reaches the agent's observation while driving + steering hard."""
    d = _sim.settle(car)
    ctrl = np.zeros(car.nu); ctrl[DRIVE] = 1.0; ctrl[STEER] = 0.4
    s = _sim.sensors(car, _sim.run(car, d, ctrl, 500))
    bad = [k for k, v in s.items() if not np.all(np.isfinite(v))]
    assert not bad, f"non-finite sensors: {bad}"


def test_motor_encoder_finite_under_load(car):
    """The typed WheelAngularVelocityReading and its derived estimates must stay finite under load."""
    d = _sim.settle(car)
    ctrl = np.zeros(car.nu); ctrl[DRIVE] = 1.0; ctrl[STEER] = 0.4
    encoder = sl.motor_encoder_readings(sl.read(car, _sim.run(car, d, ctrl, 500)))
    assert np.all(np.isfinite(encoder.angular_velocity)), (
        f"non-finite wheel angular velocity: {encoder.angular_velocity}"
    )
    assert np.isfinite(sl.average_wheel_angular_velocity(encoder))
    assert np.isfinite(sl.estimated_linear_speed_ms(encoder))


def test_deterministic(car):
    """Same start + same ctrl -> identical trajectory. RL breaks on hidden nondeterminism."""
    def rollout():
        d = _sim.settle(car)
        ctrl = np.zeros(car.nu); ctrl[DRIVE] = 1.0; ctrl[STEER] = 0.2
        return _sim.run(car, d, ctrl, 300).qpos.copy()
    assert np.array_equal(rollout(), rollout()), "trajectory not reproducible"
