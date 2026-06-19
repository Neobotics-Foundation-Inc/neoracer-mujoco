"""
Physics conformance: the standardized gate every car must clear before it can be
trained on. Each test maps to one distinct way a model can brick training or be
exploited by an RL agent. If one fails, the message tells you which failure mode.

Run:  pytest validation/ -v
"""

import mujoco
import numpy as np

import _sim
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


# --- it responds to control correctly ---------------------------------------

def test_throttle_drives_forward(car):
    d = _sim.settle(car)
    x0 = d.body("car").xpos[0].copy()
    ctrl = np.zeros(car.nu); ctrl[DRIVE] = 1.0
    _sim.run(car, d, ctrl, 400)
    assert d.body("car").xpos[0] - x0 > 0.1, "full throttle produced no forward motion"


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


def test_deterministic(car):
    """Same start + same ctrl -> identical trajectory. RL breaks on hidden nondeterminism."""
    def rollout():
        d = _sim.settle(car)
        ctrl = np.zeros(car.nu); ctrl[DRIVE] = 1.0; ctrl[STEER] = 0.2
        return _sim.run(car, d, ctrl, 300).qpos.copy()
    assert np.array_equal(rollout(), rollout()), "trajectory not reproducible"


# --- sensor and physical realism checks -------------------------------------

def test_suspension_sag_at_rest(car):
    """Each corner compresses 1–13 mm at rest. < 1 mm = spring too stiff (suspension
    is decorative). >= 14 mm = hitting the jounce limit (range ceiling is +15 mm).
    Catches mass-removal bugs that under-load the springs and spring-constant regressions."""
    d = _sim.settle(car)
    s = _sim.sensors(car, d)
    for corner in ("fl", "fr", "rl", "rr"):
        sag_mm = float(s[f"{corner}_susp_pos"][0]) * 1000
        assert 1.0 <= sag_mm < 14.0, (
            f"{corner} sag = {sag_mm:.1f} mm — expected 1–13 mm "
            f"(< 1 mm: spring too stiff; >= 14 mm: hitting jounce limit)"
        )


def test_visual_geoms_carry_no_mass(car):
    """Cosmetic mesh geoms must declare mass='0'. Without it MuJoCo assigns
    density × volume — a wheel shell at 1000 kg/m³ adds ~0.5 kg per wheel.
    Check each wheel body stays below 0.3 kg (URDF ground truth is 0.057–0.059 kg)."""
    for name in ("fl_wheel_body", "fr_wheel_body", "rl_wheel_body", "rr_wheel_body"):
        body_id = car.body(name).id
        mass = float(car.body_mass[body_id])
        assert mass < 0.3, (
            f"{name} mass = {mass:.4f} kg — expected < 0.3 kg "
            f"(a visual geom is likely missing mass='0')"
        )


def test_lidar_sane_at_rest(car):
    """At rest, each LiDAR beam returns a positive finite distance (> 5 cm) or -1
    (no hit). A reading of 0.0 means the beam origin is inside a collision geom.
    Catches wrong site height or orientation that fires the beam into the car's own body."""
    d = _sim.settle(car)
    s = _sim.sensors(car, d)
    lidar_names = [name for name in _sim.REQUIRED_SENSORS if name.startswith("lidar_")]
    for name in lidar_names:
        v = float(s[name][0])
        assert np.isfinite(v), f"{name} = NaN at rest"
        assert v > 0.05 or v == -1.0, (
            f"{name} = {v:.3f} m — expected > 0.05 m or -1 (no hit); "
            f"a near-zero reading means the beam fires into the car's own geometry"
        )
