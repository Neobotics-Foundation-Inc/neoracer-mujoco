"""
Pure-logic + one physics-integration check for
neoracer_mujoco.control.wall_following.

Mirrors test_track_centering.py's style: synthetic SensorReadings for the
pure-logic tests (fast, no compiled model), one physics test against the
real straight_corridor.xml car for an end-to-end sanity check.
"""

import mujoco
import numpy as np
import pytest

from neoracer_mujoco import compose
from neoracer_mujoco import sensors as sl
from neoracer_mujoco.control.wall_following import (
    SIDE_BEAM,
    WallFollowingConfig,
    WallFollowingController,
    compute_signals,
)
from neoracer_mujoco.sensors import ALL_BEAMS, SensorReadings

DT = 0.02


def _sensors(**overrides) -> SensorReadings:
    """All 8 lidar beams defaulted to 1.0 m, overridden per-beam by name."""
    values = {name: np.array([1.0]) for name in ALL_BEAMS}
    for name, value in overrides.items():
        values[name] = np.array([value])
    return SensorReadings(**values)


def test_left_and_right_are_mirror_images():
    """Same magnitude of 'too far from the wall' on each side must produce
    exactly opposite steering commands -- also exercises both follow_side
    values end to end."""
    right = WallFollowingController(WallFollowingConfig(follow_side="right"))
    left = WallFollowingController(WallFollowingConfig(follow_side="left"))
    right_control = right.compute(_sensors(lidar_270=1.0), DT)
    left_control = left.compute(_sensors(lidar_090=1.0), DT)
    assert left_control.steer == pytest.approx(-right_control.steer)
    assert left_control.throttle == pytest.approx(right_control.throttle)


def test_too_far_from_wall_steers_toward_it():
    cfg = WallFollowingConfig(follow_side="right", target_distance=0.3)
    control = WallFollowingController(cfg).compute(_sensors(lidar_270=1.0), DT)
    assert control.steer < 0.0, (
        "too far from the right wall must steer right (negative)"
    )


def test_too_close_to_wall_steers_away():
    cfg = WallFollowingConfig(follow_side="right", target_distance=0.3)
    control = WallFollowingController(cfg).compute(_sensors(lidar_270=0.1), DT)
    assert control.steer > 0.0, "too close to the right wall must steer left (positive)"


def test_at_target_distance_and_parallel_is_neutral():
    cfg = WallFollowingConfig(follow_side="right", target_distance=0.3)
    sensors = _sensors(lidar_270=0.3, lidar_315=0.3, lidar_225=0.3)
    control = WallFollowingController(cfg).compute(sensors, DT)
    assert control.steer == pytest.approx(0.0, abs=1e-9)


def test_nose_angled_toward_wall_corrects_away():
    """At target distance, but front-side beam short vs rear-side => nose
    angled at the wall => steer away from it (left, positive)."""
    cfg = WallFollowingConfig(follow_side="right", target_distance=0.3)
    sensors = _sensors(lidar_270=0.3, lidar_315=0.2, lidar_225=0.4)
    signals = compute_signals(sensors, cfg)
    assert signals.distance_error == pytest.approx(0.0)
    assert signals.wall_skew > 0.0
    control = WallFollowingController(cfg).compute(sensors, DT)
    assert control.steer > 0.0


def test_throttle_decreases_for_close_forward_obstacle():
    cfg = WallFollowingConfig(follow_side="right", target_distance=0.3)
    controller = WallFollowingController(cfg)
    clear = controller.compute(
        _sensors(lidar_270=0.3, lidar_000=cfg.max_beam_range), DT
    )
    controller.reset()
    blocked = controller.compute(_sensors(lidar_270=0.3, lidar_000=0.1), DT)
    assert blocked.throttle < clear.throttle
    assert blocked.throttle == pytest.approx(0.0, abs=1e-9)


def test_throttle_decreases_for_sharp_steering_correction():
    cfg = WallFollowingConfig(follow_side="right", target_distance=0.3)
    straight = WallFollowingController(cfg).compute(_sensors(lidar_270=0.3), DT)
    sharp_controller = WallFollowingController(cfg)
    sensors = _sensors(lidar_270=1.0)  # far from wall -> large correction
    sharp = None
    for _ in range(200):
        sharp = sharp_controller.compute(sensors, DT)
    assert sharp.throttle < straight.throttle


def test_invalid_readings_trigger_safe_fallback():
    cfg = WallFollowingConfig(follow_side="right")
    sensors = SensorReadings(
        lidar_270=np.array([float("nan")]),
        lidar_315=np.array([float("nan")]),
        lidar_225=np.array([float("nan")]),
        lidar_000=np.array([1.0]),
    )
    control = WallFollowingController(cfg).compute(sensors, DT)
    assert control.throttle == pytest.approx(
        cfg.target_throttle * cfg.safe_fallback_throttle_fraction
    )
    assert abs(control.steer) < 0.05, (
        "fallback must not issue an extreme steering command"
    )


def test_no_hit_reading_is_not_treated_as_corrupt():
    """-1 (legitimate no-hit) must NOT trip the corrupt-data fallback."""
    cfg = WallFollowingConfig(follow_side="right")
    sensors = _sensors(lidar_270=-1.0, lidar_315=-1.0, lidar_225=-1.0)
    assert compute_signals(sensors, cfg).n_corrupt == 0


def test_steer_output_is_clamped():
    cfg = WallFollowingConfig(follow_side="right", target_distance=0.3)
    controller = WallFollowingController(cfg)
    sensors = _sensors(lidar_270=cfg.max_beam_range)  # maximally far -> saturates
    control = None
    for _ in range(200):  # let the slew-rate limiter reach steady state
        control = controller.compute(sensors, DT)
    assert abs(control.steer) <= cfg.steer_limit + 1e-9
    assert control.steer == pytest.approx(-cfg.steer_limit, abs=1e-6)


def test_steer_slew_rate_limits_first_step():
    cfg = WallFollowingConfig(
        follow_side="right", target_distance=0.3, steer_slew_limit=1.0
    )
    controller = WallFollowingController(cfg)
    sensors = _sensors(
        lidar_270=cfg.max_beam_range
    )  # would otherwise saturate instantly
    control = controller.compute(sensors, DT)
    max_first_step_delta = cfg.steer_slew_limit * DT
    assert abs(control.steer) <= max_first_step_delta + 1e-9
    assert abs(control.steer) < cfg.steer_limit, (
        "slew must prevent an instant jump to the limit"
    )


def test_reset_clears_derivative_and_slew_history():
    cfg = WallFollowingConfig(follow_side="right", target_distance=0.3)
    controller = WallFollowingController(cfg)
    offset_sensors = _sensors(lidar_270=1.0)
    for _ in range(50):
        controller.compute(offset_sensors, DT)
    assert controller._prev_steer != 0.0  # history has built up

    controller.reset()
    assert controller._prev_steer == 0.0
    assert controller._prev_distance_error == 0.0

    control = controller.compute(_sensors(lidar_270=0.3), DT)
    assert control.steer == pytest.approx(0.0, abs=1e-9)


# --- physics integration: real compiled car + corridor -----------------------


def _load_corridor_model() -> mujoco.MjModel:
    return compose("straight_corridor")


def test_controller_converges_to_target_distance_on_corridor():
    """Runs the real controller against the real compiled car+corridor: it
    should settle near the configured standoff distance from the right wall
    without flipping, going non-finite, or exceeding the steer limit."""
    model = _load_corridor_model()
    data = mujoco.MjData(model)
    dt = model.opt.timestep

    data.ctrl[:] = 0
    for _ in range(400):
        mujoco.mj_step(model, data)

    cfg = WallFollowingConfig(follow_side="right")
    controller = WallFollowingController(cfg)
    steer_ctrlrange = model.actuator("steer_servo").ctrlrange

    side_distances = []
    for _ in range(1500):
        sensors = sl.read(model, data)
        control = controller.compute(sensors, dt)
        data.ctrl[0:4] = control.throttle
        data.ctrl[4] = control.steer
        mujoco.mj_step(model, data)

        assert steer_ctrlrange[0] - 1e-9 <= control.steer <= steer_ctrlrange[1] + 1e-9
        assert np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))
        side_distances.append(float(getattr(sensors, SIDE_BEAM["right"])[0]))

    upright_cos = float(data.body("car").xmat[8])
    assert upright_cos > 0.7, "car should not have flipped during the run"

    quarter = len(side_distances) // 4
    last_quarter_mean = sum(side_distances[-quarter:]) / quarter
    assert last_quarter_mean == pytest.approx(cfg.target_distance, abs=0.1)
