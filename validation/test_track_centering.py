"""
Pure-logic + one physics-integration check for examples/track_centering.py.

The pure-logic tests build synthetic SensorReadings directly (no compiled
model needed — mirrors test_logic.py's style) so they run fast and pin down
the controller's sign conventions and safety behavior in isolation. The one
physics test at the bottom compiles the car onto the straight_corridor.xml
validation track and checks the controller doesn't misbehave against the
real simulator (finite values, steering clamped, stays upright, doesn't
immediately hit a wall).
"""

import mujoco
import numpy as np
import pytest
import sensor_logger as sl
from track_centering import (
    ALL_BEAMS,
    TrackCenteringConfig,
    TrackCenteringController,
    compute_signals,
)

DT = 0.02


def _sensors(**overrides) -> sl.SensorReadings:
    """Build a SensorReadings with all 8 lidar beams defaulted to 1.0 m,
    overridden per-beam by name (e.g. lidar_090=0.5)."""
    values = {name: np.array([1.0]) for name in ALL_BEAMS}
    for name, value in overrides.items():
        values[name] = np.array([value])
    return sl.SensorReadings(**values)


def test_equal_left_right_is_approximately_neutral():
    cfg = TrackCenteringConfig()
    controller = TrackCenteringController(cfg)
    control = controller.compute(_sensors(), DT)
    assert control.steer == pytest.approx(0.0, abs=1e-9)


def test_closer_left_boundary_steers_right():
    cfg = TrackCenteringConfig()
    controller = TrackCenteringController(cfg)
    sensors = _sensors(lidar_045=0.3, lidar_090=0.3, lidar_135=0.3)  # left wall close
    control = controller.compute(sensors, DT)
    assert control.steer < 0.0, "closer left wall must steer right (negative)"


def test_closer_right_boundary_steers_left():
    cfg = TrackCenteringConfig()
    controller = TrackCenteringController(cfg)
    sensors = _sensors(lidar_225=0.3, lidar_270=0.3, lidar_315=0.3)  # right wall close
    control = controller.compute(sensors, DT)
    assert control.steer > 0.0, "closer right wall must steer left (positive)"


def test_heading_toward_left_wall_corrects_right():
    """Laterally centered (equal left/right means) but front-left beam
    reading short relative to rear-left => nose angled at the left wall =>
    controller should correct away from it (steer right, negative)."""
    cfg = TrackCenteringConfig()
    controller = TrackCenteringController(cfg)
    sensors = _sensors(lidar_045=0.5, lidar_090=1.0, lidar_135=1.5)
    signals = compute_signals(sensors, cfg)
    assert signals.lateral_error == pytest.approx(
        0.0
    )  # mean(0.5,1,1.5)=1 == right mean 1
    assert signals.heading_error < 0.0
    control = controller.compute(sensors, DT)
    assert control.steer < 0.0


def test_heading_toward_right_wall_corrects_left():
    cfg = TrackCenteringConfig()
    controller = TrackCenteringController(cfg)
    sensors = _sensors(lidar_315=0.5, lidar_270=1.0, lidar_225=1.5)
    signals = compute_signals(sensors, cfg)
    assert signals.lateral_error == pytest.approx(0.0)
    assert signals.heading_error > 0.0
    control = controller.compute(sensors, DT)
    assert control.steer > 0.0


def test_steer_output_is_clamped():
    cfg = TrackCenteringConfig()
    controller = TrackCenteringController(cfg)
    sensors = _sensors(
        lidar_045=10.0,
        lidar_090=10.0,
        lidar_135=10.0,
        lidar_225=0.05,
        lidar_270=0.05,
        lidar_315=0.05,
    )
    for _ in range(200):  # let the slew-rate limiter reach steady state
        control = controller.compute(sensors, DT)
    assert abs(control.steer) <= cfg.steer_limit + 1e-9
    assert control.steer == pytest.approx(cfg.steer_limit, abs=1e-6)


def test_invalid_readings_trigger_safe_fallback():
    cfg = TrackCenteringConfig()
    controller = TrackCenteringController(cfg)
    # All 8 beams missing/NaN -> corrupt count 8 > max_corrupt_beams (4).
    sensors = sl.SensorReadings(
        **{name: np.array([float("nan")]) for name in ALL_BEAMS}
    )
    control = controller.compute(sensors, DT)
    assert control.throttle == pytest.approx(
        cfg.target_throttle * cfg.safe_fallback_throttle_fraction
    )
    assert abs(control.steer) < 0.05, (
        "fallback must not issue an extreme steering command"
    )


def test_invalid_readings_via_empty_arrays_also_trigger_fallback():
    cfg = TrackCenteringConfig()
    controller = TrackCenteringController(cfg)
    sensors = sl.SensorReadings()  # all lidar fields default to empty arrays
    control = controller.compute(sensors, DT)
    assert control.throttle == pytest.approx(
        cfg.target_throttle * cfg.safe_fallback_throttle_fraction
    )


def test_no_hit_reading_is_not_treated_as_corrupt():
    """-1 (legitimate no-hit) must NOT trip the corrupt-data fallback — it's
    real data meaning 'clear', not a sensor fault."""
    cfg = TrackCenteringConfig()
    sensors = _sensors(**{name: -1.0 for name in ALL_BEAMS})
    signals = compute_signals(sensors, cfg)
    assert signals.n_corrupt == 0
    assert signals.lateral_error == pytest.approx(0.0)


def test_throttle_decreases_for_sharp_steering_correction():
    cfg = TrackCenteringConfig()
    straight = TrackCenteringController(cfg).compute(_sensors(), DT)
    sharp_controller = TrackCenteringController(cfg)
    sharp = None
    sensors = _sensors(lidar_045=0.2, lidar_090=0.2, lidar_135=0.2)
    for _ in range(200):
        sharp = sharp_controller.compute(sensors, DT)
    assert sharp.throttle < straight.throttle


def test_throttle_decreases_for_close_forward_obstacle():
    cfg = TrackCenteringConfig()
    controller = TrackCenteringController(cfg)
    clear = controller.compute(_sensors(lidar_000=cfg.max_beam_range), DT)
    controller.reset()
    blocked = controller.compute(_sensors(lidar_000=0.1), DT)
    assert blocked.throttle < clear.throttle
    assert blocked.throttle == pytest.approx(0.0, abs=1e-9)


def test_reset_clears_derivative_and_slew_history():
    cfg = TrackCenteringConfig()
    controller = TrackCenteringController(cfg)
    offset_sensors = _sensors(lidar_045=0.2, lidar_090=0.2, lidar_135=0.2)
    for _ in range(50):
        controller.compute(offset_sensors, DT)
    assert controller._prev_steer != 0.0  # history has built up

    controller.reset()
    assert controller._prev_steer == 0.0
    assert controller._prev_lateral_error == 0.0

    # With history cleared, a single neutral reading must be exactly neutral
    # again — no lingering pull from the pre-reset offset.
    control = controller.compute(_sensors(), DT)
    assert control.steer == pytest.approx(0.0, abs=1e-9)


# --- physics integration: real compiled car + corridor -----------------------


def _load_corridor_model() -> mujoco.MjModel:
    import os

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scene = mujoco.MjSpec.from_file(
        os.path.join(repo, "assets", "tracks", "straight_corridor.xml")
    )
    car = mujoco.MjSpec.from_file(os.path.join(repo, "assets", "neoracer.xml"))
    scene.worldbody.add_frame().attach_body(car.body("car"), "", "")
    return scene.compile()


def test_controller_stable_on_corridor():
    """Runs the real controller against the real compiled car+corridor for a
    few seconds: values stay finite, steering respects the actuator limit,
    the car stays upright, and it doesn't immediately plow into a wall."""
    model = _load_corridor_model()
    data = mujoco.MjData(model)
    dt = model.opt.timestep

    data.ctrl[:] = 0
    for _ in range(400):
        mujoco.mj_step(model, data)
    data.qpos[1] += 0.25  # deliberate offset toward the left wall
    mujoco.mj_forward(model, data)

    cfg = TrackCenteringConfig()
    controller = TrackCenteringController(cfg)
    steer_ctrlrange = model.actuator("steer_servo").ctrlrange

    min_front = float("inf")
    for _ in range(1500):
        sensors = sl.read(model, data)
        signals = compute_signals(sensors, cfg)
        control = controller.compute(sensors, dt)

        data.ctrl[0:4] = control.throttle
        data.ctrl[4] = control.steer
        mujoco.mj_step(model, data)

        assert steer_ctrlrange[0] - 1e-9 <= control.steer <= steer_ctrlrange[1] + 1e-9
        assert np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))
        min_front = min(min_front, signals.front_distance)

    upright_cos = float(data.body("car").xmat[8])
    assert upright_cos > 0.7, "car should not have flipped during the run"
    assert min_front > 0.05, "controller should not have driven straight into a wall"
