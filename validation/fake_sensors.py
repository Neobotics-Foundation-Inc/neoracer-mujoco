"""Build SensorReadings for the pure-logic tests, without the simulator.

Frame: +X forward, +Y left, +Z up.
"""

from dataclasses import replace

import numpy as np

from neoracer_mujoco.sensors import LIDAR_BEAM_ORDER, SensorReadings

_ZERO = SensorReadings(
    imu_accel=np.zeros(3),
    imu_gyro=np.zeros(3),
    imu_quat=np.array([1.0, 0.0, 0.0, 0.0]),  # level, no rotation
    imu_linvel=np.zeros(3),
    steer_cmd_pos=0.0,
    fl_steer_pos=0.0,
    fr_steer_pos=0.0,
    fl_wheel_vel=0.0,
    fr_wheel_vel=0.0,
    rl_wheel_vel=0.0,
    rr_wheel_vel=0.0,
    fl_susp_pos=0.0,
    fr_susp_pos=0.0,
    rl_susp_pos=0.0,
    rr_susp_pos=0.0,
    lidar_000=0.0,
    lidar_045=0.0,
    lidar_090=0.0,
    lidar_135=0.0,
    lidar_180=0.0,
    lidar_225=0.0,
    lidar_270=0.0,
    lidar_315=0.0,
)

# Each left beam and the right beam at the same angle. lidar_000 and lidar_180
# lie on the X axis, so a mirror image does not change them.
_MIRROR_PAIRS = (
    ("lidar_045", "lidar_315"),
    ("lidar_090", "lidar_270"),
    ("lidar_135", "lidar_225"),
)


def fake_readings(**overrides: float) -> SensorReadings:
    """Return the reading of a level car at rest: every field is zero, except
    imu_quat, which is the identity, and the scalar fields named in
    overrides, which get the given value."""
    return replace(_ZERO, **overrides)


def beams(**overrides: float) -> SensorReadings:
    """Return fake_readings with every lidar beam at 1.0 m, except the beams
    named in overrides, which get the given value (m)."""
    return fake_readings(**{name: 1.0 for name in LIDAR_BEAM_ORDER} | overrides)


def random_beams(rng: np.random.Generator) -> SensorReadings:
    """Return fake_readings with a random value for each lidar beam. Most
    values are distances in [0.05, 4.0) m. About one value in ten is -1 (no
    hit), NaN or Inf, so the controllers also see bad data."""
    values = rng.uniform(0.05, 4.0, len(LIDAR_BEAM_ORDER))
    special = rng.random(len(LIDAR_BEAM_ORDER)) < 0.1
    values[special] = rng.choice([-1.0, np.nan, np.inf], int(special.sum()))
    return fake_readings(**dict(zip(LIDAR_BEAM_ORDER, values.tolist(), strict=True)))


def mirror(readings: SensorReadings) -> SensorReadings:
    """Return the readings of the scene mirrored left to right (+Y becomes
    -Y). Each left beam swaps its value with the right beam at the same
    angle."""
    swapped = {}
    for left, right in _MIRROR_PAIRS:
        swapped[left] = getattr(readings, right)
        swapped[right] = getattr(readings, left)
    return replace(readings, **swapped)
