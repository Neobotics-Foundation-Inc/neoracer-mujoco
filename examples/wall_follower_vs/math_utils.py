"""
Shared math helpers for the vehicle-space wall follower trial harnesses.
Experimental; not part of `pytest validation/`.
"""

from __future__ import annotations

import numpy as np


def compose_yaw_perturbation(q: np.ndarray, yaw: float) -> list[float]:
    """Compose an extra yaw-about-+Z rotation (angle `yaw`, radians) onto
    quaternion `q` via the Hamilton product q * dq, returning the new
    [w, x, y, z] quaternion as a plain list.

    Used by closed_loop_trials.py and loop_trials.py to apply a seeded random
    spawn-yaw perturbation before each trial.

    This is the manual quaternion-multiply formula, extracted verbatim from
    both call sites (previously duplicated) rather than swapped for
    mujoco.mju_mulQuat: the two are mathematically equivalent but NOT
    bit-identical (mju_mulQuat's internal operation ordering differs by
    ~1 ULP from this formula's), and this function feeds the seeded random
    spawn pose used by the Stage D 100-trial validation -- so the exact
    original arithmetic is preserved here rather than switched for cleanliness.
    """
    half = yaw / 2.0
    dq = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])
    w0, x0, y0, z0 = q
    w1, x1, y1, z1 = dq
    return [
        w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
        w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
        w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
        w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
    ]
