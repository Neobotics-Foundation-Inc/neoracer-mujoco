"""Scripted reference controller: pure pursuit + speed PD on the ground-truth
centerline. Needed by T1 (drivability), T5 (projection), T7 (wall-ride calib).

Consumes env.state.s (the committed projection anchor) so it always looks ahead
from where the car actually is on the track, never a global search.
"""

import numpy as np

from .config import DriveConfig


def _yaw(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def scripted_action(env, cfg: DriveConfig = DriveConfig(), bias=0.0):
    """One [steer, throttle] action in [-1,1]^2.

    bias in [-1,1]: shifts the pursuit target laterally by bias * w toward a wall
    (used by the T7 wall-ride variant). 0 = track the centerline.
    """
    track = env.track
    M, L = track.C.shape[0], track.L
    s = env.state.s

    pos = env.data.qpos[0:2].copy()
    yaw = _yaw(env.data.qpos[3:7])

    # lookahead point on the centerline (optionally biased toward a boundary)
    i_look = int((s + cfg.lookahead) / L * M) % M
    target = track.C[i_look] + bias * track.w[i_look] * track.Nrm[i_look]

    # pure pursuit: curvature to reach target expressed in the car frame
    rel = target - pos
    c, sn = np.cos(yaw), np.sin(yaw)
    y_local = -sn * rel[0] + c * rel[1]
    Ld = np.linalg.norm(rel)
    curvature = 2.0 * y_local / max(Ld * Ld, 1e-6)
    steer = float(np.clip(curvature * cfg.wheelbase, -1.0, 1.0))

    # speed target from the tightest curvature over the next kappa_ahead_dist
    i_now = int(s / L * M) % M
    n_ahead = max(1, round(cfg.kappa_ahead_dist * M / L))
    idxs = (i_now + np.arange(n_ahead)) % M
    kappa_ahead = np.max(np.abs(track.kappa[idxs]))
    v_target = np.sqrt(cfg.a_lat_max / max(kappa_ahead, cfg.kappa_floor))
    v_target = float(np.clip(v_target, cfg.v_min, cfg.v_max))

    # current forward speed (body-frame vx) and tanh-clipped PD throttle
    vx, vy = float(env.data.qvel[0]), float(env.data.qvel[1])
    v_fwd = c * vx + sn * vy
    throttle = float(np.tanh(cfg.kp_speed * (v_target - v_fwd)))

    return np.array([steer, throttle])
