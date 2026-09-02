"""
Dense synthetic LiDAR raycasting against a live MuJoCo scene -- experimental,
used ONLY by this controller's closed-loop harness. Does NOT modify
assets/neoracer.xml: the model's own 8 physical rangefinder beams are
untouched; this casts extra rays with mujoco.mj_ray() from the same
lidar_mount site instead, to emulate a real dense sensor (e.g. Lakibeam L1,
270 deg FOV, up to 1080 points) at any sample count N for testing how the
vehicle_space_controller behaves as N changes.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

FOV_DEG = 270.0
MAX_RANGE_M = 12.0


def cast_dense_lidar(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    n: int,
    car_body_name: str = "car",
    site_name: str = "lidar_mount",
    fov_deg: float = FOV_DEG,
    max_range: float = MAX_RANGE_M,
) -> tuple[np.ndarray, np.ndarray]:
    """Cast N rays from the car's lidar_mount site across the real fov_deg
    FOV, symmetric about the body's own +X (forward) axis. Returns
    (angles_rad, ranges_m) in vehicle-frame physical angle order, +left,
    matching synthetic_scenes.cast_scan()'s convention exactly so the same
    controller code path is exercised in both. -1.0 = no return within
    max_range (same no-hit convention as the model's own rangefinders)."""
    site_id = model.site(site_name).id
    origin = data.site_xpos[site_id].copy()
    body_id = model.body(car_body_name).id
    body_mat = data.xmat[body_id].reshape(3, 3)
    forward = body_mat[:, 0]
    left = body_mat[:, 1]

    half = math.radians(fov_deg) / 2.0
    angles = np.linspace(-half, half, n)
    ranges = np.full(n, -1.0)
    geomid = np.zeros(1, dtype=np.int32)

    for i, theta in enumerate(angles):
        direction = math.cos(theta) * forward + math.sin(theta) * left
        dist = mujoco.mj_ray(model, data, origin, direction, None, 1, body_id, geomid)
        if 0.0 <= dist <= max_range:
            ranges[i] = dist
    return angles, ranges
