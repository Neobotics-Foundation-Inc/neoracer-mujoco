"""Helpers for composing a drivable scene: a car placed on a procedural track."""

from typing import Callable

import mujoco

from .car import load_car_spec


def build_track_scene(
    add_track: Callable[[mujoco.MjSpec], None],
    xml_path: str = None,
) -> mujoco.MjSpec:
    """Load the car model and drop it onto a procedurally built track.

    ``add_track`` is one of the ``add_*_track`` functions from ``spec_assets.tracks``;
    it adds its own ground plane, walls and checkpoint sites to the spec. The car
    model's own (small) ground plane is removed first so the track's ground is used.

    Returns the uncompiled MjSpec so callers can tweak it before ``compile()``.
    """
    spec = load_car_spec(xml_path) if xml_path else load_car_spec()

    for geom in list(spec.worldbody.geoms):
        if geom.type == mujoco.mjtGeom.mjGEOM_PLANE:
            spec.delete(geom)

    spec.option.timestep = 0.005
    spec.option.gravity = [0, 0, -9.81]
    spec.worldbody.add_light(pos=[0, 0, 8])

    add_track(spec)
    return spec
