import mujoco
from utils import DEFAULT_PHYSICS


class Plane:
    def __init__(self, spec: mujoco.MjSpec, physics=DEFAULT_PHYSICS):
        self._build(spec, physics)

    def _build(self, spec, physics):
        spec.option.gravity = physics["gravity"]
        spec.worldbody.add_light(pos=[0, 0, 5], dir=[0, 0, -1])
        spec.worldbody.add_geom(
            name="ground",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[30, 30, 0.1],
            friction=physics["ground_friction"],
        )
