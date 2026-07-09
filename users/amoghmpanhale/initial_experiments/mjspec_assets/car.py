import mujoco
from users.amoghmpanhale.initial_experiments.utils import DEFAULT_PHYSICS

WHEEL_POSITIONS = [
    ("fl_wheel", [0.8,  0.6, -0.2]),   # front-left
    ("fr_wheel", [0.8, -0.6, -0.2]),   # front-right
    ("rl_wheel", [-0.8,  0.6, -0.2]),  # rear-left
    ("rr_wheel", [-0.8, -0.6, -0.2]),  # rear-right
]


class Car:
    def __init__(self, spec: mujoco.MjSpec, parent_body, physics=DEFAULT_PHYSICS):
        self._build(spec, parent_body, physics)

    def _build(self, spec, parent_body, physics):
        car = parent_body.add_body(name="car", pos=[0, 0, 0.5])
        car.add_freejoint()
        car.add_geom(
            name="car_body",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[1, 0.5, 0.2],
        )
        for name, pos in WHEEL_POSITIONS:
            self._build_wheel(car, name, pos, physics["wheel_friction"])

    def _build_wheel(self, car_body, name, pos, friction):
        wheel = car_body.add_body(name=name, pos=pos)
        wheel.add_joint(
            name=f"{name}_joint",
            type=mujoco.mjtJoint.mjJNT_HINGE,
            axis=[0, 1, 0],
        )
        wheel.add_geom(
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[0.2, 0.1],
            euler=[90, 0, 0],
            friction=friction,
            condim=6,
        )

    def get_velocity(self, data: mujoco.MjData):
        return data.qvel[0:3]

    def get_position(self, data: mujoco.MjData):
        return data.qpos[0:3]
