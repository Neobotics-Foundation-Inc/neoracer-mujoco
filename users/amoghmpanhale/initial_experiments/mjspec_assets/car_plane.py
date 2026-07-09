'''
This file shows how to build a simple car from scratch using the MjSpec API.
It is designed and commented so that you can read through it and understand
how to create your own custom mjspec models. Ideally the actual Car class
would not be oversimplified like this. I just didn't follow until I did it
myself and had Claude teach me :-)
'''

import mujoco
import mujoco.viewer
import time


class CarPlane:
    def __init__(self):
        self.friction = [1.5, 0.05, 0.01]  # [sliding, torsional, rolling] friction coefficients
        self.gravity = [0, 0, -9.81]

        # MjSpec is a scene builder. You describe bodies, geoms, joints, etc.
        # programmatically, then call .compile() to get an MjModel.
        self.spec = self._build_spec()

        # MjModel is the compiled, static description of the world — geometry,
        # mass, joint structure. It doesn't change during simulation.
        self.model = self.spec.compile()

        # MjData holds the live simulation state: positions (qpos), velocities
        # (qvel), control inputs (ctrl), forces, etc. This is what changes each step.
        self.data = mujoco.MjData(self.model)

    def _build_spec(self):
        spec = mujoco.MjSpec()
        spec.option.gravity = self.gravity

        # worldbody is the root of the scene tree. Everything gets added to it
        # (or to bodies nested inside it).
        spec.worldbody.add_light(pos=[0, 0, 5], dir=[0, 0, -1])

        # Geoms are the physical/visual shapes. A plane geom with no parent body
        # is static — it won't move.
        spec.worldbody.add_geom(
            name="ground",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[30, 30, 0.1],
            friction=self.friction,
        )

        # Bodies are the movable objects. pos is relative to the parent body.
        car = spec.worldbody.add_body(name="car", pos=[0, 0, 0.5])

        # A freejoint gives this body all 6 degrees of freedom (x, y, z + rotation).
        # Without a joint, a body is welded to its parent and can't move.
        car.add_freejoint()

        # Each body can have one or more geoms for its shape/mass/collision.
        # size for a box is [half-length, half-width, half-height].
        car.add_geom(
            name="car_body",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[1, 0.5, 0.2],
        )

        # Wheels are child bodies of the car. Their pos is relative to the car body.
        # [front/rear, left/right, up/down] — wheels sit below and to the sides.
        for name, pos in [
            ("fl_wheel", [0.8,  0.6, -0.2]),   # front-left
            ("fr_wheel", [0.8, -0.6, -0.2]),   # front-right
            ("rl_wheel", [-0.8,  0.6, -0.2]),  # rear-left
            ("rr_wheel", [-0.8, -0.6, -0.2]),  # rear-right
        ]:
            wheel = car.add_body(name=name, pos=pos)

            # A hinge joint lets this body rotate around one axis relative to its
            # parent. axis=[0,1,0] means it spins around the Y axis — i.e. it rolls
            # forward/backward like a real wheel.
            wheel.add_joint(
                name=f"{name}_joint",
                type=mujoco.mjtJoint.mjJNT_HINGE,
                axis=[0, 1, 0],
            )

            # Cylinder geom for the wheel. size=[radius, half-length].
            # euler=[90,0,0] rotates it 90° so the flat face points sideways
            # (cylinders are vertical by default).
            # condim=6 enables full friction: sliding + torsional + rolling.
            wheel.add_geom(
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[0.2, 0.1],
                euler=[90, 0, 0],
                friction=self.friction,
                condim=6,
            )

        return spec

    def step(self):
        # Advance the simulation by one timestep (default: 2ms).
        # This integrates physics: applies forces, resolves contacts, updates qpos/qvel.
        mujoco.mj_step(self.model, self.data)

    @property
    def velocity(self):
        # qvel holds all generalized velocities. For a freejoint, the first 3
        # are linear velocity [vx, vy, vz] in world coordinates.
        return self.data.qvel[0:3]


if __name__ == "__main__":
    car = CarPlane()
    NUM_STEPS = 5
    step = 0

    with mujoco.viewer.launch_passive(car.model, car.data) as viewer:
        while viewer.is_running():
            if step < NUM_STEPS:
                car.data.qvel[0] = 3.0

            car.step()
            viewer.sync()

            print(f"velocity: {car.velocity}")

            time.sleep(0.01)
            step += 1
