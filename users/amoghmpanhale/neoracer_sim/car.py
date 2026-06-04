"""
A box car with four cylinder wheels. Rear wheels spin (driven by motors), front
wheels also steer. ctrl is [rear_left, rear_right, front_left_steer, front_right_steer].
"""

import mujoco
from . import config as c

HINGE = mujoco.mjtJoint.mjJNT_HINGE


def add_car(spec):
    car = spec.worldbody.add_body(name="car", pos=[0, 0, c.RIDE_HEIGHT])
    car.add_freejoint()
    car.add_geom(name="body", type=mujoco.mjtGeom.mjGEOM_BOX,
                 size=c.BODY_SIZE, mass=c.BODY_MASS, rgba=[0.7, 0.7, 0.7, 1])

    x, y, z = c.WHEEL_X, c.WHEEL_Y, c.WHEEL_Z
    add_wheel(car, "rear_left",   [-x,  y, z])
    add_wheel(car, "rear_right",  [-x, -y, z])
    steered_wheel(car, "front_left",  [x,  y, z])
    steered_wheel(car, "front_right", [x, -y, z])

    motor(spec, "rear_left")
    motor(spec, "rear_right")
    steer_motor(spec, "front_left")
    steer_motor(spec, "front_right")


def add_wheel(parent, name, pos):
    """A wheel that spins."""
    wheel = parent.add_body(name=name, pos=pos)
    wheel.add_joint(name=name + "_spin", type=HINGE, axis=[0, 1, 0])
    wheel.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[c.WHEEL_RADIUS, c.WHEEL_WIDTH],
                   euler=[90, 0, 0], mass=c.WHEEL_MASS, friction=c.WHEEL_FRICTION)


def steered_wheel(parent, name, pos):
    """A wheel that turns about Z (steers), with a spinning wheel hung off it."""
    steer = parent.add_body(name=name, pos=pos)
    steer.add_joint(name=name + "_steer", type=HINGE, axis=[0, 0, 1],
                    range=[-c.STEER_LIMIT, c.STEER_LIMIT])
    steer.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.04], mass=0.2)
    add_wheel(steer, name + "_wheel", [0, 0, 0])


def motor(spec, name):
    """Torque motor on a wheel's spin joint."""
    a = spec.add_actuator()
    a.name = name + "_motor"
    a.trntype = mujoco.mjtTrn.mjTRN_JOINT
    a.target = name + "_spin"
    a.gear[0] = c.MOTOR_GEAR
    a.gainprm[0] = 1


def steer_motor(spec, name):
    """Position servo on a wheel's steer joint."""
    a = spec.add_actuator()
    a.name = name + "_steer_motor"
    a.trntype = mujoco.mjtTrn.mjTRN_JOINT
    a.target = name + "_steer"
    a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
    a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
    a.gainprm[0] = c.STEER_KP
    a.biasprm[1] = -c.STEER_KP
