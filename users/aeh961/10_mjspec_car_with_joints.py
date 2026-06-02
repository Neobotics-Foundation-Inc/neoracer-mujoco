import time
import math

import mujoco
import mujoco.viewer


# Creates a motor actuator that applies torque to a wheel joint.
# Used for the rear drive wheels.
def add_motor_actuator(spec, name, joint, gear):
    actuator = spec.add_actuator()
    actuator.name = name
    actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
    actuator.target = joint.name
    actuator.gear[0] = gear
    actuator.gainprm[0] = 1
    return actuator


# Creates a position actuator that tries to move a steering joint
# to a target angle specified through data.ctrl.
def add_position_actuator(spec, name, joint, kp):
    actuator = spec.add_actuator()
    actuator.name = name
    actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
    actuator.target = joint.name

    actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
    actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE

    actuator.gainprm[0] = kp
    actuator.biasprm[1] = -kp

    return actuator


# Creates a wheel assembly.
#
# Rear wheel:
#   car
#    └── wheel
#
# Front wheel:
#   car
#    └── steering body
#          └── wheel
#
# Returns:
#   wheel_joint  -> wheel rotation
#   steer_joint  -> steering rotation (None for rear wheels)
def add_wheel(parent, name, pos, steerable=False):

    # Front wheels need a steering mount.
    if steerable:

        steer_body = parent.add_body(
            name=f"{name}_steer",
            pos=pos,
        )

        # Steering rotates around vertical (Z) axis.
        steer_joint = steer_body.add_joint(
            name=f"{name}_steer_joint",
            type=mujoco.mjtJoint.mjJNT_HINGE,
            axis=[0, 0, 1],
            range=[-0.6, 0.6],
        )

        # Small visible marker and mass carrier.
        steer_body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.04],
            mass=0.2,
            rgba=[1, 0, 0, 1],
        )

        wheel_body = steer_body.add_body(
            name=f"{name}_wheel",
        )

    else:

        steer_joint = None

        wheel_body = parent.add_body(
            name=f"{name}_wheel",
            pos=pos,
        )

    # Wheel rotation around axle.
    wheel_joint = wheel_body.add_joint(
        name=f"{name}_wheel_joint",
        type=mujoco.mjtJoint.mjJNT_HINGE,
        axis=[0, 1, 0],
    )

    # Physical wheel geometry.
    wheel_body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[0.22, 0.08],
        euler=[90, 0, 0],
        mass=1,
        friction=[1.5, 0.1, 0.1],
        rgba=[0, 0, 1, 1] if steerable else [0.2, 0.2, 0.2, 1],
    )

    return wheel_joint, steer_joint


# Builds the entire simulation world.
#
# World
# ├── Ground
# ├── Car Body
# ├── Front Left Steering Assembly
# ├── Front Right Steering Assembly
# ├── Rear Left Wheel
# ├── Rear Right Wheel
# ├── Rear Wheel Motors
# └── Front Steering Controllers
def build_car_spec():

    spec = mujoco.MjSpec()

    # Global simulation settings.
    spec.option.timestep = 0.005
    spec.option.gravity = [0, 0, -9.81]

    # Scene lighting.
    spec.worldbody.add_light(
        pos=[0, 0, 5],
    )

    # Ground plane.
    spec.worldbody.add_geom(
        name="ground",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[30, 30, 0.1],
        friction=[1.5, 0.1, 0.1],
    )

    # Main vehicle body.
    car = spec.worldbody.add_body(
        name="car",
        pos=[0, 0, 0.45],
    )

    # Allows the vehicle to move freely in the world.
    car.add_freejoint()

    car.add_geom(
        name="car_body",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[1.2, 0.5, 0.18],
        mass=20,
        rgba=[0.7, 0.7, 0.7, 1],
    )

    # Front steering assemblies.
    front_left_wheel_joint, front_left_steer_joint = add_wheel(
        car,
        name="front_left",
        pos=[0.8, 0.45, -0.22],
        steerable=True,
    )

    front_right_wheel_joint, front_right_steer_joint = add_wheel(
        car,
        name="front_right",
        pos=[0.8, -0.45, -0.22],
        steerable=True,
    )

    # Rear drive wheels.
    rear_left_wheel_joint, _ = add_wheel(
        car,
        name="rear_left",
        pos=[-0.8, 0.45, -0.22],
        steerable=False,
    )

    rear_right_wheel_joint, _ = add_wheel(
        car,
        name="rear_right",
        pos=[-0.8, -0.45, -0.22],
        steerable=False,
    )

    # Rear wheel motors.
    add_motor_actuator(
        spec,
        name="rear_left_motor",
        joint=rear_left_wheel_joint,
        gear=8,
    )

    add_motor_actuator(
        spec,
        name="rear_right_motor",
        joint=rear_right_wheel_joint,
        gear=8,
    )

    # Front wheel steering controllers.
    add_position_actuator(
        spec,
        name="front_left_steer_position",
        joint=front_left_steer_joint,
        kp=40,
    )

    add_position_actuator(
        spec,
        name="front_right_steer_position",
        joint=front_right_steer_joint,
        kp=40,
    )

    return spec


# Build editable specification.
spec = build_car_spec()

# Compile specification into runnable physics model.
model = spec.compile()

# Runtime state (positions, velocities, controls, etc.).
data = mujoco.MjData(model)


# Main simulation loop.
with mujoco.viewer.launch_passive(model, data) as viewer:

    t = 0

    while viewer.is_running():

        # Rear-wheel drive.
        data.ctrl[0] = 1.2
        data.ctrl[1] = 1.2

        # Automatic steering oscillation.
        steering = 0.35 * math.sin(t)

        data.ctrl[2] = steering
        data.ctrl[3] = steering

        # Advance physics.
        mujoco.mj_step(model, data)

        # Update viewer.
        viewer.sync()

        t += 0.02

        time.sleep(0.005)