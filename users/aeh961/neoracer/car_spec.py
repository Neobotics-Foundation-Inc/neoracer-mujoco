import mujoco


def add_motor_actuator(spec, name, joint, gear):
    actuator = spec.add_actuator()
    actuator.name = name
    actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
    actuator.target = joint.name
    actuator.gear[0] = gear
    actuator.gainprm[0] = 1
    return actuator


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


def add_wheel(parent, name, pos, steerable=False):
    if steerable:
        steer_body = parent.add_body(name=f"{name}_steer", pos=pos)

        steer_joint = steer_body.add_joint(
            name=f"{name}_steer_joint",
            type=mujoco.mjtJoint.mjJNT_HINGE,
            axis=[0, 0, 1],
            range=[-0.6, 0.6],
        )

        steer_body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.04],
            mass=0.2,
            rgba=[1, 0, 0, 1],
        )

        wheel_body = steer_body.add_body(name=f"{name}_wheel")

    else:
        steer_joint = None
        wheel_body = parent.add_body(name=f"{name}_wheel", pos=pos)

    wheel_joint = wheel_body.add_joint(
        name=f"{name}_wheel_joint",
        type=mujoco.mjtJoint.mjJNT_HINGE,
        axis=[0, 1, 0],
    )

    wheel_body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[0.22, 0.08],
        euler=[90, 0, 0],
        mass=1,
        friction=[1.5, 0.1, 0.1],
        rgba=[0, 0, 1, 1] if steerable else [0.2, 0.2, 0.2, 1],
    )

    return wheel_joint, steer_joint


def build_demo_car_spec():
    spec = mujoco.MjSpec()

    spec.option.timestep = 0.005
    spec.option.gravity = [0, 0, -9.81]

    spec.worldbody.add_light(pos=[0, 0, 5])

    spec.worldbody.add_geom(
        name="ground",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[30, 30, 0.1],
        friction=[1.5, 0.1, 0.1],
    )

    car = spec.worldbody.add_body(name="car", pos=[0, 0, 0.45])
    car.add_freejoint()

    car.add_geom(
        name="car_body",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[1.2, 0.5, 0.18],
        mass=20,
        rgba=[0.7, 0.7, 0.7, 1],
    )

    _, front_left_steer_joint = add_wheel(
        car, "front_left", [0.8, 0.45, -0.22], steerable=True
    )

    _, front_right_steer_joint = add_wheel(
        car, "front_right", [0.8, -0.45, -0.22], steerable=True
    )

    rear_left_wheel_joint, _ = add_wheel(
        car, "rear_left", [-0.8, 0.45, -0.22], steerable=False
    )

    rear_right_wheel_joint, _ = add_wheel(
        car, "rear_right", [-0.8, -0.45, -0.22], steerable=False
    )

    add_motor_actuator(spec, "rear_left_motor", rear_left_wheel_joint, gear=8)
    add_motor_actuator(spec, "rear_right_motor", rear_right_wheel_joint, gear=8)

    add_position_actuator(
        spec, "front_left_steer_position", front_left_steer_joint, kp=40
    )

    add_position_actuator(
        spec, "front_right_steer_position", front_right_steer_joint, kp=40
    )

    return spec