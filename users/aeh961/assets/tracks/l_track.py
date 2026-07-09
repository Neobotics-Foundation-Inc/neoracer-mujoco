import mujoco


def add_l_track(spec: mujoco.MjSpec) -> None:
    spec.worldbody.add_geom(
        name="l_track_ground",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[30, 30, 0.1],
        rgba=[0.18, 0.18, 0.18, 1],
        friction=[1.5, 0.1, 0.1],
    )

    walls = [
        ("bottom_wall", [0, -4, 0.2], [8, 0.15, 0.35]),
        ("left_wall", [-8, 0, 0.2], [0.15, 4, 0.35]),
        ("right_lower_wall", [8, -1.5, 0.2], [0.15, 2.5, 0.35]),
        ("middle_wall", [2, 1, 0.2], [6, 0.15, 0.35]),
        ("top_wall", [-3, 4, 0.2], [5, 0.15, 0.35]),
        ("upper_right_wall", [2, 2.5, 0.2], [0.15, 1.5, 0.35]),
    ]

    for name, pos, size in walls:
        spec.worldbody.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=pos,
            size=size,
            rgba=[0.8, 0.8, 0.8, 1],
        )

    for i, point in enumerate(get_l_track_checkpoints()):
        spec.worldbody.add_site(
            name=f"l_checkpoint_{i + 1}",
            pos=[point[0], point[1], 0.15],
            size=[0.25],
            rgba=[0, 1, 0, 1],
        )

    spec.worldbody.add_geom(
        name="l_start_finish",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[-6, -3.2, 0.02],
        size=[0.08, 0.7, 0.02],
        rgba=[1, 1, 1, 1],
        contype=0,
        conaffinity=0,
    )


def get_l_track_checkpoints() -> list[list[float]]:
    return [
        [-6, -3.2],
        [-2, -3.2],
        [4, -3.2],
        [6.5, -2],
        [6, 0.5],
        [2, 0.4],
        [0, 2.2],
        [-3, 3.2],
        [-6.5, 2.5],
        [-6.5, -1],
    ]


def get_l_track_spawn_pose() -> dict:
    return {
        "position": [-6, -3.2, 0.45],
        "heading": 0.0,
    }