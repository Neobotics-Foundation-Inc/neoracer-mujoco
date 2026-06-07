import math

import mujoco


def add_oval_track(
    spec: mujoco.MjSpec,
    track_length: float = 14.0,
    track_width: float = 7.0,
    wall_height: float = 0.35,
    wall_thickness: float = 0.15,
) -> None:
    """
    Add a simple oval-style track to an existing MjSpec scene.

    This is an early environment asset, not a final NeoRacer track.
    It creates:
    - ground plane
    - outside boundary walls
    - inside boundary walls
    - start/finish marker
    - checkpoint markers
    """

    # Ground
    spec.worldbody.add_geom(
        name="track_ground",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[30, 30, 0.1],
        rgba=[0.2, 0.2, 0.2, 1],
        friction=[1.5, 0.1, 0.1],
    )

    half_length = track_length / 2
    half_width = track_width / 2

    # Outer boundary walls
    wall_specs = [
        ("outer_top", [0, half_width, wall_height / 2], [half_length, wall_thickness, wall_height]),
        ("outer_bottom", [0, -half_width, wall_height / 2], [half_length, wall_thickness, wall_height]),
        ("outer_left", [-half_length, 0, wall_height / 2], [wall_thickness, half_width, wall_height]),
        ("outer_right", [half_length, 0, wall_height / 2], [wall_thickness, half_width, wall_height]),
    ]

    for name, pos, size in wall_specs:
        spec.worldbody.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=pos,
            size=size,
            rgba=[0.8, 0.8, 0.8, 1],
        )

    # Inner boundary walls to create a lane
    inner_length = track_length * 0.45
    inner_width = track_width * 0.35

    inner_wall_specs = [
        ("inner_top", [0, inner_width, wall_height / 2], [inner_length, wall_thickness, wall_height]),
        ("inner_bottom", [0, -inner_width, wall_height / 2], [inner_length, wall_thickness, wall_height]),
        ("inner_left", [-inner_length, 0, wall_height / 2], [wall_thickness, inner_width, wall_height]),
        ("inner_right", [inner_length, 0, wall_height / 2], [wall_thickness, inner_width, wall_height]),
    ]

    for name, pos, size in inner_wall_specs:
        spec.worldbody.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=pos,
            size=size,
            rgba=[0.4, 0.4, 0.4, 1],
        )

    # Start/finish line
    spec.worldbody.add_geom(
        name="start_finish_line",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0, -half_width + 0.7, 0.02],
        size=[0.08, 0.8, 0.02],
        rgba=[1, 1, 1, 1],
        contype=0,
        conaffinity=0,
    )

    # Visual checkpoint markers
    checkpoint_positions = get_oval_checkpoints(track_length, track_width)

    for i, pos in enumerate(checkpoint_positions):
        spec.worldbody.add_site(
            name=f"checkpoint_{i + 1}",
            pos=[pos[0], pos[1], 0.15],
            size=[0.25],
            rgba=[0, 1, 0, 1],
        )


def get_oval_checkpoints(
    track_length: float = 14.0,
    track_width: float = 7.0,
) -> list[list[float]]:
    """
    Return checkpoint positions around the oval track.

    These are simple centerline-style waypoints.
    Later these can be used for rewards, lap counting, and reset logic.
    """

    half_length = track_length / 2
    half_width = track_width / 2

    return [
        [0, -half_width + 0.9],
        [half_length - 0.9, -half_width + 0.9],
        [half_length - 0.9, 0],
        [half_length - 0.9, half_width - 0.9],
        [0, half_width - 0.9],
        [-half_length + 0.9, half_width - 0.9],
        [-half_length + 0.9, 0],
        [-half_length + 0.9, -half_width + 0.9],
    ]


def get_oval_track_spawn_pose(
    track_length: float = 14.0,
    track_width: float = 7.0,
) -> dict:
    """Spawn centered in the bottom lane, facing +x toward the first checkpoint."""
    # Lane center sits halfway between the outer wall (track_width/2) and the
    # inner wall (track_width * 0.35) so the car doesn't clip a wall on reset.
    lane_center_y = -(track_width / 2 + track_width * 0.35) / 2
    return {
        "position": [0.0, lane_center_y, 0.15],
        "heading": 0.0,
    }