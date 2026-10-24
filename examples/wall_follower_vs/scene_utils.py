"""
Scene loading for the trial scripts of the vehicle-space wall follower.

Frame: +X forward, +Y left, +Z up.
"""

import os
from typing import Literal

import mujoco

from neoracer_mujoco import assets
from neoracer_mujoco.scene import Scene

type TrackName = Literal["straight_corridor.xml", "loop_corridor.xml"]


def load_track_scene(track: TrackName) -> Scene:
    """Compile assets/tracks/<track> with the car at the track's start."""
    return assets.compose(
        mujoco.MjSpec.from_file(os.path.join(assets.TRACK_DIR, track))
    )
