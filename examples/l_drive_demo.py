"""Drive the car around the L-shaped track with the arrow keys (or WASD)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assets.spec_assets.driving import run_keyboard_demo
from assets.spec_assets.scene import build_track_scene
from assets.spec_assets.tracks.l_track import add_l_track, get_l_track_spawn_pose


def main():
    spec = build_track_scene(add_l_track)
    run_keyboard_demo(spec, spawn=get_l_track_spawn_pose())


if __name__ == "__main__":
    main()
