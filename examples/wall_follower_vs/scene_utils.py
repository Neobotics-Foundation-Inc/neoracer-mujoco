"""
Scene loader for the vehicle-space wall follower's own trial harnesses --
narrowly scoped to the tracks this feature drives on (straight_corridor.xml,
loop_corridor.xml).

run_racecar_core.load_scene() is zero-argument (always straight_corridor.xml,
no spawn offset) on both `main` and this PR branch. The per-track,
spawn-offset version this feature was actually developed and Stage-D-
validated against -- load_scene(track), with loop_corridor.xml's
(0.0, -2.5, 0.0) bottom-straight offset -- exists only on the separate,
unmerged aeh961/racecar-core-adapter-experiments branch (examples/
run_racecar_core.py, commit b23cc48) and was never merged into main. PR #29's
loop_corridor.xml harnesses (loop_trials.py, watch_loop.py,
stage_d_100trial.py) were carried over assuming that API, so a clean checkout
of this PR branch could not reproduce its own loop_corridor.xml runs.

This module ports that behavior verbatim (same offset value, same
frame-based placement mechanism) so PR #29 stays reproducible from a clean
checkout without modifying run_racecar_core.py or depending on the
experimental branch. Not independently redesigned.
"""

from __future__ import annotations

import os

import mujoco

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_TRACK = "straight_corridor.xml"

# Ported verbatim from aeh961/racecar-core-adapter-experiments's
# _TRACK_SPAWN_OFFSET (examples/run_racecar_core.py, commit b23cc48):
# loop_corridor.xml's bottom straight sits at y=-2.5, midway between its
# inner (y=-2) and outer (y=-3) walls -- see that file's own docstring.
_TRACK_SPAWN_OFFSET = {
    "loop_corridor.xml": (0.0, -2.5, 0.0),
}


def load_track_scene(track: str = DEFAULT_TRACK) -> mujoco.MjModel:
    """Car centered in a walled corridor (assets/tracks/<track>), offset per
    _TRACK_SPAWN_OFFSET so it spawns centered in that track's own corridor,
    facing +X. Tracks not listed use no offset (straight_corridor.xml's
    convention)."""
    scene = mujoco.MjSpec.from_file(os.path.join(_REPO, "assets", "tracks", track))
    car = mujoco.MjSpec.from_file(os.path.join(_REPO, "assets", "neoracer.xml"))
    frame = scene.worldbody.add_frame()
    frame.pos = _TRACK_SPAWN_OFFSET.get(track, (0.0, 0.0, 0.0))
    frame.attach_body(car.body("car"), "", "")
    return scene.compile()
