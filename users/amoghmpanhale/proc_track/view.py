"""Watch the scripted controller lap a generated track in the MuJoCo viewer.

macOS needs mjpython for the passive viewer (not python3):

    mjpython users/amoghmpanhale/proc_track/view.py --seed 7

Autopilots with drive.scripted_action; on lap/crash it regenerates the next
seed so you can eyeball a stream of tracks. Ctrl-C or close the window to quit.
"""

import argparse
import sys
import time
from pathlib import Path

# make `import proc_track` work when run as a loose script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco.viewer

from proc_track.drive import scripted_action
from proc_track.env import ProcTrackEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--phase", type=int, default=3)
    args = ap.parse_args()

    env = ProcTrackEnv(phase=args.phase)
    seed = args.seed
    env.reset(seed)
    dt = env.cfg.frame_skip * 0.002  # control-step wall-clock target

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = env.car_bid
        viewer.cam.distance = 3.0
        viewer.cam.elevation = -40

        while viewer.is_running():
            loop_start = time.time()

            action = scripted_action(env)
            _, _, terminated, truncated, info = env.step(action)
            viewer.sync()

            if terminated or truncated:
                tag = "lap" if info["total_progress"] >= env.track.L else "crash"
                print(f"seed {seed}: {tag} (progress {info['total_progress']:.1f}/{env.track.L:.1f} m)")
                seed += 1
                env.reset(seed)

            remaining = dt - (time.time() - loop_start)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
