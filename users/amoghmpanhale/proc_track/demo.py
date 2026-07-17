"""Guided tour of proc_track: generate a track, inspect it, plot it, drive it.

No viewer needed -- runs under plain python3 and saves a top-down PNG:

    python3 users/amoghmpanhale/proc_track/demo.py --seed 7
    python3 users/amoghmpanhale/proc_track/demo.py --seed 3 --phase 1 --jitter 0.6

One track flows through all four sections: the same track you see plotted is the
one the car drives. Tweak --phase / --jitter and rerun to explore track design.
"""

import argparse
import sys
from pathlib import Path

# make `import proc_track` work when run as a loose script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from proc_track.config import (
    EnvConfig,
    GenConfig,
    ProjConfig,
    WallConfig,
    gen_config_for_phase,
)
from proc_track.drive import scripted_action
from proc_track.env import ProcTrackEnv


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def build_env(phase, jitter):
    """An env whose generator uses the phase preset plus a custom jitter.

    phase 0-3 sets the curvature ceiling + min width; jitter controls how wiggly
    the layout is (higher = wilder, more S-bends, more generation rejects).
    """
    gcfg = gen_config_for_phase(phase)
    gcfg = GenConfig(**{**gcfg.__dict__, "jitter_std": jitter})
    cfg = EnvConfig(gen=gcfg, walls=WallConfig(), proj=ProjConfig())
    return ProcTrackEnv(cfg=cfg)


# ---------- 1. Inspect the generated track ----------

def inspect(track):
    section("1. GENERATE")
    # a Track is just immutable numpy arrays -- everything downstream reads these
    print(f"  lap length L      : {track.L:.2f} m")
    print(f"  centerline C      : {track.C.shape}  (x,y per sample)")
    print(f"  max |curvature|   : {np.abs(track.kappa).max():.3f} 1/m "
          f"(tightest corner radius {1 / np.abs(track.kappa).max():.2f} m)")
    print(f"  half-width w      : {track.w.min():.2f}-{track.w.max():.2f} m")


# ---------- 2. Plot it top-down (centerline + walls) ----------

def plot_track(track, seed):
    section("2. PLOT")
    import matplotlib.pyplot as plt

    # boundaries are just centerline offset by +/- half-width along the normal
    left = track.C + track.w[:, None] * track.Nrm
    right = track.C - track.w[:, None] * track.Nrm

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(track.C[:, 0], track.C[:, 1], "k--", lw=1, label="centerline")
    ax.plot(left[:, 0], left[:, 1], "b", lw=2)
    ax.plot(right[:, 0], right[:, 1], "b", lw=2, label="walls")
    ax.plot(track.C[0, 0], track.C[0, 1], "go", ms=10, label="s=0")
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title(f"seed {seed}  (L={track.L:.1f} m)")

    out = Path.cwd() / f"track_seed{seed}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"  saved {out}")


# ---------- 3. Drive it with the scripted controller ----------

def drive(env):
    section("3. DRIVE  (scripted pure-pursuit autopilot)")
    total_reward = 0.0
    wall_steps = 0
    obs = None
    for _ in range(3000):
        action = scripted_action(env)  # [steer, throttle], each in [-1, 1]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        wall_steps += info["wall_contact"]
        if terminated or truncated:
            break

    outcome = "LAP COMPLETE" if env.state.lapped else "crashed/stalled"
    print(f"  outcome           : {outcome}")
    print(f"  progress          : {info['total_progress']:.1f} / {env.track.L:.1f} m")
    print(f"  total reward      : {total_reward:.2f}")
    print(f"  steps in wall     : {wall_steps}")
    return obs


# ---------- 4. What the RL agent actually sees ----------

def explain_obs(obs):
    section("4. OBSERVATION  (23-dim Frenet vector the policy receives)")
    print(f"  shape {obs.shape}, dtype {obs.dtype}")
    print(f"  [0]     lateral offset / width      : {obs[0]:+.2f}")
    print(f"  [1:3]   heading error (sin,cos)     : {obs[1]:+.2f}, {obs[2]:+.2f}")
    print(f"  [3:5]   body-frame vx,vy            : {obs[3]:+.2f}, {obs[4]:+.2f}")
    print(f"  [5]     yaw rate                    : {obs[5]:+.2f}")
    print(f"  [6:10]  wheel speeds                : {np.round(obs[6:10], 2)}")
    print(f"  [10]    steering angle              : {obs[10]:+.2f}")
    print(f"  [11:13] previous action            : {np.round(obs[11:13], 2)}")
    print(f"  [13:23] curvature+width preview     : {np.round(obs[13:23], 2)}")
    print("  (no global position -> the policy can't memorize a specific track)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--phase", type=int, default=3, help="0=easy ... 3=hard")
    ap.add_argument("--jitter", type=float, default=0.3, help="track wiggle")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    env = build_env(args.phase, args.jitter)
    env.reset(args.seed)          # generates the track + places the car
    track = env.track             # the one track used everywhere below

    inspect(track)
    if not args.no_plot:
        plot_track(track, args.seed)
    obs = drive(env)              # drives the env we just reset
    explain_obs(obs)
    print()


if __name__ == "__main__":
    main()
