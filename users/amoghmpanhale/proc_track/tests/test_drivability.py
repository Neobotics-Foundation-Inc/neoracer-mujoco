"""T1 scripted drivability + T5 projection consistency (one rollout, both checks).

A classical pure-pursuit controller must lap every phase-3 track with zero
crash terminations. If it can't, the generator or tunables are broken -- fix
those, never relax this gate. T5 rides on the same trajectories.
"""

import pytest

from proc_track.drive import scripted_action
from proc_track.env import ProcTrackEnv


@pytest.fixture(scope="session")
def env():
    return ProcTrackEnv(phase=3)


# spec is 30 seeds; keep 30 -- each lap is ~700 control steps, stays well under a minute
@pytest.mark.parametrize("seed", range(30))
def test_T1_drivability_and_T5_projection(env, seed):
    env.reset(seed)
    L = env.track.L
    invalid = 0
    worst_lat_frac = 0.0
    terminated = False

    max_steps = int(env.cfg.e6_time / (0.002 * env.cfg.frame_skip))  # 3000
    for _ in range(max_steps):
        a = scripted_action(env)
        _, _, terminated, truncated, _ = env.step(a)
        proj = env.state.last_proj
        if not proj.valid:
            invalid += 1
        else:
            worst_lat_frac = max(worst_lat_frac, abs(proj.lateral) / env.track.w[proj.j])
        if terminated or truncated:
            break

    # T1: full lap, no crash termination (E1-E4)
    assert env.state.lapped, f"seed {seed}: no lap (progress {env.state.total_progress:.2f}/{L:.2f})"
    assert not terminated, f"seed {seed}: crashed (E1-E4)"

    # T5: zero invalid projections, Sum ds == L +/- 0.1, |lateral| <= w throughout
    assert invalid == 0, f"seed {seed}: {invalid} invalid projections"
    assert abs(env.state.total_progress - L) < 0.1, (
        f"seed {seed}: Sum ds {env.state.total_progress:.3f} vs L {L:.3f}"
    )
    assert worst_lat_frac <= 1.0, f"seed {seed}: |lateral|/w peaked at {worst_lat_frac:.2f}"
