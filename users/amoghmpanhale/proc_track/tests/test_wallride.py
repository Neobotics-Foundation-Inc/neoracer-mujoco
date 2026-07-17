"""T7 wall-ride calibration.

Frictionless walls make riding a wall physically cheap, so the per-step contact
penalty (c_wall) must make it economically expensive: a policy that hugs the
outer wall for a lap must earn <= 0.5x the return of a clean centerline lap.

The wall-ride reference uses the scripted controller with bias=0.9 (target
pushed toward the outer boundary) so the car laps while dragging the wall.
Measured with c_wall=0.05: wall-ride return ~0.33x clean on the max-curvature
seed -- comfortably under the 0.5 gate, so 0.05 stands (no escalation to
normal-force scaling, deferred D2).
"""

import pytest

from proc_track.drive import scripted_action
from proc_track.env import ProcTrackEnv

WALLRIDE_BIAS = 0.9


@pytest.fixture(scope="session")
def env():
    return ProcTrackEnv(phase=3)


def _return(env, seed, bias):
    env.reset(seed)
    total = 0.0
    contact = 0
    for _ in range(3000):
        a = scripted_action(env, bias=bias)
        _, r, term, trunc, info = env.step(a)
        total += r
        contact += info["wall_contact"]
        if term or trunc:
            break
    return total, contact


# high-curvature seeds (tightest corners == strongest wall-ride temptation)
@pytest.mark.parametrize("seed", [4, 9, 13])
def test_T7_wallride_penalized(env, seed):
    clean, clean_contact = _return(env, seed, bias=0.0)
    ride, ride_contact = _return(env, seed, bias=WALLRIDE_BIAS)

    assert clean_contact == 0, f"seed {seed}: clean lap touched wall {clean_contact}x"
    assert ride_contact > 50, f"seed {seed}: wall-ride only {ride_contact} contact steps (not a ride)"
    assert ride <= 0.5 * clean, (
        f"seed {seed}: wall-ride return {ride:.2f} > 0.5 * clean {clean:.2f} "
        f"-- raise c_wall"
    )
