"""Projection unit checks on a synthetic circle (analytic ground truth).

Marching the car exactly along the centerline once around must accumulate
Sum(ds) == circumference. Also: lateral offset recovery, and the validity gate.
"""

import numpy as np

from proc_track.config import ProjConfig
from proc_track.generator import Track
from proc_track.projection import project

CFG = ProjConfig()


def circle_track(R=5.0, M=2048, w=0.5):
    """Exact circle of radius R as a Track. Counterclockwise -> kappa = +1/R."""
    ang = np.linspace(0, 2 * np.pi, M, endpoint=False)
    C = np.stack([R * np.cos(ang), R * np.sin(ang)], axis=1)
    T = np.stack([-np.sin(ang), np.cos(ang)], axis=1)  # ccw tangent
    Nrm = np.stack([-T[:, 1], T[:, 0]], axis=1)
    kappa = np.full(M, 1.0 / R)
    L = 2 * np.pi * R
    s = np.arange(M) * L / M
    return Track(C=C, T=T, Nrm=Nrm, kappa=kappa, w=np.full(M, w), s=s, L=L, seed=0)


class _State:
    def __init__(self, s):
        self.s = s


def test_sum_ds_equals_circumference():
    track = circle_track()
    state = _State(0.0)
    total = 0.0
    # march one full lap on the centerline, one step per sample
    for i in range(track.C.shape[0]):
        res = project(track, state, track.C[i], CFG)
        assert res.valid
        total += res.ds
    # close the loop back to the start sample
    total += project(track, state, track.C[0], CFG).ds
    assert abs(total - track.L) < 0.1, f"Sum ds {total:.3f} vs L {track.L:.3f}"


def test_lateral_offset_recovered():
    track = circle_track()
    state = _State(track.s[100])
    offset = 0.2
    xy = track.C[100] + offset * track.Nrm[100]
    res = project(track, state, xy, CFG)
    assert res.valid
    assert res.j == 100
    assert abs(res.lateral - offset) < 1e-6


def test_gate_rejects_far_point_and_holds_s():
    track = circle_track()
    state = _State(track.s[50])
    far = track.C[50] + 10.0 * track.Nrm[50]  # way outside gate_scale*w + margin
    res = project(track, state, far, CFG)
    assert not res.valid
    assert res.ds == 0.0
    assert np.isinf(res.lateral)
    assert state.s == track.s[50]  # s not committed on invalid


def test_backwards_gives_negative_ds():
    track = circle_track()
    state = _State(track.s[100])
    res = project(track, state, track.C[95], CFG)  # behind, within back window
    assert res.valid
    assert res.ds < 0
