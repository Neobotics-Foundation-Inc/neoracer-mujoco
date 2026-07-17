"""Generator conformance: T3 determinism, T6 robustness, V1-V4 per track.

Seed counts are the plan's; T6 stays under a couple seconds locally.
"""

import time

import numpy as np
import pytest

from proc_track.config import gen_config_for_phase
from proc_track.generator import (
    TrackGenerationError,
    sample_valid_track,
    wrap,
)


PHASE3 = gen_config_for_phase(3)


def _assert_valid(track, cfg):
    """V1-V4 asserted on a finished track (must already satisfy `check`)."""
    kappa, w, s, L = track.kappa, track.w, track.s, track.L
    # V1
    assert np.max(np.abs(kappa)) <= cfg.kappa_max + 1e-9
    # V2
    assert np.max(w * np.abs(kappa)) <= cfg.wk_max + 1e-9
    # V4
    assert cfg.L_min <= L <= cfg.L_max
    # V3: no far-apart pair closer than the clearance
    clearance = 2 * w.max() + cfg.clearance_margin
    from scipy.spatial import cKDTree

    for i, j in cKDTree(track.C).query_pairs(r=clearance):
        if abs(wrap(s[j] - s[i], L)) > cfg.arc_sep_threshold:
            pytest.fail(f"V3 violated: samples {i},{j}")


def test_track_shapes_and_arc():
    track = sample_valid_track(7, PHASE3)
    M = PHASE3.M
    assert track.C.shape == (M, 2)
    assert track.T.shape == (M, 2)
    assert track.Nrm.shape == (M, 2)
    assert track.kappa.shape == (M,)
    assert track.w.shape == (M,)
    # s[i] = i * L / M
    assert np.allclose(track.s, np.arange(M) * track.L / M)
    # tangents unit, normals are left-rotated tangents
    assert np.allclose(np.linalg.norm(track.T, axis=1), 1.0)
    assert np.allclose(track.Nrm, np.stack([-track.T[:, 1], track.T[:, 0]], 1))


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 999])
def test_validity_V1_V4(seed):
    _assert_valid(sample_valid_track(seed, PHASE3), PHASE3)


@pytest.mark.parametrize("seed", [0, 3, 17, 123, 2024])
def test_T3_determinism(seed):
    a = sample_valid_track(seed, PHASE3)
    b = sample_valid_track(seed, PHASE3)
    for name in ("C", "T", "Nrm", "kappa", "w", "s"):
        assert np.array_equal(getattr(a, name), getattr(b, name)), name
    assert a.L == b.L


def test_T6_robustness_phase0():
    """1000 seeds at phase 0 (tightest kappa_max). Failure rate < 0.1%,
    p99 gen time comfortably fast. Plan spec target is 20 ms; assert a looser
    ceiling to stay non-flaky across machines."""
    cfg = gen_config_for_phase(0)
    times = []
    failures = 0
    for seed in range(1000):
        t0 = time.perf_counter()
        try:
            sample_valid_track(seed, cfg)
        except TrackGenerationError:
            failures += 1
        times.append(time.perf_counter() - t0)

    assert failures / 1000 < 0.001, f"{failures} failures / 1000"
    p99 = np.percentile(times, 99)
    assert p99 < 0.1, f"p99 gen time {p99*1000:.1f} ms (spec target 20 ms)"
