"""Stage 6a: stateful local projection of the car onto the centerline.

Forward-biased window around the last committed arc position. Global
nearest-point projection is ambiguous where two track sections pass near each
other (S-bends, near-crossings), so after spawn we only ever search a local
window and commit forward.

The state object just needs a mutable float `s` (the last committed arc length);
env.py's EnvState supplies it. `project` mutates state.s ONLY on a valid result.
"""

from dataclasses import dataclass

import numpy as np

from .config import ProjConfig
from .generator import wrap


@dataclass
class ProjResult:
    valid: bool
    ds: float       # signed arc progress since last commit (negative = backwards)
    lateral: float  # signed offset along left normal; +inf when invalid
    j: int          # nearest sample index in the window


def project(track, state, xy, cfg: ProjConfig):
    M = track.C.shape[0]
    i0 = int(state.s / track.L * M)
    idxs = (i0 + np.arange(-cfg.window_back, cfg.window_fwd)) % M
    dists = np.linalg.norm(track.C[idxs] - xy, axis=1)
    k = int(np.argmin(dists))
    j = int(idxs[k])

    # validity gate BEFORE computing ds -- an invalid projection never
    # contributes reward, not even on a terminal step.
    if dists[k] > cfg.gate_scale * track.w[j] + cfg.gate_margin:
        return ProjResult(valid=False, ds=0.0, lateral=np.inf, j=j)

    ds = float(wrap(track.s[j] - state.s, track.L))
    lateral = float(np.dot(xy - track.C[j], track.Nrm[j]))
    state.s = float(track.s[j])  # commit ONLY on valid projection
    return ProjResult(valid=True, ds=ds, lateral=lateral, j=j)
