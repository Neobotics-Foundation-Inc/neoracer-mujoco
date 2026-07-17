"""Stages 1-3: generate a valid closed-loop racetrack centerline from one seed.

Pure NumPy/SciPy, no MuJoCo. Output is the immutable Track object that every
downstream stage (walls, projection, reward, obs) consumes.

Determinism contract: one rng per sample_valid_track call, consumed in a fixed
order (centerline draws, then width draws) per attempt. Repair uses no rng, so
identical seeds give bit-identical tracks (test T3).
"""

from dataclasses import dataclass

import numpy as np
import scipy.interpolate
from scipy.spatial import cKDTree

from .config import GenConfig


class TrackGenerationError(RuntimeError):
    """Raised when max_attempts is exhausted without a valid track."""


@dataclass(frozen=True)
class Track:
    C: np.ndarray      # (M, 2) centerline positions
    T: np.ndarray      # (M, 2) unit tangents
    Nrm: np.ndarray    # (M, 2) unit left normals (T rotated +90 deg)
    kappa: np.ndarray  # (M,)   signed curvature, + = turning left
    w: np.ndarray      # (M,)   half-width
    s: np.ndarray      # (M,)   s[i] = i * L / M
    L: float
    seed: int


def wrap(a, L):
    """Signed shortest arc difference in (-L/2, L/2]."""
    return ((a + L / 2) % L) - L / 2


# ----- Stage 1: centerline control points -----

def respace_min_gap(theta, min_gap):
    """Push sorted angles apart so no consecutive gap is below min_gap. No rng."""
    theta = np.sort(theta).copy()
    for i in range(1, len(theta)):
        if theta[i] - theta[i - 1] < min_gap:
            theta[i] = theta[i - 1] + min_gap
    return theta


def make_centerline(rng, cfg: GenConfig):
    """Draw perturbed radial control points, Laplacian-smooth, return (n, 2) points."""
    n = int(rng.integers(cfg.n_ctrl_min, cfg.n_ctrl_max + 1))
    theta = np.sort(rng.uniform(0, 2 * np.pi, n))
    theta = respace_min_gap(theta, cfg.min_angular_gap)
    r = rng.uniform(cfg.r_min, cfg.r_max, n)
    # smooth the radius sequence (circular) -- kills adjacent radius jumps that
    # force high curvature, without shrinking the loop like point smoothing does
    for _ in range(cfg.radius_smooth_rounds):
        r = 0.5 * r + 0.25 * (np.roll(r, 1) + np.roll(r, -1))
    pts = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
    pts += rng.normal(0.0, cfg.jitter_std, pts.shape)
    for _ in range(cfg.laplacian_rounds):
        pts = 0.5 * pts + 0.25 * (np.roll(pts, 1, 0) + np.roll(pts, -1, 0))
    return pts


def fit_spline(pts):
    """Periodic cubic spline through the control points."""
    n = len(pts)
    t_knots = np.linspace(0.0, 1.0, n + 1)
    return scipy.interpolate.CubicSpline(
        t_knots, np.vstack([pts, pts[:1]]), bc_type="periodic"
    )


# ----- Stage 2: arc-length reparameterization -----

def build_track_arrays(spline, M, oversample):
    """Resample the spline to uniform arc length. Curvature from central
    differences on the uniform-s samples (never from spline t-derivatives)."""
    P = M * oversample
    t = np.linspace(0.0, 1.0, P, endpoint=False)
    p = spline(t)
    seg = np.linalg.norm(np.roll(p, -1, 0) - p, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    L = float(cum[-1])
    s_uniform = np.linspace(0.0, L, M, endpoint=False)
    t_of_s = np.interp(s_uniform, cum[:-1], t)
    C = spline(t_of_s)
    dC = spline(t_of_s, 1)  # d/dt, not d/ds -- normalize away
    T = dC / np.linalg.norm(dC, axis=1, keepdims=True)
    Nrm = np.stack([-T[:, 1], T[:, 0]], axis=1)  # left normal
    ds = L / M
    dT = (np.roll(T, -1, 0) - np.roll(T, 1, 0)) / (2 * ds)
    kappa = dT[:, 0] * Nrm[:, 0] + dT[:, 1] * Nrm[:, 1]  # signed
    return C, T, Nrm, kappa, s_uniform, L


# ----- Stage 3: width -----

def draw_width_randoms(rng):
    """Draw the harmonic width randoms ONCE per attempt (before repair loop)."""
    k = int(rng.integers(2, 6))
    phase = rng.uniform(0, 2 * np.pi, k)
    amp = rng.uniform(0.03, 0.12, k) / np.arange(1, k + 1)  # 1/f falloff
    return k, phase, amp


def width_from_draws(draws, cfg: GenConfig):
    """Low-frequency harmonic half-width, smooth and periodic. Clip enforces the
    hard floor/ceiling. Length M, independent of the (repaired) centerline."""
    _, phase, amp = draws
    u = np.linspace(0, 2 * np.pi, cfg.M, endpoint=False)
    w = cfg.w_base + sum(
        a * np.sin((i + 1) * u + p) for i, (a, p) in enumerate(zip(amp, phase))
    )
    return np.clip(w, cfg.w_min, cfg.w_max)


# ----- validation + repair -----

@dataclass
class Violations:
    curvature_idx: np.ndarray      # sample indices violating V1/V2
    proximity_pairs: list          # (i, j) sample-index pairs violating V3
    length_ok: bool                # V4

    def __bool__(self):
        return (
            len(self.curvature_idx) > 0
            or len(self.proximity_pairs) > 0
            or not self.length_ok
        )


def check(C, kappa, w, s, L, cfg: GenConfig):
    curv = np.where(
        (np.abs(kappa) > cfg.kappa_max) | (w * np.abs(kappa) > cfg.wk_max)
    )[0]

    clearance = 2 * w.max() + cfg.clearance_margin
    pairs = cKDTree(C).query_pairs(r=clearance, output_type="ndarray")
    if len(pairs) > 0:
        arc = np.abs(wrap(s[pairs[:, 1]] - s[pairs[:, 0]], L))
        prox = pairs[arc > cfg.arc_sep_threshold]
    else:
        prox = pairs
    prox = [(int(i), int(j)) for i, j in prox]

    length_ok = cfg.L_min <= L <= cfg.L_max
    return Violations(curv, prox, length_ok)


def repair(pts, violations: Violations, C, w, cfg: GenConfig):
    """Curvature repair first, then proximity (order matters). No rng."""
    pts = pts.copy()
    n = len(pts)

    if len(violations.curvature_idx) > 0:
        targets = set()
        for si in violations.curvature_idx:
            cp = int(np.argmin(np.linalg.norm(pts - C[si], axis=1)))
            targets.add(cp)
        smoothed = pts.copy()
        for cp in targets:
            smoothed[cp] = 0.5 * pts[cp] + 0.25 * (
                pts[(cp - 1) % n] + pts[(cp + 1) % n]
            )
        pts = smoothed

    clearance = 2 * w.max() + cfg.clearance_margin
    for i, j in violations.proximity_pairs:
        a = int(np.argmin(np.linalg.norm(pts - C[i], axis=1)))
        b = int(np.argmin(np.linalg.norm(pts - C[j], axis=1)))
        if a == b:
            continue
        diff = pts[a] - pts[b]
        dist = np.linalg.norm(diff)
        if dist < 1e-9:
            continue
        actual = np.linalg.norm(C[i] - C[j])
        shift = min(0.5 * (clearance - actual), cfg.repair_disp_cap)
        if shift <= 0:
            continue
        direction = diff / dist
        pts[a] = pts[a] + direction * shift
        pts[b] = pts[b] - direction * shift

    return pts


def sample_valid_track(seed: int, cfg: GenConfig) -> Track:
    rng = np.random.default_rng(seed)
    for _ in range(cfg.max_attempts):
        pts = make_centerline(rng, cfg)
        width_draws = draw_width_randoms(rng)  # once per attempt, before repair
        for _ in range(cfg.max_repair):
            C, T, Nrm, kappa, s, L = build_track_arrays(
                fit_spline(pts), cfg.M, cfg.oversample
            )
            w = width_from_draws(width_draws, cfg)
            violations = check(C, kappa, w, s, L, cfg)
            if not violations:
                return Track(C=C, T=T, Nrm=Nrm, kappa=kappa, w=w, s=s, L=L, seed=seed)
            pts = repair(pts, violations, C, w, cfg)
    raise TrackGenerationError(seed)
