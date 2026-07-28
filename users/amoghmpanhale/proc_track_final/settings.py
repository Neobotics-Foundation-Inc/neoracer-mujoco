"""Every tunable number used by track generation, in one place.

The defaults are measured, not guessed: they produce a raw (pre-repair) valid
track roughly 1 attempt in 3 at difficulty 0, so the 20-attempt budget in
`generate_track` is never the thing that fails.
"""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class TrackSettings:
    """Tunables for one track generation run. All lengths in meters."""

    # --- Stage 1: the ring of random control points the loop is drawn through ---
    control_points_min: int = 10
    control_points_max: int = 17
    ring_radius_min: float = 6.0        # control points start on a ring of this
    ring_radius_max: float = 9.0        # radius before being jittered
    radius_smoothing_rounds: int = 2    # blur the radii so neighbors don't jump
    point_jitter_meters: float = 6.0    # std dev of the random shove per point
    point_smoothing_rounds: int = 2     # blur the points to soften sharp corners
    min_angle_between_points: float = 0.15  # radians, stops points stacking up

    # --- Stage 2: how finely the smooth loop is sampled ---
    samples: int = 2048             # points in the finished centerline
    oversample_factor: int = 4      # extra spline samples used to measure length

    # --- Stage 3: how the corridor width varies around the loop ---
    half_width_base: float = 0.50
    half_width_min: float = 0.35    # difficulty raises this (wider = easier)
    half_width_max: float = 0.80

    # --- Rejection thresholds: what makes a track invalid ---
    max_curvature: float = 1.0          # 1/m; difficulty lowers this
    max_width_times_curvature: float = 0.8   # corridor must not fold onto itself
    min_separation_along_loop: float = 2.5   # two samples this far apart along
    extra_separation_margin: float = 0.3     # the loop must be this much clear
    min_total_length: float = 25.0           # in space, or they nearly touch
    max_total_length: float = 90.0

    # --- Retry budget ---
    # 60 attempts because difficulty 0 is the strict end: measured over 20 seeds,
    # 20 attempts leaves 7 of them with no valid track, 60 leaves none.
    max_attempts: int = 60          # fresh random tracks before giving up
    max_repairs_per_attempt: int = 5  # nudge-and-recheck passes per track
    max_repair_shift: float = 0.3   # cap on how far one repair moves a point


# Difficulty 0-3: easy tracks are gentle and wide, hard tracks are tight and
# narrow. Only these two knobs change; everything else stays fixed so tracks
# across a curriculum stay recognizably the same family.
_MAX_CURVATURE_BY_DIFFICULTY = (0.4, 0.6, 0.8, 1.0)
_MIN_HALF_WIDTH_BY_DIFFICULTY = (0.60, 0.50, 0.40, 0.35)


def settings_for_difficulty(difficulty: int, base=TrackSettings()) -> TrackSettings:
    """Return `base` with the curvature/width limits set for difficulty 0 (easiest)
    through 3 (hardest)."""
    if difficulty not in (0, 1, 2, 3):
        raise ValueError(f"difficulty must be 0-3, got {difficulty}")
    return replace(
        base,
        max_curvature=_MAX_CURVATURE_BY_DIFFICULTY[difficulty],
        half_width_min=_MIN_HALF_WIDTH_BY_DIFFICULTY[difficulty],
    )
