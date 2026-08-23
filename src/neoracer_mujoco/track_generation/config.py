"""Tunables shared across track generation. All lengths in meters.

Numbers used by exactly one stage live in that stage's module, not here.
"""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class TrackSettings:
    """The knobs `generate_track` reads. Defaults are measured, not guessed."""

    # How finely the smooth loop is sampled. Everything downstream -- curvature,
    # the self-touch check, the emitted walls -- is defined at these samples.
    samples: int = 2048
    oversample_factor: int = 4  # extra spline samples used to measure length

    # --- what makes a candidate track invalid (see reject.py) ---
    max_curvature: float = 1.0  # 1/m; the difficulty table below lowers this
    max_width_times_curvature: float = 0.8  # corridor must not fold onto itself
    min_total_length: float = 25.0
    max_total_length: float = 90.0
    # Two samples more than min_separation_along_loop apart *along the loop*
    # must be 2*half_width + extra_separation_margin apart *in space*, or the
    # loop nearly touches itself and there is no room for walls between.
    min_separation_along_loop: float = 2.5
    extra_separation_margin: float = 0.3

    # --- corridor width band (see shape.py) ---
    half_width_base: float = 0.50
    half_width_min: float = 0.35  # the difficulty table below raises this
    half_width_max: float = 0.80

    # --- retry budget ---
    # 60 attempts because difficulty 0 is the strict end: measured over 20
    # seeds, 20 attempts leaves 7 of them with no valid track, 60 leaves none.
    max_attempts: int = 60  # fresh random tracks before giving up
    max_repairs_per_attempt: int = 5  # nudge-and-recheck passes per track


# Easy tracks are gentle and wide, hard tracks are tight and narrow. Only these
# two knobs change with difficulty; everything else stays fixed so tracks across
# a curriculum stay recognizably the same family.
_MAX_CURVATURE_BY_DIFFICULTY = (0.4, 0.6, 0.8, 1.0)
_MIN_HALF_WIDTH_BY_DIFFICULTY = (0.60, 0.50, 0.40, 0.35)


def settings_for_difficulty(difficulty: int, base=None) -> TrackSettings:
    """`base` (default: stock TrackSettings) with the curvature and width limits
    set for difficulty 0 (easiest) through 3 (hardest)."""
    if difficulty not in (0, 1, 2, 3):
        raise ValueError(f"difficulty must be 0-3, got {difficulty}")
    return replace(
        base if base is not None else TrackSettings(),
        max_curvature=_MAX_CURVATURE_BY_DIFFICULTY[difficulty],
        half_width_min=_MIN_HALF_WIDTH_BY_DIFFICULTY[difficulty],
    )
