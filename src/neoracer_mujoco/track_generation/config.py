"""Tunables shared across track generation. All lengths in meters.

Numbers used by exactly one stage live in that stage's module, not here.
"""

from collections.abc import Mapping
from copy import replace
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

# 0 is gentle and wide, 3 is tight and narrow.
type Difficulty = Literal[0, 1, 2, 3]


@dataclass(frozen=True)
class TrackSettings:
    """The knobs `generate_track` reads. Defaults are measured, not guessed."""

    # How finely the smooth loop is sampled. Everything downstream -- curvature,
    # the self-touch check, the emitted walls -- is defined at these samples.
    samples: int = 2048
    oversample_factor: int = 4  # extra spline samples used to measure length

    # --- control points (see shape.draw_control_points) ---
    control_points_min: int = 16
    control_points_max: int = 24
    edge_length_min: float = 2.0  # m; the walk's step from one point to the next
    edge_length_max: float = 4.0  # m
    # rad; std dev of the noise on each turn of the walk. More gives more and
    # sharper bends both ways, and more rejected candidates.
    turn_jitter: float = 0.9

    # --- what makes a candidate track invalid (see reject.py) ---
    max_curvature: float = 1.0  # 1/m; settings_for_difficulty lowers this
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
    half_width_min: float = 0.35  # settings_for_difficulty raises this
    half_width_max: float = 0.80

    # --- retry budget ---
    # 60 attempts because difficulty 1 is the strict end: over seeds 0-199, 20
    # attempts leave 13 of them with no valid track, 60 leave none.
    max_attempts: int = 60  # fresh random tracks before giving up
    max_repairs_per_attempt: int = 5  # nudge-and-recheck passes per track
    # m; the most that one repair pass moves a control point pair apart, so one
    # bad pair cannot change the whole loop. Larger moves take several passes.
    max_repair_shift: float = 0.3

    def __post_init__(self) -> None:
        # These invariants have no Python type, so they are checked once here
        # (AGENTS.md rule 2). The generation code then trusts them.
        rules = {
            "samples >= 3": self.samples >= 3,
            "oversample_factor >= 1": self.oversample_factor >= 1,
            "3 <= control_points_min <= control_points_max": (
                3 <= self.control_points_min <= self.control_points_max
            ),
            "0 < edge_length_min <= edge_length_max": (
                0 < self.edge_length_min <= self.edge_length_max
            ),
            "turn_jitter >= 0": self.turn_jitter >= 0,
            "max_curvature >= 0": self.max_curvature >= 0,
            "max_width_times_curvature > 0": self.max_width_times_curvature > 0,
            "0 < min_total_length <= max_total_length": (
                0 < self.min_total_length <= self.max_total_length
            ),
            "separation limits >= 0": (
                self.min_separation_along_loop >= 0
                and self.extra_separation_margin >= 0
            ),
            "0 < half_width_min <= half_width_max": (
                0 < self.half_width_min <= self.half_width_max
            ),
            "max_attempts >= 1": self.max_attempts >= 1,
            "max_repairs_per_attempt >= 1": self.max_repairs_per_attempt >= 1,
            "max_repair_shift >= 0": self.max_repair_shift >= 0,
        }
        broken = [rule for rule, holds in rules.items() if not holds]
        if broken:
            raise ValueError(f"TrackSettings needs {', '.join(broken)}")


# Easy tracks are gentle and wide, hard tracks are tight and narrow. Only these
# three knobs change with difficulty. All other knobs stay the same.
_MAX_CURVATURE: Mapping[Difficulty, float] = MappingProxyType(
    {0: 0.4, 1: 0.6, 2: 0.8, 3: 1.0}
)  # 1/m
_MIN_HALF_WIDTH: Mapping[Difficulty, float] = MappingProxyType(
    {0: 0.60, 1: 0.50, 2: 0.40, 3: 0.35}
)  # m
# rad. The curvature limit of difficulty 0 rejects most walks drawn with the
# default 0.9: over seeds 0-199, 41 gave no track. With 0.6, every seed
# gives one, with as many bends to the right.
_TURN_JITTER: Mapping[Difficulty, float] = MappingProxyType(
    {0: 0.6, 1: 0.9, 2: 0.9, 3: 0.9}
)


def settings_for_difficulty(difficulty: Difficulty) -> TrackSettings:
    """Stock TrackSettings with the curvature and width limits and the turn
    noise of difficulty. To change other knobs too, use copy.replace on the
    result."""
    return replace(
        TrackSettings(),
        max_curvature=_MAX_CURVATURE[difficulty],
        half_width_min=_MIN_HALF_WIDTH[difficulty],
        turn_jitter=_TURN_JITTER[difficulty],
    )
