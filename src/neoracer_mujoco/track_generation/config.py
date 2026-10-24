"""Tunables shared across track generation. Quantities are in SI units unless
a name says otherwise.

Numbers used by exactly one stage live in that stage's module, not here.
"""

from collections.abc import Mapping
from copy import replace
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

type Difficulty = Literal[0, 1, 2, 3]
"""Difficulty 0 is gentle and wide, and 3 is tight and narrow."""


@dataclass(frozen=True)
class TrackSettings:
    """The knobs `generate_track` reads. Defaults are measured, not guessed.

    shape.draw_control_points reads the knobs of the walk through the control
    points, reject.py the limits that make a candidate track invalid, and
    shape.py the corridor's width band.

    Two samples more than min_separation_along_loop apart *along the loop* must
    be 2*half_width + extra_separation_margin apart *in space*, or the loop
    nearly touches itself and there is no room for walls between."""

    samples: int = 2048
    """How finely the smooth loop is sampled. Everything downstream is defined
    at these samples: the curvature, the self-touch check and the emitted
    walls."""
    oversample_factor: int = 4
    """Extra spline samples used to measure the length."""

    control_points_min: int = 16
    control_points_max: int = 24
    edge_length_min: float = 2.0
    """The shortest step of the walk from one point to the next."""
    edge_length_max: float = 4.0
    """The longest step of the walk."""
    turn_jitter: float = 0.9
    """The standard deviation of the noise on each turn of the walk (rad). More
    gives more and sharper bends both ways, and more rejected candidates."""

    max_curvature: float = 1.0
    """The curvature limit (1/m). settings_for_difficulty lowers it."""
    max_width_times_curvature: float = 0.8
    """The limit that keeps the corridor from folding onto itself."""
    min_total_length: float = 25.0
    max_total_length: float = 90.0
    min_separation_along_loop: float = 2.5
    extra_separation_margin: float = 0.3

    half_width_base: float = 0.50
    half_width_min: float = 0.35
    """The narrowest half-width. settings_for_difficulty raises it."""
    half_width_max: float = 0.80

    max_attempts: int = 60
    """How many fresh random tracks to draw before giving up. It is 60 because
    difficulty 1 is the strict end: over seeds 0-199, 20 attempts leave 13 of
    them with no valid track, and 60 leave none."""
    max_repairs_per_attempt: int = 5
    """The nudge-and-recheck passes per track."""
    max_repair_shift: float = 0.3
    """The most that one repair pass moves a control point pair apart, so that
    one bad pair cannot change the whole loop. Larger moves take several
    passes."""

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


_MAX_CURVATURE: Mapping[Difficulty, float] = MappingProxyType(
    {0: 0.4, 1: 0.6, 2: 0.8, 3: 1.0}
)
"""Each difficulty's max_curvature (1/m)."""
_MIN_HALF_WIDTH: Mapping[Difficulty, float] = MappingProxyType(
    {0: 0.60, 1: 0.50, 2: 0.40, 3: 0.35}
)
_TURN_JITTER: Mapping[Difficulty, float] = MappingProxyType(
    {0: 0.6, 1: 0.9, 2: 0.9, 3: 0.9}
)
"""Each difficulty's turn_jitter (rad). The curvature limit of difficulty 0
rejects most walks drawn with the default 0.9: over seeds 0-199, 41 gave no
track. With 0.6, every seed gives one, with as many bends to the right."""


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
