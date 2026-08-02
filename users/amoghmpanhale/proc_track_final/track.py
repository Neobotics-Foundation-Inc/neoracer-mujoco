"""Procedural generation of closed-loop racetracks: the Track and the loop that
makes one. This is the file to read first; each step lives in its own module
under stages/, named for the order it runs in.

The whole procedure, start to finish:

  1. Scatter a ring of random points around the origin.  stages/s1_control_points.py
  2. Draw a smooth closed spline through them - the loop,
     then resample it evenly in meters and measure its
     tangent and curvature at every sample.              stages/s2_centerline.py
  3. Give the corridor a half-width that varies smoothly
     around the loop.                                    stages/s3_width.py
  4. Check the result against four rules.                stages/s4_validation.py
  5. If it fails, nudge the offending control points     stages/s5_repair.py
     and go back to step 2. If nudging does not fix it
     in a few passes, throw the track away and redraw
     from step 1.

Pure NumPy/SciPy. No MuJoCo, no simulator assumptions - the output is geometry.

Same seed in, byte-identical track out: one random generator per call, drawn
from in a fixed order, and repair uses no randomness at all.
"""

from dataclasses import dataclass

import numpy as np

from .settings import TrackSettings, settings_for_difficulty
from .stages.s1_control_points import draw_control_points
from .stages.s2_centerline import fit_closed_spline, sample_evenly_by_distance
from .stages.s3_width import draw_width_waves, half_width_from_waves
from .stages.s4_validation import find_problems
from .stages.s5_repair import repair


class TrackGenerationError(RuntimeError):
    """Every attempt was rejected. Loosen the thresholds in TrackSettings."""


@dataclass(frozen=True)
class Track:
    """A finished racetrack: N samples marching once around a closed loop, evenly
    spaced along the loop so sample i and sample i+1 are always the same distance
    apart. Index N-1 wraps around to index 0."""

    center: np.ndarray            # (N, 2) centerline positions, meters
    tangent: np.ndarray           # (N, 2) unit vector pointing forward along the loop
    left_normal: np.ndarray       # (N, 2) unit vector 90 deg left of the tangent
    curvature: np.ndarray         # (N,) 1/meters, positive where the loop turns left
    half_width: np.ndarray        # (N,) corridor half-width, meters
    distance_along: np.ndarray    # (N,) distance from sample 0 travelling forward
    total_length: float           # meters around the whole loop
    seed: int

    @property
    def left_edge(self) -> np.ndarray:
        """(N, 2) points on the left-hand boundary of the corridor."""
        return self.center + self.half_width[:, None] * self.left_normal

    @property
    def right_edge(self) -> np.ndarray:
        """(N, 2) points on the right-hand boundary of the corridor."""
        return self.center - self.half_width[:, None] * self.left_normal


def generate_track(seed: int, difficulty: int = 0, settings: TrackSettings = None) -> Track:
    """Generate one closed-loop racetrack and return it as a Track.

    seed        same seed gives the same track, always.
    difficulty  0 (gentle and wide) to 3 (tight and narrow). Ignored when
                `settings` is passed - set the limits there instead.
    settings    full TrackSettings override for callers who want every knob.

    Draws a fresh random track, repairs it up to max_repairs_per_attempt times,
    and starts over with a new random track if the repairs do not converge.
    Raises TrackGenerationError once max_attempts is spent.
    """
    if settings is None:
        settings = settings_for_difficulty(difficulty)
    random = np.random.default_rng(seed)

    for _ in range(settings.max_attempts):
        points = draw_control_points(random, settings)
        half_width = half_width_from_waves(draw_width_waves(random), settings)

        for _ in range(settings.max_repairs_per_attempt):
            center, tangent, left_normal, curvature, distance_along, total_length = (
                sample_evenly_by_distance(fit_closed_spline(points),
                                          settings.samples,
                                          settings.oversample_factor)
            )
            problems = find_problems(center, curvature, half_width, distance_along,
                                     total_length, settings)
            if not problems:
                return Track(center=center, tangent=tangent, left_normal=left_normal,
                             curvature=curvature, half_width=half_width,
                             distance_along=distance_along,
                             total_length=total_length, seed=seed)
            points = repair(points, problems, center, half_width, settings)

    raise TrackGenerationError(
        f"no valid track after {settings.max_attempts} attempts (seed {seed}, "
        f"difficulty {difficulty})"
    )
