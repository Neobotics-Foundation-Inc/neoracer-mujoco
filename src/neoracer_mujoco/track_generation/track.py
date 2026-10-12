"""The Track a caller gets back, and the loop that makes one.

Read this file first. Draw a candidate (shape.py), check it (reject.py), nudge
it and re-check, or throw it away and redraw. Pure NumPy/SciPy -- no MuJoCo, no
simulator assumptions. mjcf.py turns the result into scenery.
"""

from dataclasses import dataclass

import numpy as np

from .config import TrackSettings
from .reject import find_problems, repair
from .shape import (
    Points,
    Samples,
    draw_control_points,
    draw_width_waves,
    fit_closed_spline,
    half_width_from_waves,
    sample_evenly_by_distance,
)


@dataclass(frozen=True)
class Track:
    """A finished racetrack: N samples marching once around a closed loop,
    evenly spaced so sample i and sample i+1 are always the same distance
    apart. Index N-1 wraps around to index 0."""

    center: Points  # centerline positions, meters
    tangent: Points  # unit vector pointing forward along the loop
    left_normal: Points  # unit vector 90 deg left of the tangent
    curvature: Samples  # 1/meters, positive where the loop turns left
    half_width: Samples  # corridor half-width, meters
    distance_along: Samples  # distance from sample 0 travelling forward
    total_length: float  # meters around the whole loop
    seed: int


@dataclass(frozen=True)
class NoValidTrack:
    """generate_track rejected every attempt. The limits in TrackSettings ask
    for a track that the generator cannot draw. Loosen them."""

    seed: int
    attempts: int


type Generated = Track | NoValidTrack


def left_edge(track: Track) -> Points:
    """Points on the left-hand boundary of the corridor."""
    return track.center + track.half_width[:, None] * track.left_normal


def right_edge(track: Track) -> Points:
    """Points on the right-hand boundary of the corridor."""
    return track.center - track.half_width[:, None] * track.left_normal


def generate_track(seed: int, settings: TrackSettings) -> Generated:
    """Generate one closed-loop racetrack.

        generate_track(7, settings_for_difficulty(2))

    seed      same seed gives the same track, always.
    settings  the limits a track must meet; settings_for_difficulty gives
              them for difficulty 0 (gentle and wide) to 3 (tight and narrow).

    Draws a fresh random track, repairs it up to max_repairs_per_attempt times,
    and starts over with a new random track if the repairs do not converge.
    Returns NoValidTrack once max_attempts is spent.

    Reproducible: one random generator per call, drawn from in a fixed order,
    and repair uses no randomness at all.
    """
    random = np.random.default_rng(seed)

    # Each attempt is a whole track drawn from scratch. Most are thrown away.
    for _ in range(settings.max_attempts):
        # Scatter a ring of random points for the loop to be drawn through.
        # This is the only place a track's shape is decided at random.
        points = draw_control_points(random, settings)

        # Pick how wide the corridor is at every point around the loop. Drawn
        # once per attempt, not once per repair pass below, so a track that
        # gets repaired keeps the width it was checked against.
        half_width = half_width_from_waves(draw_width_waves(random), settings)

        # Now try to make those points into a valid track, nudging them a few
        # times before giving up and drawing a whole new set.
        for _ in range(settings.max_repairs_per_attempt):
            # Draw a smooth closed curve through the points, then walk it in
            # even steps of distance, noting where each step lands, which way
            # it faces, and how hard it is turning.
            line = sample_evenly_by_distance(
                fit_closed_spline(points), settings.samples, settings.oversample_factor
            )

            # Is it drivable? Corners not too tight, the loop not doubling
            # back onto itself, total length in range. See reject.py.
            problems = find_problems(line, half_width, settings)

            # Nothing wrong with it -- this is the track.
            if (
                len(problems.too_curvy) == 0
                and len(problems.too_close) == 0
                and problems.length_ok
            ):
                return Track(
                    center=line.center,
                    tangent=line.tangent,
                    left_normal=line.left_normal,
                    curvature=line.curvature,
                    half_width=half_width,
                    distance_along=line.distance_along,
                    total_length=line.total_length,
                    seed=seed,
                )

            # Something was wrong. Shift the handful of control points behind
            # each complaint and go round again -- the rest of the loop keeps
            # its shape, so a single bad corner does not cost the whole track.
            points = repair(points, problems, line.center, half_width, settings)

    # Every attempt was rejected, each after its repairs failed to save it.
    # The limits are asking for something this generator cannot draw.
    return NoValidTrack(seed=seed, attempts=settings.max_attempts)
