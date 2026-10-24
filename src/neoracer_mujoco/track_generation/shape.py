"""Drawing one candidate track: random control points, a smooth loop through
them sampled evenly in meters, and a corridor half-width around it.

Everything here is geometry. reject.py decides whether the result is usable.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import scipy.interpolate

from .config import TrackSettings

type Points = npt.NDArray[np.float64]  # shape (N, 2), meters
type Samples = npt.NDArray[np.float64]  # shape (N,), one value per sample


def draw_control_points(random: np.random.Generator, settings: TrackSettings) -> Points:
    """Draw the (n, 2) control points a track is drawn through, in loop order.

    A closed walk of n steps. Each step has a random length and turns from the
    last by 2 pi / n, the turn of a regular polygon, plus noise of standard
    deviation turn_jitter, so the walk turns both ways. Where several turns to
    the right come together, the loop bends inward: an S-bend or a chicane. The
    turns are shifted to add up to exactly one full turn to the left, so the
    loop goes once around, counterclockwise, unless a turn of more than pi,
    which points like a turn to the right, reverses it: find_problems rejects
    those. Subtracting the mean step from every step makes the walk end where
    it started.

    This is the only randomness in a track's shape -- everything downstream is
    deterministic given these points.
    """
    s = settings
    count = int(random.integers(s.control_points_min, s.control_points_max + 1))
    turn = random.normal(0.0, s.turn_jitter, count)
    heading = np.cumsum(turn + 2 * np.pi / count - turn.mean())
    length = random.uniform(s.edge_length_min, s.edge_length_max, count)
    steps = length[:, None] * np.stack([np.cos(heading), np.sin(heading)], axis=1)
    return np.cumsum(steps - steps.mean(axis=0), axis=0)


# --- the centerline --------------------------------------------------------


def shortest_loop_distance(distance: Samples, total_length: float) -> Samples:
    """Distance around a loop taken the short way: a value in
    [-total_length/2, total_length/2).

    Going 99% of the way forward around a 100 m loop is the same as going 1 m
    backwards, and this returns -1.
    """
    return ((distance + total_length / 2) % total_length) - total_length / 2


def fit_closed_spline(points: Points) -> scipy.interpolate.CubicSpline:
    """Cubic spline through every control point, joining back up smoothly.

    Parameterized by a dimensionless 0..1 that runs once around the loop, NOT
    by distance -- equal steps in that parameter are not equal steps in meters,
    which is what sample_evenly_by_distance exists to fix.
    """
    knots = np.linspace(0.0, 1.0, len(points) + 1)
    closed_points = np.vstack([points, points[:1]])
    return scipy.interpolate.CubicSpline(knots, closed_points, bc_type="periodic")


def measure_distance_along_spline(
    spline: scipy.interpolate.CubicSpline, sample_count: int
) -> tuple[Samples, Samples]:
    """Walk the spline in `sample_count` equal parameter steps, adding up the
    straight-line hops between them.

    Returns (parameter, distance_traveled), the lookup table converting a
    distance in meters into the spline parameter sitting at it.
    distance_traveled has one extra entry, for the whole closed loop.
    """
    parameter = np.linspace(0.0, 1.0, sample_count, endpoint=False)
    points = spline(parameter)
    hop_lengths = np.linalg.norm(np.roll(points, -1, 0) - points, axis=1)
    return parameter, np.concatenate([[0.0], np.cumsum(hop_lengths)])


@dataclass(frozen=True)
class Centerline:
    """A closed loop sampled every total_length / N meters. Index N-1 wraps
    around to index 0."""

    center: Points  # positions
    tangent: Points  # unit vector pointing forward along the loop
    left_normal: Points  # unit vector 90 degrees left of the tangent
    curvature: Samples  # 1/m, positive where the loop turns left
    distance_along: Samples  # m from sample 0, traveling forward
    total_length: float  # m around the whole loop


def sample_evenly_by_distance(
    spline: scipy.interpolate.CubicSpline, samples: int, oversample_factor: int
) -> Centerline:
    """Sample the spline so consecutive samples are equally spaced *in meters*.

    Sampled directly, a spline bunches up in the corners. Rather than
    reparameterize it by arc length analytically, this measures its length
    numerically at oversample_factor times the output density and interpolates
    the parameter back out -- one table lookup instead of a solve, with error
    bounded by the dense spacing.
    """
    dense_parameter, distance_at_dense = measure_distance_along_spline(
        spline, samples * oversample_factor
    )
    total_length = float(distance_at_dense[-1])

    distance_along = np.linspace(0.0, total_length, samples, endpoint=False)
    parameter = np.interp(distance_along, distance_at_dense[:-1], dense_parameter)
    center = spline(parameter)

    # The spline's derivative is per unit parameter, not per meter, so its
    # length is meaningless here -- only its direction matters.
    tangent = spline(parameter, 1)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    left_normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)

    # Curvature is how fast the tangent rotates per meter traveled, signed by
    # which way it rotates. This central difference assumes the samples are
    # evenly spaced, which is why that had to happen first.
    spacing = total_length / samples
    tangent_change = (np.roll(tangent, -1, 0) - np.roll(tangent, 1, 0)) / (2 * spacing)
    curvature = np.sum(tangent_change * left_normal, axis=1)
    return Centerline(
        center, tangent, left_normal, curvature, distance_along, total_length
    )


# --- corridor width --------------------------------------------------------
# Width is a function of position around the loop, not of the shape there.
# reject.py is what throws out a width the corners cannot take.


@dataclass(frozen=True)
class WidthWaves:
    """Sine wave w completes w+1 cycles around the loop."""

    phases: Samples  # rad, one per wave
    amplitudes: Samples  # m, one per wave


def draw_width_waves(random: np.random.Generator) -> WidthWaves:
    """Draw the random ingredients of the width profile: a few sine waves, each
    with a random phase and an amplitude that shrinks as the wave gets faster,
    so the width drifts slowly and never flickers.

    Separate from half_width_from_waves because the two have different
    lifetimes: waves are drawn once per attempt and reused across every repair
    pass, so a repaired track keeps the width it was validated against.
    """
    wave_count = int(random.integers(2, 6))
    phases = random.uniform(0, 2 * np.pi, wave_count)
    amplitudes = random.uniform(0.03, 0.12, wave_count) / np.arange(1, wave_count + 1)
    return WidthWaves(phases, amplitudes)


def half_width_from_waves(waves: WidthWaves, settings: TrackSettings) -> Samples:
    """Sum the width waves onto the base half-width and clamp to the allowed
    band, giving one half-width per sample.

    Harmonics rather than smoothed noise: wave `w` completes exactly w+1 cycles
    over the loop, so the profile closes on itself exactly -- no seam to patch
    where sample N-1 meets sample 0.
    """
    angle_around_loop = np.linspace(0, 2 * np.pi, settings.samples, endpoint=False)
    half_width = settings.half_width_base + sum(
        amplitude * np.sin((wave + 1) * angle_around_loop + phase)
        for wave, (amplitude, phase) in enumerate(
            zip(waves.amplitudes, waves.phases, strict=True)
        )
    )
    return np.clip(half_width, settings.half_width_min, settings.half_width_max)
