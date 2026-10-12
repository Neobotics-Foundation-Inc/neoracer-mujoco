"""Drawing one candidate track: random control points, a smooth loop through
them sampled evenly in meters, and a corridor half-width around it.

Everything here is geometry. reject.py decides whether the result is usable.
"""

import numpy as np
import scipy.interpolate

from .config import TrackSettings

# --- control point scatter -------------------------------------------------
# Only draw_control_points reads these, so they live here rather than in the
# shared TrackSettings.
CONTROL_POINTS_MIN = 10
CONTROL_POINTS_MAX = 17
RING_RADIUS_MIN = 6.0  # control points start on a ring of this radius...
RING_RADIUS_MAX = 9.0  # ...before being jittered
RADIUS_SMOOTHING_ROUNDS = 2  # blur the radii so neighbors don't jump
POINT_JITTER_METERS = 6.0  # std dev of the random shove per point
POINT_SMOOTHING_ROUNDS = 2  # blur the points to soften sharp corners
MIN_ANGLE_BETWEEN_POINTS = 0.15  # radians, stops points stacking up


def spread_out_angles(angles, min_gap):
    """Sort `angles` and push each one forward until it is at least `min_gap`
    past the previous.

    Two control points at nearly the same angle would sit almost on top of each
    other and force the loop into a hairpin.
    """
    angles = np.sort(angles).copy()
    for i in range(1, len(angles)):
        if angles[i] - angles[i - 1] < min_gap:
            angles[i] = angles[i - 1] + min_gap
    return angles


def smooth_around_loop(values, rounds):
    """Blur a periodic sequence `rounds` times: each entry becomes half itself
    plus a quarter of each neighbor, wrapping at the ends.

    A low-pass filter, so it removes the sharp jumps that become tight corners.
    Works on a 1D sequence of numbers or an (n, 2) sequence of points alike.
    """
    for _ in range(rounds):
        values = 0.5 * values + 0.25 * (np.roll(values, 1, 0) + np.roll(values, -1, 0))
    return values


def draw_control_points(random):
    """Scatter the (n, 2) control points a track is drawn through, in loop order.

    Polar, not cartesian: each point is an angle plus a radius, so sorting by
    angle puts them in loop order for free and the loop always encloses the
    origin. Then the radii are smoothed so neighboring points sit at similar
    distances and the loop does not zigzag in and out; each point gets an
    independent random shove, which is what makes one track differ from the
    next; and the points are smoothed to round off the corners that created.

    This is the only randomness in a track's shape -- everything downstream is
    deterministic given these points.
    """
    count = int(random.integers(CONTROL_POINTS_MIN, CONTROL_POINTS_MAX + 1))
    angles = spread_out_angles(
        random.uniform(0, 2 * np.pi, count), MIN_ANGLE_BETWEEN_POINTS
    )
    radii = random.uniform(RING_RADIUS_MIN, RING_RADIUS_MAX, count)
    radii = smooth_around_loop(radii, RADIUS_SMOOTHING_ROUNDS)

    points = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)
    points += random.normal(0.0, POINT_JITTER_METERS, points.shape)
    return smooth_around_loop(points, POINT_SMOOTHING_ROUNDS)


# --- the centerline --------------------------------------------------------


def shortest_loop_distance(distance, total_length):
    """Distance around a loop taken the short way: a value in
    (-total_length/2, total_length/2].

    Going 99% of the way forward around a 100 m loop is the same as going 1 m
    backwards, and this returns -1.
    """
    return ((distance + total_length / 2) % total_length) - total_length / 2


def fit_closed_spline(points):
    """Cubic spline through every control point, joining back up smoothly.

    Parameterized by a dimensionless 0..1 that runs once around the loop, NOT
    by distance -- equal steps in that parameter are not equal steps in meters,
    which is what sample_evenly_by_distance exists to fix.
    """
    knots = np.linspace(0.0, 1.0, len(points) + 1)
    closed_points = np.vstack([points, points[:1]])
    return scipy.interpolate.CubicSpline(knots, closed_points, bc_type="periodic")


def measure_distance_along_spline(spline, sample_count):
    """Walk the spline in `sample_count` equal parameter steps, adding up the
    straight-line hops between them.

    Returns (parameter, distance_travelled), the lookup table converting a
    distance in meters into the spline parameter sitting at it.
    distance_travelled has one extra entry, for the whole closed loop.
    """
    parameter = np.linspace(0.0, 1.0, sample_count, endpoint=False)
    points = spline(parameter)
    hop_lengths = np.linalg.norm(np.roll(points, -1, 0) - points, axis=1)
    return parameter, np.concatenate([[0.0], np.cumsum(hop_lengths)])


def sample_evenly_by_distance(spline, samples, oversample_factor):
    """Sample the spline so consecutive samples are equally spaced *in meters*.

    Sampled directly, a spline bunches up in the corners. Rather than
    reparameterize it by arc length analytically, this measures its length
    numerically at oversample_factor times the output density and interpolates
    the parameter back out -- one table lookup instead of a solve, with error
    bounded by the dense spacing.

    Returns (center, tangent, left_normal, curvature, distance_along,
    total_length):
      center        (N, 2) positions
      tangent       (N, 2) unit vector pointing forward along the loop
      left_normal   (N, 2) unit vector 90 degrees left of the tangent
      curvature     (N,) 1/meters, positive where the loop turns left
      distance_along(N,) meters from sample 0, travelling forward
      total_length  meters around the whole loop
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

    # Curvature is how fast the tangent rotates per meter travelled, signed by
    # which way it rotates. This central difference assumes the samples are
    # evenly spaced, which is why that had to happen first.
    spacing = total_length / samples
    tangent_change = (np.roll(tangent, -1, 0) - np.roll(tangent, 1, 0)) / (2 * spacing)
    curvature = np.sum(tangent_change * left_normal, axis=1)
    return center, tangent, left_normal, curvature, distance_along, total_length


# --- corridor width --------------------------------------------------------
# Width is a function of position around the loop, not of the shape there.
# reject.py is what throws out a width the corners cannot take.


def draw_width_waves(random):
    """Draw the random ingredients of the width profile: a few sine waves, each
    with a random phase and an amplitude that shrinks as the wave gets faster,
    so the width drifts slowly and never flickers. Returns (phases, amplitudes).

    Separate from half_width_from_waves because the two have different
    lifetimes: waves are drawn once per attempt and reused across every repair
    pass, so a repaired track keeps the width it was validated against.
    """
    wave_count = int(random.integers(2, 6))
    phases = random.uniform(0, 2 * np.pi, wave_count)
    amplitudes = random.uniform(0.03, 0.12, wave_count) / np.arange(1, wave_count + 1)
    return phases, amplitudes


def half_width_from_waves(waves, settings: TrackSettings):
    """Sum the width waves onto the base half-width and clamp to the allowed
    band, giving one half-width per sample.

    Harmonics rather than smoothed noise: wave `w` completes exactly w+1 cycles
    over the loop, so the profile closes on itself exactly -- no seam to patch
    where sample N-1 meets sample 0.
    """
    phases, amplitudes = waves
    angle_around_loop = np.linspace(0, 2 * np.pi, settings.samples, endpoint=False)
    half_width = settings.half_width_base + sum(
        amplitude * np.sin((wave + 1) * angle_around_loop + phase)
        for wave, (amplitude, phase) in enumerate(zip(amplitudes, phases))
    )
    return np.clip(half_width, settings.half_width_min, settings.half_width_max)
