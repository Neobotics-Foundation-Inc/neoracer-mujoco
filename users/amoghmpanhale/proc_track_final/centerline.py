"""Stage 2: turn control points into a smooth loop sampled evenly in meters.

The output of this stage is the geometry a car actually drives on: positions,
which way the track points, and how hard it is turning, at samples that are all
the same distance apart. Nothing here is random.
"""

import numpy as np
import scipy.interpolate


def shortest_loop_distance(distance, total_length):
    """Distance around a loop taken the short way: a value in
    (-total_length/2, total_length/2]. Going 99% of the way forward around a
    100 m loop is the same as going 1 m backwards, and this returns -1."""
    return ((distance + total_length / 2) % total_length) - total_length / 2


def fit_closed_spline(points):
    """Fit a cubic spline that passes through every control point and joins back
    up smoothly at the start.

    Parameterized by a dimensionless 0..1 that runs once around the loop - NOT
    by distance. Equal steps in that parameter are not equal steps in meters,
    which is exactly what `sample_evenly_by_distance` exists to fix.
    """
    knots = np.linspace(0.0, 1.0, len(points) + 1)
    closed_points = np.vstack([points, points[:1]])
    return scipy.interpolate.CubicSpline(knots, closed_points, bc_type="periodic")


def measure_distance_along_spline(spline, sample_count):
    """Walk the spline in `sample_count` equal parameter steps and add up the
    straight-line hops between them.

    Returns (parameter, distance_travelled) where distance_travelled[i] is the
    length of spline from its start up to parameter[i], plus one final entry for
    the whole closed loop. That is the lookup table converting a distance in
    meters into the spline parameter that sits at it.
    """
    parameter = np.linspace(0.0, 1.0, sample_count, endpoint=False)
    points = spline(parameter)
    hop_lengths = np.linalg.norm(np.roll(points, -1, 0) - points, axis=1)
    return parameter, np.concatenate([[0.0], np.cumsum(hop_lengths)])


def sample_evenly_by_distance(spline, samples, oversample_factor):
    """Sample the spline so consecutive samples are equally spaced *in meters*.

    Densely measures the spline's length first (see
    `measure_distance_along_spline`), then looks up which parameter value sits
    at each evenly spaced distance we actually want.

    Returns (center, tangent, left_normal, curvature, distance_along,
    total_length), all defined at those evenly spaced samples:
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
    # length is meaningless here - only its direction matters.
    tangent = spline(parameter, 1)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    left_normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)

    # Curvature is how fast the tangent rotates per meter travelled, signed by
    # which way it rotates. Measured by central difference on the evenly spaced
    # samples (spacing is total_length / samples), which is why the even spacing
    # above had to happen first.
    spacing = total_length / samples
    tangent_change = (np.roll(tangent, -1, 0) - np.roll(tangent, 1, 0)) / (2 * spacing)
    curvature = np.sum(tangent_change * left_normal, axis=1)
    return center, tangent, left_normal, curvature, distance_along, total_length
