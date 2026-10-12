"""Checks on neoracer_mujoco.track_generation: that every generated track is closed,
evenly sampled, within its own limits and reproducible (the geometry half), and
that it turns into a MuJoCo scene the car can actually be dropped onto (the
MJCF half).

The geometry checks are contract checks on the generator; the MJCF checks are
conformance checks on the composed scene, in the same spirit as
test_conformance.py -- they guard the conventions (wall material name, spawn
clearance, geom budget) that the rest of the toolbox silently depends on.
"""

import functools
import itertools
import warnings

import mujoco
import numpy as np
import pytest

from neoracer_mujoco import assets, sim
from neoracer_mujoco.collision import car_hit_wall, wall_geom_ids
from neoracer_mujoco.track_generation import (
    Difficulty,
    Track,
    TrackGenerationError,
    TrackSettings,
    generate_track,
    left_edge,
    right_edge,
    settings_for_difficulty,
    to_mjcf,
    to_spec,
)
from neoracer_mujoco.track_generation.mjcf import decimate_closed, to_start_frame

# Every geom in a composed scene: walls, floor, and the car's own. Generated
# walls dominate it. Measured across difficulties 0-3, seeds 0-7: 305 worst
# case, so a 600 ceiling catches a decimation regression (one box per sample
# would be ~4000) without tripping on an unusually long loop.
MAX_GEOMS = 600


# Generating a track takes a second or so and several tests below want the same
# (seed, difficulty) one. Caching keeps this file's runtime in seconds rather
# than minutes; it is safe precisely because generation is deterministic, which
# test_same_seed_gives_the_same_track independently checks.
@functools.cache
def _track(seed: int, difficulty: Difficulty) -> Track:
    return generate_track(seed, settings_for_difficulty(difficulty))


@functools.cache
def _scene(seed: int, difficulty: Difficulty) -> mujoco.MjModel:
    """The compiled scene, car included, for one generated track."""
    return assets.compose(to_spec(_track(seed, difficulty)))


# --- the generator: pure geometry ---------------------------------------------


@pytest.mark.parametrize("difficulty", [0, 3])
@pytest.mark.parametrize("seed", range(4))
def test_generated_track_is_valid(seed, difficulty):
    settings = settings_for_difficulty(difficulty)
    track = _track(seed, difficulty)

    assert np.abs(track.curvature).max() <= settings.max_curvature
    assert (track.half_width * np.abs(track.curvature)).max() <= (
        settings.max_width_times_curvature
    )
    assert settings.min_total_length <= track.total_length <= settings.max_total_length
    assert track.half_width.min() >= settings.half_width_min
    assert track.half_width.max() <= settings.half_width_max

    hops = np.linalg.norm(np.roll(track.center, -1, 0) - track.center, axis=1)
    expected = track.total_length / len(track.center)
    # chord vs arc, so hops sit just under the spacing but must all agree
    assert np.allclose(hops, hops[0], rtol=0.05)
    assert hops.mean() == pytest.approx(expected, rel=0.02)

    assert np.allclose(np.linalg.norm(track.tangent, axis=1), 1.0)
    assert np.allclose(
        np.sum(track.tangent * track.left_normal, axis=1), 0.0, atol=1e-9
    )
    assert left_edge(track).shape == track.center.shape


def test_same_seed_gives_the_same_track():
    settings = settings_for_difficulty(0)
    a, b = generate_track(3, settings), generate_track(3, settings)
    assert np.array_equal(a.center, b.center)
    assert np.array_equal(a.half_width, b.half_width)
    assert not np.array_equal(a.center, generate_track(4, settings).center)


def test_harder_difficulty_allows_tighter_turns():
    easy = settings_for_difficulty(0)
    hard = settings_for_difficulty(3)
    assert hard.max_curvature > easy.max_curvature
    assert hard.half_width_min < easy.half_width_min


def test_impossible_limits_raise_rather_than_hang():
    impossible = TrackSettings(max_curvature=0.0, max_attempts=2)
    with pytest.raises(TrackGenerationError):
        generate_track(0, impossible)


# --- the MJCF layer -----------------------------------------------------------


@pytest.mark.parametrize("seed", range(3))
def test_generated_track_compiles_with_the_car_on_it(seed):
    model = _scene(seed, 3)
    assert model.body("car").id > 0
    assert model.ngeom > 10


@pytest.mark.parametrize("seed", range(3))
def test_generated_walls_use_the_wall_material(seed):
    """collision.py finds walls by material name, so a rename in mjcf.py would
    silently disable car_hit_wall/reset_if_wall_hit on generated tracks."""
    model = _scene(seed, 3)
    assert len(wall_geom_ids(model)) > 0


@pytest.mark.parametrize("difficulty", [0, 3])
@pytest.mark.parametrize("seed", range(3))
def test_car_spawns_clear_of_the_walls(seed, difficulty):
    """The start transform must put the car on sample 0 facing +X, not clipped
    into a wall or squeezed against one."""
    track = _track(seed, difficulty)
    model = _scene(seed, difficulty)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    assert not car_hit_wall(model, data)

    corridor_width = 2 * track.half_width[0]
    nearest = min(
        float(np.linalg.norm(data.geom_xpos[geom_id][:2]))
        for geom_id in wall_geom_ids(model)
    )
    assert nearest >= 0.4 * corridor_width


@pytest.mark.parametrize("difficulty", [0, 3])
@pytest.mark.parametrize("seed", range(3))
def test_geom_count_stays_within_budget(seed, difficulty):
    """Decimation regression guard -- see MAX_GEOMS. One static box per
    centerline sample would be thousands, and every one of them is broadphase
    cost on every step of every RL rollout."""
    model = _scene(seed, difficulty)
    assert model.ngeom < MAX_GEOMS


@pytest.mark.parametrize("difficulty", [0, 3])
def test_walls_track_the_generated_edges_closely(difficulty):
    """The decimated wall polyline must not bow away from the fine edge by more
    than the tolerance to_mjcf was asked for."""
    track = _track(0, difficulty)
    tolerance = 0.01

    for edge in (left_edge(track), right_edge(track)):
        edge = to_start_frame(track, edge)
        kept = decimate_closed(edge, 0.5, tolerance)
        for i, j in itertools.pairwise(kept):
            first, last = edge[i], edge[j % len(edge)]
            chord = last - first
            length = np.linalg.norm(chord)
            interior = edge[i + 1 : j] - first
            if len(interior) == 0 or length < 1e-12:
                continue
            across = chord[0] * interior[:, 1] - chord[1] * interior[:, 0]
            assert np.abs(across).max() / length <= tolerance + 1e-9


def test_same_seed_gives_byte_identical_mjcf():
    """Determinism has to survive the MJCF layer too -- the start transform and
    the decimation both run on the generated geometry, and either one drifting
    would make two runs of the same seed different scenes."""
    settings = settings_for_difficulty(3)
    first = to_mjcf(generate_track(5, settings))
    second = to_mjcf(generate_track(5, settings))
    assert first == second
    assert first != to_mjcf(generate_track(6, settings))


@pytest.mark.parametrize("seed", range(2))
def test_generated_track_keeps_the_car_solver_options(seed):
    """The same law as test_assets.py for the hand-written tracks: when the car
    is attached, MuJoCo keeps the track's <option>, so it must match the car's."""
    car = assets.load()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = assets.compose(to_spec(_track(seed, 1)))
    assert [str(w.message) for w in caught] == []
    assert model.opt.cone == car.opt.cone
    assert model.opt.impratio == car.opt.impratio


@pytest.mark.parametrize("seed", range(2))
def test_car_comes_to_rest_on_a_generated_track(seed):
    model = _scene(seed, 1)
    data = mujoco.MjData(model)
    sim.run(model, data, 0.0, sim.SETTLE_STEPS)
    car = data.body("car")
    assert 0.05 < float(car.xpos[2]) < 0.2, f"car z = {car.xpos[2]:.3f} m"
    assert float(np.linalg.norm(data.qvel[:3])) < 0.05
