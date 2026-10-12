"""Laws for neoracer_mujoco.track_generation.

The geometry tests check that each generated track is closed, evenly sampled,
inside the limits of its settings and the same for the same seed. The MJCF
tests check the scene that the car drives on: the wall material, the spawn
clearance and the geom count. Other modules depend on these conventions.
"""

import functools
import itertools
from typing import assert_never

import mujoco
import numpy as np
import pytest
from scenes import assert_car_rests, assert_keeps_car_solver_options

from neoracer_mujoco import assets
from neoracer_mujoco.collision import car_touches_wall, wall_geom_ids
from neoracer_mujoco.track_generation import (
    Difficulty,
    NoValidTrack,
    Track,
    TrackSettings,
    WallParams,
    generate_track,
    left_edge,
    right_edge,
    settings_for_difficulty,
    to_mjcf,
    to_spec,
)
from neoracer_mujoco.track_generation.mjcf import decimate_closed, to_start_frame

# The largest geom count of a composed scene: walls, floor and car. Over
# difficulties 0-3 and seeds 0-7, the largest measured count is 305. One box for
# each centerline sample gives about 4000. So this limit finds a decimation error
# and still allows a long track.
MAX_GEOMS = 600


def _generate(seed: int, settings: TrackSettings) -> Track:
    match generate_track(seed, settings):
        case Track() as track:
            return track
        case NoValidTrack() as none:
            pytest.fail(f"no track: {none}")
        case other:
            assert_never(other)


# The generation of one track takes about a second, and several tests use the
# same (seed, difficulty). The cache keeps the run time of this file short. The
# cache is correct only because the generation is deterministic.
# test_same_seed_gives_the_same_track checks that.
@functools.cache
def _track(seed: int, difficulty: Difficulty) -> Track:
    return _generate(seed, settings_for_difficulty(difficulty))


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
    """A new generation gives the same track as the cached one. A different
    seed gives a different track."""
    fresh = _generate(2, settings_for_difficulty(0))
    assert np.array_equal(fresh.center, _track(2, 0).center)
    assert np.array_equal(fresh.half_width, _track(2, 0).half_width)
    assert not np.array_equal(_track(3, 0).center, _track(2, 0).center)


def test_harder_difficulty_allows_tighter_turns():
    """Each step up in difficulty allows a larger curvature and a narrower
    track."""
    difficulties: tuple[Difficulty, ...] = (0, 1, 2, 3)
    for easy, hard in itertools.pairwise(map(settings_for_difficulty, difficulties)):
        assert hard.max_curvature > easy.max_curvature
        assert hard.half_width_min < easy.half_width_min


@pytest.mark.parametrize(
    "bad",
    [
        {"samples": 0},
        {"oversample_factor": 0},
        {"control_points_min": 20},
        {"ring_radius_min": 0.0},
        {"min_total_length": 100.0},
        {"half_width_min": 0.9},
        {"max_attempts": 0},
        {"max_repair_shift": -0.1},
    ],
)
def test_settings_reject_values_that_make_no_track(bad):
    with pytest.raises(ValueError, match="TrackSettings needs"):
        TrackSettings(**bad)


def test_impossible_limits_return_no_valid_track():
    impossible = TrackSettings(max_curvature=0.0, max_attempts=2)
    assert generate_track(0, impossible) == NoValidTrack(seed=0, attempts=2)


# --- the MJCF layer -----------------------------------------------------------


@pytest.mark.parametrize("seed", range(3))
def test_generated_track_compiles_with_the_car_on_it(seed):
    assert _scene(seed, 3).body("car").id > 0


@pytest.mark.parametrize("seed", range(3))
def test_generated_walls_use_the_wall_material(seed):
    """collision.py finds the walls by the name of their material. If
    mjcf.py renames the material, car_touches_wall finds no wall on a
    generated track. The walls are some of the geoms, but not all: the floor
    and the car are not walls."""
    model = _scene(seed, 3)
    assert 0 < len(wall_geom_ids(model)) < model.ngeom


@pytest.mark.parametrize("difficulty", [0, 3])
@pytest.mark.parametrize("seed", range(3))
def test_car_spawns_clear_of_the_walls(seed, difficulty):
    """The start transform puts the car on sample 0, facing +X. The car does
    not touch a wall and is not near one."""
    track = _track(seed, difficulty)
    model = _scene(seed, difficulty)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    assert not car_touches_wall(model, data, model.body("car").id, wall_geom_ids(model))

    corridor_width = 2 * track.half_width[0]
    nearest = min(
        float(np.linalg.norm(data.geom_xpos[geom_id][:2]))
        for geom_id in wall_geom_ids(model)
    )
    assert nearest >= 0.4 * corridor_width


@pytest.mark.parametrize("difficulty", [0, 3])
@pytest.mark.parametrize("seed", range(3))
def test_geom_count_stays_within_budget(seed, difficulty):
    """The decimation keeps the geom count below MAX_GEOMS. One box for each
    centerline sample gives thousands of geoms, and MuJoCo checks each geom
    for contacts at each step."""
    model = _scene(seed, difficulty)
    assert model.ngeom < MAX_GEOMS


@pytest.mark.parametrize("difficulty", [0, 3])
def test_walls_track_the_generated_edges_closely(difficulty):
    """No point of the fine edge is farther from the decimated wall line
    than WallParams().max_deviation."""
    track = _track(0, difficulty)
    tolerance = WallParams().max_deviation

    for edge in (left_edge(track), right_edge(track)):
        edge = to_start_frame(track, edge)
        kept = decimate_closed(edge, WallParams().segment_length, tolerance)
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
    """The start transform and the decimation run on the generated track. A
    new generation of the same seed gives the same MJCF text. A different
    seed gives a different text."""
    cached = to_mjcf(_track(0, 3))
    assert to_mjcf(_generate(0, settings_for_difficulty(3))) == cached
    assert to_mjcf(_track(1, 3)) != cached


@pytest.mark.parametrize("seed", range(2))
def test_generated_track_keeps_the_car_solver_options(seed):
    assert_keeps_car_solver_options(to_spec(_track(seed, 3)))


@pytest.mark.parametrize("seed", range(2))
def test_car_comes_to_rest_on_a_generated_track(seed):
    assert_car_rests(_scene(seed, 3))
