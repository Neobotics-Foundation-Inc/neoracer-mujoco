"""Laws for neoracer_mujoco.track_generation.

The geometry tests check that each generated track is closed, evenly sampled,
inside the limits of its settings and the same for the same seed. The MJCF
tests check the scene that the car drives on: the walls body, the spawn
clearance and the geom count. Other modules depend on these conventions.
"""

import functools
import itertools
from typing import assert_never

import mujoco
import numpy as np
import pytest
from scenes import assert_car_rests, assert_keeps_car_solver_options

from neoracer_mujoco import Scene, assets
from neoracer_mujoco.collision import car_touches_wall
from neoracer_mujoco.track_generation import (
    Difficulty,
    NoValidTrack,
    Track,
    TrackSettings,
    WallParams,
    generate_track,
    left_edge,
    mjcf,
    right_edge,
    settings_for_difficulty,
    to_mjcf,
    to_spec,
)
from neoracer_mujoco.track_generation.mjcf import (
    decimate_closed,
    lap_distance,
    split_to,
    to_start_frame,
)
from neoracer_mujoco.truth import WALLS_BODY

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
def _scene(seed: int, difficulty: Difficulty) -> Scene:
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
    scene = _scene(seed, 3)
    assert scene.model.bind_scalar(scene.car.body).id > 0


@pytest.mark.parametrize("seed", range(3))
def test_generated_walls_are_in_the_walls_body(seed):
    """The wall contact sensor counts contacts with the walls body only (see
    truth.py). Every wall box is in it; the floor is not."""
    spec = to_spec(_track(seed, 3))
    walls = spec.body(WALLS_BODY)
    assert walls is not None
    assert len(walls.geoms) == spec.compile().ngeom - 1
    assert all(geom.name.startswith("track_wall_") for geom in walls.geoms)


@pytest.mark.parametrize("difficulty", [0, 3])
@pytest.mark.parametrize("seed", range(3))
def test_car_spawns_clear_of_the_walls(seed, difficulty):
    """The start transform puts the car on sample 0, facing +X. The car does
    not touch a wall and is not near one."""
    track = _track(seed, difficulty)
    scene = _scene(seed, difficulty)
    walls = scene.spec.body(WALLS_BODY)
    assert walls is not None
    data = mujoco.MjData(scene.model)
    mujoco.mj_forward(scene.model, data)

    assert not car_touches_wall(data, scene.car)

    corridor_width = 2 * track.half_width[0]
    nearest = min(
        float(np.linalg.norm(data.bind_scalar(geom).xpos[:2])) for geom in walls.geoms
    )
    assert nearest >= 0.4 * corridor_width


@pytest.mark.parametrize("difficulty", [0, 3])
@pytest.mark.parametrize("seed", range(3))
def test_geom_count_stays_within_budget(seed, difficulty):
    """The decimation keeps the geom count below MAX_GEOMS. One box for each
    centerline sample gives thousands of geoms, and MuJoCo checks each geom
    for contacts at each step."""
    model = _scene(seed, difficulty).model
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


# --- fixed-size walls and the lap distance -------------------------------------

FIXED = WallParams(boxes_per_wall=192)


def test_a_fixed_box_count_gives_every_track_the_same_model_size():
    """With boxes_per_wall, tracks of every difficulty compile to models with
    the same geoms and bodies, so a batch can step one per row."""
    sizes = set()
    for seed, difficulty in itertools.product(range(3), (0, 3)):
        model = assets.compose(to_spec(_track(seed, difficulty), FIXED)).model
        sizes.add((model.ngeom, model.nbody, model.nq, model.nsensordata))
    assert len(sizes) == 1


@pytest.mark.parametrize("difficulty", [0, 3])
def test_split_walls_still_track_the_edges(difficulty):
    """After split_to, every box stays within max_deviation of the edge and
    the boxes cover the whole loop."""
    track = _track(1, difficulty)
    walls = WallParams()
    for edge in (left_edge(track), right_edge(track)):
        edge = to_start_frame(track, edge)
        kept = decimate_closed(edge, walls.segment_length, walls.max_deviation)
        kept = split_to(edge, kept, 192, walls.max_deviation)
        assert len(kept) - 1 == 192 and kept[0] == 0 and kept[-1] == len(edge)
        for i, j in itertools.pairwise(kept):
            first, last = edge[i], edge[j % len(edge)]
            chord = last - first
            interior = edge[i + 1 : j] - first
            if len(interior) == 0:
                continue
            across = chord[0] * interior[:, 1] - chord[1] * interior[:, 0]
            assert (
                np.abs(across).max() / np.linalg.norm(chord)
                <= walls.max_deviation + 1e-9
            )


def test_split_to_refuses_fewer_boxes_than_a_wall_needs():
    with pytest.raises(ValueError, match="boxes_per_wall"):
        split_to(np.zeros((15, 2)), [0, 5, 10, 15], 2, 0.01)


def test_a_split_where_the_edge_bends_stays_within_max_deviation():
    """Here the middle sample of the first box would leave a second half
    10.75 mm from the edge, more than max_deviation = 10 mm, so split_to
    takes another sample."""
    points = np.array(
        [(0, 0), (0.124, 0), (0.248, 0.0035), (0.372, -0.009), (0.496, 0)]
    )
    kept = split_to(points, [0, 4, 5], 3, max_deviation=0.01)
    assert len(kept) == 4
    for i, j in itertools.pairwise(kept):
        assert mjcf._max_deviation(points, i, j) <= 0.01


@pytest.mark.parametrize("difficulty", [0, 3])
def test_lap_distance_finds_the_sample_across_the_corridor(difficulty):
    """A point on the centerline, or moved across the corridor by up to 80% of
    its half-width, is at the distance along the loop of its sample."""
    track = _track(2, difficulty)
    samples = np.arange(0, len(track.center), 7)
    for side in (-0.8, 0.0, 0.8):
        moved = track.center + side * track.half_width[:, None] * track.left_normal
        xy = to_start_frame(track, moved)[samples]
        np.testing.assert_array_equal(
            lap_distance(track, xy), track.distance_along[samples]
        )
