"""Laws for neoracer_mujoco.track_generation.

The geometry tests check that each generated track is closed, evenly sampled,
inside the limits of its settings and the same for the same seed. The MJCF
tests check the scene that the car drives on: the walls body, the spawn
clearance and the geom count. Other modules depend on these conventions.
"""

import functools
import itertools
from copy import replace
from typing import assert_never

import mujoco
import numpy as np
import pytest
from cases import case
from scenes import assert_car_rests, assert_takes_car_solver_options

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
    right_edge,
    settings_for_difficulty,
    to_mjcf,
    to_spec,
)
from neoracer_mujoco.track_generation import track as track_module
from neoracer_mujoco.track_generation.mjcf import (
    decimate_closed,
    lap_distance,
    split_to,
    to_start_frame,
)
from neoracer_mujoco.truth import WALLS_BODY

MAX_GEOMS = 600
"""The largest geom count of a composed scene: walls, floor and car. Over
difficulties 0-3 and seeds 0-7, the largest measured count is 298. One box for
each centerline sample gives about 4000. So this limit finds a decimation error
and still allows a long track."""


def _generate(seed: int, settings: TrackSettings) -> Track:
    match generate_track(seed, settings):
        case Track() as track:
            return track
        case NoValidTrack() as none:
            pytest.fail(f"no track: {none}")
        case other:
            assert_never(other)


@functools.cache
def _track(seed: int, difficulty: Difficulty) -> Track:
    """The generation of one track takes up to a second, and several tests use
    the same (seed, difficulty). The cache keeps the run time of this file
    short. The cache is correct only because the generation is deterministic.
    test_same_seed_gives_the_same_track_and_mjcf checks that."""
    return _generate(seed, settings_for_difficulty(difficulty))


@functools.cache
def _scene(seed: int, difficulty: Difficulty) -> Scene:
    """The compiled scene, car included, for one generated track."""
    return assets.compose(to_spec(_track(seed, difficulty)))


def test_generated_track_is_valid():
    """For seeds 0 to 3 at difficulties 0 and 3, the track keeps to the
    limits of its settings and is evenly sampled, with unit tangents normal
    to the left normals."""
    for seed, difficulty in itertools.product(range(4), (0, 3)):
        with case(f"seed {seed}, difficulty {difficulty}"):
            _check_track(_track(seed, difficulty), settings_for_difficulty(difficulty))


def _check_track(track: Track, settings: TrackSettings) -> None:
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


def _farthest_from_wall(points: np.ndarray, i: int, j: int) -> float:
    """The farthest any point strictly between i and j sits from the wall
    segment that joins them (m); j == len(points) is the wrap to point 0.
    Measured to the segment, not the line through it, so a wall that stops
    short of the edge counts too."""
    a, b = points[i], points[j % len(points)]
    interior = points[i + 1 : j]
    if len(interior) == 0:
        return 0.0
    ab = b - a
    t = np.clip((interior - a) @ ab / (ab @ ab), 0.0, 1.0)
    return float(np.linalg.norm(interior - (a + t[:, None] * ab), axis=1).max())


def _right_bends(track: Track) -> list[float]:
    """The angle (rad) of each stretch of the loop that turns right."""
    turn = track.curvature * track.total_length / len(track.curvature)  # rad
    right = turn < 0
    start = int(np.argmin(right))  # a sample that turns left, if there is one
    bends, angle = [], 0.0
    for i in range(start, start + len(turn)):
        if right[i % len(turn)]:
            angle -= turn[i % len(turn)]
        elif angle > 0:
            bends.append(angle)
            angle = 0.0
    return [*bends, angle] if angle > 0 else bends


def test_generated_tracks_bend_both_ways():
    """Each loop goes once around counterclockwise: its turns add up to one
    full turn to the left. The loops are not convex: some stretches turn
    right, by more than 22.5 degrees at difficulty 0 and by more than 60
    degrees at difficulty 3."""
    for difficulty, bend in [(0, np.pi / 8), (3, np.pi / 3)]:
        tracks = [_track(seed, difficulty) for seed in range(4)]
        for track in tracks:
            turn = track.curvature.sum() * track.total_length / len(track.curvature)
            assert turn == pytest.approx(2 * np.pi, abs=0.01), difficulty
        largest = max(max(_right_bends(t), default=0.0) for t in tracks)
        assert largest > bend, difficulty


def test_a_clockwise_loop_is_rejected(monkeypatch):
    """A walk can go around clockwise: 16 steps of -pi/8, with two of them
    +15 pi/8 instead, sum to +2 pi. The same circle is a track
    counterclockwise and no track clockwise."""
    angle = np.linspace(0.0, 2 * np.pi, 16, endpoint=False)
    circle = 8.0 * np.stack([np.cos(angle), np.sin(angle)], axis=1)  # 50 m
    settings = replace(settings_for_difficulty(1), max_attempts=2)
    for clockwise in (False, True):
        points = circle[::-1] if clockwise else circle
        monkeypatch.setattr(
            track_module, "draw_control_points", lambda *_, points=points: points
        )
        generated = generate_track(0, settings)
        assert isinstance(generated, NoValidTrack) == clockwise, clockwise


def test_same_seed_gives_the_same_track_and_mjcf():
    """A new generation gives the same track as the cached one, and the start
    transform and the decimation, which run on it, give the same MJCF text.
    A different seed gives a different track and text."""
    fresh = _generate(0, settings_for_difficulty(3))
    cached = _track(0, 3)
    assert np.array_equal(fresh.center, cached.center)
    assert np.array_equal(fresh.half_width, cached.half_width)
    assert to_mjcf(fresh) == to_mjcf(cached)
    assert not np.array_equal(_track(1, 3).center, cached.center)
    assert to_mjcf(_track(1, 3)) != to_mjcf(cached)


def test_harder_difficulty_allows_tighter_turns():
    """Each step up in difficulty allows a larger curvature and a narrower
    track."""
    difficulties: tuple[Difficulty, ...] = (0, 1, 2, 3)
    for easy, hard in itertools.pairwise(map(settings_for_difficulty, difficulties)):
        assert hard.max_curvature > easy.max_curvature
        assert hard.half_width_min < easy.half_width_min


def test_settings_reject_values_that_make_no_track():
    for make in [
        lambda: TrackSettings(samples=0),
        lambda: TrackSettings(oversample_factor=0),
        lambda: TrackSettings(control_points_min=30),
        lambda: TrackSettings(edge_length_min=0.0),
        lambda: TrackSettings(turn_jitter=-0.1),
        lambda: TrackSettings(min_total_length=100.0),
        lambda: TrackSettings(half_width_min=0.9),
        lambda: TrackSettings(max_attempts=0),
        lambda: TrackSettings(max_repair_shift=-0.1),
    ]:
        with pytest.raises(ValueError, match="TrackSettings needs"):
            make()


def test_impossible_limits_return_no_valid_track():
    impossible = TrackSettings(max_curvature=0.0, max_attempts=2)
    assert generate_track(0, impossible) == NoValidTrack(seed=0, attempts=2)


def test_a_generated_scene_is_drivable():
    """For seeds 0 to 2 at difficulties 0 and 3: the walls body holds every
    wall box and not the floor, as the wall contact sensor counts contacts
    with the walls body only (truth.py). The decimation keeps the geom count
    below MAX_GEOMS: one box for each centerline sample gives thousands of
    geoms, and MuJoCo checks each geom for contacts at each step. The start
    transform puts the car on sample 0, facing +X, where it touches no wall
    and is not near one."""
    for seed, difficulty in itertools.product(range(3), (0, 3)):
        with case(f"seed {seed}, difficulty {difficulty}"):
            track = _track(seed, difficulty)
            spec = to_spec(track)
            walls = spec.body(WALLS_BODY)
            assert walls is not None
            assert len(walls.geoms) == spec.compile().ngeom - 1
            assert all(geom.name.startswith("track_wall_") for geom in walls.geoms)

            scene = _scene(seed, difficulty)
            assert scene.model.ngeom < MAX_GEOMS
            data = mujoco.MjData(scene.model)
            mujoco.mj_forward(scene.model, data)
            assert not car_touches_wall(data, scene.car)
            scene_walls = scene.spec.body(WALLS_BODY)
            assert scene_walls is not None
            nearest = min(
                float(np.linalg.norm(data.bind_scalar(geom).xpos[:2]))
                for geom in scene_walls.geoms
            )
            assert nearest >= 0.4 * 2 * track.half_width[0]


def test_walls_track_the_generated_edges_closely():
    """No point of the fine edge is farther from the decimated wall line
    than WallParams().max_deviation."""
    walls = WallParams()
    for difficulty in (0, 3):
        track = _track(0, difficulty)
        for edge in (left_edge(track), right_edge(track)):
            edge = to_start_frame(track, edge)
            kept = decimate_closed(edge, walls.segment_length, walls.max_deviation)
            for i, j in itertools.pairwise(kept):
                deviation = _farthest_from_wall(edge, i, j)
                assert deviation <= walls.max_deviation + 1e-9, difficulty


def test_the_car_rests_on_a_generated_track_with_its_solver_options():
    track = _track(0, 3)
    assert_takes_car_solver_options(to_spec(track))
    assert_car_rests(_scene(0, 3))


FIXED = WallParams(boxes_per_wall=192)


def test_a_fixed_box_count_gives_every_track_the_same_model_size():
    """With boxes_per_wall, tracks of every difficulty compile to models with
    the same geoms and bodies, so a batch can step one per row."""
    sizes = set()
    for seed, difficulty in itertools.product(range(2), (0, 3)):
        spec = to_spec(_track(seed, difficulty), FIXED)
        model = assets.compose(spec, visuals=False).model
        sizes.add((model.ngeom, model.nbody, model.nq, model.nsensordata))
    assert len(sizes) == 1


def test_split_walls_still_track_the_edges():
    """After split_to, every box stays within max_deviation of the edge and
    the boxes cover the whole loop."""
    walls = WallParams()
    for difficulty in (0, 3):
        track = _track(1, difficulty)
        for edge in (left_edge(track), right_edge(track)):
            edge = to_start_frame(track, edge)
            kept = decimate_closed(edge, walls.segment_length, walls.max_deviation)
            kept = split_to(edge, kept, 192, walls.max_deviation)
            assert len(kept) - 1 == 192 and kept[0] == 0 and kept[-1] == len(edge)
            for i, j in itertools.pairwise(kept):
                deviation = _farthest_from_wall(edge, i, j)
                assert deviation <= walls.max_deviation + 1e-9, difficulty


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
        assert _farthest_from_wall(points, i, j) <= 0.01


def test_repeated_splits_stay_within_max_deviation():
    """A half made by one split was never checked by decimate_closed, so the
    next split of it must check both of its halves too. Here the middle
    samples would give a box 10.75 mm from the edge."""
    y = [0, 0, 0, 0, 0.005, -0.006, 0.0045, 0, 0]
    points = np.array([(0.062 * i, v) for i, v in enumerate(y)])
    kept = decimate_closed(points, 0.5, 0.01)
    assert kept == [0, 8, 9]
    kept = split_to(points, kept, 5, max_deviation=0.01)
    assert len(kept) == 6
    for i, j in itertools.pairwise(kept):
        assert _farthest_from_wall(points, i, j) <= 0.01


def test_lap_distance_finds_the_sample_across_the_corridor():
    """A point on the centerline, or moved across the corridor by up to 80% of
    its half-width, is at the distance along the loop of its sample."""
    for difficulty in (0, 3):
        track = _track(2, difficulty)
        samples = np.arange(0, len(track.center), 7)
        for side in (-0.8, 0.0, 0.8):
            moved = track.center + side * track.half_width[:, None] * track.left_normal
            xy = to_start_frame(track, moved)[samples]
            np.testing.assert_array_equal(
                lap_distance(track, xy),
                track.distance_along[samples],
                err_msg=f"difficulty {difficulty}, side {side}",
            )
