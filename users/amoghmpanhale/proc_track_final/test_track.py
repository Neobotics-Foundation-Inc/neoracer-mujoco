"""Checks that every generated track is closed, evenly sampled, within its own
limits, and reproducible. Run: python3 -m pytest users/amoghmpanhale/proc_track_final
"""

import numpy as np
import pytest

from . import TrackSettings, generate_track, settings_for_difficulty

@pytest.mark.parametrize("difficulty", [0, 3])
@pytest.mark.parametrize("seed", range(4))
def test_generated_track_is_valid(seed, difficulty):
    settings = settings_for_difficulty(difficulty)
    track = generate_track(seed, difficulty=difficulty)

    assert np.abs(track.curvature).max() <= settings.max_curvature
    assert (track.half_width * np.abs(track.curvature)).max() <= settings.max_width_times_curvature
    assert settings.min_total_length <= track.total_length <= settings.max_total_length
    assert track.half_width.min() >= settings.half_width_min
    assert track.half_width.max() <= settings.half_width_max

    hops = np.linalg.norm(np.roll(track.center, -1, 0) - track.center, axis=1)
    expected = track.total_length / len(track.center)
    # chord vs arc, so hops sit just under the spacing but must all agree
    assert np.allclose(hops, hops[0], rtol=0.05)
    assert hops.mean() == pytest.approx(expected, rel=0.02)

    assert np.allclose(np.linalg.norm(track.tangent, axis=1), 1.0)
    assert np.allclose(np.sum(track.tangent * track.left_normal, axis=1), 0.0, atol=1e-9)
    assert track.left_edge.shape == track.center.shape


def test_same_seed_gives_the_same_track():
    a, b = generate_track(3), generate_track(3)
    assert np.array_equal(a.center, b.center)
    assert np.array_equal(a.half_width, b.half_width)
    assert not np.array_equal(a.center, generate_track(4).center)


def test_harder_difficulty_allows_tighter_turns():
    easy = settings_for_difficulty(0)
    hard = settings_for_difficulty(3)
    assert hard.max_curvature > easy.max_curvature
    assert hard.half_width_min < easy.half_width_min
    with pytest.raises(ValueError):
        settings_for_difficulty(4)


def test_impossible_limits_raise_rather_than_hang():
    from .track import TrackGenerationError

    impossible = TrackSettings(max_curvature=0.0, max_attempts=2)
    with pytest.raises(TrackGenerationError):
        generate_track(0, settings=impossible)
