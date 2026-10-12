"""Tests for examples/track_centering_demo.py."""

import pytest

from examples.track_centering_demo import DemoParams, run


@pytest.mark.parametrize(
    "bad",
    [{"steps": 3}, {"print_every_steps": 0}, {"viewer_duration_s": 0.0}],
)
def test_demo_params_reject_values_that_give_no_report(bad):
    with pytest.raises(ValueError):
        DemoParams(**bad)


def test_headless_run_centers_the_car():
    """The default run starts 0.25 m off center. The controller drives down
    the corridor, stays within its steer limits, keeps the car upright and
    reduces the lateral error from the first to the last quarter."""
    report = run(DemoParams())
    assert report.finite_throughout and report.upright_throughout
    assert report.steer_within_limits
    assert report.distance_traveled_x > 5.0  # m in 6 s
    assert (
        report.last_quarter_mean_abs_lateral_error
        < report.first_quarter_mean_abs_lateral_error
    )
