"""Tests for examples/track_centering_demo.py."""

import pytest

from examples.track_centering_demo import DemoParams


@pytest.mark.parametrize(
    "bad",
    [{"steps": 3}, {"print_every_steps": 0}, {"viewer_duration_s": 0.0}],
)
def test_demo_params_reject_values_that_give_no_report(bad):
    with pytest.raises(ValueError):
        DemoParams(**bad)
