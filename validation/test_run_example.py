"""Tests for examples/run.py."""

import pytest

from examples.run import MODES, DriveParams, _command


@pytest.mark.parametrize("bad", [{"print_every_steps": 0}, {"speed_limit_ms": 0.0}])
def test_drive_params_reject_values_that_break_the_loop(bad):
    with pytest.raises(ValueError):
        DriveParams(**bad)


@pytest.mark.parametrize("mode", list(MODES))
def test_command_coasts_at_the_speed_limit_and_steers_in_bounds(mode):
    """Below the speed limit the throttle is throttle_nm; at or above it the
    throttle is 0. The steer never leaves [-steer_amp_rad, steer_amp_rad]."""
    params = MODES[mode]
    for t in [0.0, 0.4, 1.3, 2.9, 7.7]:  # s
        for speed in [0.0, 0.5, 1.5, 3.0]:  # m/s
            command = _command(params, t, speed)
            limit = params.speed_limit_ms
            coasting = limit is not None and speed >= limit
            assert command.throttle == (0.0 if coasting else params.throttle_nm)
            assert abs(command.steer) <= params.steer_amp_rad
