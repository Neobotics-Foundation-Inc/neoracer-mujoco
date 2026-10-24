"""Tests for examples/run.py."""

import pytest

from examples.run import MODES, DriveParams, _command


def test_drive_params_reject_values_that_break_the_loop():
    for make in [
        lambda: DriveParams(print_every_steps=0),
        lambda: DriveParams(speed_limit_ms=0.0),
    ]:
        with pytest.raises(ValueError):
            make()


def test_command_coasts_at_the_speed_limit_and_steers_in_bounds():
    """In every mode, below the speed limit the throttle is params.throttle;
    at or above it the throttle is 0. The steer never leaves
    [-steer_amp_rad, steer_amp_rad]."""
    for mode, params in MODES.items():
        for t in [0.0, 0.4, 1.3, 2.9, 7.7]:
            for speed in [0.0, 0.5, 1.5, 3.0]:
                command = _command(params, t, speed)
                limit = params.speed_limit_ms
                coasting = limit is not None and speed >= limit
                case = f"{mode} at t={t} s, {speed} m/s"
                assert command.throttle == (0.0 if coasting else params.throttle), case
                assert abs(command.steer) <= params.steer_amp_rad, case
