"""Tests for examples/run.py."""

import pytest
from cases import case

from examples.run import MODES, DriveParams, _command


def test_drive_params_reject_values_that_break_the_loop():
    with pytest.raises(ValueError):
        DriveParams(print_every_steps=0)


def test_command_asks_for_the_mode_s_speed_and_steers_in_bounds():
    """In every mode, the command asks for params.speed at every time, and
    its steer never leaves [-steer_amp_rad, steer_amp_rad]."""
    for mode, params in MODES.items():
        for t in [0.0, 0.4, 1.3, 2.9, 7.7]:
            with case(f"{mode} at t={t} s"):
                command = _command(params, t)
                assert command.target == params.speed
                assert abs(command.steer) <= params.steer_amp_rad
