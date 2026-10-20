"""
Laws for the keyboard control of the manual driving demo (examples/manual_drive.py).

update() is pure, so these tests call it directly, without a window or the simulator.
"""

import itertools

import numpy as np

from examples.manual_drive import ManualDriveParams, Pedal, SteerKey, update
from neoracer_mujoco.control.steering import ControlCommand

PARAMS = ManualDriveParams()
DT = 0.01  # s; one frame at 100 Hz
STOPPED = ControlCommand(throttle=0.0, steer=0.0)


def _hold(
    command: ControlCommand, pedal: Pedal, steer_key: SteerKey, frames: int
) -> list[ControlCommand]:
    """The commands of `frames` frames with the same keys held."""
    commands = [command]
    for _ in range(frames):
        commands.append(update(PARAMS, commands[-1], DT, pedal, steer_key))
    return commands


def _mirror(command: ControlCommand) -> ControlCommand:
    return ControlCommand(throttle=-command.throttle, steer=-command.steer)


def test_held_throttle_increases_to_the_cap_and_stays_there():
    throttles = [c.throttle for c in _hold(STOPPED, "forward", "center", 1000)]
    assert all(b >= a for a, b in itertools.pairwise(throttles))
    assert throttles[1] > throttles[0]
    assert throttles[-1] == PARAMS.throttle.max
    assert max(throttles) == PARAMS.throttle.max


def test_released_throttle_decreases_to_exactly_zero():
    full = ControlCommand(throttle=PARAMS.throttle.max, steer=0.0)
    throttles = [c.throttle for c in _hold(full, "coast", "center", 1000)]
    assert throttles[1] < throttles[0]
    assert all(b <= a for a, b in itertools.pairwise(throttles))
    assert throttles[-1] == 0.0
    assert min(throttles) == 0.0


def test_released_steer_returns_to_exactly_center():
    left = _hold(STOPPED, "coast", "left", 1000)[-1]
    assert left.steer == PARAMS.steering.max
    steers = [c.steer for c in _hold(left, "coast", "center", 1000)]
    assert all(b <= a for a, b in itertools.pairwise(steers))
    assert steers[-1] == 0.0
    assert min(steers) == 0.0


def test_mirrored_keys_give_a_mirrored_command():
    """Reverse and right on a mirrored command give the mirror of forward and
    left, from any command."""
    rng = np.random.default_rng(0)
    pairs: list[tuple[Pedal, SteerKey, Pedal, SteerKey]] = [
        ("forward", "left", "reverse", "right"),
        ("coast", "center", "coast", "center"),
        ("reverse", "center", "forward", "center"),
    ]
    for _ in range(5):
        command = ControlCommand(
            throttle=float(rng.uniform(-PARAMS.throttle.max, PARAMS.throttle.max)),
            steer=float(rng.uniform(-PARAMS.steering.max, PARAMS.steering.max)),
        )
        for pedal, steer_key, mirror_pedal, mirror_steer_key in pairs:
            direct = update(PARAMS, command, DT, pedal, steer_key)
            mirrored = update(
                PARAMS, _mirror(command), DT, mirror_pedal, mirror_steer_key
            )
            assert mirrored == _mirror(direct), (command, pedal, steer_key)
