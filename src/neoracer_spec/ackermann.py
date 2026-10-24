"""
Ideal Ackermann steering: the angle of each front wheel for a steering
input, the angle of a virtual wheel at the middle of the front axle.

Frame: +X forward, +Y left; angles in rad, + = left.
"""

import math


def left_wheel(wheelbase: float, track: float, steer: float) -> float:
    """The left front wheel's angle (rad) for a steering input of steer
    (rad). The left wheel is the inner one in a left turn, so it turns more
    than steer, and the outer one in a right turn."""
    t = math.tan(steer)
    return math.atan2(wheelbase * t, wheelbase - t * track / 2)


def right_wheel(wheelbase: float, track: float, steer: float) -> float:
    """The right front wheel's angle (rad): the left's mirror image."""
    return -left_wheel(wheelbase, track, -steer)


def steer_for_inner(wheelbase: float, track: float, inner: float) -> float:
    """The steering input (rad) that turns the inner front wheel to inner > 0
    (rad): the inverse of left_wheel."""
    return math.atan(wheelbase / (wheelbase / math.tan(inner) + track / 2))
