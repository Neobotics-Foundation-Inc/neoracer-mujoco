"""
Laws for neoracer_spec.tree: a change deep in a tree of frozen dataclasses,
tuples and mappings gives a new tree and leaves the old one as it is.
"""

from dataclasses import dataclass, field
from types import MappingProxyType

import pytest
from cases import case

from neoracer_spec.neoracer import NEORACER
from neoracer_spec.tree import at


@dataclass(frozen=True)
class Gains:
    by_axle: tuple[float, float] = (1.0, 2.0)
    by_name: MappingProxyType[str, float] = field(
        default_factory=lambda: MappingProxyType({"easy": 1.0, "hard": 3.0})
    )


@dataclass(frozen=True)
class Tree:
    gains: tuple[Gains, ...] = (Gains(), Gains())


def test_a_set_value_reads_back_and_the_old_tree_is_unchanged():
    """The new tree has the value at the path, the old one its own value,
    and the two share every branch off the path."""
    tree = Tree()
    cases = [
        (lambda t: t.gains[0].by_axle[1], 5.0, lambda t: t.gains[1]),
        (lambda t: t.gains[-1].by_name["hard"], 4.0, lambda t: t.gains[0]),
    ]
    for i, (where, value, off_path) in enumerate(cases):
        with case(i):
            new = at(tree, where).set(value)
            assert where(new) == value
            assert where(tree) != value
            assert off_path(new) is off_path(tree)
    new = at(tree, lambda t: t.gains[-1].by_name["hard"]).set(4.0)
    assert type(new.gains[-1].by_name) is MappingProxyType


def test_setting_the_value_there_gives_an_equal_tree():
    tree = NEORACER
    for where in (
        lambda c: c.chassis.wheelbase.value,
        lambda c: c.servo.low.value.voltage,
        lambda c: c.lidar.position.value,
    ):
        assert at(tree, where).set(where(tree)) == tree


def test_apply_is_set_of_the_function_of_the_value():
    focus = at(NEORACER, lambda c: c.chassis.track.value)
    track = NEORACER.chassis.track.value
    assert focus.apply(lambda x: x * 1.1) == focus.set(track * 1.1)


def test_a_record_is_checked_again_when_it_is_rebuilt():
    """Spec.__post_init__ refuses a full lock outside the fit range."""
    with pytest.raises(ValueError, match="fit_range"):
        at(NEORACER, lambda c: c.steering.inner_lock_degrees.value).set(31.0)


def test_a_where_that_computes_is_refused():
    """A where that does more than read a path, such as arithmetic, a
    constant, a test or a loop over the tree, raises."""
    for where in (
        lambda c: c.chassis.mass.value + 1.0,
        lambda c: 0.29,
        lambda c: c.chassis.track if c.chassis.mass.value else c.chassis.wheelbase,
        lambda c: list(c.lidar.position.value),
    ):
        with pytest.raises(TypeError):
            at(NEORACER, where)
