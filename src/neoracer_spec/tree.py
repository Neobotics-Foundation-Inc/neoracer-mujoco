"""
A change deep in a tree of frozen dataclasses, tuples and mappings, which
leaves the tree as it is and returns a new one:

    wide = at(NEORACER, lambda car: car.chassis.track.value).set(0.20)

where reads a path from the tree's root. at runs it once on a recorder
that notes each attribute and item it reads, and set and apply rebuild the
tree along that path: a dataclass with dataclasses.replace, which checks it
again (__post_init__), a tuple as a new tuple, and a mapping as a new
MappingProxyType. Every branch off the path is shared with the old tree.

A type checker reads where as a function of the tree, so it checks the path
and the new value's type. It does not check that where only reads, so at
runs where on the tree as well and raises TypeError unless it returns the
node at the end of the recorded path. A where that computes, such as
lambda car: car.mass.value + 1, a slice or a choice by a test of the tree,
fails, unless it chooses between nodes that are one object.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, assert_never, cast


def at[T, V](tree: T, where: Callable[[T], V]) -> Focus[T, V]:
    """The node of tree that where reads."""
    found: object = where(cast(T, _Recorder(())))
    if not isinstance(found, _Recorder):
        raise TypeError(_ONLY_READ)
    # The recorder cannot see an identity test (is), which no object can
    # intercept, nor an item that is a new value, such as a slice. Either
    # makes where return other than the recorded path's node.
    if where(tree) is not _node(tree, found._tree_steps):
        raise TypeError(_ONLY_READ)
    return Focus(tree, found._tree_steps)


@dataclass(frozen=True)
class Focus[T, V]:
    """A node of tree, by the steps from its root."""

    tree: T
    steps: tuple[Step, ...]

    def set(self, value: V) -> T:
        """tree with value at the node."""
        return _rebuild(self.tree, self.steps, lambda _: value)

    def apply(self, f: Callable[[V], V]) -> T:
        """tree with f of the node at the node."""
        return _rebuild(self.tree, self.steps, f)


@dataclass(frozen=True)
class Attr:
    name: str


@dataclass(frozen=True)
class Item:
    key: object


type Step = Attr | Item

_ONLY_READ = "where may only read attributes and items of its argument"


class _Recorder:
    """Stands in for the tree in where, and notes each step it takes."""

    __slots__ = ("_tree_steps",)

    def __init__(self, steps: tuple[Step, ...]) -> None:
        self._tree_steps = steps

    def __getattr__(self, name: str) -> _Recorder:
        return _Recorder((*self._tree_steps, Attr(name)))

    def __getitem__(self, key: object) -> _Recorder:
        return _Recorder((*self._tree_steps, Item(key)))

    # A where that compares, tests or loops over the tree computes; it does
    # not read a path. Without these, a comparison would be by identity, a
    # test would take one branch and a loop would never end. != calls __eq__.
    def __eq__(self, other: object) -> bool:
        raise TypeError(_ONLY_READ)

    def __bool__(self) -> bool:
        raise TypeError(_ONLY_READ)

    def __iter__(self) -> Iterator[object]:
        raise TypeError(_ONLY_READ)


def _node(tree: Any, steps: tuple[Step, ...]) -> Any:
    for step in steps:
        match step:
            case Attr(name=name):
                tree = getattr(tree, name)
            case Item(key=key):
                tree = tree[key]
    return tree


def _rebuild(node: Any, steps: tuple[Step, ...], f: Callable[[Any], Any]) -> Any:
    if not steps:
        return f(node)
    step, rest = steps[0], steps[1:]
    match step:
        case Attr(name=name):
            return dataclasses.replace(
                node, **{name: _rebuild(getattr(node, name), rest, f)}
            )
        case Item(key=key) if isinstance(node, tuple):
            i = range(len(node))[cast(int, key)]
            return (*node[:i], _rebuild(node[i], rest, f), *node[i + 1 :])
        case Item(key=key) if isinstance(node, Mapping):
            return MappingProxyType({**node, key: _rebuild(node[key], rest, f)})
        case Item():
            raise TypeError(f"no item of a {type(node).__name__} can be replaced")
        case _:
            assert_never(step)
