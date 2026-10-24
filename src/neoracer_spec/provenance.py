"""
Where each value of a car's specification comes from.

A value of the specification is a Sourced[T]: the value, its status (how we
know it) and the values that other sources give for it, when they disagree.

    measured(0.28764, CAD, conflicts=(Alternative(0.280, DOCS_CHASSIS),))
    estimated(1.5, "no tire has been tested")

walk lists every Sourced value of a specification by its path, so that code
and tests can ask which values are estimated or disputed.
"""

import dataclasses
from collections.abc import Iterator
from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    name: str
    """A short name: "CAD", "docs", "firmware"."""
    where: str
    """Enough to find it: a file, a page, a person and a date."""


@dataclass(frozen=True)
class Measured:
    """Measured on the car or on its CAD model."""

    source: Source


@dataclass(frozen=True)
class Calibrated:
    """Fitted so that the car's own readings or commands agree with a
    reference: a scale factor of its odometry, the mounting angle of a
    sensor, the angle a steering command gives."""

    source: Source
    reference: str
    """What it was fitted against: "the LiDAR", "a tape"."""


@dataclass(frozen=True)
class Documented:
    """Read from a datasheet, the docs, the firmware or a config file."""

    source: Source


@dataclass(frozen=True)
class Assumed:
    """Implied by a source but not stated by it."""

    source: Source
    why: str


@dataclass(frozen=True)
class Estimated:
    """Nobody has measured it or read it anywhere."""

    why: str


@dataclass(frozen=True)
class Chosen:
    """The model's own value, not a property of the car."""

    why: str


type Status = Measured | Calibrated | Documented | Assumed | Estimated | Chosen


@dataclass(frozen=True)
class Alternative[T]:
    """A value that another source gives."""

    value: T
    source: Source


@dataclass(frozen=True)
class Sourced[T]:
    value: T
    status: Status
    note: str = ""
    """Why this value, when it needs saying."""
    conflicts: tuple[Alternative[T], ...] = ()
    """The values that other sources give. The spec is INCONSISTENT there."""


def measured[T](
    value: T, source: Source, note: str = "", conflicts: tuple[Alternative[T], ...] = ()
) -> Sourced[T]:
    return Sourced(value, Measured(source), note, conflicts)


def calibrated[T](
    value: T,
    source: Source,
    reference: str,
    note: str = "",
    conflicts: tuple[Alternative[T], ...] = (),
) -> Sourced[T]:
    return Sourced(value, Calibrated(source, reference), note, conflicts)


def documented[T](
    value: T, source: Source, note: str = "", conflicts: tuple[Alternative[T], ...] = ()
) -> Sourced[T]:
    return Sourced(value, Documented(source), note, conflicts)


def assumed[T](
    value: T, source: Source, why: str, conflicts: tuple[Alternative[T], ...] = ()
) -> Sourced[T]:
    return Sourced(value, Assumed(source, why), "", conflicts)


def estimated[T](
    value: T, why: str, conflicts: tuple[Alternative[T], ...] = ()
) -> Sourced[T]:
    return Sourced(value, Estimated(why), "", conflicts)


def chosen[T](value: T, why: str) -> Sourced[T]:
    return Sourced(value, Chosen(why))


def walk(spec: object, path: str = "") -> Iterator[tuple[str, Sourced[object]]]:
    """Every Sourced value inside spec, a tree of frozen dataclasses, with its
    dotted path: ("chassis.wheelbase", Sourced(...))."""
    if isinstance(spec, Sourced):
        yield path, spec
    elif dataclasses.is_dataclass(spec) and not isinstance(spec, type):
        for field in dataclasses.fields(spec):
            name = f"{path}.{field.name}" if path else field.name
            yield from walk(getattr(spec, field.name), name)
