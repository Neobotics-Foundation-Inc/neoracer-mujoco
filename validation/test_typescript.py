"""
Laws for neoracer_spec.typescript: every dataclass of Spec and
DerivedParams has an interface with each of its fields, and the members of
a union of dataclasses carry their kind.
"""

import dataclasses
import re
import typing

import pytest

from neoracer_spec import derive, provenance, typescript
from neoracer_spec.neoracer import NEORACER
from neoracer_spec.spec import Spec

TEXT = typescript.module()


def _interfaces() -> dict[str, list[str]]:
    """Each interface's name and its fields' names."""
    return {
        name: re.findall(r"^  readonly (\w+):", body, flags=re.MULTILINE)
        for name, body in re.findall(
            r"^export interface (\w+)(?:<\w+>)? \{\n(.*?)^\}",
            TEXT,
            re.MULTILINE | re.DOTALL,
        )
    }


def _dataclasses(root: type) -> set[type]:
    found = {root}
    for hint in typing.get_type_hints(root).values():
        for t in [hint, *typing.get_args(hint)]:
            if dataclasses.is_dataclass(t) and isinstance(t, type):
                found |= _dataclasses(t)
    return found


def test_each_dataclass_is_an_interface_with_its_fields():
    interfaces = _interfaces()
    for cls in _dataclasses(Spec) | _dataclasses(derive.DerivedParams):
        assert dataclasses.is_dataclass(cls)
        fields = [f.name for f in dataclasses.fields(cls)]
        got = [f for f in interfaces[cls.__name__] if f != "kind"]
        assert got == fields, cls.__name__


def test_the_statuses_carry_their_kind():
    interfaces = _interfaces()
    statuses = typing.get_args(provenance.Status.__value__)
    for status in statuses:
        assert interfaces[status.__name__][0] == "kind"
    assert "kind" not in interfaces["Sourced"]
    names = " | ".join(status.__name__ for status in statuses)
    assert f"export type Status = {names};" in TEXT


def test_the_values_are_checked_against_their_types():
    assert TEXT.count("as const satisfies Spec;") == 1
    assert TEXT.count("as const satisfies DerivedParams;") == 1
    assert f"slots: {derive.derive(NEORACER).lidar.slots}," in TEXT


def test_a_type_with_no_typescript_is_refused():
    with pytest.raises(TypeError, match="no TypeScript type"):
        typescript._type(dict[str, float])
