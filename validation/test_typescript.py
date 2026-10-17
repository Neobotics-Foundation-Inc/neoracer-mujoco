"""
Laws for neoracer_spec.typescript: js/src/spec.ts is what it writes, every
dataclass of Spec and DerivedParams has an interface with each of its
fields, and the members of a union of dataclasses carry their kind.
"""

import dataclasses
import re
import typing
from pathlib import Path

import pytest

from neoracer_spec import derive, provenance, typescript
from neoracer_spec.neoracer import NEORACER
from neoracer_spec.spec import Spec

SPEC_TS = Path(__file__).resolve().parent.parent / "js" / "src" / "spec.ts"
TEXT = typescript.module()


def test_the_checked_in_typescript_is_what_the_generator_writes():
    """After a change to the spec, run
    python -m neoracer_spec.typescript js/src/spec.ts."""
    assert SPEC_TS.read_text(encoding="utf-8") == TEXT


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


@pytest.mark.parametrize(
    "cls",
    sorted(
        _dataclasses(Spec) | _dataclasses(derive.DerivedParams),
        key=lambda c: c.__name__,
    ),
    ids=lambda c: c.__name__,
)
def test_each_dataclass_is_an_interface_with_its_fields(cls):
    fields = [f.name for f in dataclasses.fields(cls)]
    assert [f for f in _interfaces()[cls.__name__] if f != "kind"] == fields


def test_the_statuses_carry_their_kind():
    interfaces = _interfaces()
    for status in typing.get_args(provenance.Status.__value__):
        assert interfaces[status.__name__][0] == "kind"
    assert "kind" not in interfaces["Sourced"]
    assert "export type Status = Measured | Documented | Assumed" in TEXT


def test_the_values_are_checked_against_their_types():
    assert TEXT.count("as const satisfies Spec;") == 1
    assert TEXT.count("as const satisfies DerivedParams;") == 1
    assert f"slots: {derive.derive(NEORACER).lidar.slots}," in TEXT


def test_a_type_with_no_typescript_is_refused():
    with pytest.raises(TypeError, match="no TypeScript type"):
        typescript._type(dict[str, float])
