"""
Laws for neoracer_spec.typescript: every dataclass of Spec and
DerivedParams has an interface with each of its fields, a field's docstring
is its TSDoc comment, and the members of a union of dataclasses carry their
kind.
"""

import ast
import dataclasses
import inspect
import itertools
import re
import textwrap
import typing

import pytest
from cases import case

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


def test_a_fields_docstring_is_the_comment_above_it():
    """The first paragraph of a field's docstring, on one line, is the TSDoc
    comment just above the field in its interface."""
    documented = 0
    for cls in _dataclasses(Spec) | _dataclasses(derive.DerivedParams):
        (node,) = ast.parse(textwrap.dedent(inspect.getsource(cls))).body
        assert isinstance(node, ast.ClassDef)
        for field, after in itertools.pairwise(node.body):
            if not (
                isinstance(field, ast.AnnAssign)
                and isinstance(field.target, ast.Name)
                and isinstance(after, ast.Expr)
                and isinstance(after.value, ast.Constant)
                and isinstance(after.value.value, str)
            ):
                continue
            name = field.target.id
            text = " ".join(after.value.value.strip().split("\n\n")[0].split())
            with case(f"{cls.__name__}.{name}"):
                assert f"  /** {text} */\n  readonly {name}:" in TEXT
            documented += 1
    assert documented > 0


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


def test_a_tuple_of_any_length_wraps_an_element_type_that_needs_it():
    """[] binds tighter than | and than readonly, so an element type with
    either is in parentheses."""
    for t, ts in [
        (tuple[float, ...], "readonly number[]"),
        (tuple[float | str, ...], "readonly (number | string)[]"),
        (tuple[tuple[float, float], ...], "readonly (readonly [number, number])[]"),
    ]:
        with case(ts):
            assert typescript._type(t) == ts


def test_a_type_with_no_typescript_is_refused():
    with pytest.raises(TypeError, match="no TypeScript type"):
        typescript._type(dict[str, float])
