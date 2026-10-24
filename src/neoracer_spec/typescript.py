"""
The specification and its derivation as TypeScript, for the JS package: an
interface for each dataclass, a type for each type alias, and the
NeoRacer's Spec (NEORACER) and DerivedParams (NEORACER_PARAMS) as values
that the TypeScript compiler checks against them.

    float, int          number
    str, bool           string, boolean
    Literal["a", "b"]   "a" | "b"
    tuple[A, B]         readonly [A, B]
    tuple[A, ...]       readonly A[]
    X | Y               X | Y
    Sourced[T]          Sourced<T>, from the generic dataclass

A dataclass that is one of a union of dataclasses, as Measured is of
Status, gets a field `kind`, its name, so that TypeScript tells them apart.
Field names stay as in Python. The JS package's spec.ts is what module()
writes; neoracer_mujoco.javascript writes it when the package is built.
"""

import ast
import dataclasses
import functools
import inspect
import itertools
import json
import math
import sys
import textwrap
import types
import typing
from collections.abc import Iterator, Mapping

from . import derive
from .neoracer import NEORACER
from .spec import Spec

type _Named = type | typing.TypeAliasType


def module() -> str:
    """The TypeScript module of the types of Spec and DerivedParams, with
    NEORACER and derive(NEORACER) as NEORACER_PARAMS."""
    named = _named((Spec, derive.DerivedParams))
    tagged = _tagged(named)
    spec = _value(NEORACER, tagged)
    params = _value(derive.derive(NEORACER), tagged)
    neoracer = (
        "/** The NeoRacer, each value with its source. */\n"
        f"export const NEORACER = {spec} as const satisfies Spec;\n"
    )
    neoracer_params = (
        "/** What the simulators run for the NeoRacer: derive(NEORACER). */\n"
        f"export const NEORACER_PARAMS = {params} as const satisfies DerivedParams;\n"
    )
    declarations = [_declaration(t, tagged) for t in named]
    return "\n".join([*declarations, neoracer, neoracer_params])


def _named(roots: tuple[type, ...]) -> list[_Named]:
    """The dataclasses and type aliases that roots reach, each after those it
    names."""
    seen: list[_Named] = []

    def visit(t: object) -> None:
        if typing.get_origin(t) is typing.Literal:
            return
        named = isinstance(t, typing.TypeAliasType) or (
            dataclasses.is_dataclass(t) and isinstance(t, type)
        )
        if named and t in seen:
            return
        for part in _parts(t):
            visit(part)
        if named:
            seen.append(t)

    for root in roots:
        visit(root)
    return seen


def _parts(t: object) -> Iterator[object]:
    """The types that t names directly."""
    if isinstance(t, typing.TypeAliasType):
        yield t.__value__
    elif dataclasses.is_dataclass(t) and isinstance(t, type):
        yield from typing.get_type_hints(t).values()
    else:
        yield from typing.get_args(t)
        origin = typing.get_origin(t)
        if origin is not None and origin is not typing.Literal:
            yield origin


def _tagged(named: list[_Named]) -> set[type]:
    """The dataclasses that are members of a union of dataclasses."""
    tagged: set[type] = set()
    for t in named:
        hints = (
            [t.__value__]
            if isinstance(t, typing.TypeAliasType)
            else typing.get_type_hints(t).values()
        )
        for hint in hints:
            members = _members(hint)
            if len(members) > 1 and all(dataclasses.is_dataclass(m) for m in members):
                tagged.update(m for m in members if isinstance(m, type))
    return tagged


def _members(t: object) -> tuple[object, ...]:
    """The members of the union t, or () if t is not a union."""
    if isinstance(t, typing.TypeAliasType):
        return _members(t.__value__)
    if typing.get_origin(t) in (types.UnionType, typing.Union):
        return typing.get_args(t)
    return ()


def _value(x: object, tagged: set[type], indent: str = "") -> str:
    """x as a TypeScript literal."""
    inner = indent + "  "
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, int | float):
        if not math.isfinite(x):
            raise ValueError(f"{x} has no TypeScript literal")
        return repr(x)
    if isinstance(x, str):
        return json.dumps(x, ensure_ascii=False)
    if isinstance(x, tuple):
        items = [_value(v, tagged, inner) for v in x]
        if all("\n" not in i for i in items) and sum(map(len, items)) < 60:
            return f"[{', '.join(items)}]"
        return "[\n" + "".join(f"{inner}{i},\n" for i in items) + f"{indent}]"
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        fields = [
            (f.name, _value(getattr(x, f.name), tagged, inner))
            for f in dataclasses.fields(x)
        ]
        if type(x) in tagged:
            fields.insert(0, ("kind", json.dumps(type(x).__name__)))
        body = "".join(f"{inner}{name}: {value},\n" for name, value in fields)
        return "{\n" + body + f"{indent}}}"
    raise TypeError(f"no TypeScript literal for {x!r}")


def _declaration(t: _Named, tagged: set[type]) -> str:
    if isinstance(t, typing.TypeAliasType):
        params = "".join(f"<{p.__name__}>" for p in t.__type_params__)
        where = t.__module__
        assert where is not None, f"a type statement sets {t}'s module"
        alias = f"export type {t.__name__}{params} = {_type(t.__value__)};"
        return "\n".join([*doc_comment(sys.modules[where], t.__name__), alias]) + "\n"
    params = ", ".join(p.__name__ for p in getattr(t, "__type_params__", ()))
    name = f"{t.__name__}<{params}>" if params else t.__name__
    lines = [*_tsdoc(_docstring(t)), f"export interface {name} {{"]
    if t in tagged:
        lines.append(f"  readonly kind: {json.dumps(t.__name__)};")
    (cls,) = ast.parse(textwrap.dedent(inspect.getsource(t))).body
    assert isinstance(cls, ast.ClassDef)
    docs = _attribute_docs(cls.body)
    for field, hint in typing.get_type_hints(t).items():
        lines += [f"  {line}" for line in _tsdoc(docs.get(field, ""))]
        lines.append(f"  readonly {field}: {_type(hint)};")
    lines.append("}")
    return "\n".join(lines) + "\n"


def doc_comment(module: types.ModuleType, name: str) -> list[str]:
    """The TSDoc comment of the docstring of module's constant or type alias
    name: one line, or none if it has no docstring."""
    return _tsdoc(_module_docs(module).get(name, ""))


@functools.cache
def _module_docs(module: types.ModuleType) -> Mapping[str, str]:
    return _attribute_docs(ast.parse(inspect.getsource(module)).body)


def _attribute_docs(body: list[ast.stmt]) -> Mapping[str, str]:
    """The docstring of each field, constant or type alias in body: the
    string just after it."""
    docs = {}
    for node, after in itertools.pairwise(body):
        match node, after:
            case (
                ast.AnnAssign(target=ast.Name(id=name))
                | ast.Assign(targets=[ast.Name(id=name)])
                | ast.TypeAlias(name=ast.Name(id=name)),
                ast.Expr(value=ast.Constant(value=str(doc))),
            ):
                docs[name] = doc
            case _:
                pass
    return docs


def _type(t: object) -> str:
    """t as a TypeScript type."""
    if isinstance(t, typing.TypeVar):
        return t.__name__
    if isinstance(t, typing.TypeAliasType):
        return t.__name__
    if t is float or t is int:
        return "number"
    if t is str:
        return "string"
    if t is bool:
        return "boolean"
    origin, args = typing.get_origin(t), typing.get_args(t)
    if origin is typing.Literal:
        return " | ".join(json.dumps(a) for a in args)
    if origin in (types.UnionType, typing.Union):
        return " | ".join(_type(a) for a in args)
    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            inner = _type(args[0])
            return f"readonly ({inner})[]" if " | " in inner else f"readonly {inner}[]"
        return f"readonly [{', '.join(_type(a) for a in args)}]"
    if dataclasses.is_dataclass(origin) and isinstance(origin, type):
        return f"{origin.__name__}<{', '.join(_type(a) for a in args)}>"
    if dataclasses.is_dataclass(t) and isinstance(t, type):
        return t.__name__
    raise TypeError(f"no TypeScript type for {t!r}")


def _docstring(t: type) -> str:
    """t's docstring, or "" if it has none."""
    doc = t.__doc__ or ""
    # A dataclass with no docstring gets one from its signature.
    return "" if doc.startswith(f"{t.__name__}(") else doc


def _tsdoc(doc: str) -> list[str]:
    """The first paragraph of doc as a TSDoc comment: one line, or none if
    doc is empty."""
    text = " ".join(doc.strip().split("\n\n")[0].split())
    return [f"/** {text} */"] if text else []
