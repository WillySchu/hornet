"""The typed tree: what a program means, built from the parser's tree by semantic.analyze().

Every expression has a concrete `type`. Each node has one meaning: names refer to Symbols, the
parser's overloaded forms are split (a call, a struct literal, a builtin; array, slice, str, and
dict indexing), and every implicit operation is a node of its own (widening into a sum, boxing a
variant, a literal becoming a slice, zero values). Nodes are immutable. dump() renders a program
deterministically, one node per line.
"""

import itertools
from dataclasses import dataclass, field, fields
from typing import Any, Optional

from symbols import Symbol
from typesys import Type


_node_numbers = itertools.count()


@dataclass(frozen=True)
class _Node:
    """Every node gets a number when created: passes key per-node facts on it."""
    nid: int = field(default=-1, kw_only=True, compare=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, 'nid', next(_node_numbers))


@dataclass(frozen=True)
class Expr(_Node):
    type: Type


@dataclass(frozen=True)
class Stmt(_Node):
    pass


# -- expressions: values

@dataclass(frozen=True)
class IntLit(Expr):
    value: int


@dataclass(frozen=True)
class BoolLit(Expr):
    value: bool


@dataclass(frozen=True)
class StrLit(Expr):
    value: str


@dataclass(frozen=True)
class NoneLit(Expr):
    """A null pointer (type is a pointer), or a sum's `none` variant before widening (type none)."""


@dataclass(frozen=True)
class Local(Expr):
    """A local variable or parameter."""
    symbol: Symbol


@dataclass(frozen=True)
class ZeroValue(Expr):
    """The type's zero value (for a sum, its `none` variant; a dict's is NewEmptyDict)."""


@dataclass(frozen=True)
class NewEmptyDict(Expr):
    pass


@dataclass(frozen=True)
class EmptySlice(Expr):
    pass


@dataclass(frozen=True)
class ArrayLiteral(Expr):
    elements: tuple


@dataclass(frozen=True)
class SliceLiteral(Expr):
    """A slice over new storage holding the elements."""
    elements: tuple


@dataclass(frozen=True)
class DictLiteral(Expr):
    entries: tuple  # of (key, value)


@dataclass(frozen=True)
class StructLiteral(Expr):
    fields: tuple  # every field, in declaration order


# -- expressions: conversions

@dataclass(frozen=True)
class WidenToSum(Expr):
    """A variant (or `none`) as a value of the sum type."""
    value: Expr


@dataclass(frozen=True)
class BoxVariant(Expr):
    """`&Variant(...)` where a pointer to the sum is expected: new heap storage holding the variant."""
    value: Expr


@dataclass(frozen=True)
class Payload(Expr):
    """The variant held by a sum known to hold it (type is the variant)."""
    sum: Expr


@dataclass(frozen=True)
class TagTest(Expr):
    """Whether a sum holds `variant`."""
    sum: Expr
    variant: Type


@dataclass(frozen=True)
class IntCast(Expr):
    value: Expr


@dataclass(frozen=True)
class StrFromByte(Expr):
    value: Expr


@dataclass(frozen=True)
class StrFromBytes(Expr):
    value: Expr


@dataclass(frozen=True)
class BytesFromStr(Expr):
    value: Expr


# -- expressions: access

@dataclass(frozen=True)
class FieldAccess(Expr):
    base: Expr
    name: str
    index: int
    through_pointer: bool


@dataclass(frozen=True)
class ArrayIndex(Expr):
    base: Expr
    index: Expr


@dataclass(frozen=True)
class SliceIndex(Expr):
    base: Expr
    index: Expr


@dataclass(frozen=True)
class StrIndex(Expr):
    base: Expr
    index: Expr


@dataclass(frozen=True)
class DictLookup(Expr):
    """`d[k]`: panics if the key is missing. As an assignment target, inserts or overwrites."""
    dict: Expr
    key: Expr


@dataclass(frozen=True)
class SliceOf(Expr):
    """`x[low:high]` of an array, slice, or str (`kind`); a missing bound is None."""
    kind: str
    base: Expr
    low: Optional[Expr]
    high: Optional[Expr]


@dataclass(frozen=True)
class AddressOf(Expr):
    place: Expr


@dataclass(frozen=True)
class Deref(Expr):
    pointer: Expr


# -- expressions: operators and calls

@dataclass(frozen=True)
class Unary(Expr):
    op: Any
    operand: Expr


@dataclass(frozen=True)
class Binary(Expr):
    """Integer and bool operators, comparisons, and short-circuit `and`/`or`."""
    op: Any
    left: Expr
    right: Expr


@dataclass(frozen=True)
class StrConcat(Expr):
    left: Expr
    right: Expr


@dataclass(frozen=True)
class StrCompare(Expr):
    op: Any
    left: Expr
    right: Expr


@dataclass(frozen=True)
class DictContains(Expr):
    key: Expr
    dict: Expr


@dataclass(frozen=True)
class ElementContains(Expr):
    """`x in a` for an array or slice."""
    value: Expr
    container: Expr


@dataclass(frozen=True)
class Call(Expr):
    """A call of a Hornet function (`kind` 'function'), extern, or intrinsic."""
    name: str
    kind: str
    args: tuple


@dataclass(frozen=True)
class Len(Expr):
    value: Expr


@dataclass(frozen=True)
class Append(Expr):
    slice: Expr
    value: Expr


@dataclass(frozen=True)
class Print(Expr):
    value: Expr


@dataclass(frozen=True)
class DictDelete(Expr):
    dict: Expr
    key: Expr


# -- statements

@dataclass(frozen=True)
class Declare(Stmt):
    symbol: Symbol
    init: Expr  # always present: a zero value is explicit


@dataclass(frozen=True)
class Assign(Stmt):
    """`target` is a Local, FieldAccess, ArrayIndex, SliceIndex, DictLookup (insert), or Deref."""
    target: Expr
    value: Expr


@dataclass(frozen=True)
class CompoundAssign(Stmt):
    """`target op= value`, the target's address evaluated once."""
    target: Expr
    op: Any
    value: Expr


@dataclass(frozen=True)
class ExprStmt(Stmt):
    expr: Expr


@dataclass(frozen=True)
class Return(Stmt):
    value: Optional[Expr]


@dataclass(frozen=True)
class If(Stmt):
    cond: Expr
    then_body: tuple
    else_body: tuple


@dataclass(frozen=True)
class Match(Stmt):
    """Branch on which variant `subject` (a sum) holds; arms are (variant type, body)."""
    subject: Expr
    arms: tuple
    else_body: Optional[tuple]


@dataclass(frozen=True)
class While(Stmt):
    cond: Expr
    body: tuple


@dataclass(frozen=True)
class For(Stmt):
    """`for init; cond; step:`; a variable declared in `init` is fresh on each iteration."""
    init: Optional[Stmt]
    cond: Expr
    step: Stmt
    body: tuple


@dataclass(frozen=True)
class ForIn(Stmt):
    """`for ... in` over an array, slice, str, or dict (`kind`); bindings are fresh each iteration."""
    kind: str
    iterable: Expr
    bindings: tuple  # of Symbol
    body: tuple


@dataclass(frozen=True)
class Break(Stmt):
    pass


@dataclass(frozen=True)
class Continue(Stmt):
    pass


@dataclass(frozen=True)
class Function:
    name: str
    params: tuple  # of Symbol
    return_type: Type
    body: tuple


@dataclass(frozen=True)
class Program:
    functions: tuple
    structs: Any  # name -> StructInfo
    sum_types: Any  # name -> SumTypeInfo
    symbols: Any = field(compare=False)  # SymbolTable


# -- dump

def dump(program: Program) -> str:
    """Deterministic text: one function per block, one node per line, children indented."""
    lines = []
    for fn in program.functions:
        params = ', '.join(f"{s}: {s.type}" for s in fn.params)
        lines.append(f"function {fn.name}({params}) -> {fn.return_type}")
        for stmt in fn.body:
            _dump_node(stmt, 1, lines)
        lines.append("")
    return "\n".join(lines)


def _label(node) -> str:
    attrs = []
    for f in fields(node):
        value = getattr(node, f.name)
        if f.name in ('type', 'nid') or value is None or isinstance(value, (Expr, Stmt, tuple)):
            continue
        if isinstance(value, (bool, int, str)) and f.name == 'value':
            text = repr(value)
        elif hasattr(value, 'name') and not isinstance(value, (Symbol, Type)):
            text = value.name  # an operator enum
        else:
            text = str(value)
        attrs.append(f"{f.name}={text}")
    label = type(node).__name__ + ''.join(' ' + a for a in attrs)
    return label + (f" : {node.type}" if isinstance(node, Expr) else '')


def _dump_node(node, depth: int, lines: list) -> None:
    lines.append("  " * depth + _label(node))
    for f in fields(node):
        value = getattr(node, f.name)
        if isinstance(value, (Expr, Stmt)):
            lines.append("  " * (depth + 1) + f"{f.name}:")
            _dump_node(value, depth + 2, lines)
        elif isinstance(value, tuple) and value and f.name != 'bindings':
            lines.append("  " * (depth + 1) + f"{f.name}:")
            for item in value:
                if isinstance(item, tuple):  # (key, value) entries or (variant, body) arms
                    head, rest = item
                    if isinstance(head, Type):
                        lines.append("  " * (depth + 2) + f"is {head}:")
                        for s in rest:
                            _dump_node(s, depth + 3, lines)
                    else:
                        _dump_node(head, depth + 2, lines)
                        _dump_node(rest, depth + 3, lines)
                elif isinstance(item, (Expr, Stmt)):
                    _dump_node(item, depth + 2, lines)
        elif f.name == 'bindings':
            lines.append("  " * (depth + 1) + "bindings: " + ', '.join(f"{s}: {s.type}" for s in value))
