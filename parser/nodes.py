"""Definitions for AST nodes."""

import itertools
from dataclasses import dataclass, field, fields
from typing import Optional, Any, Union, List, Tuple

from ops import UnaryOp, BinaryOp

# What this module offers: the node classes, in the order they are defined, and stamp_file.
__all__ = [
    'Node', 'Constant', 'BoolLiteral', 'NoneLiteral', 'StringLiteral', 'ByteLiteral', 'Variable', 'ArrayLiteral',
    'DictLiteral', 'Index', 'SliceLiteral', 'Slice', 'Call', 'Unary', 'Cast', 'Binary', 'Return', 'ArrayTypeExpr',
    'SliceTypeExpr', 'PointerTypeExpr', 'DictTypeExpr', 'QualifiedTypeExpr', 'VarDecl', 'Assign', 'Field',
    'StructField', 'MethodDef', 'StructDef', 'ExprStmt', 'IsCheck', 'If', 'Match', 'While', 'For', 'ForIn', 'Defer',
    'Break', 'Continue', 'Param', 'Function', 'ExternFunctionDecl', 'IntrinsicDecl', 'ConstDecl', 'EnumMember',
    'EnumDef',
    'TypeAlias', 'SumTypeDef', 'ImportDecl', 'FromImportDecl', 'Program', 'stamp_file',
]

_PRETTY_MAX_WIDTH = 88
_HIDDEN_FIELDS = {'line', 'col', 'file', 'nid', 'could_be_multiplication'}
_PRETTY_INDENT = "    "


def _pretty_scalar(value: Any) -> str:
    """Render a non-Node field; operators render as their symbol."""
    if isinstance(value, (UnaryOp, BinaryOp)):
        value = value.symbol()
    return repr(value)


def _pretty_value(value: Any, indent: int) -> str:
    """Render a Node, list, or scalar at depth `indent`; first line unindented."""
    if isinstance(value, Node):
        return _pretty_node(value, indent)
    if isinstance(value, list):
        return _pretty_list(value, indent)
    return _pretty_scalar(value)


def _pretty_list(items: list, indent: int) -> str:
    """Render a list on one line if it fits, else one item per line."""
    if not items:
        return "[]"
    rendered = [_pretty_value(item, indent + 1) for item in items]
    if not any('\n' in r for r in rendered):
        candidate = f"[{', '.join(rendered)}]"
        if indent * len(_PRETTY_INDENT) + len(candidate) <= _PRETTY_MAX_WIDTH:
            return candidate
    inner = ",\n".join(f"{_PRETTY_INDENT * (indent + 1)}{r}" for r in rendered)
    return f"[\n{inner},\n{_PRETTY_INDENT * indent}]"


def _pretty_node(node: 'Node', indent: int) -> str:
    """Render `ClassName(field=value, ...)` via dataclasses.fields."""
    class_name = type(node).__name__
    field_names = [f.name for f in fields(node) if f.name not in _HIDDEN_FIELDS]
    if not field_names:
        return f"{class_name}()"

    rendered = {name: _pretty_value(getattr(node, name), indent + 1) for name in field_names}
    if not any('\n' in r for r in rendered.values()):
        candidate = f"{class_name}({', '.join(f'{n}={rendered[n]}' for n in field_names)})"
        if indent * len(_PRETTY_INDENT) + len(candidate) <= _PRETTY_MAX_WIDTH:
            return candidate

    inner = ",\n".join(f"{_PRETTY_INDENT * (indent + 1)}{n}={rendered[n]}" for n in field_names)
    return f"{class_name}(\n{inner},\n{_PRETTY_INDENT * indent})"


def stamp_file(node, file: str) -> None:
    """Set `file` on `node` and every descendant that lacks one."""
    stack = [node]
    while stack:
        n = stack.pop()
        if isinstance(n, Node):
            if n.file is None:
                n.file = file
            stack.extend(getattr(n, f.name) for f in fields(n))
        elif isinstance(n, (list, tuple)):
            stack.extend(n)


_node_numbers = itertools.count()


@dataclass
class Node:
    """AST base. pretty() renders any node generically. Every node gets a number (`nid`) when it is
    created: passes key per-node facts on it, never on object identity."""
    line: int = field(default=0, kw_only=True, compare=False, repr=False)
    col: int = field(default=0, kw_only=True, compare=False, repr=False)
    file: Optional[str] = field(default=None, kw_only=True, compare=False, repr=False)
    nid: int = field(default=-1, kw_only=True, compare=False, repr=False)

    def __post_init__(self):
        self.nid = next(_node_numbers)

    def pretty(self) -> str:
        return _pretty_node(self, indent=0)


@dataclass
class Constant(Node):
    """A number literal: an integer (one written with a fraction, `1.5`, is a parse error)."""
    value: int


@dataclass
class BoolLiteral(Node):
    """`true` / `false`."""
    value: bool


@dataclass
class NoneLiteral(Node):
    """`none`: a pointer to nothing, or a sum type's `none` variant. Typed Type.NONE."""


@dataclass
class StringLiteral(Node):
    """`'...'`; `value` is unescaped. Strings are bytes, not Unicode."""
    value: str


@dataclass
class ByteLiteral(Node):
    """`"a"`: a single uint8."""
    value: int


@dataclass
class Variable(Node):
    """Variable reference."""
    name: str


@dataclass
class ArrayLiteral(Node):
    """`[e1, ...]` or typed `[N]T[...]` / `[]T[...]`. `could_be_multiplication` marks a typed one
    whose tokens could also be an indexed literal times an index (`[x][0] * ys[1]` beside
    `[2][1]*P[...]`): it is read as the literal, and an error in its type says so."""
    elements: List[Node] = field(default_factory=list)
    type_expr: Optional['ArrayTypeExpr'] = None
    could_be_multiplication: bool = field(default=False, compare=False, repr=False)


@dataclass
class DictLiteral(Node):
    """`dict[K]V{k: v, ...}`."""
    key_type: Union[str, 'ArrayTypeExpr', 'SliceTypeExpr', 'PointerTypeExpr', 'DictTypeExpr']
    value_type: Union[str, 'ArrayTypeExpr', 'SliceTypeExpr', 'PointerTypeExpr', 'DictTypeExpr']
    entries: List[Tuple[Node, Node]] = field(default_factory=list)


@dataclass
class Index(Node):
    """`array[index]`; multi-dimensional indexing nests."""
    array: Node
    index: Node


@dataclass
class SliceLiteral(Node):
    """`[]T[e1, ...]`: a new slice holding the elements."""
    element_type: Any
    elements: List[Node]


@dataclass
class Slice(Node):
    """`array[low:high]`: a view over [low, high); either bound optional."""
    array: Node
    low: Optional[Node] = None
    high: Optional[Node] = None


@dataclass
class Call(Node):
    """`name(args)` or `name(f=v, ...)`; with a `receiver`, a method call or a module-qualified one.
    Struct literals share this shape."""
    name: str
    args: List[Node] = field(default_factory=list)
    kwargs: Optional[List[Tuple[str, Node]]] = None
    receiver: Optional[Node] = None


@dataclass
class Unary(Node):
    op: UnaryOp
    operand: Node


@dataclass
class Cast(Node):
    """`T(expr)`: explicit scalar cast."""
    target_type: str
    expr: Node


@dataclass
class Binary(Node):
    op: BinaryOp
    left: Node
    right: Node


@dataclass
class Return(Node):
    """`return [expr]`."""
    value: Optional[Node] = None


@dataclass
class ArrayTypeExpr(Node):
    """`[N]T`; nests row-major. `size` is an int, or a constant expression that semantic analysis
    evaluates."""
    size: Union[int, 'Node']
    element_type: Union[str, 'ArrayTypeExpr', 'SliceTypeExpr']


@dataclass
class SliceTypeExpr(Node):
    """`[]T`."""
    element_type: Union[str, ArrayTypeExpr, 'SliceTypeExpr']


@dataclass
class PointerTypeExpr(Node):
    """`*T`."""
    pointee_type: Union[str, ArrayTypeExpr, SliceTypeExpr, 'PointerTypeExpr']


@dataclass
class DictTypeExpr(Node):
    """`dict[K]V`."""
    key_type: Union[str, ArrayTypeExpr, SliceTypeExpr, PointerTypeExpr, 'DictTypeExpr']
    value_type: Union[str, ArrayTypeExpr, SliceTypeExpr, PointerTypeExpr, 'DictTypeExpr']


@dataclass
class QualifiedTypeExpr(Node):
    """`module.Name`; semantic analysis resolves it to the declaration's key (scopes.py)."""
    module: str
    name: str


@dataclass
class VarDecl(Node):
    """`T name [= init]`."""
    name: str
    var_type: Union[str, ArrayTypeExpr, SliceTypeExpr]
    init: Optional[Node] = None


@dataclass
class Assign(Node):
    """`target = value`, or `target op= value` (`op` the operator); the target is a name, field,
    index, or dereference, as written."""
    target: Node
    value: Node
    op: Optional[BinaryOp] = None


@dataclass
class Field(Node):
    """`base.name`; pointers auto-deref."""
    base: Node
    name: str


@dataclass
class StructField(Node):
    """Struct field `T name`; zero-initialized."""
    name: str
    field_type: Union[str, 'ArrayTypeExpr', 'SliceTypeExpr']


@dataclass
class MethodDef(Node):
    """Struct method; first param is the untyped receiver (`*name` for a pointer receiver)."""
    receiver_name: str
    name: str
    return_type: Optional[Union[str, ArrayTypeExpr, SliceTypeExpr]]
    params: List['Param'] = field(default_factory=list)
    body: List[Node] = field(default_factory=list)
    receiver_is_pointer: bool = False


@dataclass
class StructDef(Node):
    """`type Name struct:`; field order fixes layout."""
    name: str
    fields: List[StructField] = field(default_factory=list)
    methods: List[MethodDef] = field(default_factory=list)


@dataclass
class ExprStmt(Node):
    """Expression statement."""
    expr: Node


@dataclass
class IsCheck(Node):
    """`SUBJECT is T`: whether a sum holds variant T, or an enum is member T. Three shapes: `NAME is T`
    (variable_name; the one that can narrow NAME), `EXPR is T` (subject), and `EXPR is T as NAME`
    (both: NAME is declared as a copy of EXPR), which is for an `if` condition or a `match`."""
    variable_name: Optional[str]
    type_name: Union[str, QualifiedTypeExpr, ArrayTypeExpr, SliceTypeExpr, PointerTypeExpr]
    subject: Optional[Node] = None

    @property
    def binds(self) -> bool:
        """Whether this is the `EXPR is T as NAME` shape."""
        return self.variable_name is not None and self.subject is not None


@dataclass
class If(Node):
    """`if`; `elif` nests in else_body."""
    condition: Node
    then_body: List[Node]
    else_body: Optional[List[Node]] = None


@dataclass
class Match(Node):
    """`match NAME:` / `match EXPR as NAME:`. Each arm is (`NAME is T` check, body), tried in order;
    the first arm's check carries the subject expression, if any. Positioned at the first arm."""
    variable_name: str
    subject: Optional[Node]
    arms: List[Tuple[Node, List[Node]]]
    else_body: Optional[List[Node]] = None


@dataclass
class While(Node):
    """`while cond:`."""
    condition: Node
    body: List[Node]


@dataclass
class For(Node):
    """`for init; cond; increment:`."""
    init: Node
    condition: Node
    increment: Node
    body: List[Node]


@dataclass
class ForIn(Node):
    """`for a[, b] in iterable:`."""
    binding_names: List[str]
    iterable: Node
    body: List[Node]


@dataclass
class Defer(Node):
    """`defer CALL`: the call's receiver and arguments are evaluated here; the call is made when the
    block this statement is in is left."""
    call: Node


@dataclass
class Break(Node):
    """Exits the innermost loop."""


@dataclass
class Continue(Node):
    """Continues the innermost loop."""


@dataclass
class Param(Node):
    """Parameter `T name`."""
    name: str
    type: Union[str, ArrayTypeExpr, SliceTypeExpr]


@dataclass
class Function(Node):
    """`def [T] name(params):`; return_type None means no value, 'never' that it doesn't return."""
    name: str
    return_type: Optional[Union[str, ArrayTypeExpr, SliceTypeExpr]]
    params: List[Param] = field(default_factory=list)
    body: List[Node] = field(default_factory=list)


@dataclass
class ExternFunctionDecl(Node):
    """`extern [T] name(params)`: a linker symbol, never mangled."""
    name: str
    return_type: Optional[Union[str, ArrayTypeExpr, SliceTypeExpr]]
    params: List[Param] = field(default_factory=list)


@dataclass
class IntrinsicDecl(Node):
    """`intrinsic T name(params)`: compiler-provided operation. original_name survives mangling."""
    name: str
    original_name: str
    return_type: Optional[Union[str, ArrayTypeExpr, SliceTypeExpr, PointerTypeExpr]]
    params: List[Param] = field(default_factory=list)


@dataclass
class ConstDecl(Node):
    """Top-level `const T NAME = expr`; expr must be a compile-time constant."""
    name: str
    const_type: Union[str, ArrayTypeExpr, SliceTypeExpr]
    value: Node


@dataclass
class EnumMember(Node):
    """One member's name in an enum declaration."""
    name: str


@dataclass
class EnumDef(Node):
    """`type Name enum:` and its members, one per line; a member's value is its position. Methods,
    as a struct has, follow the members."""
    name: str
    members: List[EnumMember]
    methods: List[MethodDef] = field(default_factory=list)


@dataclass
class TypeAlias(Node):
    """`type Name = T`: an interchangeable alias."""
    name: str
    target_type: Union[str, ArrayTypeExpr, SliceTypeExpr]


@dataclass
class SumTypeDef(Node):
    """`type Name is A | B`: tagged sum type. Variant order fixes discriminants."""
    name: str
    variants: List[
        Union[str, QualifiedTypeExpr, ArrayTypeExpr, SliceTypeExpr, PointerTypeExpr]] = field(default_factory=list)


@dataclass
class ImportDecl(Node):
    """`import 'path' [as name]`."""
    path: str
    qualifier: str


@dataclass
class FromImportDecl(Node):
    """`from 'path' import a [as b], ...`."""
    path: str
    names: List[Tuple[str, str]] = field(default_factory=list)


@dataclass
class Program(Node):
    functions: List[Function] = field(default_factory=list)
    structs: List[StructDef] = field(default_factory=list)
    type_aliases: List[TypeAlias] = field(default_factory=list)
    sum_types: List[SumTypeDef] = field(default_factory=list)
    extern_functions: List[ExternFunctionDecl] = field(default_factory=list)
    imports: List[ImportDecl] = field(default_factory=list)
    from_imports: List[FromImportDecl] = field(default_factory=list)
    intrinsics: List[IntrinsicDecl] = field(default_factory=list)
    consts: List[ConstDecl] = field(default_factory=list)
    enums: List[EnumDef] = field(default_factory=list)

    def __repr__(self) -> str:
        return self.pretty()
