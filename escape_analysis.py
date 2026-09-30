"""Escape analysis: which locals must live on the heap.

Flow-insensitive, field-insensitive points-to analysis over one
function. Each storage location t (local, param, for-in binding, `&S()`
literal, heap literal, or EXT = memory not owned by this frame) has
H[t]: the locations whose addresses may be stored inside t. An
expression's value is the set of locations it may point into.

Escaping: returned values, anything stored into EXT, and, transitively,
anything stored inside an escaping location. Escaping locations may in
turn hold EXT (callee writes).

Parameter i's value points into PARAM(i): memory the caller owns, whose
contents are external and into which stores escape (like EXT). The
function's summary says whether each PARAM(i) escapes. At a call with a
summary, only arguments whose parameter escapes do; for the others, the
callee may still leak or overwrite what their memory holds, so their
contents escape and their memory may come to hold EXT. Calls without a
summary (externs, intrinsics) escape every argument.

Storage declared inside a loop (body declarations, the loop variable, for-in bindings,
`&S()` literals) is fresh on each iteration, so it also needs the heap when a location
declared outside that loop may come to hold its address: the address would outlive the
iteration that created it."""

import dataclasses

from typing import Optional, Union

from parser import (
    ArrayLiteral,
    Assign,
    Binary,
    BoolLiteral,
    Break,
    ByteLiteral,
    Call,
    Cast,
    Constant,
    Continue,
    DerefAssign,
    DictLiteral,
    ExprStmt,
    Field,
    FieldAssign,
    For,
    ForIn,
    Function,
    If,
    Index,
    IndexAssign,
    IsCheck,
    Node,
    NoneLiteral,
    Return,
    Slice,
    StringLiteral,
    Unary,
    VarDecl,
    Variable,
    While,
)
from ops import UnaryOp
from typesys import StructInfo, Type, TypeKind
from typesys import type_byte_width

# Composites larger than this are heap-allocated regardless of escape.
_STACK_ARRAY_LIMIT_BYTES = 16384


def is_heap_allocated(t: Type, structs: dict[str, StructInfo], sum_types: dict) -> bool:
    """Size-based promotion only; see IRFunctionBuilder._is_heap_allocated."""
    return t.kind in (TypeKind.ARRAY, TypeKind.STRUCT, TypeKind.SUM) and type_byte_width(t, structs, sum_types) > _STACK_ARRAY_LIMIT_BYTES


# id() of a VarDecl/Param/`&S()` Call, or (id(ForIn), binding index).
DeclId = Union[int, tuple]


EXT = 'EXT'


def _param(i: int) -> tuple:
    return ('param', i)


def _is_external(t) -> bool:
    return t == EXT or (isinstance(t, tuple) and t[0] == 'param')


def _heap(node: Node) -> tuple:
    return ('heap', id(node))


class EscapeAnalyzer:
    def __init__(self, fn: Function, structs: dict[str, StructInfo], summaries: Optional[dict] = None):
        self.fn = fn
        self.structs = structs
        self.summaries = summaries or {}
        self.H: dict = {EXT: {EXT}}
        self.esc: set = {EXT}
        self.changed = False

    def analyze(self) -> set[DeclId]:
        while True:
            self.changed = False
            for i, p in enumerate(self.fn.params):
                self._add(_param(i), {EXT})
                self._add(id(p), {_param(i)})
            self.walk_statements(self.fn.body)
            self._close_escapes()
            if not self.changed:
                break
        heap = self.esc | self._iteration_escapes()
        return {t for t in heap
                if t != EXT and not (isinstance(t, tuple) and isinstance(t[0], str))}

    def _iteration_escapes(self) -> set:
        """Locations declared inside some loop whose address may be held outside it."""
        out = set()
        for inner in _loop_scopes(self.fn.body):
            reached = [u for t, held in self.H.items() if t not in inner for u in held if u in inner]
            while reached:
                u = reached.pop()
                if u not in out:
                    out.add(u)
                    reached.extend(v for v in self.H.get(u, ()) if v in inner)
        return out

    def param_escapes(self) -> list[bool]:
        """After analyze(): whether each parameter's pointed-into memory escapes."""
        return [_param(i) in self.esc for i in range(len(self.fn.params))]

    # -- lattice helpers

    def _add(self, t, vals: set) -> None:
        cur = self.H.setdefault(t, set())
        if not vals <= cur:
            cur |= vals
            self.changed = True

    def _escape(self, vals: set) -> None:
        if not vals <= self.esc:
            self.esc |= vals
            self.changed = True

    def _store(self, targets: set, vals: set) -> None:
        for t in targets:
            if _is_external(t):
                self._escape(vals)
            else:
                self._add(t, vals)

    def _contents(self, targets: set) -> set:
        out = set()
        for t in targets:
            out |= self.H.get(t, set())
        return out

    def _close_escapes(self) -> None:
        stack = list(self.esc)
        while stack:
            t = stack.pop()
            if t != EXT:
                self._add(t, {EXT})
            for u in self.H.get(t, ()):
                if u not in self.esc:
                    self.esc.add(u)
                    self.changed = True
                    stack.append(u)

    # -- expressions

    @staticmethod
    def _kind(expr: Node) -> Optional[TypeKind]:
        t = getattr(expr, 'resolved_type', None)
        return t.kind if t is not None else None

    def loc(self, expr: Node) -> set:
        """Locations `expr`'s storage may reside in."""
        if isinstance(expr, Variable):
            return {expr.decl_id} if expr.decl_id is not None else {EXT}
        if isinstance(expr, Field):
            if self._kind(expr.base) == TypeKind.POINTER:
                return self.vals(expr.base)
            return self.loc(expr.base)
        if isinstance(expr, Index):
            self.vals(expr.index)
            if self._kind(expr.array) == TypeKind.ARRAY:
                return self.loc(expr.array)
            return self.vals(expr.array)
        if isinstance(expr, Unary) and expr.op == UnaryOp.DEREFERENCE:
            return self.vals(expr.operand)
        if isinstance(expr, Call) and expr.name in self.structs:
            t = id(expr)
            self._add(t, self.vals(expr))
            return {t}
        # Rvalue: materialized in fresh, non-promotable storage.
        t = _heap(expr)
        self._add(t, self.vals(expr))
        return {t}

    def vals(self, expr: Node) -> set:
        """Locations `expr`'s value may point into. Evaluates every
        subexpression, so escaping call arguments are always recorded."""
        if expr is None:
            return set()
        if isinstance(expr, Variable):
            return self._contents(self.loc(expr))
        if isinstance(expr, Field):
            if self._kind(expr.base) == TypeKind.POINTER:
                return self._contents(self.vals(expr.base))
            return self.vals(expr.base)
        if isinstance(expr, Index):
            self.vals(expr.index)
            k = self._kind(expr.array)
            if k == TypeKind.ARRAY:
                return self.vals(expr.array)
            if k == TypeKind.STR:
                self.vals(expr.array)
                return set()
            return self._contents(self.vals(expr.array))
        if isinstance(expr, Slice):
            self.vals(expr.low)
            self.vals(expr.high)
            if self._kind(expr.array) == TypeKind.ARRAY:
                return self.loc(expr.array)
            return self.vals(expr.array)
        if isinstance(expr, Unary):
            if expr.op == UnaryOp.ADDRESS_OF:
                return self.loc(expr.operand)
            if expr.op == UnaryOp.DEREFERENCE:
                return self._contents(self.vals(expr.operand))
            self.vals(expr.operand)
            return set()
        if isinstance(expr, Binary):
            self.vals(expr.left)
            self.vals(expr.right)
            return set()
        if isinstance(expr, Cast):
            return self.vals(expr.expr)
        if isinstance(expr, ArrayLiteral):
            t = _heap(expr)
            elems = set()
            for e in expr.elements:
                elems |= self.vals(e)
            self._add(t, elems)
            return {t} | elems
        if isinstance(expr, DictLiteral):
            t = _heap(expr)
            for k, v in expr.entries:
                self._add(t, self.vals(k) | self.vals(v))
            return {t}
        if isinstance(expr, Call):
            return self._call(expr)
        if isinstance(expr, IsCheck):
            self.vals(expr.subject)
            return set()
        if isinstance(expr, (Constant, BoolLiteral, NoneLiteral, StringLiteral, ByteLiteral)):
            return set()
        raise TypeError(f"escape analysis: unhandled expression {type(expr).__name__}")

    def _call(self, expr: Call) -> set:
        args = list(expr.args) + [v for _, v in (expr.kwargs or [])]
        arg_vals = [self.vals(a) for a in args]
        if expr.name in self.structs:
            return set().union(*arg_vals)
        if expr.name in ('print', 'len', 'del'):
            return set()
        if expr.name == 'append':
            t = _heap(expr)
            old = arg_vals[0]
            self._add(t, self._contents(old))
            elems = set().union(*arg_vals[1:])
            self._store(old | {t}, elems)
            return old | {t}
        summary = self.summaries.get(expr.name)
        if summary is None or expr.kwargs or len(summary) != len(arg_vals):
            for v in arg_vals:
                self._escape(v)
            return {EXT}
        result = {EXT}
        for escapes, v in zip(summary, arg_vals):
            if escapes:
                self._escape(v)
                result |= v
            else:
                self._escape(self._contents(v))
                self._store(v, {EXT})
        return result

    # -- statements

    def walk_statements(self, statements: list[Node]) -> None:
        for stmt in statements:
            self.walk_statement(stmt)

    def _block(self, body) -> None:
        if body is not None:
            self.walk_statements(body)

    def walk_statement(self, stmt: Node) -> None:
        if isinstance(stmt, VarDecl):
            self._add(id(stmt), self.vals(stmt.init) if stmt.init is not None else set())
        elif isinstance(stmt, Assign):
            self._store({stmt.decl_id} if stmt.decl_id is not None else {EXT}, self.vals(stmt.value))
        elif isinstance(stmt, IndexAssign):
            self._store(self.loc(Index(array=stmt.array, index=stmt.index)), self.vals(stmt.value))
        elif isinstance(stmt, FieldAssign):
            self._store(self.loc(Field(base=stmt.base, name=stmt.name)), self.vals(stmt.value))
        elif isinstance(stmt, DerefAssign):
            self._store(self.vals(stmt.pointer), self.vals(stmt.value))
        elif isinstance(stmt, Return):
            self._escape(self.vals(stmt.value))
        elif isinstance(stmt, ExprStmt):
            self.vals(stmt.expr)
        elif isinstance(stmt, If):
            cond = stmt.condition
            if isinstance(cond, IsCheck) and cond.binding_decl is not None:
                self.walk_statement(cond.binding_decl)
            else:
                self.vals(cond)
            self._block(stmt.then_body)
            self._block(stmt.else_body)
        elif isinstance(stmt, While):
            self.vals(stmt.condition)
            self._block(stmt.body)
        elif isinstance(stmt, For):
            self.walk_statement(stmt.init)
            self.vals(stmt.condition)
            self._block(stmt.body)
            self.walk_statement(stmt.increment)
        elif isinstance(stmt, ForIn):
            k = self._kind(stmt.iterable)
            it = self.vals(stmt.iterable)
            elems = it if k == TypeKind.ARRAY else self._contents(it)
            for i in range(len(stmt.binding_names)):
                self._add((id(stmt), i), elems)
            self.walk_statements(stmt.body)
        elif isinstance(stmt, (Break, Continue)):
            pass
        else:
            self.vals(stmt)


def _children(node) -> list:
    out = []
    for f in dataclasses.fields(node):
        value = getattr(node, f.name)
        for v in value if isinstance(value, (list, tuple)) else [value]:
            if isinstance(v, Node):
                out.append(v)
            elif isinstance(v, tuple):
                out.extend(x for x in v if isinstance(x, Node))
    return out


def _declared_within(nodes: list) -> set:
    """Locations declared anywhere in `nodes` (DeclIds, including `&S()` literal locations)."""
    out = set()
    stack = list(nodes)
    while stack:
        node = stack.pop()
        if isinstance(node, VarDecl):
            out.add(id(node))
        elif isinstance(node, ForIn):
            out.update((id(node), i) for i in range(len(node.binding_names)))
        elif isinstance(node, Call):
            out.add(id(node))
        elif isinstance(node, IsCheck) and node.binding_decl is not None:
            stack.append(node.binding_decl)
        stack.extend(_children(node))
    return out


def _loop_scopes(statements: list) -> list:
    """For each loop in `statements`, the set of locations declared inside it (loop variable and
    for-in bindings included)."""
    scopes = []
    stack = list(statements)
    while stack:
        node = stack.pop()
        if isinstance(node, ForIn):
            # The iterable is evaluated once, before the first iteration.
            scopes.append(_declared_within(node.body) | {(id(node), i) for i in range(len(node.binding_names))})
        elif isinstance(node, (While, For)):
            scopes.append(_declared_within(_children(node)))
        stack.extend(_children(node))
    return scopes


def compute_escape_summaries(functions: list[Function], structs: dict[str, StructInfo]) -> dict:
    """Function name -> per-parameter escape flags, iterated to a fixed point (handles recursion)."""
    summaries = {fn.name: [False] * len(fn.params) for fn in functions}
    changed = True
    while changed:
        changed = False
        for fn in functions:
            analyzer = EscapeAnalyzer(fn, structs, summaries)
            analyzer.analyze()
            flags = analyzer.param_escapes()
            if flags != summaries[fn.name]:
                summaries[fn.name] = flags
                changed = True
    return summaries


def analyze_array_escapes(fn: Function, structs: dict[str, StructInfo], summaries: Optional[dict] = None) -> set[DeclId]:
    """DeclIds in `fn` whose storage must be heap-allocated. Requires semantic analysis.
    Without `summaries`, every call escapes its arguments."""
    return EscapeAnalyzer(fn, structs, summaries).analyze()
