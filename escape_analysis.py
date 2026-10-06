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

Runs on the typed tree (typed_ast.py). Storage declared inside a loop (body declarations, the
loop variable, for-in bindings, `&S()` literals) is fresh on each iteration, so it also needs the heap when a location
declared outside that loop may come to hold its address: the address would outlive the
iteration that created it."""

from dataclasses import fields
from typing import Optional, Union

import typed_ast as t
from typesys import StructInfo, Type, TypeKind
from typesys import type_byte_width

# Composites larger than this are heap-allocated regardless of escape.
_STACK_ARRAY_LIMIT_BYTES = 16384


def is_heap_allocated(type_: Type, structs: dict[str, StructInfo], sum_types: dict) -> bool:
    """Size-based promotion only; escape analysis adds the rest (TypedFunctionBuilder.heap)."""
    return type_.kind in (TypeKind.ARRAY, TypeKind.STRUCT, TypeKind.SUM) and \
        type_byte_width(type_, structs, sum_types) > _STACK_ARRAY_LIMIT_BYTES


# A declaration's Symbol.id, or ('literal', nid) for an `&S()` struct literal's storage.
DeclId = Union[int, tuple]


EXT = 'EXT'


def _param(i: int) -> tuple:
    return ('param', i)


def _is_external(loc) -> bool:
    return loc == EXT or (isinstance(loc, tuple) and loc[0] == 'param')


def _heap(node) -> tuple:
    return ('heap', node.nid)


def _literal(node) -> tuple:
    """An `S(...)` struct literal's own storage, which may need the heap like a variable's."""
    return ('literal', node.nid)


_NO_POINTERS = (t.IntLit, t.EnumMember, t.BoolLit, t.StrLit, t.NoneLit, t.ZeroValue, t.NewEmptyDict, t.EmptySlice)
_OPERATORS = (t.Unary, t.Binary, t.StrConcat, t.StrCompare, t.TagTest, t.Len, t.DictContains, t.ElementContains,
              t.StrIndex, t.Print, t.Panic, t.DictDelete, t.EnumFromInt, t.EnumContains, t.EnumName)


class EscapeAnalyzer:
    def __init__(self, fn: t.Function, structs: dict[str, StructInfo], summaries: Optional[dict] = None):
        self.fn = fn
        self.structs = structs
        self.summaries = summaries or {}
        self.H: dict = {EXT: {EXT}}
        self.esc: set = {EXT}
        self.changed = False

    def analyze(self) -> set[DeclId]:
        while True:
            self.changed = False
            for i, symbol in enumerate(self.fn.params):
                self._add(_param(i), {EXT})
                self._add(symbol.id, {_param(i)})
            self.walk_statements(self.fn.body)
            self._close_escapes()
            if not self.changed:
                break
        heap = self.esc | self._iteration_escapes()
        return {loc for loc in heap if loc != EXT and not (isinstance(loc, tuple) and loc[0] in ('heap', 'param'))}

    def _iteration_escapes(self) -> set:
        """Locations declared inside some loop whose address may be held outside it."""
        out = set()
        for inner in _loop_scopes(self.fn.body):
            reached = [u for loc, held in self.H.items() if loc not in inner for u in held if u in inner]
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

    def _add(self, loc, vals: set) -> None:
        cur = self.H.setdefault(loc, set())
        if not vals <= cur:
            cur |= vals
            self.changed = True

    def _escape(self, vals: set) -> None:
        if not vals <= self.esc:
            self.esc |= vals
            self.changed = True

    def _store(self, targets: set, vals: set) -> None:
        for loc in targets:
            if _is_external(loc):
                self._escape(vals)
            else:
                self._add(loc, vals)

    def _contents(self, targets: set) -> set:
        out = set()
        for loc in targets:
            out |= self.H.get(loc, set())
        return out

    def _close_escapes(self) -> None:
        stack = list(self.esc)
        while stack:
            loc = stack.pop()
            if loc != EXT:
                self._add(loc, {EXT})
            for u in self.H.get(loc, ()):
                if u not in self.esc:
                    self.esc.add(u)
                    self.changed = True
                    stack.append(u)

    # -- expressions

    def loc(self, e) -> set:
        """Locations `e`'s storage may reside in."""
        if isinstance(e, t.Local):
            return {e.symbol.id}
        if isinstance(e, t.FieldAccess):
            return self.vals(e.base) if e.through_pointer else self.loc(e.base)
        if isinstance(e, t.ArrayIndex):
            self.vals(e.index)
            return self.loc(e.base)
        if isinstance(e, t.SliceIndex):
            self.vals(e.index)
            return self.vals(e.base)
        if isinstance(e, t.DictLookup):
            # A dict's entries live in its shared heap table, which any copy of the dict reaches.
            self.vals(e.key)
            self.vals(e.dict)
            return {EXT}
        if isinstance(e, t.Deref):
            return self.vals(e.pointer)
        if isinstance(e, t.Payload):
            return self.loc(e.sum)
        if isinstance(e, t.StructLiteral):
            loc = _literal(e)
            self._add(loc, self.vals(e))
            return {loc}
        # Rvalue: materialized in fresh, non-promotable storage.
        loc = _heap(e)
        self._add(loc, self.vals(e))
        return {loc}

    def vals(self, e) -> set:
        """Locations `e`'s value may point into. Evaluates every subexpression, so escaping call
        arguments are always recorded."""
        if e is None or isinstance(e, _NO_POINTERS):
            return set()
        if isinstance(e, t.Local):
            return self._contents(self.loc(e))
        if isinstance(e, t.Bind):  # a declaration, then a test
            self._add(e.symbol.id, self.vals(e.value))
            self.vals(e.test)
            return set()
        if isinstance(e, t.FieldAccess):
            return self._contents(self.vals(e.base)) if e.through_pointer else self.vals(e.base)
        if isinstance(e, t.ArrayIndex):
            self.vals(e.index)
            return self.vals(e.base)
        if isinstance(e, t.SliceIndex):
            self.vals(e.index)
            return self._contents(self.vals(e.base))
        if isinstance(e, t.DictLookup):
            self.vals(e.dict)
            self.vals(e.key)
            return {EXT}
        if isinstance(e, t.SliceOf):
            self.vals(e.low)
            self.vals(e.high)
            return self.loc(e.base) if e.kind == 'array' else self.vals(e.base)
        if isinstance(e, t.AddressOf):
            return self.loc(e.place)
        if isinstance(e, t.BoxVariant):
            loc = _heap(e)
            self._add(loc, self.vals(e.value))
            return {loc}
        if isinstance(e, t.Deref):
            return self._contents(self.vals(e.pointer))
        if isinstance(e, (t.Payload, t.WidenToSum)):
            return self.vals(e.sum if isinstance(e, t.Payload) else e.value)
        if isinstance(e, (t.IntCast, t.StrFromByte, t.StrFromBytes)):
            return self.vals(e.value)
        if isinstance(e, _OPERATORS):
            for child in _children(e):
                self.vals(child)
            return set()
        if isinstance(e, t.StructLiteral):
            return set().union(*(self.vals(f) for f in e.fields)) if e.fields else set()
        if isinstance(e, (t.ArrayLiteral, t.SliceLiteral)):
            loc = _heap(e)
            elems = set()
            for x in e.elements:
                elems |= self.vals(x)
            self._add(loc, elems)
            return {loc} | elems
        if isinstance(e, t.DictLiteral):
            loc = _heap(e)
            for k, v in e.entries:
                self._add(loc, self.vals(k) | self.vals(v))
            return {loc}
        if isinstance(e, t.Append):
            loc = _heap(e)
            old = self.vals(e.slice)
            self._add(loc, self._contents(old))
            self._store(old | {loc}, self.vals(e.value))
            return old | {loc}
        if isinstance(e, (t.BytesFromStr, t.StrRawPtr, t.StrRawLen, t.StrFromRawParts)):
            # Runtime and intrinsic operations: treated as calls without a summary.
            for child in _children(e):
                self._escape(self.vals(child))
            return {EXT}
        if isinstance(e, t.Call):
            return self._call(e)
        raise TypeError(f"escape analysis: unhandled expression {type(e).__name__}")

    def _call(self, e: t.Call) -> set:
        arg_vals = [self.vals(a) for a in e.args]
        summary = self.summaries.get(e.name)
        if summary is None or len(summary) != len(arg_vals):
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

    def walk_statements(self, statements) -> None:
        for s in statements or ():
            self.walk_statement(s)

    def walk_statement(self, s) -> None:
        if isinstance(s, t.Declare):
            self._add(s.symbol.id, self.vals(s.init))
        elif isinstance(s, (t.Assign, t.CompoundAssign)):
            self._store(self.loc(s.target), self.vals(s.value))
        elif isinstance(s, t.Return):
            self._escape(self.vals(s.value))
        elif isinstance(s, t.ExprStmt):
            self.vals(s.expr)
        elif isinstance(s, t.If):
            self.vals(s.cond)
            self.walk_statements(s.then_body)
            self.walk_statements(s.else_body)
        elif isinstance(s, t.Match):
            self.vals(s.subject)
            for _, body in s.arms:
                self.walk_statements(body)
            self.walk_statements(s.else_body)
        elif isinstance(s, t.While):
            self.vals(s.cond)
            self.walk_statements(s.body)
        elif isinstance(s, t.For):
            if s.init is not None:
                self.walk_statement(s.init)
            self.vals(s.cond)
            self.walk_statements(s.body)
            self.walk_statement(s.step)
        elif isinstance(s, t.ForIn):
            it = self.vals(s.iterable)
            elems = it if s.kind == 'array' else self._contents(it)
            for symbol in s.bindings:
                self._add(symbol.id, elems)
            self.walk_statements(s.body)
        elif isinstance(s, (t.Break, t.Continue)):
            pass
        else:
            raise TypeError(f"escape analysis: unhandled statement {type(s).__name__}")


def _children(node) -> list:
    out = []
    for f in fields(node):
        value = getattr(node, f.name)
        for v in value if isinstance(value, tuple) else (value,):
            for x in v if isinstance(v, tuple) else (v,):
                if isinstance(x, (t.Expr, t.Stmt)):
                    out.append(x)
    return out


def _declared_within(nodes: list) -> set:
    """Locations declared anywhere in `nodes` (DeclIds, including `&S()` literal locations)."""
    out = set()
    stack = list(nodes)
    while stack:
        node = stack.pop()
        if isinstance(node, (t.Declare, t.Bind)):
            out.add(node.symbol.id)
        elif isinstance(node, t.ForIn):
            out.update(symbol.id for symbol in node.bindings)
        elif isinstance(node, t.StructLiteral):
            out.add(_literal(node))
        stack.extend(_children(node))
    return out


def _loop_scopes(statements) -> list:
    """For each loop in `statements`, the set of locations declared inside it (loop variable and
    for-in bindings included)."""
    scopes = []
    stack = list(statements)
    while stack:
        node = stack.pop()
        if isinstance(node, t.ForIn):
            # The iterable is evaluated once, before the first iteration.
            scopes.append(_declared_within(list(node.body)) | {symbol.id for symbol in node.bindings})
        elif isinstance(node, (t.While, t.For)):
            scopes.append(_declared_within(_children(node)))
        stack.extend(_children(node))
    return scopes


def compute_escape_summaries(functions, structs: dict[str, StructInfo]) -> dict:
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


def analyze_array_escapes(
        fn: t.Function, structs: dict[str, StructInfo], summaries: Optional[dict] = None) -> set[DeclId]:
    """DeclIds in typed function `fn` whose storage must be heap-allocated. Without `summaries`,
    every call escapes its arguments."""
    return EscapeAnalyzer(fn, structs, summaries).analyze()
