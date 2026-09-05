"""Escape Analysis"""

from typing import Optional

from parser import (
    ArrayLiteral,
    Assign,
    Binary,
    Call,
    ExprStmt,
    Field,
    FieldAssign,
    Function,
    If,
    Index,
    IndexAssign,
    Node,
    Return,
    Slice,
    Unary,
    VarDecl,
    Variable,
    While,
)
from semantic import type_from_name, Type, TypeKind, StructInfo
from codegen.utils import type_byte_width


# Fixed, hardcoded threshold for size-based stack safety. Any array-typed local
# or parameter whose total flattened footprint (type_byte_width) exceeds this
# many bytes is heap-allocated instead of living inline in its own stack slot.
#
# 16KB, not the more "principled" 4KB single-page size: generous enough that
# ordinary matrix-shaped code (e.g. a [50][50]int, exactly 10000 bytes) isn't
# quietly promoted to the heap by surprise, while leaving a wide safety margin
# against the default ~8MB stack budget even under moderate recursion.
_STACK_ARRAY_LIMIT_BYTES = 16384


def is_heap_allocated(t: Type, structs: dict[str, StructInfo]) -> bool:
    """Whether a value of type `t` is heap-allocated rather than stored
    inline in its own stack slot purely because of its OWN size -- true
    for an array OR STRUCT type whose total footprint (type_byte_width)
    exceeds _STACK_ARRAY_LIMIT_BYTES, false for every scalar type and
    every array/struct under the limit. A struct gets the same size-
    based treatment an array does -- both are value types whose
    footprint is a genuine, unbounded property of their own declared
    shape, so the identical risk (one huge local or parameter blowing
    the stack) applies equally to both. Purely a function of the type
    itself, never stored anywhere -- anywhere codegen has the Type, it
    can just call this directly.

    This is NOT the only reason a particular array ends up heap-
    allocated -- see analyze_array_escapes below for the other,
    independent trigger (a small array backing a slice that escapes
    the function still needs to survive past its return, regardless of
    size) -- so a caller deciding whether a SPECIFIC, NAMED variable
    needs heap allocation should go through
    CodeGenerator._is_heap_allocated instead, which combines this size
    check with that escape-analysis result."""
    return t.kind in (TypeKind.ARRAY, TypeKind.STRUCT) and type_byte_width(t, structs) > _STACK_ARRAY_LIMIT_BYTES


def _unwrap_slices(expr: Node) -> Node:
    """Unwraps a chain of Slice nodes (re-slicing, `s[0:3][0:2]`)
    down to whatever is actually being sliced underneath -- a bare
    Variable, an Index (reading an element out of an aggregate),
    or a Field (reading a struct field)."""
    while isinstance(expr, Slice):
        expr = expr.array
    return expr


def root_variable_name(expr: Node) -> Optional[str]:
    """Unwraps a chain of Index, Slice, AND Field nodes down to
    whatever bare Variable, if any, ultimately sits underneath."""
    while isinstance(expr, (Index, Slice, Field)):
        expr = expr.base if isinstance(expr, Field) else expr.array
    return expr.name if isinstance(expr, Variable) else None


class EscapeAnalyzer:
    def __init__(self, fn: Function, param_types: list[Type], structs: dict[str, StructInfo], aliases: dict[str, Type]):
        self.fn = fn
        self.structs = structs
        self.aliases = aliases

        self.array_decls: set[int] = set()
        self.slice_decls: set[int] = set()
        self.decl_types: dict[int, Type] = {}
        self.direct_backing: dict[int, set[int]] = {}
        self.slice_deps: dict[int, set[int]] = {}
        self.escaping_slices: set[int] = set()
        self.escaping_arrays: set[int] = set()
        self.aggregate_slot_ids: dict[tuple[int, str], int] = {}

        self.scopes: list[dict[str, int]] = [{}]

        for p, p_type in zip(fn.params, param_types):
            self.declare(p.name, id(p), p_type)

        self._AGGREGATE_ELEMENTS_SLOT = '[]'  # the one shared slot for a WHOLE
        # aggregate declaration -- an array-/slice-of-slices (used by
        # indexed_slot_of) or a struct containing a slice-typed field
        # (used by field_slot_of), at any nesting depth -- regardless of
        # which specific index or field is involved. The SAME sentinel
        # serves both, since a given declaration is always either array/
        # slice-shaped or struct-shaped, never both. Chosen because '['
        # and ']' can never appear in a Hornet identifier, so this can
        # never collide with a real field name.

    def analyze(self) -> set[int]:
        self.walk_statements(self.fn.body)

        result: set[int] = set(self.escaping_arrays)
        visited: set[int] = set()
        stack: list[int] = list(self.escaping_slices)
        while stack:
            slice_id = stack.pop()
            if slice_id in visited:
                continue
            visited.add(slice_id)
            result |= self.direct_backing.get(slice_id, set())
            for dep in self.slice_deps.get(slice_id, set()):
                if dep not in visited:
                    stack.append(dep)
        return result

    def declare(self, name: str, decl_id: int, decl_type: Type) -> None:
        self.scopes[-1][name] = decl_id
        self.decl_types[decl_id] = decl_type
        if decl_type.kind == TypeKind.ARRAY:
            self.array_decls.add(decl_id)
        if decl_type.kind == TypeKind.SLICE:
            self.slice_decls.add(decl_id)
            self.direct_backing.setdefault(decl_id, set())
            self.slice_deps.setdefault(decl_id, set())

    def resolve(self, name: str) -> Optional[int]:
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None

    def slot_node_id(self, container_id: int, slot: str) -> int:
        """Gives each distinct (container_id, slot) pair a unique node id,
        synthesizing one the first time that exact pair is seen and returning
        the same one every time after."""
        key = (container_id, slot)
        if key not in self.aggregate_slot_ids:
            node_id = -(len(self.aggregate_slot_ids) + 1)  # always negative;
            # id() is always a positive address in CPython, so this can
            # never collide with a real declaration's own node id.
            self.aggregate_slot_ids[key] = node_id
            self.slice_decls.add(node_id)
            self.direct_backing.setdefault(node_id, set())
            self.slice_deps.setdefault(node_id, set())
        return self.aggregate_slot_ids[key]

    def indexed_slot_of(self, base_expr: Node) -> Optional[int]:
        """Recognizes `base_expr` as something that, indexed ONE more
        time, produces a slice. Resolves it to whatever ROOT Variable
        underlies the whole chain (see root_variable_name), returning
        that root's shared indexed-elements slot id (see slot_node_id).

        Returns None if base_expr's type isn't an array or slice, if
        indexing it one more time wouldn't yield a slice, or if the
        chain doesn't resolve to a bare Variable at its root."""
        base_type = base_expr.resolved_type
        if base_type is None or base_type.kind not in (TypeKind.ARRAY, TypeKind.SLICE):
            return None
        element_type = base_type.element_type
        if element_type is None or element_type.kind != TypeKind.SLICE:
            return None
        root_name = root_variable_name(base_expr)
        if root_name is None:
            return None
        return self.whole_value_node_of(root_name)

    def field_slot_of(self, field_expr: Field) -> Optional[int]:
        """Recognizes field_expr as a struct field access whose value
        needs this analysis's tracking: the field is slice-typed, or is
        an aggregate containing a slice at some depth (see
        _contains_slice). Resolves to the ROOT Variable underlying the
        whole access chain (see root_variable_name) and returns that
        root's shared aggregate-elements slot id (see slot_node_id) --
        the SAME slot indexed_slot_of gives an array/slice-of-slices
        declaration.

        Deliberately ONE combined slot per root declaration, not one
        per distinct field path (`p.a` and `p.b` share it, even though
        a field name, unlike a dynamic index, could in principle get
        its own precise slot): per-field precision would need a
        struct-to-struct copy of just one field (`i = outer.inner`) to
        propagate only that field's own slots to `i`'s -- correct, but
        a larger mechanism than this analysis's "resolve to exactly one
        node" shape supports without a bigger refactor. Lumping every
        field into one shared slot is sound (a write into any field
        still makes the WHOLE declaration escape when it needs to),
        just coarser than necessary for two logically-independent
        slice fields on the same struct.

        Returns None if field_expr's base isn't struct-typed, the
        struct is unknown, or the field doesn't exist (all three
        already guaranteed impossible by the time semantic analysis
        has passed -- this stays defensive rather than assuming), if
        the field's type doesn't contain a slice, or if the chain
        doesn't resolve to a bare Variable at its root."""
        base_type = field_expr.base.resolved_type
        if base_type is None or base_type.kind != TypeKind.STRUCT:
            return None
        struct_info = self.structs.get(base_type.struct_name)
        if struct_info is None or field_expr.name not in struct_info.fields:
            return None
        field_type = struct_info.fields[field_expr.name]
        if not self._contains_slice(field_type):
            return None
        root_name = root_variable_name(field_expr)
        if root_name is None:
            return None
        return self.whole_value_node_of(root_name)

    def whole_value_node_of(self, name: str) -> Optional[int]:
        """Resolves `name` to the node id tracking its value, if it's a
        slice or contains one. Returns None if `name` doesn't resolve
        to anything, or resolves to a type that isn't a slice and
        doesn't contain one."""
        decl_id = self.resolve(name)
        if decl_id is None:
            return None
        decl_type = self.decl_types.get(decl_id)
        if decl_type is not None:
            if decl_type.kind in (TypeKind.ARRAY, TypeKind.SLICE):
                element_type = decl_type.element_type
                if element_type is not None and self._contains_slice(element_type):
                    return self.slot_node_id(decl_id, self._AGGREGATE_ELEMENTS_SLOT)
            elif decl_type.kind == TypeKind.STRUCT and self._contains_slice(decl_type):
                return self.slot_node_id(decl_id, self._AGGREGATE_ELEMENTS_SLOT)
        if decl_id in self.slice_decls:
            return decl_id
        return None

    def _contains_slice(self, t: Type) -> bool:
        """True if `t` is itself a slice, or contains one at ANY depth
        of further nesting.

        Recursing into a STRUCT's fields is safe from infinite
        recursion even for a self-referential struct: the SLICE case
        above is always checked first and returns True immediately
        without recursing further, so this can never recurse back into
        the same struct through a slice field. semantic.py's cycle
        detection guarantees the only way a struct could reach itself
        again is through a slice field (a direct or array-embedded
        self-reference is rejected), so any cycle that could exist here
        is guaranteed to pass through a slice-typed field first."""
        if t.kind == TypeKind.SLICE:
            return True
        if t.kind == TypeKind.ARRAY:
            return self._contains_slice(t.element_type)
        if t.kind == TypeKind.STRUCT:
            struct_info = self.structs.get(t.struct_name)
            if struct_info is None:
                return False
            return any(self._contains_slice(field_type) for field_type in struct_info.fields.values())
        return False

    def contribution(self, value_expr: Node) -> tuple[Optional[int], Optional[int]]:
        """Returns (array_decl_id, slice_decl_id) -- whichever ONE of
        the two value_expr's aliasing actually resolves to (never
        both), or (None, None) if it isn't backed by any of this
        function's declarations at all (a fresh literal, `none`, an
        ordinary user-function call's return value, ...). An
        aggregate's slot id (see AGGREGATES AND SLOTS in
        analyze_array_escapes) is returned as the second element here
        exactly like a bare slice Variable's id would be -- callers
        don't need to know it came from indexing into an aggregate
        rather than reading a plain slice variable directly."""
        if isinstance(value_expr, Slice):
            # Re-slicing never changes what backs a value, so unwrap
            # any further re-slicing FIRST (`s[0:3][0:2]`) down to
            # whatever's actually being sliced. This stops at, rather
            # than through, an Index or Field: those need indexed_
            # slot_of's/field_slot_of's aggregate-slot resolution tried
            # first, falling back to the plain root-declaration check
            # below when that doesn't apply -- e.g. `matrix[1][0:2]`
            # (slicing a plain sub-array row, no slices involved) has
            # inner = Index(matrix, 1), but indexed_slot_of(matrix)
            # correctly returns None (indexing matrix once more yields
            # another array, not a slice), so this falls through to
            # resolving matrix itself, like the plain-Variable case
            # below.
            inner = _unwrap_slices(value_expr.array)
            slot_id = None
            if isinstance(inner, Index):
                slot_id = self.indexed_slot_of(inner.array)
            elif isinstance(inner, Field):
                slot_id = self.field_slot_of(inner)
            if slot_id is not None:
                return None, slot_id
            base_name = root_variable_name(inner)
            if base_name is not None:
                base_id = self.resolve(base_name)
                if base_id in self.array_decls:
                    return base_id, None
                if base_id in self.slice_decls:
                    return None, base_id
        elif isinstance(value_expr, Variable):
            node_id = self.whole_value_node_of(value_expr.name)
            if node_id is not None:
                return None, node_id
        elif isinstance(value_expr, Index):
            # Reading an element back out of a declared aggregate --
            # `rows[i]` -- resolves to that aggregate's indexed-
            # elements slot.
            slot_id = self.indexed_slot_of(value_expr.array)
            if slot_id is not None:
                return None, slot_id
        elif isinstance(value_expr, Field):
            # Reading a slice-typed (or slice-containing) field back
            # out of a declared struct -- `p.values` -- resolves to
            # that struct's combined aggregate-elements slot,
            # structurally identical to the Index case above.
            slot_id = self.field_slot_of(value_expr)
            if slot_id is not None:
                return None, slot_id
        elif isinstance(value_expr, Call) and value_expr.name == 'append':
            # append's first argument might reuse ITS OWN backing
            # storage (the reuse path -- see gen_append_call_into), so
            # whatever that argument resolves to is this call's
            # contribution too. Recursing here means append's first
            # argument gets the same treatment any other slice-valued
            # expression does -- an aggregate element, an unnamed slice
            # expression, or another append call's result -- not just
            # a named slice variable.
            return self.contribution(value_expr.args[0])
        return None, None

    def scan_expr_for_escaping_calls(self, expr: Node) -> None:
        if isinstance(expr, Call):
            if expr.name not in ('print', 'len', 'append'):
                for arg in expr.args:
                    array_id, slice_id = self.contribution(arg)
                    if array_id is not None:
                        self.escaping_arrays.add(array_id)
                    if slice_id is not None:
                        self.escaping_slices.add(slice_id)
            for arg in expr.args:
                self.scan_expr_for_escaping_calls(arg)
        elif isinstance(expr, Binary):
            self.scan_expr_for_escaping_calls(expr.left)
            self.scan_expr_for_escaping_calls(expr.right)
        elif isinstance(expr, Unary):
            self.scan_expr_for_escaping_calls(expr.operand)
        elif isinstance(expr, Index):
            self.scan_expr_for_escaping_calls(expr.array)
            self.scan_expr_for_escaping_calls(expr.index)
        elif isinstance(expr, Slice):
            self.scan_expr_for_escaping_calls(expr.array)
            if expr.low is not None:
                self.scan_expr_for_escaping_calls(expr.low)
            if expr.high is not None:
                self.scan_expr_for_escaping_calls(expr.high)
        elif isinstance(expr, ArrayLiteral):
            for element in expr.elements:
                self.scan_expr_for_escaping_calls(element)
        elif isinstance(expr, Field):
            # A pre-existing gap this closes alongside the struct-
            # field escape work: `foo(bar()).x` (a nested call under a
            # Field access) needs bar()'s own argument-escaping check
            # just as much as any other sub-expression -- never reached
            # before, regardless of whether the field itself ends up
            # being slice-relevant.
            self.scan_expr_for_escaping_calls(expr.base)
        # Variable, Constant, BoolLiteral, StringLiteral, NoneLiteral:
        # leaves, nothing further to recurse into.

    def walk_statements(self, statements: list[Node]) -> None:
        for stmt in statements:
            if isinstance(stmt, VarDecl):
                var_type = type_from_name(stmt.var_type, self.structs, self.aliases)
                self.declare(stmt.name, id(stmt), var_type)
                if stmt.init is not None:
                    target_node = self.whole_value_node_of(stmt.name)
                    if target_node is not None:
                        array_id, slice_id = self.contribution(stmt.init)
                        if array_id is not None:
                            self.direct_backing[target_node].add(array_id)
                        if slice_id is not None:
                            self.slice_deps[target_node].add(slice_id)
                    self.scan_expr_for_escaping_calls(stmt.init)
            elif isinstance(stmt, Assign):
                target_node = self.whole_value_node_of(stmt.name)
                if target_node is not None:
                    array_id, slice_id = self.contribution(stmt.value)
                    if array_id is not None:
                        self.direct_backing[target_node].add(array_id)
                    if slice_id is not None:
                        self.slice_deps[target_node].add(slice_id)
                self.scan_expr_for_escaping_calls(stmt.value)
            elif isinstance(stmt, IndexAssign):
                slot_id = self.indexed_slot_of(stmt.array)
                if slot_id is not None:
                    array_id, slice_id = self.contribution(stmt.value)
                    if array_id is not None:
                        self.direct_backing[slot_id].add(array_id)
                    if slice_id is not None:
                        self.slice_deps[slot_id].add(slice_id)
                self.scan_expr_for_escaping_calls(stmt.array)
                self.scan_expr_for_escaping_calls(stmt.index)
                self.scan_expr_for_escaping_calls(stmt.value)
            elif isinstance(stmt, FieldAssign):
                # Mirrors IndexAssign one kind of access over --
                # field_slot_of needs an actual Field node, not stmt
                # directly: root_variable_name's isinstance check only
                # recognizes Index/Slice/Field, not FieldAssign, so
                # duck-typing stmt.base/stmt.name through it silently
                # fails to unwrap anything, always returning None -- a
                # real bug this construction fixes, found by testing (a
                # struct escaping via return, with a slice field
                # previously written through FieldAssign, failed to
                # promote its backing array) rather than by inspection.
                slot_id = self.field_slot_of(Field(base=stmt.base, name=stmt.name))
                if slot_id is not None:
                    array_id, slice_id = self.contribution(stmt.value)
                    if array_id is not None:
                        self.direct_backing[slot_id].add(array_id)
                    if slice_id is not None:
                        self.slice_deps[slot_id].add(slice_id)
                self.scan_expr_for_escaping_calls(stmt.base)
                self.scan_expr_for_escaping_calls(stmt.value)
            elif isinstance(stmt, Return):
                if stmt.value is not None:
                    array_id, slice_id = self.contribution(stmt.value)
                    if array_id is not None:
                        self.escaping_arrays.add(array_id)
                    if slice_id is not None:
                        self.escaping_slices.add(slice_id)
                    self.scan_expr_for_escaping_calls(stmt.value)
            elif isinstance(stmt, ExprStmt):
                self.scan_expr_for_escaping_calls(stmt.expr)
            elif isinstance(stmt, If):
                self.scan_expr_for_escaping_calls(stmt.condition)
                self.scopes.append({})
                self.walk_statements(stmt.then_body)
                self.scopes.pop()
                if stmt.else_body is not None:
                    self.scopes.append({})
                    self.walk_statements(stmt.else_body)
                    self.scopes.pop()
            elif isinstance(stmt, While):
                self.scan_expr_for_escaping_calls(stmt.condition)
                self.scopes.append({})
                self.walk_statements(stmt.body)
                self.scopes.pop()
            # Break, Continue: nothing to do.


def analyze_array_escapes(fn: Function, param_types: list[Type], structs: dict[str, StructInfo], aliases: dict[str, Type]) -> set[int]:
    """Returns the set of id()s -- of this function's VarDecl or Param
    nodes -- for array-typed declarations that need to be heap-
    allocated because a slice backed by them might outlive this
    function's call, REGARDLESS of their own size. This closes the
    memory-safety gap size-based heap promotion alone leaves open: a
    small, stack-allocated array, sliced and returned (directly, via a
    named slice variable, or after an `append` that reused its backing
    storage), leaves the returned slice's pointer dangling into a
    torn-down stack frame.

    An intraprocedural, FLOW-INSENSITIVE analysis: it doesn't reason
    about execution order or which branch actually runs -- every
    assignment to a given slice variable anywhere in the function is
    unioned into one combined "what might this be backed by" answer.
    This is a deliberate simplification (a variable reused for two
    logically-unrelated slices gets treated as if it could be either
    one everywhere), avoiding a genuine fixed-point dataflow pass over
    branches and loops for a precision level ordinary code doesn't
    often need. The same treatment covers a slice stored as an element
    of an AGGREGATE (an array-/slice-of-slices, or a struct with a
    slice field -- see AGGREGATES AND SLOTS below), composing
    correctly at any depth (`matrix[i][j] = arr[:]`, `s1[0:3][0:2]`).

    Two further limitations, deliberately out of scope:
      - Purely INTRAprocedural: a slice passed to any user-defined
        function call is conservatively treated as escaping
        unconditionally, without looking at what the callee does with
        it. A real interprocedural version would need a per-function
        escape summary computed via fixed-point iteration over the
        call graph (Hornet allows recursion) -- a substantially larger
        undertaking, left for its own follow-up.
      - A slice stored through indirection this language doesn't
        actually have (e.g. a pointer) was never in scope.

    ALGORITHM, two phases:
      1. Walk the function body once (recursing into If/While bodies
         with a scope stack, so shadowed names resolve correctly),
         building:
           - direct_backing: for each trackable node (a slice-typed
             declaration, or an aggregate's slot), which array-typed
             declaration(s) it's ever directly sliced from.
           - slice_deps: for each trackable node, which OTHER
             trackable node(s) it might be derived from (re-slicing, a
             slice-to-slice copy, `append`, or reading an aggregate
             element back out).
           - escaping_slices / escaping_arrays: nodes directly marked
             escaping, from a `return` value or a slice/array argument
             to a user-defined call (found via a full recursive scan
             of every sub-expression, not just top-level ones --
             `return foo(bar(s))` still needs `s` checked as bar's
             argument).

         AGGREGATES AND SLOTS: an "aggregate" is any declaration that
         can hold MULTIPLE independently-accessed values, at least one
         possibly slice-typed -- an array-/slice-of-slices, or a
         struct. A "slot" identifies which part is being accessed;
         slot_node_id gives every distinct (declaration, slot) pair a
         stable synthesized node id, registered in the same
         slice_decls/direct_backing/slice_deps structures a bare slice
         declaration uses -- nothing downstream needs to know whether
         a node id came from a real declaration or a slot.

         Only one KIND of slot exists: indexed_slot_of recognizes
         `rows[i]` or any chain of Index/Slice steps where indexing the
         base ONE more time would yield a slice, and maps the root
         declaration to a single SHARED slot (_AGGREGATE_ELEMENTS_SLOT)
         regardless of which index is involved -- the same "one
         combined blob per declaration" treatment a bare slice variable
         gets, extended to an aggregate's elements. The "one more level
         would yield a slice" guard checks the immediate base's own
         element_type directly, NOT a recursive "does this eventually
         contain a slice" walk -- conflating the two was a real bug:
         for `rows[0][0]` where rows: [1][]int, the outer Index's base
         (`rows[0]`) is itself slice-typed, and indexing it once more
         yields an INT, not a slice -- reading a plain int has nothing
         to do with rows' role as a slice-holder, and must NOT resolve
         to rows' slot just because rows contains a slice somewhere.
         whole_value_node_of's own "contains a slice at any depth"
         check (see _contains_slice) is the right question for a
         WHOLE-VALUE read (`return rows`); indexed_slot_of's narrower,
         one-level check is right for "would indexing this once more
         give a slice" -- both are needed, for different callers.

         WHOLE-VALUE READS OF AN AGGREGATE GO THROUGH THE SAME SLOT,
         not a separate one: `return rows`, `rows2 = rows`, and `rows`
         as a call argument all resolve to indexed_slot_of's own node
         (via whole_value_node_of). This matters concretely:
         `rows`/`rows2` alias the same backing storage (a slice
         descriptor copy is shallow), so a later `rows[0] = arr[:]`
         must stay visible through `rows2[0]` and through a bare
         `return rows2` for this analysis to stay sound. An earlier
         version gave whole-aggregate reads their own separate node
         instead of sharing indexed_slot_of's, and `return rows` (with
         nothing ever indexing into it) silently stopped resolving to
         anything -- caught by testing, not inspection.

         FIELD access (`s.my_ints`) got its own analogous function,
         field_slot_of, doing the same recognition one kind of access
         over. It deliberately does NOT give each field its own precise
         slot -- `s.a` and `s.b` share one slot, the same "one combined
         blob" treatment array elements get. Per-field precision was
         the original plan, but a struct-to-struct copy of just PART
         of a struct (`i = outer.inner`) would need to precisely
         propagate only that field's own slots -- correct, but a
         larger mechanism than contribution()'s "resolve to exactly
         one node" shape supports without a bigger refactor.
         whole_value_node_of got the analogous extension: a bare
         Variable referring to a struct falls back to it exactly like
         an aggregate-of-slices does.

         An array-typed aggregate (`[N][]int`) is registered in BOTH
         array_decls (its unrelated existing role, e.g. `rows[0:2]`)
         and, via its slot(s), the slice-tracking structure -- two
         independent roles a single declaration can have.
      2. Compute the transitive closure from escaping_slices, following
         slice_deps edges (BFS via an explicit stack, not recursion, so
         it can't stack-overflow on a pathological chain), unioning in
         direct_backing at every node reached, plus escaping_arrays
         found directly. The result is exactly the set of array
         declarations that need to survive past this function's
         return.
    """
    analyzer = EscapeAnalyzer(fn, param_types, structs, aliases)
    return analyzer.analyze()