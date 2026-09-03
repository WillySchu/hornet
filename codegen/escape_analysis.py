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
# 16KB, not the more "principled" 4KB single-page size: generous
# enough that ordinary matrix-shaped code (e.g. a [50][50]int, exactly
# 10000 bytes) doesn't get quietly promoted to the heap by surprise,
# while still leaving a wide safety margin against the default ~8MB
# stack budget even under moderate recursion (8MB / 16KB = 512
# same-sized frames before exhaustion).
_STACK_ARRAY_LIMIT_BYTES = 16384


def is_heap_allocated(t: Type, structs: dict[str, StructInfo]) -> bool:
    """Whether a value of type `t` is heap-allocated rather than stored
    inline in its own stack slot purely because of its OWN size -- true
    for an array OR STRUCT type whose total footprint (type_byte_width)
    exceeds _STACK_ARRAY_LIMIT_BYTES, false for every scalar type and
    every array/struct under the limit. A struct gets exactly the same
    size-based treatment an array already does -- both are value types
    whose own footprint is a genuine, unbounded property of their own
    declared shape (an array's size, or a struct's own field list),
    not something this compiler controls -- so the identical risk
    (one huge local or parameter blowing the stack on its own) applies
    equally to both, and gets the identical fix. Purely a function of
    the type itself, not of any per-variable state, so it's never
    stored anywhere -- anywhere codegen already has the Type (via
    _local_type or _type_of), it can just call this directly.

    This is NOT the only reason a particular array ends up heap-
    allocated any more -- see analyze_array_escapes below for the
    other, independent trigger (a small array that backs a slice which
    escapes the function it's declared in still needs to survive past
    that function's own return, regardless of its size) -- so a
    caller deciding whether a SPECIFIC, NAMED variable needs heap
    allocation should go through CodeGenerator._is_heap_allocated
    instead, which combines this size check with that escape-analysis
    result; this function alone only ever answers the size half of
    that question."""
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
        # indexed_slot_of) or a struct containing a slice-typed field, at
        # any depth of nesting (used by field_slot_of below) -- regardless
        # of which specific index or field is involved (see AGGREGATES AND
        # SLOTS above for why); the SAME sentinel serves both kinds of
        # aggregate, not two separate ones, since a given declaration is
        # always either array/slice-shaped or struct-shaped, never both,
        # so there's never a risk of the two meanings colliding for the
        # same declaration. Chosen because '[' and ']' can never appear in
        # a Hornet identifier, so this can never collide with an actual
        # field name either, if a per-field sentinel were ever needed.

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
        """Recognizes `base_expr` as something that, indexed ONE more time,
        produces a slice. Then resolves it to whatever ROOT Variable underlies
        the whole chain (see root_variable_name), returning that root's own
        shared indexed-elements slot id (see slot_node_id).

        Returns None if base_expr's own type isn't an array or slice at all, if
        indexing it one more time wouldn't yield a slice, or if the chain
        doesn't resolve to a bare Variable at its root."""
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
        # TODO(will): Fix this docstring
        """Recognizes field_expr as a struct field access that reads or writes a
        value needing this analysis's own tracking. This means:
            1. The field itself is slice-typed, or is an aggregate (array or
               struct) that itself contains a slice at some depth (see
               _contains_slice).
            2. It resolves to whatever ROOT Variable underlies the whole access
               chain (see root_variable_name, which unwraps Field, Index, and
               Slice together, in any mix).
        It returns that root's own shared aggregate-elements slot id (see slot_node_id) --
        the SAME slot indexed_slot_of gives an array/slice-of-slices
        declaration, and whole_value_node_of gives a struct considered
        as a whole. Deliberately ONE combined slot per root
        declaration, not a separate one per distinct field path (`p.a`
        and `p.b` share the SAME slot, even though a field name -- unlike
        a dynamic array index -- is known statically and so COULD in
        principle get its own precise one): giving every field its own
        slot would mean a struct-to-struct copy of just a PART of a
        struct (`i = outer.inner`, copying one field's own sub-struct
        without touching outer's OTHER fields) needs to precisely
        propagate only the sub-struct's own field slots to `i`'s own
        -- correct, but a genuinely larger mechanism than this
        analysis's existing "resolve to exactly one node" shape
        supports without a much bigger refactor. Lumping every field
        of a given declaration into one shared slot instead keeps this
        an incremental extension of the exact same machinery indexed_
        slot_of already established, at the cost of the identical kind
        of precision loss already accepted for array elements (`rows[0]`
        and `rows[1]` already share one slot too) -- sound (a write
        into any field still correctly makes anything the WHOLE
        declaration reaches escape when it needs to), just coarser
        than necessary in the specific case of two logically-
        independent slice fields on the same struct.

        Returns None if field_expr's base isn't struct-typed, that
        struct is unknown, or field_expr.name isn't a real field of it
        (all three already guaranteed impossible by the time semantic
        analysis has passed -- this stays defensive rather than
        assuming), if the field's own type doesn't contain a slice at
        all, or if the chain doesn't resolve to a bare Variable at its
        root."""
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
        """Resolves `name` to the node id associated with the name if the node
        is a slice or contains a slice. Returns None if `name` doesn't resolve
        to anything or resolves to a non slice or aggregate type that does not
        contain a slice."""
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

        Recursing into a STRUCT's own fields  is safe from infinite recursion
        even for a self-referential struct. The SLICE case above is always
        checked FIRST and returns True immediately without recursing any
        further, so this can never recurse back into the SAME struct through a
        slice field. Semantic.py's cycle detection guarantees the only way a
        struct could ever reach itself again at all is through a slice field,
        since a direct or array-embedded self-reference is rejected. Any struct
        cycle that could exist by the time this runs is therefore guaranteed to
        pass through a slice-typed field, which this returns True for without
        descending any further."""
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
        # TODO(will): Clean up this documentation.
        """Returns (array_decl_id, slice_decl_id) -- whichever ONE of
        the two value_expr's own aliasing actually resolves to (never
        both), or (None, None) if it isn't backed by any of this
        function's own declarations at all (a fresh literal, `none`,
        an ordinary user-function call's own return value, ...). An
        aggregate's own slot id (see AGGREGATES AND SLOTS above) is
        returned as the second element here exactly like a bare slice
        Variable's own id would be -- callers don't need to know or
        care that it came from indexing into an aggregate rather than
        reading a plain slice variable directly."""
        if isinstance(value_expr, Slice):
            # Re-slicing never changes what backs a value, so unwrap
            # any further re-slicing FIRST (`s[0:3][0:2]`) down to
            # whatever's actually being sliced -- see _unwrap_slices's
            # own docstring for why this stops at, rather than through,
            # an Index or Field: those need indexed_slot_of's/field_
            # slot_of's own aggregate-slot resolution tried FIRST, with
            # a fallback to the plain root-declaration check right
            # below when that resolution doesn't apply -- e.g.
            # `matrix[1][0:2]` (slicing a plain SUB-ARRAY row out of a
            # multi-dimensional array with no slices involved anywhere)
            # has inner = Index(matrix, 1), but indexed_slot_of(matrix)
            # correctly returns None (indexing matrix one more time
            # yields another array, not a slice) -- so this must still
            # fall through to resolving matrix itself as the raw array
            # that needs to escape, exactly like the plain-Variable
            # case just below already does.
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
            # e.g. `rows[i]` -- resolves to that aggregate's own
            # indexed-elements slot.
            slot_id = self.indexed_slot_of(value_expr.array)
            if slot_id is not None:
                return None, slot_id
        elif isinstance(value_expr, Field):
            # Reading a slice-typed (or slice-containing) field back
            # out of a declared struct -- e.g. `p.values` -- resolves
            # to that struct's own combined aggregate-elements slot,
            # structurally identical to the Index case just above, one
            # kind of access over.
            slot_id = self.field_slot_of(value_expr)
            if slot_id is not None:
                return None, slot_id
        elif isinstance(value_expr, Call) and value_expr.name == 'append':
            # append's own first argument might reuse ITS OWN backing
            # storage (the reuse path -- see gen_append_call_into), so
            # whatever that argument itself resolves to is exactly
            # this call's own contribution too. Recursing into
            # contribution() here (rather than only handling a bare
            # Variable) means append's first argument gets the SAME
            # treatment any other slice-valued expression already
            # does -- an aggregate element (`append(rows[0], v)`), an
            # unnamed slice expression (`append(arr[0:2], v)`), or
            # even another append call's own result -- not just a
            # named slice variable.
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
            # field escape work, not something new introduced by it:
            # `foo(bar()).x` (a nested call underneath a Field access)
            # needs bar()'s own argument-escaping check just as much
            # as any other sub-expression does -- this was never
            # reached at all before, regardless of whether the field
            # itself ends up being slice-relevant.
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
                # Mirrors IndexAssign exactly, one kind of access over
                # -- field_slot_of needs an actual Field node, not
                # stmt directly: root_variable_name's own isinstance
                # check (which field_slot_of calls into) only
                # recognizes Index/Slice/Field, not FieldAssign, so
                # duck-typing stmt.base/stmt.name through it silently
                # fails to unwrap anything at all, always returning
                # None -- a real bug this construction fixes, found by
                # testing (a struct escaping via return, with a slice
                # field previously written through FieldAssign, failed
                # to promote its own backing array at all) rather than
                # by inspection.
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
    """Returns the set of id()s -- of this function's own VarDecl or
    Param nodes -- for array-typed declarations that need to be heap-
    allocated because a slice backed by them might outlive this
    function's own call, REGARDLESS of their own size. This is what
    closes the real memory-safety gap size-based heap promotion alone
    left open: a small, stack-allocated array, sliced and returned
    (directly, or via a named slice variable, or after an `append`
    that happened to reuse its backing storage), leaves the returned
    slice's own pointer field dangling into a stack frame that's
    already been torn down by the time anything reads it again.

    An intraprocedural, FLOW-INSENSITIVE analysis: it doesn't reason
    about the order statements execute in, or which branch of an
    if/while actually runs -- every assignment to a given slice
    variable ANYWHERE in the function is unioned together into one
    combined "what might this be backed by" answer, used uniformly
    wherever that variable is read. This is a real, deliberate
    simplification (a variable reused for two logically-unrelated
    slices at different points in the same function gets treated as
    if it could be either one everywhere), not an oversight -- it
    avoids needing a genuine fixed-point dataflow pass over branches
    and loops, which true flow sensitivity would require even before
    any function calls enter the picture, for a level of precision
    ordinary code doesn't often need. The SAME flow-insensitive
    treatment now also covers a slice stored as an element of an
    AGGREGATE -- today, specifically an array- or slice-of-slices
    (`rows[i] = arr[:]`) -- see AGGREGATES AND SLOTS below for what
    that word is doing here and why it's phrased more generally than
    "array-of-slices" alone, and for how this now composes correctly
    at ANY depth: `matrix[i][j] = arr[:]` (an aggregate reached
    through a further Index, not a bare Variable) and `s1[0:3][0:2]`
    (a Slice reached through another Slice, with no intermediate
    named variable at all) both resolve to the correct root
    declaration's own shared slot, not just the single-hop case. Two
    further, real limitations, each deliberately out of scope for now
    rather than silently mishandled:
      - Purely INTRAprocedural: a slice passed as an argument to any
        user-defined function call is conservatively treated as
        escaping unconditionally, without looking at what the callee
        actually does with it (does it store it somewhere that
        outlives the call, or just read it and let it go). A real
        interprocedural version would need a per-function escape
        SUMMARY, computed for every function in dependency order --
        and since Hornet allows recursion, computing those soundly
        needs a genuine fixed-point iteration over the call graph, not
        a single pass. That's a substantially larger undertaking than
        this, and left for its own, separate follow-up.
      - A slice stored as an element of something that ISN'T itself a
        declared local or parameter -- e.g. through a pointer-like
        indirection this language doesn't actually have -- was never
        in scope to begin with and remains so.

    The algorithm itself has two phases:
      1. Walk the function body once (recursing into every If's own
         then_body/else_body and every While's own body, maintaining a
         scope stack so a name resolves to the SPECIFIC declaration it
         actually refers to at that point -- Hornet allows shadowing a
         name in a nested block, so a plain name-based lookup would
         risk conflating two entirely different variables), building:
           - direct_backing: for each trackable node (a slice-typed
             declaration OR an aggregate's own slot -- see AGGREGATES
             AND SLOTS below), which array-typed declaration(s) it's
             ever directly sliced from (`s = arr[low:high]`, or
             `s = matrix[i][low:high]` -- indexing into a multi-
             dimensional array's own row is still a view into the SAME
             backing storage, not a copy, so this unwraps nested Index
             nodes down to their root Variable rather than requiring a
             bare one).
           - slice_deps: for each trackable node, which OTHER trackable
             node(s) it might in turn be derived from (re-slicing a
             slice, a plain slice-to-slice copy, `append` -- which
             might reuse its own first argument's backing array -- or
             reading an element back out of an aggregate).
           - escaping_slices / escaping_arrays: nodes directly marked
             as escaping, from a `return` statement's own value or a
             slice/array-typed argument passed to a user-defined call
             (found via a full recursive scan of every expression for
             a nested Call, not just ones at a statement's own top
             level -- `return foo(bar(s))` still needs `s` to be
             checked as bar's own argument even though the RETURN
             itself is really about foo's result, not s directly).

         AGGREGATES AND SLOTS: this is the piece that exists purely to
         make the NEXT thing built on top of this analysis (struct
         support) a small addition rather than a third near-copy of
         very similar logic -- see the module's own note on why this
         was worth doing as its own, standalone step before struct
         work starts. An "aggregate" is any declaration that can hold
         MULTIPLE independently-accessed values, at least one of which
         might be slice-typed -- today, that's exactly the array- and
         slice-of-slices case, but a struct is exactly this too, just
         with named fields instead of indices. A "slot" identifies
         WHICH part of the aggregate is being accessed; slot_node_id
         gives every distinct (aggregate declaration, slot) pair its
         own stable, synthesized node id (a small negative integer,
         guaranteed distinct from every real id() -- id() is always a
         positive address in CPython -- and from every other slot's own
         id, generated once per pair and memoized in aggregate_slot_ids
         for every later request), then registers THAT id in the very
         SAME slice_decls/direct_backing/slice_deps structures a bare
         slice-typed declaration already uses. Nothing downstream --
         not the transitive closure walk, not contribution()'s own
         callers -- needs to know or care whether a node id it's
         holding came from id()-ing a real declaration or from
         slot_node_id: they're both just integers in the same graph.

         Only one KIND of slot exists right now: indexed_slot_of
         recognizes `rows[i]` (an Index) or `rows[i] = ...`
         (IndexAssign's own target) -- and, more generally, any chain
         of Index and/or Slice operations reached through however many
         further Index/Slice steps precede it (`matrix[i][j]`,
         `s1[0:3][0:2]`, any mix, at any depth -- see root_variable_
         name, which unwraps the whole chain down to whatever bare
         Variable underlies it) -- where indexing the immediate base
         ONE more time would yield a slice-typed result, and maps the
         ROOT declaration to the slot key _AGGREGATE_ELEMENTS_SLOT -- a
         single, SHARED slot for the whole declaration, regardless of
         which actual index evaluates to what at runtime, or how many
         levels of indexing/re-slicing separate a particular access
         from that root, since indices are dynamic values this
         analysis can't distinguish without real per-index tracking (a
         different, larger undertaking, and not attempted here -- see
         the limitations list above). This is exactly the "one
         combined blob per declaration" flow-insensitive treatment a
         bare slice variable already gets, just extended to an
         aggregate's elements collectively, at whatever depth they're
         reached from.

         The "one more level would yield a slice" guard is checked
         directly against the immediate base's own resolved_type (its
         element_type, specifically), NOT via a recursive "does this
         eventually contain a slice somewhere" walk -- those are
         answering two different questions, and conflating them is a
         real bug an earlier version of this had: for `rows[0][0]`
         where rows: [1][]int, the outer Index's own base is `rows[0]`
         (itself slice-typed), and indexing that ONE more time yields
         an INT, not a slice -- reading a plain int value OUT of a
         slice has nothing to do with rows' own role as a slice-
         holding aggregate, and must NOT resolve to rows' own slot
         just because rows, considered as a whole, happens to contain
         a slice somewhere. whole_value_node_of's OWN check (does the
         aggregate, taken as a whole, contain a slice at ANY depth of
         array nesting -- see _contains_slice) is the right question
         for a WHOLE-VALUE read (`return rows`, no indexing at all);
         indexed_slot_of's own, narrower, one-level check is the right
         question for "would indexing this ONE more time give me a
         slice" -- and both are needed, for different callers, rather
         than one subsuming the other.

         WHOLE-VALUE READS OF AN AGGREGATE ALSO GO THROUGH THE SAME
         SLOT, not a separate one: `return rows`, `rows2 = rows`, and
         `rows` passed as a call argument all resolve to indexed_slot_
         of's own node (via whole_value_node_of, contribution()'s and
         the VarDecl/Assign target-resolution's shared entry point for
         "what node tracks whatever this name's value is"), exactly
         like reading `rows[i]` does -- not a second, disconnected node
         that happens to sit next to it. This matters concretely, not
         just tidily: `rows` and `rows2` in `rows2 = rows` alias the
         SAME backing storage (copying a slice descriptor is a shallow,
         alias-preserving copy, the same as anywhere else in this
         language), so a later `rows[0] = arr[:]` has to still be
         visible through `rows2[0]` -- and, just as much, through a
         bare `return rows2` -- for this analysis to stay sound. This
         wasn't a hypothetical worth hardening pre-emptively: an
         earlier version of this exact refactor gave whole-aggregate
         reads their OWN separate node instead of sharing indexed_slot_
         of's, and `return rows` (with nothing ever indexing into it
         at all) silently stopped resolving to anything -- caught by
         testing the refactor against the very scenarios it was
         supposed to preserve, not found by inspection.

         FIELD access (`s.my_ints`) got its own analogous function --
         field_slot_of -- doing the same shape of recognition (does
         accessing this thing resolve to a declaration whose type
         needs the aggregate-elements slot treatment) one kind of
         access over. It deliberately does NOT give each field its
         own separate, precise slot the way a field name's static
         (unlike a dynamic array index) would in principle allow --
         `s.a` and `s.b` share the SAME slot as each other, and as any
         other field of the same declaration, exactly the "one
         combined blob" treatment `rows[i]` and `rows[j]` already get.
         Per-field precision was the original plan, but building it
         out surfaced a real complication: a struct-to-struct copy of
         just PART of a struct (`i = outer.inner`, copying one field's
         own sub-struct without touching outer's OTHER fields) would
         need to precisely propagate only the sub-struct's own field
         slots to `i`'s own -- correct, but a genuinely larger
         mechanism than contribution()'s existing "resolve to exactly
         one node" shape supports without a much bigger refactor.
         Lumping every field into one shared slot per declaration
         keeps this an incremental extension of exactly the same
         machinery indexed_slot_of already established, at the cost of
         the identical kind of precision loss already accepted for
         array elements -- sound, just coarser than strictly necessary
         for two logically-independent slice fields on the same
         struct. whole_value_node_of got the analogous extension too --
         a bare `Variable` referring to a struct falls back to it
         exactly like an aggregate-of-slices does, so `return s`
         resolves to that struct's own single combined slot the same
         deliberate way `return rows` resolves to indexed_slot_of's.

         An array-typed aggregate (`[N][]int`) is registered in BOTH
         array_decls (for its own, unrelated existing role -- e.g.
         `rows[0:2]`, slicing the aggregate itself, still works via the
         existing array_decls/root_variable_name path) and, via its
         slot(s), the unified slice-tracking structure (for its role as
         a holder of slice elements) -- these are two independent
         things a single declaration can be, not a conflict between
         them.
      2. Compute the transitive closure from escaping_slices, following
         slice_deps edges (an ordinary graph reachability walk -- BFS
         via an explicit stack, not recursion, so it can't stack-
         overflow on a pathologically long dependency chain), unioning
         in direct_backing at every node reached along the way, plus
         escaping_arrays found directly. The result is exactly the set
         of array declarations that need to survive past this
         function's own return.
    """
    analyzer = EscapeAnalyzer(fn, param_types, structs, aliases)
    return analyzer.analyze()
