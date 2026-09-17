"""The real-IR dispatch layer: gen_expr_ir inspects an expression
node's type or kind and hands off to whichever feature file
(arrays_slices, structs, strings, scalars) owns that case's own real-
IR construction, building an IRValue rather than emitting Instructions
directly. Also holds _ir_load/_ir_composite_operand_address/_ir_expr_
binary/_ir_binary, the shared leaves gen_expr_ir's own Index/Field/
Binary cases are built from.

Used to also hold gen_expr_into and gen_binary_into, the old-style
counterparts this class's own docstring used to describe -- both
removed entirely as part of this arc's old-style dead-code cleanup
(see ir.py's own top docstring), since nothing called either of them
anymore once gen_expr_ir itself had a real-IR case for everything
that's actually reachable (see gen_expr_ir's own docstring for exactly
which shapes those are, and why the ones that aren't are provably
unreachable rather than merely untested)."""

from codegen.errors import CodegenError
from codegen.ir import (
    IRBinOp, IRValue, IRConst, IRLoad, IRMove, IRJump, IRLabel, IRStaticDataAddress, IRUnOp, IRCast
)
from codegen.utils import type_of
from typing import Optional
from parser import (
    ArrayLiteral,
    Binary,
    BinaryOp,
    BoolLiteral,
    Call,
    Cast,
    Constant,
    Field,
    Index,
    Node,
    StringLiteral,
    Unary,
    Variable,
)
from semantic import Type, TypeKind


class DispatchMixin:

    def gen_expr_ir(self, expr: Node) -> tuple[list, Optional[IRValue]]:
        """Builds real IR for the node kinds that have it -- a bare
        Constant/BoolLiteral (see below), a scalar Variable (see
        below), a scalar-typed Index/Field read (address and load both
        real IR now -- see _ir_index_address/_ir_field_address for the
        address, _ir_load for the load through it), Binary (see
        _ir_expr_binary), a len(x) call (see _ir_len_call, its own
        dedicated case, not routed through _ir_call
        at all -- len isn't an ordinary function call), and an
        ordinary scalar-or-void-returning Call (see _ir_call) -- and
        raises CodegenError explicitly for everything else, with no
        fallback left to defer to (see ir.py's own module docstring
        for IRRaw's own removal, once this was the last remaining
        site).

        Every other node kind reaching here is a genuine bug, not a
        deliberate scope boundary anymore -- see the note below on
        exactly why ArrayLiteral/Slice/NoneLiteral/a composite-
        returning Call never actually reach this point in practice.

        This method does NOT "cover" an ArrayLiteral, Slice,
        NoneLiteral, or a composite-returning Call the way an earlier
        version of this docstring claimed -- when the old-style
        fallback still existed here, gen_expr_into itself defensively
        REJECTED every one of those (they don't fit in a single
        register), so reaching that fallback with one of them crashed,
        it didn't handle it. A REAL BUG, found this way: a bare `none`
        or a bare, discarded composite-returning Call used directly as
        a statement (`none` or `makeArray()` alone on a line) used to
        crash outright. The actual fix lives one level up, in
        gen_statement_ir's own ExprStmt dispatch -- explicit
        NoneLiteral/append/ordinary-composite-Call cases there route
        around this method entirely for exactly those shapes, the
        same "handle it before it ever reaches the generic case" shape
        that method's own ArrayLiteral/Slice cases already used. An
        ArrayLiteral or Slice reaching this method directly (not via a
        bare ExprStmt) still isn't reachable in practice: every other
        AST position either has its own, earlier real-IR case (a
        VarDecl's own initializer, a function-call argument, ...) or
        is one semantic.py itself doesn't allow these two shapes to
        appear in at all.

        Returns (ir, value) -- value is None only for a void call,
        which can only legally appear via a bare ExprStmt (see
        gen_statement_ir), never as another expression's operand.

        A Constant/BoolLiteral or a Variable reference each cost ZERO
        instructions here -- the former is just IRConst, a compile-
        time value IR already had a case for since it was first
        designed (see ir.py), simply never actually produced until
        now: this replaces what used to be the old-style gen_expr_
        into's own Mov-emitting Constant/BoolLiteral case, producing
        an IRConst that lowers to the identical Mov later instead.
        The latter's `expr.name` already has its own persistent Temp
        (see _bind_local), so reading it is just handing back that
        same Temp -- materializing either kind of value into a real
        register only happens later, lazily, wherever something
        actually needs it. Never reached for a composite-typed
        Variable (array/slice/struct): every real-IR caller of this
        whole class (gen_statement_ir's own cases, _ir_array_address,
        _ir_struct_address, _ir_slice_address, ...) already has its
        own dedicated handling for a composite-typed Variable before
        this method could ever be reached with one at all."""
        if isinstance(expr, Constant):
            return [], IRConst(expr.value, type_of(expr))
        if isinstance(expr, BoolLiteral):
            return [], IRConst(1 if expr.value else 0, Type.BOOL)
        if isinstance(expr, StringLiteral):
            # A static .data label's own address -- IRStaticDataAddress
            # directly, not the old-style gen_string_literal_into's own
            # IRRaw-wrapped LeaQ: this IS exactly the leaf
            # IRStaticDataAddress exists for (see its own docstring),
            # just never converted until
            # now. Registers this literal's own content for later
            # emission the identical way the old-style version used
            # to -- a fresh label per occurrence, even for
            # identical content, no deduplication -- just inlined here
            # rather than delegating to that method for what's now a
            # two-line operation with no old-style instruction sequence
            # left to wrap at all.
            t = self._new_temp(Type.STR)
            label = self.new_label("str")
            self.string_literals.append((label, expr.value))
            return [IRStaticDataAddress(dst=t, label=label)], t
        if isinstance(expr, Variable):
            return [], self._local_temp(expr.name)
        if isinstance(expr, Index) and type_of(expr).kind not in (TypeKind.ARRAY, TypeKind.STRUCT):
            result = self._ir_index_address(expr)
            if result is None:
                raise CodegenError(
                    f"_ir_index_address returned None for a scalar-typed Index read "
                    f"({expr!r}) -- expected to always succeed for a reachable base")
            addr_ir, addr_value = result
            return self._ir_load(addr_ir, addr_value, type_of(expr))
        if isinstance(expr, Field) and type_of(expr).kind not in (TypeKind.ARRAY, TypeKind.SLICE, TypeKind.STRUCT):
            result = self._ir_field_address(expr)
            if result is None:
                raise CodegenError(
                    f"_ir_field_address returned None for a scalar-typed Field read "
                    f"({expr!r}) -- expected to always succeed for a reachable base")
            addr_ir, addr_value = result
            return self._ir_load(addr_ir, addr_value, type_of(expr))
        if isinstance(expr, Binary):
            return self._ir_expr_binary(expr)
        if isinstance(expr, Call) and expr.name == 'print':
            # A dedicated case, not routed through _ir_call at all --
            # print isn't an ordinary function call (it always needs
            # an address for its argument, unlike an ordinary call's
            # own by-value/by-address split, and its second
            # argument -- a type descriptor -- has no corresponding
            # Hornet expression to run gen_expr_ir on at all), so it
            # needs its own entry point the same way len's own
            # exclusion from _ir_call already does. Never falls
            # through to the catch-all below -- unlike len,
            # _ir_print_call's own contract is total, matching
            # semantic.py's own guarantee that this is always exactly
            # one, well-formed argument by the time codegen ever sees
            # it.
            return self._ir_print_call(expr)
        if isinstance(expr, Call) and expr.name == 'len':
            # A dedicated case, not routed through _ir_call at all --
            # len isn't an ordinary function call (no calling
            # convention, no argument-register placement), so it
            # needs its own entry point the same way print's own
            # exclusion from _ir_call already does. Falls through to
            # the ordinary catch-all below when out of scope (see
            # _ir_len_call's own docstring) -- _ir_call's own name
            # exclusion ('len', alongside 'print') already keeps that
            # dispatch from ever re-attempting this as an ordinary
            # call.
            result = self._ir_len_call(expr)
            if result is not None:
                return result
        if isinstance(expr, Call) and expr.name not in ('print', 'len') and type_of(expr).kind not in (
                TypeKind.ARRAY, TypeKind.SLICE, TypeKind.STRUCT):
            return self._ir_call(expr)
        if isinstance(expr, Unary):
            # IRUnOp already existed, fully lowered (via gen_unary_op,
            # the same already-proven old-style function IRBinOp's own
            # lowering already reuses for binary operators) and
            # already register-allocator-supported -- it was just
            # never constructed anywhere. Recursing into expr.operand
            # FIRST is what makes a chained operator (`~-2`) compose
            # correctly: the inner Unary node builds its own IRUnOp
            # through this identical case, so the outer one's own
            # operand is already an ordinary Temp by the time it runs,
            # no different from any other nested expression in this
            # arc.
            operand_ir, operand_value = self.gen_expr_ir(expr.operand)
            t = self._new_temp(type_of(expr))
            return operand_ir + [IRUnOp(dst=t, op=expr.op, operand=operand_value)], t
        if isinstance(expr, Cast):
            # IRCast is new -- see its own docstring for why an
            # ordinary IRMove into a differently-typed Temp doesn't
            # suffice here, unlike every other "produce a value of a
            # specific type" case in this arc. Lowering reuses gen_
            # cast_narrowing_into completely unchanged, the identical
            # "wrap the one proven old-style helper, don't reimplement
            # it" shape IRUnOp's own lowering already uses for gen_
            # unary_op.
            src_ir, src_value = self.gen_expr_ir(expr.expr)
            t = self._new_temp(type_of(expr))
            return src_ir + [IRCast(dst=t, src=src_value)], t
        raise CodegenError(
            f"No real-IR case for expression of type {type(expr).__name__}: {expr!r} -- "
            f"every language construct this arc's own tests exercise (including every "
            f"shape print() can take, the last remaining user of what used to be this "
            f"method's own IRRaw-wrapped fallback) is confirmed to reach real IR without "
            f"ever falling back here. A genuine bug if this fires, not a deliberate scope "
            f"boundary -- there is no old-style fallback left to silently defer to anymore."
        )

    def _ir_load(self, addr_ir: list, addr_value, value_type) -> tuple[list, IRValue]:
        """Shared by gen_expr_ir's Index/Field cases: given an
        address already built as real IR (see _ir_index_address/_ir_
        field_address, this method's own two callers), IRLoads
        value_type's own width through it. This method itself has no
        fallback of any kind -- it's a pure two-instruction leaf over
        whatever address its caller already produced."""
        t = self._new_temp(value_type)
        return addr_ir + [IRLoad(dst=t, address=addr_value)], t

    def _ir_composite_operand_address(self, expr: Node, value_type: Type):
        """Builds (without lowering) an ARRAY- or STRUCT-typed
        equality operand's own address as real IR -- returns (ir,
        address), or None when out of scope. Both of _ir_expr_binary's
        own equality operands need this identical dispatch, so it's
        factored out here rather than duplicated inline the way _ir_
        indexable_base/_ir_call_arguments each dispatch their own,
        different single operand -- this one genuinely needs the same
        logic run twice within one call.

        In dispatch order: a Variable/Field/Index (an existing
        address, via _ir_array_address/_ir_struct_address); a bare
        bracketed-list literal (ARRAY only -- _ir_materialize_array_
        literal; there is no SLICE/STRUCT equivalent, since a struct
        literal can never be compared this way at all -- see below);
        an ordinary composite-returning Call (_ir_materialize_
        composite_call, shared by both ARRAY and STRUCT).

        A struct-literal Call needs no case here, unlike the ARRAY
        side's own ArrayLiteral one: semantic.py already rejects a
        struct literal as a Binary operand outright (`Point(1,2) ==
        Point(1,2)` fails to type-check, naming the same narrow
        whitelist of allowed positions this arc has already run into
        at a Field/Index/Slice base) -- there is no gap here for
        codegen to close, the identical situation _ir_materialize_
        array_literal's own docstring already documents for those
        other base positions.

        A REAL BUG, found and fixed here rather than carried forward:
        `[1,2,3] == [1,2,4]` and an ordinary array/struct-returning
        call used directly as an equality operand both used to raise
        a hard CodegenError, tracing back to the OLD-style gen_array_
        address_into itself -- which never handled an ArrayLiteral or
        Call operand at all. This was never supported, even old-style,
        the identical situation a composite-returning call used
        directly as an addressable base was in before this arc's own
        earlier work there."""
        if isinstance(expr, (Variable, Field, Index)):
            address_fn = self._ir_array_address if value_type.kind == TypeKind.ARRAY else self._ir_struct_address
            return address_fn(expr)
        if value_type.kind == TypeKind.ARRAY and isinstance(expr, ArrayLiteral):
            return self._ir_materialize_array_literal(expr)
        if self._is_ordinary_composite_call(expr):
            return self._ir_materialize_composite_call(expr, value_type)
        return None

    def _ir_expr_binary(self, expr: Binary) -> tuple[list, IRValue]:
        """Builds real IR for every Binary shape this compiler
        supports: short-circuit AND/OR (already real IR, via
        _ir_short_circuit), slice-vs-none comparison (real IR too, for
        every reachable slice-typed base shape, confirmed
        exhaustively -- see _ir_slice_none_comparison's own docstring;
        a None here, genuinely never observed, raises CodegenError
        rather than silently propagating further up the call chain),
        string concat/compare (also real IR now, via
        _ir_string_concat/_ir_string_compare -- composed entirely from
        ordinary IRCall/IRBinOp, since IRCall's own lowering is
        already generic over any callee name, external C library
        functions included, with no new IR concept needed), array/
        struct equality (now real IR too, for a Variable/Field/Index,
        a bare bracketed-list literal (ARRAY only), or an ordinary
        composite-returning Call, in any combination on either side --
        see _ir_composite_operand_address/_ir_composite_equal; a
        struct-literal Call is the one shape genuinely still out of
        scope here, moot in practice since semantic.py already rejects
        it as a Binary operand outright, not just here -- so once this
        branch is entered at all, i.e. type_of(expr.left).kind is
        ARRAY/STRUCT, both operands' own addresses are expected to
        always resolve; a CodegenError, not a silent fall-through to
        the ordinary scalar case just below, is what a None here now
        gets), or the
        ordinary arithmetic/comparison case (already real IR, via
        _ir_binary)."""
        if expr.op == BinaryOp.AND:
            return self._ir_short_circuit(expr, short_circuit_value=0, label_prefix="and")
        if expr.op == BinaryOp.OR:
            return self._ir_short_circuit(expr, short_circuit_value=1, label_prefix="or")
        if type_of(expr.left) == Type.STR:
            if expr.op == BinaryOp.ADD:
                return self._ir_string_concat(expr)
            if expr.op in (BinaryOp.EQUAL, BinaryOp.NOT_EQUAL):
                return self._ir_string_compare(expr)
        if expr.op in (BinaryOp.EQUAL, BinaryOp.NOT_EQUAL):
            if type_of(expr.left).kind == TypeKind.SLICE or type_of(expr.right).kind == TypeKind.SLICE:
                result = self._ir_slice_none_comparison(expr)
                if result is None:
                    raise CodegenError(
                        f"_ir_slice_none_comparison returned None for a slice-vs-"
                        f"none comparison ({expr.left!r} {expr.op} {expr.right!r}) "
                        f"-- expected to always succeed for a reachable slice-typed "
                        f"base, with no old-style fallback remaining to catch it")
                return result
            if type_of(expr.left).kind in (TypeKind.ARRAY, TypeKind.STRUCT):
                value_type = type_of(expr.left)
                left_result = self._ir_composite_operand_address(expr.left, value_type)
                right_result = self._ir_composite_operand_address(expr.right, value_type)
                if left_result is None or right_result is None:
                    raise CodegenError(
                        f"_ir_composite_operand_address returned None for an "
                        f"ARRAY/STRUCT equality operand ({expr.left!r} or "
                        f"{expr.right!r}) -- expected to always succeed once "
                        f"type_of(expr.left).kind is ARRAY/STRUCT at all, since a "
                        f"struct-literal Call (the one shape it doesn't cover) is "
                        f"already rejected as a Binary operand by semantic.py before "
                        f"codegen ever runs")
                left_ir, left_addr = left_result
                right_ir, right_addr = right_result
                mismatch_label = self.new_label("eq_mismatch")
                done_label = self.new_label("eq_done")
                cmp_ir = self._ir_composite_equal(left_addr, right_addr, value_type, mismatch_label)
                t = self._new_temp(Type.BOOL)
                ir = left_ir + right_ir + cmp_ir + [
                    IRMove(dst=t, src=IRConst(1 if expr.op == BinaryOp.EQUAL else 0, Type.BOOL)),
                    IRJump(done_label),
                    IRLabel(mismatch_label),
                    IRMove(dst=t, src=IRConst(0 if expr.op == BinaryOp.EQUAL else 1, Type.BOOL)),
                    IRLabel(done_label),
                ]
                return ir, t
        return self._ir_binary(expr)


    def _ir_binary(self, expr: Binary) -> tuple[list, object]:
        """Builds (without lowering) the ordinary arithmetic/comparison
        case's IR: evaluate each operand (via gen_expr_ir, so a nested
        migrated sub-expression -- another Binary, a scalar Call --
        stays real IR instead of being immediately, separately
        lowered), combine via IRBinOp. lower_ir reuses gen_binary_op
        unchanged as this op's own instruction-selection rule, so the
        arithmetic itself isn't reimplemented here. Returns
        (ir, t_result)."""
        left_ir, left_value = self.gen_expr_ir(expr.left)
        right_ir, right_value = self.gen_expr_ir(expr.right)
        t_result = self._new_temp(type_of(expr))
        ir = left_ir + right_ir + [
            IRBinOp(dst=t_result, op=expr.op, left=left_value, right=right_value),
        ]
        return ir, t_result
