"""The real-IR dispatch layer: gen_expr_ir inspects an expression
node's type or kind and hands off to whichever feature file
(arrays_slices, structs, strings, scalars) owns that case's own real-
IR construction, building an IRValue rather than emitting Instructions
directly. Also holds _ir_load/_ir_composite_operand_address/_ir_expr_
binary/_ir_binary, the shared leaves gen_expr_ir's own Index/Field/
Binary cases are built from."""

from ir.errors import IRError
from ir.ir import (
    IRBinOp, IRValue, IRConst, IRLoad, IRMove, IRJump, IRLabel, IRStaticDataAddress, IRUnOp, IRCast
)
from ir.utils import COMPOSITE_KINDS, is_composite_addressable, type_of
from typing import Optional
from parser import (
    ArrayLiteral,
    Binary,
    BinaryOp,
    BoolLiteral,
    ByteLiteral,
    Call,
    Cast,
    Constant,
    Field,
    Index,
    IsCheck,
    Node,
    NoneLiteral,
    StringLiteral,
    Unary,
    UnaryOp,
    Variable,
)
from semantic import Type, TypeKind


class DispatchMixin:
    def gen_expr_ir(self, expr: Node) -> tuple[list, Optional[IRValue]]:
        """Builds real IR for the node kinds that have it -- a bare
        Constant/BoolLiteral, a scalar Variable, a scalar-typed Index/
        Field read (via _ir_index_address/_ir_field_address plus _ir_
        load), Binary (via _ir_expr_binary), a len(x)/print(x) call
        (their own dedicated entry points, not routed through _ir_
        call -- neither is an ordinary function call), an ordinary
        scalar-or-void-returning Call (via _ir_call), Unary, Cast,
        IsCheck (via _ir_is_check, ir/sum_types.py -- a sum type's own
        runtime discriminant test, never a general expression; see
        IsCheck's own docstring in parser.py), and ADDRESS_OF/
        DEREFERENCE (via _ir_address_of/_ir_dereference, ir/
        pointers.py -- checked ahead of the generic Unary case just
        below, since neither shape fits it: ADDRESS_OF never evaluates
        its own operand as a value at all, and DEREFERENCE needs an
        IRLoad, not an IRUnOp) -- and raises IRError for everything
        else.

        An ArrayLiteral, Slice, NoneLiteral, StringLiteral, or a
        composite-returning Call used as a bare statement is routed
        around this method entirely by gen_statement_ir's own ExprStmt
        dispatch; in any other position, each already has its own
        earlier real-IR case (a VarDecl initializer, a function-call
        argument, ...) or is a shape semantic.py doesn't allow there
        at all -- so none of the five actually reach this method's own
        IRError in practice.

        Returns (ir, value) -- value is None only for a void call,
        which can only legally appear via a bare ExprStmt, never as
        another expression's operand.

        A Constant/BoolLiteral or a Variable reference each cost ZERO
        instructions here: the former is just IRConst, a compile-time
        value; the latter's `expr.name` already has its own persistent
        Temp (see _bind_local), so reading it just hands back that
        Temp. Never reached for a composite-typed Variable (array/
        slice/struct/str -- see ir/strings.py's own module docstring
        for why str joined this set): every real-IR caller of this
        class already has its own dedicated handling for one before
        this method could be reached with it -- a str-typed Variable/
        Field/Index/Binary(ADD)/Call goes through _ir_str_value (ir/
        strings.py) instead, exactly as a slice-typed one already goes
        through _ir_slice_arg rather than here."""
        if isinstance(expr, Constant):
            return [], IRConst(expr.value, type_of(expr))
        if isinstance(expr, ByteLiteral):
            # Exactly Constant's own shape just above -- expr.value is
            # already a resolved Python int (0-255), computed once at
            # parse time (see ByteLiteral's own docstring in parser.py
            # for why), so there's nothing left to compute here beyond
            # wrapping it in an IRConst at its own type (always UINT8).
            return [], IRConst(expr.value, Type.UINT8)
        if isinstance(expr, BoolLiteral):
            return [], IRConst(1 if expr.value else 0, Type.BOOL)
        if isinstance(expr, Variable):
            slot_type = self._local_type(expr.name)
            if slot_type.kind == TypeKind.SUM and expr.resolved_type is not None and expr.resolved_type != slot_type:
                # Narrowed-to-SCALAR (`if n is int: print(n)`): the
                # persistent Temp below holds the WHOLE sum-typed
                # value, not the scalar alone, so that fast path is
                # wrong here. _ir_struct_address's own narrowing
                # branch already computes the correct payload address
                # for any narrowed type; this just adds the load a
                # struct-narrowed occurrence doesn't need. str never
                # reaches here -- it has its own address/value path
                # (ir/strings.py) that never calls this method at all.
                addr_ir, addr_value = self._ir_struct_address(expr)
                return self._ir_load(addr_ir, addr_value, expr.resolved_type)
            if self._is_heap_allocated(self._local_decl_id(expr.name), slot_type):
                # This variable's own address escaped past this
                # function (see _bind_local/_bind_param's own,
                # identical check) -- its permanent Temp holds a
                # pointer to a malloc'd box, not the value directly,
                # so an ordinary read has to go THROUGH it. &x itself
                # (check_unary's own ADDRESS_OF case, ir/pointers.py's
                # own _ir_address_of) reads this exact Temp directly,
                # unchanged -- it's already the pointer this case
                # dereferences.
                return self._ir_load([], self._local_temp(expr.name), slot_type)
            return [], self._local_temp(expr.name)
        if isinstance(expr, Index) and type_of(expr.array).kind == TypeKind.STR:
            # Checked BEFORE the generic scalar-Index case just below,
            # on expr.array's own type (str), not type_of(expr) (the
            # RESULT type, always UINT8 here -- which that generic
            # case's own guard would already accept, since UINT8 isn't
            # composite, routing this into _ir_index_address, built
            # for an array/slice base's own addressing and wrong for
            # str's entirely different one -- see _ir_str_index_into's
            # own docstring in ir/strings.py).
            result = self._ir_str_index_into(expr)
            if result is None:
                raise IRError(
                    f"_ir_str_index_into returned None for a str-typed Index read "
                    f"({expr!r}) -- expected to always succeed for a reachable base")
            return result
        if isinstance(expr, Index) and type_of(expr).kind not in (TypeKind.ARRAY, TypeKind.STRUCT, TypeKind.STR):
            result = self._ir_index_address(expr)
            if result is None:
                raise IRError(
                    f"_ir_index_address returned None for a scalar-typed Index read "
                    f"({expr!r}) -- expected to always succeed for a reachable base")
            addr_ir, addr_value = result
            return self._ir_load(addr_ir, addr_value, type_of(expr))
        if isinstance(expr, Field) and type_of(expr).kind not in COMPOSITE_KINDS:
            result = self._ir_field_address(expr)
            if result is None:
                raise IRError(
                    f"_ir_field_address returned None for a scalar-typed Field read "
                    f"({expr!r}) -- expected to always succeed for a reachable base")
            addr_ir, addr_value = result
            return self._ir_load(addr_ir, addr_value, type_of(expr))
        if isinstance(expr, Binary):
            return self._ir_expr_binary(expr)
        if isinstance(expr, Call) and expr.name == 'print':
            # print always needs an address for its argument (unlike
            # an ordinary call's by-value/by-address split), and its
            # second argument -- a type descriptor -- has no
            # corresponding Hornet expression to run gen_expr_ir on.
            return self._ir_print_call(expr)
        if isinstance(expr, Call) and expr.name == 'len':
            # No calling convention, no argument-register placement --
            # falls through to the ordinary catch-all below when out
            # of scope (see _ir_len_call's own docstring); _ir_call's
            # own name exclusion keeps that dispatch from re-attempting
            # this as an ordinary call.
            result = self._ir_len_call(expr)
            if result is not None:
                return result
        if isinstance(expr, Call) and self.ir_program.intrinsic_original_names.get(expr.name) in ('_raw_ptr', '_raw_len'):
            # raw_ptr/raw_len -- see IntrinsicDecl's own docstring in
            # parser.py for the whole mechanism. Neither is an
            # ordinary call at all: no calling convention, no argument
            # register placement, no callee to jump to -- _ir_str_
            # value already computes exactly the (ptr, len) pair every
            # other string operation reads, so this just asks for that
            # same pair and hands back whichever half was requested,
            # discarding the other. expr.args[0] is str-typed (already
            # confirmed by semantic.py's own argument-count/type
            # checking against the registered intrinsic signature),
            # so this always succeeds -- unlike _ir_len_call just
            # above, there's no "out of scope, fall through" case here
            # to handle.
            original_name = self.ir_program.intrinsic_original_names[expr.name]
            arg_ir, ptr_value, len_value = self._ir_str_value(expr.args[0])
            return arg_ir, (ptr_value if original_name == '_raw_ptr' else len_value)
        if isinstance(expr, Call) and expr.name not in ('print', 'len') and self.ir_program.intrinsic_original_names.get(expr.name) is None and type_of(expr).kind not in COMPOSITE_KINDS:
            return self._ir_call(expr)
        if isinstance(expr, Unary) and expr.op == UnaryOp.ADDRESS_OF:
            return self._ir_address_of(expr)
        if isinstance(expr, Unary) and expr.op == UnaryOp.DEREFERENCE:
            return self._ir_dereference(expr)
        if isinstance(expr, Unary):
            # Recursing into expr.operand FIRST is what makes a
            # chained operator (`~-2`) compose correctly.
            operand_ir, operand_value = self.gen_expr_ir(expr.operand)
            t = self.ir_program.ids.new_temp(type_of(expr))
            return operand_ir + [IRUnOp(dst=t, op=expr.op, operand=operand_value)], t
        if isinstance(expr, Cast):
            src_ir, src_value = self.gen_expr_ir(expr.expr)
            t = self.ir_program.ids.new_temp(type_of(expr))
            return src_ir + [IRCast(dst=t, src=src_value)], t
        if isinstance(expr, IsCheck):
            return self._ir_is_check(expr)
        if isinstance(expr, NoneLiteral):
            # A pointer's own "null" is just the address 0 -- no
            # descriptor to build the way a nil SLICE needs (ptr, len,
            # cap all zero, via _ir_nil_slice), so no dedicated
            # special-casing is needed the many places a slice-typed
            # target's own `none` gets intercepted earlier (VarDecl/
            # Assign/IndexAssign/FieldAssign/Return/argument-passing,
            # all gated on TypeKind.SLICE specifically, in ir/
            # statements.py and ir/arrays_slices.py) -- those never
            # reach gen_expr_ir with a NoneLiteral at all. This is
            # purely the FALLBACK for contexts with no such
            # interception: a pointer-typed VarDecl/Assign initializer
            # (an ordinary scalar target, evaluated via gen_expr_ir
            # like any other), and pointer-vs-none equality (_ir_
            # binary's own generic case, once neither operand is
            # SLICE/ARRAY/STRUCT-kind -- see _ir_expr_binary's own
            # dispatch), where the ordinary IRBinOp(EQUAL, ...) this
            # produces already works correctly once BOTH operands are
            # real IRValues, with no pointer-specific comparison logic
            # needed at all.
            return [], IRConst(0, Type.INT64)
        raise IRError(
            f"No real-IR case for expression of type {type(expr).__name__}: {expr!r}"
        )

    def _ir_load(self, addr_ir: list, addr_value, value_type) -> tuple[list, IRValue]:
        """Shared by gen_expr_ir's Index/Field cases: given an address
        already built as real IR (see _ir_index_address/_ir_field_
        address), IRLoads value_type's own width through it."""
        t = self.ir_program.ids.new_temp(value_type)
        return addr_ir + [IRLoad(dst=t, address=addr_value)], t

    def _ir_composite_operand_address(self, expr: Node, value_type: Type):
        """Builds (without lowering) an ARRAY- or STRUCT-typed
        equality operand's own address as real IR -- returns (ir,
        address), or None when out of scope. Both of _ir_expr_binary's
        own equality operands need this identical dispatch, so it's
        factored out here.

        In dispatch order: a Variable/Field/Index (an existing
        address, via _ir_array_address/_ir_struct_address); a bare
        bracketed-list literal (ARRAY only -- _ir_materialize_array_
        literal; a struct literal can never be compared this way at
        all, since semantic.py rejects it as a Binary operand
        outright); an ordinary composite-returning Call (_ir_
        materialize_composite_call, shared by both ARRAY and
        STRUCT)."""
        if is_composite_addressable(expr):
            address_fn = self._ir_array_address if value_type.kind == TypeKind.ARRAY else self._ir_struct_address
            return address_fn(expr)
        if value_type.kind == TypeKind.ARRAY and isinstance(expr, ArrayLiteral):
            return self._ir_materialize_array_literal(expr)
        if self._is_ordinary_composite_call(expr):
            return self._ir_materialize_composite_call(expr, value_type)
        return None

    def _ir_expr_binary(self, expr: Binary) -> tuple[list, IRValue]:
        """Builds real IR for every Binary shape this compiler
        supports: short-circuit AND/OR (_ir_short_circuit), slice-vs-
        none comparison (_ir_slice_none_comparison), string concat/
        compare (_ir_string_concat/_ir_string_compare), array/struct
        equality (_ir_composite_operand_address/_ir_composite_equal,
        for a Variable/Field/Index, a bare bracketed-list literal
        (ARRAY only), or an ordinary composite-returning Call, in any
        combination on either side), or the ordinary arithmetic/
        comparison case (_ir_binary)."""
        if expr.op == BinaryOp.AND:
            return self._ir_short_circuit(expr, short_circuit_value=0, label_prefix="and")
        if expr.op == BinaryOp.OR:
            return self._ir_short_circuit(expr, short_circuit_value=1, label_prefix="or")
        if type_of(expr.left) == Type.STR:
            # ADD (concatenation) is deliberately NOT dispatched here:
            # its own result IS a str, a composite, multi-value type
            # now (see ir/strings.py's own module docstring) -- it can
            # no longer flow through gen_expr_ir's own single-value
            # contract at all, so _ir_str_value (ir/strings.py) is the
            # ONLY caller that ever reaches _ir_string_concat, never
            # this method. EQUAL/NOT_EQUAL stay here: a comparison's
            # own result is an ordinary bool regardless of its
            # operands' own type, a single value gen_expr_ir's
            # contract already covers correctly.
            if expr.op in (BinaryOp.EQUAL, BinaryOp.NOT_EQUAL):
                return self._ir_string_compare(expr)
        if expr.op in (BinaryOp.EQUAL, BinaryOp.NOT_EQUAL):
            if type_of(expr.left).kind == TypeKind.SLICE or type_of(expr.right).kind == TypeKind.SLICE:
                result = self._ir_slice_none_comparison(expr)
                if result is None:
                    raise IRError(
                        f"_ir_slice_none_comparison returned None for a slice-vs-"
                        f"none comparison ({expr.left!r} {expr.op} {expr.right!r}) "
                        f"-- expected to always succeed for a reachable slice-typed base")
                return result
            if type_of(expr.left).kind in (TypeKind.ARRAY, TypeKind.STRUCT):
                value_type = type_of(expr.left)
                left_result = self._ir_composite_operand_address(expr.left, value_type)
                right_result = self._ir_composite_operand_address(expr.right, value_type)
                if left_result is None or right_result is None:
                    raise IRError(
                        f"_ir_composite_operand_address returned None for an "
                        f"ARRAY/STRUCT equality operand ({expr.left!r} or "
                        f"{expr.right!r})")
                left_ir, left_addr = left_result
                right_ir, right_addr = right_result
                mismatch_label = self.ir_program.ids.new_label("eq_mismatch")
                done_label = self.ir_program.ids.new_label("eq_done")
                cmp_ir = self._ir_composite_equal(left_addr, right_addr, value_type, mismatch_label)
                t = self.ir_program.ids.new_temp(Type.BOOL)
                ir = left_ir + right_ir + cmp_ir + [
                    IRMove(dst=t, src=IRConst(1 if expr.op == BinaryOp.EQUAL else 0, Type.BOOL)),
                    IRJump(done_label),
                    IRLabel(mismatch_label),
                    IRMove(dst=t, src=IRConst(0 if expr.op == BinaryOp.EQUAL else 1, Type.BOOL)),
                    IRJump(done_label),
                    IRLabel(done_label),
                ]
                return ir, t
        return self._ir_binary(expr)

    def _ir_binary(self, expr: Binary) -> tuple[list, object]:
        """Builds (without lowering) the ordinary arithmetic/comparison
        case's IR: evaluate each operand via gen_expr_ir, combine via
        IRBinOp. Returns (ir, t_result)."""
        left_ir, left_value = self.gen_expr_ir(expr.left)
        right_ir, right_value = self.gen_expr_ir(expr.right)
        t_result = self.ir_program.ids.new_temp(type_of(expr))
        ir = left_ir + right_ir + [
            IRBinOp(dst=t_result, op=expr.op, left=left_value, right=right_value),
        ]
        return ir, t_result
