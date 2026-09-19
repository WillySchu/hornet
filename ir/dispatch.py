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
from ir.utils import COMPOSITE_KINDS, type_of
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
        Constant/BoolLiteral/StringLiteral, a scalar Variable, a
        scalar-typed Index/Field read (via _ir_index_address/_ir_
        field_address plus _ir_load), Binary (via _ir_expr_binary), a
        len(x)/print(x) call (their own dedicated entry points, not
        routed through _ir_call -- neither is an ordinary function
        call), an ordinary scalar-or-void-returning Call (via _ir_
        call), Unary, and Cast -- and raises IRError for everything
        else.

        An ArrayLiteral, Slice, NoneLiteral, or a composite-returning
        Call used as a bare statement is routed around this method
        entirely by gen_statement_ir's own ExprStmt dispatch; in any
        other position, each already has its own earlier real-IR case
        (a VarDecl initializer, a function-call argument, ...) or is a
        shape semantic.py doesn't allow there at all -- so none of the
        four actually reach this method's own IRError in practice.

        Returns (ir, value) -- value is None only for a void call,
        which can only legally appear via a bare ExprStmt, never as
        another expression's operand.

        A Constant/BoolLiteral or a Variable reference each cost ZERO
        instructions here: the former is just IRConst, a compile-time
        value; the latter's `expr.name` already has its own persistent
        Temp (see _bind_local), so reading it just hands back that
        Temp. Never reached for a composite-typed Variable (array/
        slice/struct) -- every real-IR caller of this class already
        has its own dedicated handling for one before this method
        could be reached with it."""
        if isinstance(expr, Constant):
            return [], IRConst(expr.value, type_of(expr))
        if isinstance(expr, BoolLiteral):
            return [], IRConst(1 if expr.value else 0, Type.BOOL)
        if isinstance(expr, StringLiteral):
            # A fresh label per occurrence, even for identical content
            # -- no deduplication.
            t = self.ir_program.ids.new_temp(Type.STR)
            label = self.ir_program.ids.new_label("str")
            self.ir_program.string_literals.append((label, expr.value))
            return [IRStaticDataAddress(dst=t, label=label)], t
        if isinstance(expr, Variable):
            return [], self._local_temp(expr.name)
        if isinstance(expr, Index) and type_of(expr).kind not in (TypeKind.ARRAY, TypeKind.STRUCT):
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
        if isinstance(expr, Call) and expr.name not in ('print', 'len') and type_of(expr).kind not in COMPOSITE_KINDS:
            return self._ir_call(expr)
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
            if expr.op == BinaryOp.ADD:
                return self._ir_string_concat(expr)
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
