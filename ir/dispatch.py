"""gen_expr_ir: dispatch expressions to the feature mixins."""

from ir.errors import IRError
from ir.ir import (
    IRBinOp,
    IRValue,
    IRConst,
    IRLoad,
    IRMove,
    IRJump,
    IRLabel,
    IRUnOp,
    IRCast,
)
from ir.utils import COMPOSITE_KINDS, is_composite_addressable, type_of
from typing import Optional
from parser import (
    ArrayLiteral,
    Binary,
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
    Unary,
    Variable,
)
from ops import BinaryOp, UnaryOp
from typesys import Type, TypeKind


class DispatchMixin:
    def gen_expr_ir(self, expr: Node) -> tuple[list, Optional[IRValue]]:
        """Build IR for an expression; returns (ir, value)."""
        if isinstance(expr, Constant):
            return [], IRConst(expr.value, type_of(expr))
        if isinstance(expr, ByteLiteral):
            return [], IRConst(expr.value, Type.UINT8)
        if isinstance(expr, BoolLiteral):
            return [], IRConst(1 if expr.value else 0, Type.BOOL)
        if isinstance(expr, Variable):
            slot_type = self._local_type(expr)
            if slot_type.kind == TypeKind.SUM and expr.resolved_type is not None and expr.resolved_type != slot_type:
                # A narrowed sum variable's Temp holds the whole sum; load the payload through its address.
                addr_ir, addr_value = self._ir_struct_address(expr, payload=True)
                return self._ir_load(addr_ir, addr_value, expr.resolved_type)
            if self._is_heap_allocated(self._local_decl_id(expr), slot_type):
                # Heap-promoted: the Temp holds a pointer.
                return self._ir_load([], self._local_temp(expr), slot_type)
            return [], self._local_temp(expr)
        if isinstance(expr, Index) and type_of(expr.array).kind == TypeKind.STR:
            # str indexing, checked before the generic scalar index.
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
            return self._ir_print_call(expr)
        if isinstance(expr, Call) and expr.name == 'del':
            return self._ir_del_call(expr)
        if isinstance(expr, Call) and expr.name == 'len':
            result = self._ir_len_call(expr)
            if result is not None:
                return result
        if isinstance(expr, Call) and self.ir_program.intrinsic_original_names.get(expr.name) in ('_raw_ptr', '_raw_len'):
            # Intrinsics lower inline.
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
            # null pointer
            return [], IRConst(0, Type.INT64)
        raise IRError(
            f"No real-IR case for expression of type {type(expr).__name__}: {expr!r}"
        )

    def _ir_load(self, addr_ir: list, addr_value, value_type) -> tuple[list, IRValue]:
        """Load value_type through an address."""
        t = self.ir_program.ids.new_temp(value_type)
        return addr_ir + [IRLoad(dst=t, address=addr_value)], t

    def _ir_composite_operand_address(self, expr: Node, value_type: Type):
        """Address of an array/struct equality operand, or None."""
        if is_composite_addressable(expr):
            address_fn = self._ir_array_address if value_type.kind == TypeKind.ARRAY else self._ir_struct_address
            return address_fn(expr)
        if value_type.kind == TypeKind.ARRAY and isinstance(expr, ArrayLiteral):
            return self._ir_materialize_array_literal(expr)
        if value_type.kind == TypeKind.STRUCT and isinstance(expr, Call) and expr.name in self.ir_program.struct_registry:
            return self._ir_materialize_struct_literal(expr)
        if self._is_ordinary_composite_call(expr):
            return self._ir_materialize_composite_call(expr, value_type)
        return None

    def _ir_expr_binary(self, expr: Binary) -> tuple[list, IRValue]:
        """IR for any Binary."""
        if expr.op == BinaryOp.AND:
            return self._ir_short_circuit(expr, short_circuit_value=0, label_prefix="and")
        if expr.op == BinaryOp.OR:
            return self._ir_short_circuit(expr, short_circuit_value=1, label_prefix="or")
        if expr.op == BinaryOp.IN:
            if type_of(expr.right).kind == TypeKind.DICT:
                return self._ir_dict_contains(expr.left, expr.right, type_of(expr.right))
            return self._ir_array_slice_contains(expr.left, expr.right, type_of(expr.right).element_type)
        if type_of(expr.left) == Type.STR:
            # ADD on str is handled elsewhere: its result is composite.
            if expr.op in (BinaryOp.EQUAL, BinaryOp.NOT_EQUAL):
                return self._ir_string_compare(expr)
        if expr.op in (BinaryOp.EQUAL, BinaryOp.NOT_EQUAL):
            if Type.NONE in (type_of(expr.left), type_of(expr.right)) and TypeKind.SUM in (
                    type_of(expr.left).kind, type_of(expr.right).kind):
                return self._ir_sum_none_comparison(expr)
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
        """Arithmetic/comparison via IRBinOp; returns (ir, t_result)."""
        left_ir, left_value = self.gen_expr_ir(expr.left)
        right_ir, right_value = self.gen_expr_ir(expr.right)
        t_result = self.ir_program.ids.new_temp(type_of(expr))
        ir = left_ir + right_ir + [
            IRBinOp(dst=t_result, op=expr.op, left=left_value, right=right_value),
        ]
        return ir, t_result
