"""Pointer IR: `&x` and `*p`."""

from ir.errors import IRError
from ir.ir import IRBinOp, IRConst, IRLoad, IRLocalAddress, IRStore, IRValue
from ir.utils import COMPOSITE_KINDS, type_of
from typesys import SUM_TYPE_TAG_WIDTH
from parser import Call, DerefAssign, Field, Index, Unary, Variable
from ops import BinaryOp
from typesys import Type, TypeKind


class PointersMixin:
    def _ir_address_of(self, expr: Unary) -> tuple[list, IRValue]:
        """`&x` for locals, struct literals, and field/index chains."""
        if isinstance(expr.operand, Call):
            result = self._ir_materialize_struct_literal(expr.operand)
            if result is None:
                raise IRError(
                    f"_ir_materialize_struct_literal returned None for a "
                    f"struct-literal '&' operand ({expr.operand!r}) -- "
                    f"expected to always succeed for a reachable shape"
                )
            return result
        if isinstance(expr.operand, Field):
            result = self._ir_field_address(expr.operand)
            if result is None:
                raise IRError(
                    f"_ir_field_address returned None for a Field '&' "
                    f"operand ({expr.operand!r}) -- expected to always "
                    f"succeed for a reachable shape"
                )
            return result
        if isinstance(expr.operand, Index):
            result = self._ir_index_address(expr.operand)
            if result is None:
                raise IRError(
                    f"_ir_index_address returned None for an Index '&' "
                    f"operand ({expr.operand!r}) -- expected to always "
                    f"succeed for a reachable shape"
                )
            return result
        if not isinstance(expr.operand, Variable):
            raise IRError(
                f"IRError: '&' operand is {type(expr.operand).__name__}, not a "
                f"Variable -- semantic.py's own check_unary should have already "
                f"rejected this before real-IR generation ever runs"
            )
        slot = self._local_slot(expr.operand)
        slot_type = self._local_type(expr.operand)
        slot_addr = self.ir_program.ids.new_temp(type_of(expr))
        ir = [IRLocalAddress(dst=slot_addr, slot=slot)]
        if self._is_heap_allocated(self._local_decl_id(expr.operand), slot_type):
            addr_temp = self.ir_program.ids.new_temp(type_of(expr))
            ir.append(IRLoad(dst=addr_temp, address=slot_addr))
            base_addr = addr_temp
        else:
            base_addr = slot_addr
        narrowed_type = expr.operand.resolved_type
        if slot_type.kind == TypeKind.SUM and narrowed_type is not None and narrowed_type != slot_type:
            # &n of a narrowed sum is the payload's address.
            payload_addr = self.ir_program.ids.new_temp(type_of(expr))
            ir.append(IRBinOp(
                dst=payload_addr, op=BinaryOp.ADD,
                left=base_addr, right=IRConst(SUM_TYPE_TAG_WIDTH, Type.INT64),
            ))
            return ir, payload_addr
        return ir, base_addr

    def _ir_dereference(self, expr: Unary) -> tuple[list, IRValue]:
        """`*p`: load the pointee."""
        operand_ir, operand_value = self.gen_expr_ir(expr.operand)
        return self._ir_load(operand_ir, operand_value, type_of(expr))

    def _ir_deref_assign(self, stmt: DerefAssign) -> list:
        """`*pointer = value`."""
        pointer_type = type_of(stmt.pointer)
        pointee_type = pointer_type.element_type
        ptr_ir, ptr_value = self.gen_expr_ir(stmt.pointer)

        if pointee_type.kind not in COMPOSITE_KINDS:
            if stmt.compound_op is not None:
                return ptr_ir + self._ir_compound_assign_through_address(ptr_value, stmt.compound_op, stmt.value, pointee_type)
            value_ir, value = self.gen_expr_ir(stmt.value)
            return ptr_ir + value_ir + [IRStore(address=ptr_value, value=value, value_type=pointee_type)]

        write_ir = self._ir_write_composite_value_into(ptr_value, stmt.value, pointee_type)
        if write_ir is not None:
            return ptr_ir + write_ir
        raise IRError(
            f"No real-IR case for DerefAssign with pointee kind "
            f"{pointee_type.kind} and value {stmt.value!r}"
        )
