"""Sum types: 4-byte tag, then a payload sized to the widest variant.
No constructor syntax: a variant widens to a sum on assignment.
"""

from ir.errors import IRError
from ir.ir import IRBinOp, IRCall, IRConst, IRLoad, IRLocalAddress, IRStore
from ir.utils import COMPOSITE_KINDS, type_of
from typesys import SUM_TYPE_TAG_WIDTH, type_byte_width
from parser import IsCheck, Node, Variable
from ops import BinaryOp
from typesys import Type


class SumTypesMixin:
    def _ir_write_sum_type_value_into(self, dst_address, value_expr: Node, sum_type: Type):
        """Write tag then payload for a widening into dst_address."""
        source_type = type_of(value_expr)
        variants = self.ir_program.sum_type_registry[sum_type.sum_type_name].variants
        discriminant = variants.index(source_type)

        tag_ir = [IRStore(address=dst_address, value=IRConst(discriminant, Type.INT32), value_type=Type.INT32)]

        payload_addr = self.ir_program.ids.new_temp(Type.INT64)
        payload_addr_ir = [IRBinOp(
            dst=payload_addr, op=BinaryOp.ADD,
            left=dst_address, right=IRConst(SUM_TYPE_TAG_WIDTH, Type.INT64),
        )]

        if source_type.kind in COMPOSITE_KINDS:
            payload_ir = self._ir_write_composite_value_into(payload_addr, value_expr, source_type)
            if payload_ir is None:
                return None
        else:
            value_ir, value = self.gen_expr_ir(value_expr)
            payload_ir = value_ir + [IRStore(address=payload_addr, value=value, value_type=source_type)]
        return tag_ir + payload_addr_ir + payload_ir

    def _ir_materialize_sum_type_value(self, expr: Node, sum_type: Type):
        """Materialize a widened variant; returns (ir, address)."""
        if id(expr) in self._argument_temp_slots:
            slot = self._argument_temp_slots[id(expr)]
            addr = self.ir_program.ids.new_temp(Type.INT64)
            addr_ir = [IRLocalAddress(dst=addr, slot=slot)]
        else:
            addr = self.ir_program.ids.new_temp(Type.INT64)
            size = type_byte_width(sum_type, self.ir_program.struct_registry, self.ir_program.sum_type_registry)
            addr_ir = [IRCall(dst=addr, name='malloc', args=[IRConst(size, Type.INT64)])]
        write_ir = self._ir_write_sum_type_value_into(addr, expr, sum_type)
        if write_ir is None:
            return None
        return addr_ir + write_ir, addr

    def _ir_is_check(self, expr: IsCheck) -> tuple[list, object]:
        """`x is T`: compare the tag."""
        sum_type = self._local_type(expr)
        result = self._ir_struct_address(Variable(name=expr.variable_name, decl_id=expr.decl_id))
        if result is None:
            raise IRError(
                f"_ir_struct_address returned None for an IsCheck's own variable "
                f"({expr.variable_name!r}) -- expected to always succeed for a "
                f"reachable, sum-typed variable"
            )
        addr_ir, addr_value = result
        tag_temp = self.ir_program.ids.new_temp(Type.INT32)
        load_ir = [IRLoad(dst=tag_temp, address=addr_value)]

        variants = self.ir_program.sum_type_registry[sum_type.sum_type_name].variants
        narrowed_type = expr.narrowed_type
        discriminant = variants.index(narrowed_type)

        result_temp = self.ir_program.ids.new_temp(Type.BOOL)
        compare_ir = [IRBinOp(dst=result_temp, op=BinaryOp.EQUAL, left=tag_temp, right=IRConst(discriminant, Type.INT32))]
        return addr_ir + load_ir + compare_ir, result_temp
