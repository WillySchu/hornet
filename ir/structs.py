"""Structs: fields at sequential unpadded offsets in declaration order; value semantics."""

from ir.errors import IRError
from ir.ir import IRBinOp, IRConst, IRStore, IRLoad, IRLocalAddress, IRCall
from ir.utils import COMPOSITE_KINDS, SUM_TYPE_TAG_WIDTH, type_byte_width, type_of
from parser import Node, Variable, Field, Index, Call, BinaryOp, Unary, UnaryOp
from semantic import TypeKind, Type


class StructsMixin:
    def _field_offset(self, struct_name: str, field_name: str) -> int:
        """Sum of preceding field widths."""
        offset = 0
        for name, field_type in self.ir_program.struct_registry[struct_name].fields.items():
            if name == field_name:
                return offset
            offset += type_byte_width(field_type, self.ir_program.struct_registry, self.ir_program.sum_type_registry)
        raise IRError(f"Struct '{struct_name}' has no field '{field_name}'")

    def _ir_struct_address(self, expr: Node) -> tuple[list, object]:
        """Address of a struct-typed expression."""
        if isinstance(expr, (Field, Index)) and expr.resolved_type is not None and expr.resolved_type.kind == TypeKind.POINTER:
            # Auto-deref a pointer-typed field base.
            return self.gen_expr_ir(expr)
        if isinstance(expr, Variable):
            var_type = self._local_type(expr.name)
            if var_type.kind == TypeKind.POINTER:
                # Auto-deref: use the pointer's value.
                return self.gen_expr_ir(expr)
            slot = self._local_slot(expr.name)
            struct_type = var_type
            slot_addr = self.ir_program.ids.new_temp(Type.INT64)
            ir = [IRLocalAddress(dst=slot_addr, slot=slot)]
            if self._is_heap_allocated(self._local_decl_id(expr.name), struct_type):
                addr_temp = self.ir_program.ids.new_temp(Type.INT64)
                ir.append(IRLoad(dst=addr_temp, address=slot_addr))
                base_addr = addr_temp
            else:
                base_addr = slot_addr
            if struct_type.kind == TypeKind.SUM:
                # resolved_type None marks an internal whole-value node.
                narrowed_type = expr.resolved_type
                if narrowed_type is None or narrowed_type == struct_type:
                    return ir, base_addr
                if narrowed_type not in self.ir_program.sum_type_registry[struct_type.sum_type_name].variants:
                    raise IRError(
                        f"Variable '{expr.name}' has sum type {struct_type} but its own "
                        f"resolved_type {narrowed_type} is neither that same sum type nor "
                        f"one of its own declared variants -- expected only these shapes "
                        f"for a sum-typed name"
                    )
                payload_addr = self.ir_program.ids.new_temp(Type.INT64)
                ir.append(IRBinOp(
                    dst=payload_addr, op=BinaryOp.ADD,
                    left=base_addr, right=IRConst(SUM_TYPE_TAG_WIDTH, Type.INT64),
                ))
                return ir, payload_addr
            return ir, base_addr
        if isinstance(expr, Field):
            return self._ir_field_address(expr)
        if isinstance(expr, Index):
            return self._ir_index_address(expr)
        if isinstance(expr, Unary) and expr.op == UnaryOp.DEREFERENCE:
            # `*p` as a whole struct.
            return self.gen_expr_ir(expr.operand)
        if self._is_ordinary_composite_call(expr):
            return self._ir_materialize_composite_call(expr, type_of(expr))
        if isinstance(expr, Call) and expr.name in self.ir_program.struct_registry:
            # Struct literal as a base.
            result = self._ir_materialize_struct_literal(expr)
            if result is None:
                raise IRError(
                    f"_ir_materialize_struct_literal returned None for a "
                    f"struct-literal Call used as a struct address "
                    f"({expr!r}) -- some field is out of scope for real "
                    f"IR, with no old-style fallback remaining to catch it")
            return result
        raise IRError(f"Cannot compute a struct address for: {expr!r}")

    def _ir_field_address(self, expr: Field) -> tuple[list, object]:
        """Address of expr.base.name."""
        base_type = type_of(expr.base)
        if base_type.kind == TypeKind.POINTER and base_type.element_type.kind == TypeKind.STRUCT:
            base_type = base_type.element_type
        if base_type.kind != TypeKind.STRUCT:
            raise IRError(
                f"Cannot access field '{expr.name}' on a value of "
                f"non-struct type {base_type}"
            )
        offset = self._field_offset(base_type.struct_name, expr.name)
        result = self._ir_struct_address(expr.base)
        if result is None:
            raise IRError(
                f"_ir_struct_address returned None for a Field's own STRUCT-typed "
                f"base ({expr.base!r}) -- expected to always succeed for a "
                f"reachable base")
        base_ir, base_addr = result
        if offset == 0:
            return base_ir, base_addr
        result = self.ir_program.ids.new_temp(Type.INT64)
        add_op = IRBinOp(dst=result, op=BinaryOp.ADD, left=base_addr, right=IRConst(offset, Type.INT64))
        return base_ir + [add_op], result

    def _ir_write_struct_literal_into(self, dst_address, expr: Call, struct_type: Type):
        """Write a struct literal's fields through dst_address."""
        struct_info = self.ir_program.struct_registry[struct_type.struct_name]
        field_items = list(struct_info.fields.items())
        if expr.kwargs is not None:
            provided = dict(expr.kwargs)
            entries = [(field_name, provided.get(field_name), field_type) for field_name, field_type in field_items]
        else:
            entries = [
                (field_name, arg_expr, field_type) for arg_expr, (field_name, field_type) in zip(expr.args, field_items)
            ]
        ir = []
        for field_name, arg_expr, field_type in entries:
            offset = self._field_offset(struct_type.struct_name, field_name)
            if arg_expr is None:
                # Omitted named field: zero value.
                if offset == 0:
                    field_addr = dst_address
                else:
                    field_addr = self.ir_program.ids.new_temp(Type.INT64)
                    ir.append(
                        IRBinOp(dst=field_addr, op=BinaryOp.ADD, left=dst_address, right=IRConst(offset, Type.INT64)))
                ir.extend(self._ir_write_zero_value_into(field_addr, field_type))
            elif field_type.kind in COMPOSITE_KINDS:
                if offset == 0:
                    field_addr = dst_address
                else:
                    field_addr = self.ir_program.ids.new_temp(Type.INT64)
                    ir.append(
                        IRBinOp(dst=field_addr, op=BinaryOp.ADD, left=dst_address, right=IRConst(offset, Type.INT64)))
                field_ir = self._ir_write_composite_value_into(field_addr, arg_expr, field_type)
                if field_ir is None:
                    return None
                ir.extend(field_ir)
            else:
                arg_ir, arg_value = self.gen_expr_ir(arg_expr)
                ir.extend(arg_ir)
                if offset == 0:
                    field_addr = dst_address
                else:
                    field_addr = self.ir_program.ids.new_temp(Type.INT64)
                    ir.append(
                        IRBinOp(dst=field_addr, op=BinaryOp.ADD, left=dst_address, right=IRConst(offset, Type.INT64)))
                ir.append(IRStore(address=field_addr, value=arg_value, value_type=field_type))
        return ir

    def _ir_materialize_struct_literal(self, expr: Call):
        """Materialize a struct literal; returns (ir, address)."""
        struct_type = type_of(expr)
        if id(expr) in self._argument_temp_slots:
            slot = self._argument_temp_slots[id(expr)]
            addr = self.ir_program.ids.new_temp(Type.INT64)
            addr_ir = [IRLocalAddress(dst=addr, slot=slot)]
        else:
            addr = self.ir_program.ids.new_temp(Type.INT64)
            size = type_byte_width(struct_type, self.ir_program.struct_registry, self.ir_program.sum_type_registry)
            addr_ir = [IRCall(dst=addr, name='malloc', args=[IRConst(size, Type.INT64)])]
        write_ir = self._ir_write_struct_literal_into(addr, expr, struct_type)
        if write_ir is None:
            return None
        return addr_ir + write_ir, addr

    def _check_struct_and_field_type(self, base_expr: Node, field_name: str) -> Type:
        """Declared type of field_name."""
        base_type = type_of(base_expr)
        if base_type.kind == TypeKind.POINTER and base_type.element_type.kind == TypeKind.STRUCT:
            base_type = base_type.element_type
        return self.ir_program.struct_registry[base_type.struct_name].fields[field_name]

