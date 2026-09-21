"""Struct layout and value semantics. Fields sit at sequential byte
offsets in declaration order with no padding (x86-64 doesn't require
aligned access); a struct's nominal typing -- two structs with
identical fields but different names are different types -- falls out
of Type's existing structural-equality machinery for free, needing no
new mechanism here. Struct assignment, parameter passing, and return
all copy the whole value, never alias, exactly like arrays one level
over; equality and zero-initialization work field by field, recursing
into nested structs and delegating array-typed fields back to
arrays_slices.py."""

from ir.errors import IRError
from ir.ir import IRBinOp, IRConst, IRStore, IRLoad, IRLocalAddress, IRCall
from ir.utils import COMPOSITE_KINDS, type_byte_width, type_of
from parser import Node, Variable, Field, Index, Call, BinaryOp
from semantic import TypeKind, Type


class StructsMixin:
    def _field_offset(self, struct_name: str, field_name: str) -> int:
        """Byte offset of `field_name` within struct_name: the sum of
        every preceding field's width, in declaration order. No
        padding or alignment is inserted between fields -- x86-64
        doesn't require aligned access."""
        offset = 0
        for name, field_type in self.ir_program.struct_registry[struct_name].fields.items():
            if name == field_name:
                return offset
            offset += type_byte_width(field_type, self.ir_program.struct_registry, self.ir_program.sum_type_registry)
        raise IRError(f"Struct '{struct_name}' has no field '{field_name}'")

    def _ir_struct_address(self, expr: Node) -> tuple[list, object]:
        """Builds (without lowering) the address of a struct-typed
        expr -- Variable, Field, Index, or an ordinary composite-
        returning Call (materialized via _ir_materialize_composite_
        call) -- as real IR. Field/Index delegate to _ir_field_
        address/_ir_index_address, which call back into this method
        for their own STRUCT-typed base, so a chain of arbitrary depth
        (`a.b.c`, `rows[0].f`) falls out with no special-casing.

        The Variable case is the one genuine leaf: a named struct
        variable's own address is either a fixed %rbp-relative offset
        (IRLocalAddress directly) or, if heap-allocated, the pointer
        stored at that offset (IRLocalAddress for the slot's own
        address, then an IRLoad reading the pointer through it)."""
        if isinstance(expr, Variable):
            slot = self._local_slot(expr.name)
            struct_type = self._local_type(expr.name)
            slot_addr = self.ir_program.ids.new_temp(Type.INT64)
            ir = [IRLocalAddress(dst=slot_addr, slot=slot)]
            if self._is_heap_allocated(self._local_decl_id(expr.name), struct_type):
                addr_temp = self.ir_program.ids.new_temp(Type.INT64)
                ir.append(IRLoad(dst=addr_temp, address=slot_addr))
                return ir, addr_temp
            return ir, slot_addr
        if isinstance(expr, Field):
            return self._ir_field_address(expr)
        if isinstance(expr, Index):
            return self._ir_index_address(expr)
        if self._is_ordinary_composite_call(expr):
            return self._ir_materialize_composite_call(expr, type_of(expr))
        raise IRError(f"Cannot compute a struct address for: {expr!r}")

    def _ir_field_address(self, expr: Field) -> tuple[list, object]:
        """Builds (without lowering) the address of `expr.base.expr.
        name` as real IR: the base's own address (recursively, via
        _ir_struct_address) plus expr.name's fixed byte offset, via
        IRBinOp(ADD). Skips the add when offset is 0."""
        base_type = type_of(expr.base)
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
        """Builds (without lowering) a struct literal's fields as real
        IR, written through dst_address. Returns None only when some
        field's own value is out of scope for _ir_write_composite_
        value_into.

        Named construction (expr.kwargs) is normalized to the same
        (field_name, value_expr, field_type) shape positional
        construction iterates, walking every field in declaration
        order. An omitted field's value_expr comes back None, filled
        via _ir_write_zero_value_into (a TOTAL function).

        Each field's own address is dst_address + its own _field_
        offset, skipped for the first field (offset 0). A scalar
        field's value is evaluated via gen_expr_ir and written via
        IRStore; a composite field delegates to _ir_write_composite_
        value_into. If any field's own value is out of scope, the
        whole literal falls back (returns None)."""
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
                # Omitted field in a named, partial literal (positional
                # construction is exhaustive, so arg_expr is never None
                # there) -- gets its own type's zero value via _ir_
                # write_zero_value_into, a TOTAL function that never
                # needs to fall back.
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
        """Builds (without lowering) a struct-literal Call's own
        materialized address as real IR -- returns (ir, address), or
        None when some field is out of scope for _ir_write_struct_
        literal_into. The struct-literal counterpart to _ir_
        materialize_array_literal in arrays_slices.py, sharing its
        same skeleton (a reserved slot, or malloc when none was
        reserved).

        No value_type ambiguity here, unlike an ArrayLiteral: a
        struct-literal Call's own type is unambiguously its own struct
        name, so type_of(expr) is simply, always correct."""
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
        """Returns field_name's declared type within base_expr's
        struct type. Doesn't raise on an invalid access -- already
        validated by semantic.py before codegen runs."""
        base_type = type_of(base_expr)
        return self.ir_program.struct_registry[base_type.struct_name].fields[field_name]



