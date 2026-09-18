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

from codegen.errors import CodegenError
from ir.ir import IRBinOp, IRConst, IRStore, IRLoad, IRLocalAddress, IRCall
from codegen.utils import type_byte_width, type_of
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
            offset += type_byte_width(field_type, self.ir_program.struct_registry)
        raise CodegenError(f"Struct '{struct_name}' has no field '{field_name}'")

    def _ir_struct_address(self, expr: Node) -> tuple[list, object]:
        """Builds (without lowering) the address of a struct-typed
        expr -- Variable, Field, Index, or now an ordinary composite-
        returning Call (`makePoint().x`, materialized first via _ir_
        materialize_composite_call -- see its own docstring) -- as
        real IR. Field/Index delegate to _ir_field_address/_ir_
        index_address, which call back into this method for their own
        STRUCT-typed base -- the same mutual recursion those two
        already use for an ARRAY-typed base, so a chain of arbitrary
        depth (`a.b.c`, `rows[0].f`) falls out with no special-casing.

        The Variable case is the one genuine leaf: a named struct
        variable's own address is either a fixed, compile-time %rbp-
        relative offset (IRLocalAddress directly) or, if heap-
        allocated, the pointer STORED at that offset (IRLocalAddress
        for the slot's own address, then an ordinary IRLoad reading
        the pointer through it) -- IRLocalAddress always means "the
        address of this slot," never "the value stored there," so the
        heap-allocated case composes it with IRLoad rather than
        needing its own, second meaning (see IRLocalAddress's own
        docstring)."""
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
        raise CodegenError(f"Cannot compute a struct address for: {expr!r}")

    def _ir_field_address(self, expr: Field) -> tuple[list, object]:
        """Builds (without lowering) the address of `expr.base.expr.
        name` as real IR: the base's own address (recursively, via
        _ir_struct_address), plus expr.name's fixed byte offset, as an
        ordinary IRBinOp(ADD) on two INT64 values -- offset already
        fits IRBinOp's existing type-driven lowering with no changes
        needed at all (it already derives its own operand width from
        left.type, already INT64 here). Skips the add entirely when
        offset is 0, an ordinary micro-optimization, not a correctness
        requirement -- IRBinOp(ADD, x, 0) would compute the identical
        address either way."""
        base_type = type_of(expr.base)
        if base_type.kind != TypeKind.STRUCT:
            raise CodegenError(
                f"Cannot access field '{expr.name}' on a value of "
                f"non-struct type {base_type}"
            )
        offset = self._field_offset(base_type.struct_name, expr.name)
        result = self._ir_struct_address(expr.base)
        if result is None:
            raise CodegenError(
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
        IR, written through dst_address -- an ordinary INT64-typed
        IRValue, however the caller already has it (see _ir_composite_
        call's own docstring for the same "doesn't care how" contract).
        Returns None only when some field's own value is itself out of
        scope for _ir_write_composite_value_into -- a field that's
        ITSELF composite is no longer automatically out of scope,
        unlike this method's own earlier version: mutual recursion
        through _ir_write_composite_value_into handles it, the same
        shape address computation's own Field/Index handling already
        relies on elsewhere in this arc.

        Named construction (expr.kwargs) is normalized to the same
        (field_name, value_expr, field_type) shape positional
        construction already iterates, walking every field in
        declaration order -- the same normalization gen_struct_
        literal_into's own old-style version already does. An omitted
        field's value_expr comes back None, filled via _ir_write_zero_
        value_into (a TOTAL function -- see its own docstring for why
        it never needs to fall back the way a value_expr-dependent
        write can) rather than left as untouched, garbage bytes.

        Each field's own address is dst_address + its own, already-
        correct _field_offset (computed once per field, the same
        proven helper this compiler's own field-address code
        everywhere else already calls, rather than accumulating a
        running offset by hand and risking it drifting out of sync
        with that logic) via ordinary IRBinOp -- skipped entirely for
        the first field (offset 0 needs no addition, matching _ir_
        write_slice_descriptor's own identical shortcut for its own
        ptr field). A scalar field's value is evaluated via gen_expr_ir
        and written via IRStore; a composite field delegates entirely
        to _ir_write_composite_value_into. If ANY provided field's own
        value turns out to be out of scope, the whole literal falls
        back (returns None) -- see _ir_write_array_literal_into's own
        docstring for why any IR already built for earlier fields
        being discarded is harmless, not a partial-write risk."""
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
                # An OMITTED field in a named, partial literal (only
                # possible when expr.kwargs is not None -- positional
                # construction is exhaustive, so arg_expr is never
                # None there) -- gets its own type's implicit zero
                # value via _ir_write_zero_value_into, a TOTAL
                # function (see its own docstring), so this branch
                # never needs to fall back the way the other two do.
                if offset == 0:
                    field_addr = dst_address
                else:
                    field_addr = self.ir_program.ids.new_temp(Type.INT64)
                    ir.append(
                        IRBinOp(dst=field_addr, op=BinaryOp.ADD, left=dst_address, right=IRConst(offset, Type.INT64)))
                ir.extend(self._ir_write_zero_value_into(field_addr, field_type))
            elif field_type.kind in (TypeKind.ARRAY, TypeKind.SLICE, TypeKind.STRUCT):
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
        None when some field is itself out of scope for _ir_write_
        struct_literal_into (a named/partial literal, chiefly -- see
        its own docstring). The struct-literal counterpart to _ir_
        materialize_array_literal in arrays_slices.py, sharing its
        exact same skeleton (a reserved slot, or malloc when none was
        reserved) -- see its own docstring for the full reasoning,
        which applies here unchanged. Used wherever a struct literal
        is passed directly as a function-call argument (`draw(Point(1,
        2))`) -- the identical AST position _collect_argument_temps
        has always reserved a slot for, since before this arc's own
        addressable-base work existed; this is that reservation's
        real-IR consumer finally catching up to it, not a new
        reservation of its own.

        No value_type ambiguity to worry about here, unlike an
        ArrayLiteral: a struct-literal Call's own type is
        unambiguously its own struct name, not something that can
        resolve differently by context the way a bracketed list can
        (ARRAY vs SLICE) -- so type_of(expr) is simply, always
        correct, with nothing to disambiguate."""
        struct_type = type_of(expr)
        if id(expr) in self._argument_temp_slots:
            slot = self._argument_temp_slots[id(expr)]
            addr = self.ir_program.ids.new_temp(Type.INT64)
            addr_ir = [IRLocalAddress(dst=addr, slot=slot)]
        else:
            addr = self.ir_program.ids.new_temp(Type.INT64)
            size = type_byte_width(struct_type, self.ir_program.struct_registry)
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



