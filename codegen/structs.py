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

from codegen.assembly_ast import (
    AddQ,
    CallInstr,
    Cmp,
    Instruction,
    Imm,
    Jmp,
    Jne,
    Label,
    LeaQFrame,
    Memory,
    Mov,
    MovQ,
    Push,
    Pop,
    Register,
)
from codegen.errors import CodegenError
from codegen.ir import IRRaw, IRBinOp, IRConst, IRStore
from codegen.utils import type_byte_width, type_of, leaf_type, gen_protecting_dst_across
from parser import Node, Variable, Field, Index, Call, NoneLiteral, Binary, BinaryOp
from semantic import TypeKind, Type


class StructsMixin:
    def _field_offset(self, struct_name: str, field_name: str) -> int:
        """Byte offset of `field_name` within struct_name: the sum of
        every preceding field's width, in declaration order. No
        padding or alignment is inserted between fields -- x86-64
        doesn't require aligned access."""
        offset = 0
        for name, field_type in self.struct_registry[struct_name].fields.items():
            if name == field_name:
                return offset
            offset += type_byte_width(field_type, self.struct_registry)
        raise CodegenError(f"Struct '{struct_name}' has no field '{field_name}'")

    def gen_struct_address_into(self, expr: Node, dst: Register) -> list[Instruction]:
        """Computes the ADDRESS of a struct-typed expression -- a
        Variable, a Field resolving to a nested struct (`a.inner`), or
        an Index into an array of structs (`rows[0]`) -- into `dst`.
        Mirrors gen_array_address_into one level over.

        A struct-returning Call as a field's base (`makePoint(1,
        2).x`) isn't supported -- assign it to a variable first."""
        if isinstance(expr, Variable):
            offset = self._local_offset(expr.name)
            struct_type = self._local_type(expr.name)
            if self._is_heap_allocated(self._local_decl_id(expr.name), struct_type):
                return [MovQ(src=Memory('rbp', offset), dst=dst)]
            return [LeaQFrame(offset=offset, dst=dst)]
        if isinstance(expr, Field):
            return self.gen_field_address_into(expr, dst)
        if isinstance(expr, Index):
            return self.gen_index_address_into(expr, dst)
        if isinstance(expr, Call):
            raise CodegenError(
                f"Cannot use a Call directly as the base of a field "
                f"access when it's struct-typed -- assign it to a "
                f"named variable first"
            )
        raise CodegenError(f"Cannot compute a struct address for: {expr!r}")

    def gen_field_address_into(self, expr: Field, dst: Register) -> list[Instruction]:
        """Computes the address of `expr.base.expr.name` into `dst` --
        the old-style foundation for reading (gen_expr_into's Field
        case, still used for a composite-typed field/element not yet
        migrated to IR) and writing (gen_field_assign, likewise).
        Recurses through gen_struct_address_into for chains of
        arbitrary depth (`a.b.c`, `rows[0].f`), then adds the field's
        own byte offset (skipped when zero).

        A scalar field's own address is real IR now instead -- see
        _ir_field_address, which replicates this exact same recursion
        and offset-add via IRBinOp rather than calling this method at
        all."""
        base_type = type_of(expr.base)
        if base_type.kind != TypeKind.STRUCT:
            raise CodegenError(
                f"Cannot access field '{expr.name}' on a value of "
                f"non-struct type {base_type}"
            )
        offset = self._field_offset(base_type.struct_name, expr.name)
        instructions = self.gen_struct_address_into(expr.base, dst)
        if offset:
            instructions.append(AddQ(src=Imm(offset), dst=dst))
        return instructions

    def _ir_struct_address(self, expr: Node) -> tuple[list, object]:
        """Builds (without lowering) the address of a struct-typed
        expr -- Variable, Field, or Index -- as real IR. Mirrors gen_
        struct_address_into's own three cases; Field/Index delegate to
        _ir_field_address/_ir_index_address, which call back into this
        method for their own STRUCT-typed base -- the identical mutual
        recursion gen_struct_address_into/gen_field_address_into/gen_
        index_address_into already use, so a chain of arbitrary depth
        (`a.b.c`, `rows[0].f`) falls out with no special-casing.

        The Variable case is the one genuine leaf: a named struct
        variable's own address is either a fixed, compile-time %rbp-
        relative offset or, if heap-allocated, a single pointer read
        -- neither is a computation OVER other values, so it stays a
        small, IRRaw-wrapped leaf rather than its own IR concept, the
        same deliberate scope boundary IRLoad/IRStore's own address
        capture already draws."""
        if isinstance(expr, Variable):
            offset = self._local_offset(expr.name)
            struct_type = self._local_type(expr.name)
            addr_temp = self._new_temp(Type.INT64)
            if self._is_heap_allocated(self._local_decl_id(expr.name), struct_type):
                leaf = [MovQ(src=Memory('rbp', offset), dst=Register('rax'))]
            else:
                leaf = [LeaQFrame(offset=offset, dst=Register('rax'))]
            return [IRRaw(leaf, dst=addr_temp)], addr_temp
        if isinstance(expr, Field):
            return self._ir_field_address(expr)
        if isinstance(expr, Index):
            return self._ir_index_address(expr)
        raise CodegenError(f"Cannot compute a struct address for: {expr!r}")

    def _ir_field_address(self, expr: Field) -> tuple[list, object]:
        """Builds (without lowering) the address of `expr.base.expr.
        name` as real IR: the base's own address (recursively, via
        _ir_struct_address), plus expr.name's fixed byte offset, as an
        ordinary IRBinOp(ADD) on two INT64 values -- offset already
        fits IRBinOp's existing type-driven lowering with no changes
        needed at all (it already derives its own operand width from
        left.type, already INT64 here). Skips the add entirely when
        offset is 0, matching gen_field_address_into's own shortcut."""
        base_type = type_of(expr.base)
        if base_type.kind != TypeKind.STRUCT:
            raise CodegenError(
                f"Cannot access field '{expr.name}' on a value of "
                f"non-struct type {base_type}"
            )
        offset = self._field_offset(base_type.struct_name, expr.name)
        base_ir, base_addr = self._ir_struct_address(expr.base)
        if offset == 0:
            return base_ir, base_addr
        result = self._new_temp(Type.INT64)
        add_op = IRBinOp(dst=result, op=BinaryOp.ADD, left=base_addr, right=IRConst(offset, Type.INT64))
        return base_ir + [add_op], result

    def gen_struct_call_into(self, dst_mem: Memory, expr: Call) -> list[Instruction]:
        """Calls a struct-returning function, writing the result into
        dst_mem via gen_array_call_into's hidden-pointer convention
        (which doesn't care whether the return type is an array or
        struct). Kept as a separate, named entry point purely so
        gen_struct_value_into's Call case reads the same way as its
        other cases."""
        return self.gen_array_call_into(dst_mem, expr, None)

    def gen_struct_literal_into(self, expr: Call, dst_mem: Memory, struct_type: Type) -> list[Instruction]:
        """Writes a struct literal's fields directly into dst_mem, one
        at a time -- already validated by semantic.py, so no missing/
        extra/duplicate/unknown field checking is needed here. Each
        field's value is routed by its own declared type:
        gen_array_value_into/gen_slice_value_into/gen_struct_value_
        into for a composite field, gen_expr_into for a scalar one. A
        nested struct-typed field (another struct literal) recurses
        through gen_struct_value_into like any other struct-typed
        value -- no special-casing needed here.

        dst_mem's base is push/pop-protected around EVERY field's
        write, not just ones proven to need it: a composite field's
        value can involve a function call that clobbers dst_mem.base
        as an ordinary caller-saved side effect, corrupting a LATER
        field's address computation if left unprotected. (Verified by
        a real bug: an array-returning-call field followed by another
        field, on a heap-allocated destination, silently corrupted the
        second field's write before this fix.)

        A slice-typed field's value can be `none`, checked via
        isinstance rather than its resolved type (see gen_none_into).

        Named construction (expr.kwargs) is normalized to the same
        (field_name, value_expr, field_type) shape positional
        construction already iterates, walking every field in
        declaration order; an omitted field's value_expr comes back
        None, which the loop below fills with that field's zero value
        (_gen_zero_value_into) rather than leaving dst_mem's existing
        bytes untouched.
        """
        struct_info = self.struct_registry[struct_type.struct_name]
        if expr.kwargs is not None:
            provided = dict(expr.kwargs)
            entries = [
                (field_name, provided.get(field_name), field_type)
                for field_name, field_type in struct_info.fields.items()
            ]
        else:
            field_items = list(struct_info.fields.items())
            entries = [
                (field_name, arg_expr, field_type)
                for arg_expr, (field_name, field_type) in zip(expr.args, field_items)
            ]
        protect_dst = dst_mem.base != 'rbp'
        instructions = []
        for field_name, arg_expr, field_type in entries:
            field_offset = self._field_offset(struct_type.struct_name, field_name)
            field_mem = Memory(dst_mem.base, dst_mem.offset + field_offset)
            if arg_expr is None:
                # An OMITTED field in a NAMED, partial literal (only
                # possible when expr.kwargs is not None -- positional
                # construction is exhaustive, so arg_expr is never
                # None there) -- gets its own type's implicit zero
                # value, via the exact same _gen_zero_value_into a `T
                # x` VarDecl with no initializer at all already uses.
                if protect_dst:
                    instructions.append(Push(Register(dst_mem.base)))
                instructions.extend(self._gen_zero_value_into(field_type, field_mem))
                if protect_dst:
                    instructions.append(Pop(Register(dst_mem.base)))
                continue
            if field_type.kind == TypeKind.ARRAY:
                if protect_dst:
                    instructions.append(Push(Register(dst_mem.base)))
                instructions.extend(self.gen_array_value_into(arg_expr, field_mem, field_type))
                if protect_dst:
                    instructions.append(Pop(Register(dst_mem.base)))
                continue
            if field_type.kind == TypeKind.SLICE:
                if protect_dst:
                    instructions.append(Push(Register(dst_mem.base)))
                if isinstance(arg_expr, NoneLiteral):
                    instructions.extend(self.gen_none_into(field_mem, field_type))
                else:
                    instructions.extend(self.gen_slice_value_into(arg_expr, field_mem))
                if protect_dst:
                    instructions.append(Pop(Register(dst_mem.base)))
                continue
            if field_type.kind == TypeKind.STRUCT:
                if protect_dst:
                    instructions.append(Push(Register(dst_mem.base)))
                instructions.extend(self.gen_struct_value_into(arg_expr, field_mem, field_type))
                if protect_dst:
                    instructions.append(Pop(Register(dst_mem.base)))
                continue
            if protect_dst:
                instructions.append(Push(Register(dst_mem.base)))
            instructions.extend(self.gen_expr_into(arg_expr, Register('eax')))
            if field_type == Type.STR:
                if protect_dst:
                    instructions.append(MovQ(src=Register('rax'), dst=Register('r8')))
                    instructions.append(Pop(Register(dst_mem.base)))
                    instructions.append(MovQ(src=Register('r8'), dst=field_mem))
                else:
                    instructions.append(MovQ(src=Register('rax'), dst=field_mem))
            else:
                if protect_dst:
                    if field_type == Type.INT64:
                        instructions.append(MovQ(src=Register('rax'), dst=Register('r8')))
                    else:
                        instructions.append(Mov(src=Register('eax'), dst=Register('r8d')))
                    instructions.append(Pop(Register(dst_mem.base)))
                    instructions.extend(self._gen_write_scalar_from(Register('r8d'), field_type, field_mem))
                else:
                    instructions.extend(self._gen_write_scalar_from(Register('eax'), field_type, field_mem))
        return instructions

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
        proven helper gen_struct_literal_into itself already calls,
        rather than accumulating a running offset by hand and risking
        it drifting out of sync with that method's own logic) via
        ordinary IRBinOp -- skipped entirely for the first field
        (offset 0 needs no addition, matching _ir_write_slice_
        descriptor's own identical shortcut for its own ptr field). A
        scalar field's value is evaluated via gen_expr_ir and written
        via IRStore; a composite field delegates entirely to _ir_
        write_composite_value_into. If ANY provided field's own value
        turns out to be out of scope, the whole literal falls back
        (returns None) -- see _ir_write_array_literal_into's own
        docstring for why any IR already built for earlier fields
        being discarded is harmless, not a partial-write risk."""
        struct_info = self.struct_registry[struct_type.struct_name]
        field_items = list(struct_info.fields.items())
        if expr.kwargs is not None:
            provided = dict(expr.kwargs)
            entries = [(field_name, provided.get(field_name), field_type) for field_name, field_type in field_items]
        else:
            entries = [(field_name, arg_expr, field_type) for arg_expr, (field_name, field_type) in zip(expr.args, field_items)]
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
                    field_addr = self._new_temp(Type.INT64)
                    ir.append(IRBinOp(dst=field_addr, op=BinaryOp.ADD, left=dst_address, right=IRConst(offset, Type.INT64)))
                ir.extend(self._ir_write_zero_value_into(field_addr, field_type))
            elif field_type.kind in (TypeKind.ARRAY, TypeKind.SLICE, TypeKind.STRUCT):
                if offset == 0:
                    field_addr = dst_address
                else:
                    field_addr = self._new_temp(Type.INT64)
                    ir.append(IRBinOp(dst=field_addr, op=BinaryOp.ADD, left=dst_address, right=IRConst(offset, Type.INT64)))
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
                    field_addr = self._new_temp(Type.INT64)
                    ir.append(IRBinOp(dst=field_addr, op=BinaryOp.ADD, left=dst_address, right=IRConst(offset, Type.INT64)))
                ir.append(IRStore(address=field_addr, value=arg_value, value_type=field_type))
        return ir

    def gen_struct_value_into(self, expr: Node, dst_mem: Memory, struct_type: Type) -> list[Instruction]:
        """Stores a struct-typed expression's VALUE into dst_mem,
        matching struct_type's shape -- the struct counterpart to
        gen_array_value_into, dispatched on the producing expression:
          - Call naming a struct (a struct literal): each field
            written directly -- gen_struct_literal_into. Checked
            before the ordinary Call case below via struct_registry
            membership (a name can never be both a struct and a
            function).
          - Variable: copied via gen_array_copy (handles any value's
            flat byte copy, struct included -- no field-by-field
            recursion needed), loading the source pointer first if
            heap-allocated.
          - Field (a nested struct field, `outer.inner`): source
            address computed via gen_field_address_into, then copied.
          - Index (a struct-typed array element, `rows[i]`): via
            gen_index_address_into, then copied.
          - Call, otherwise (an ordinary struct-returning function):
            via gen_struct_call_into's hidden-pointer convention.
        """
        if isinstance(expr, Call) and expr.name in self.struct_registry:
            return self.gen_struct_literal_into(expr, dst_mem, struct_type)
        if isinstance(expr, Variable):
            src_offset = self._local_offset(expr.name)
            src_type = self._local_type(expr.name)
            if self._is_heap_allocated(self._local_decl_id(expr.name), src_type):
                load_ptr = gen_protecting_dst_across(
                    dst_mem, [MovQ(src=Memory('rbp', src_offset), dst=Register('rbx'))]
                )
                return load_ptr + self.gen_array_copy(dst_mem, Memory('rbx', 0), struct_type)
            return self.gen_array_copy(dst_mem, Memory('rbp', src_offset), struct_type)
        if isinstance(expr, Field):
            addr_instructions = gen_protecting_dst_across(
                dst_mem, self.gen_field_address_into(expr, Register('rbx'))
            )
            return addr_instructions + self.gen_array_copy(dst_mem, Memory('rbx', 0), struct_type)
        if isinstance(expr, Index):
            addr_instructions = gen_protecting_dst_across(
                dst_mem, self.gen_index_address_into(expr, Register('rbx'))
            )
            return addr_instructions + self.gen_array_copy(dst_mem, Memory('rbx', 0), struct_type)
        if isinstance(expr, Call):
            return self.gen_struct_call_into(dst_mem, expr)
        raise CodegenError(f"No codegen rule for a struct-typed value: {expr!r}")

    def _check_struct_and_field_type(self, base_expr: Node, field_name: str) -> Type:
        """Returns field_name's declared type within base_expr's
        struct type. Doesn't raise on an invalid access -- already
        validated by semantic.py before codegen runs."""
        base_type = type_of(base_expr)
        return self.struct_registry[base_type.struct_name].fields[field_name]

    def _flatten_struct_fields(self, struct_name: str, base_offset: int = 0):
        """Recursively walks struct_name's fields -- including any
        nested struct field's fields, arbitrarily deep -- yielding
        (leaf_field_type, absolute_byte_offset_from_the_outermost_
        struct's_base) pairs in declaration order. Done entirely at
        compile time (a struct's shape is fixed), so _gen_struct_
        fields_equality_at_addresses can emit one flat sequence of
        comparisons with no runtime recursion. An array-typed field's
        runtime-many elements still need an actual loop
        (_gen_array_struct_equality_loop) -- only the struct nesting
        itself flattens away."""
        struct_info = self.struct_registry[struct_name]
        for field_name, field_type in struct_info.fields.items():
            field_offset = base_offset + self._field_offset(struct_name, field_name)
            if field_type.kind == TypeKind.STRUCT:
                yield from self._flatten_struct_fields(field_type.struct_name, field_offset)
            else:
                yield field_type, field_offset

    def _gen_struct_fields_equality_at_addresses(
            self,
            struct_name: str,
            left_base: Register,
            right_base: Register,
            mismatch_label: str) -> list[Instruction]:
        """Compares every one of struct_name's fields -- including
        nested struct fields, via _flatten_struct_fields -- jumping to
        mismatch_label the moment any one differs. Shared by
        gen_struct_equality_into (struct-vs-struct) and
        _gen_array_struct_equality_loop (per-element); this method
        only needs left_base/right_base to be two registers holding
        real addresses of same-typed struct_name values.

        left_base/right_base are pushed/popped around every field's
        comparison, unconditionally -- even the cheap int/bool case --
        so that whatever comparison follows (a strcmp call for a str
        field, an array-equality loop for an array field) is free to
        use any register, including left_base/right_base's own,
        without needing to coordinate."""
        instructions = []
        for field_type, offset in self._flatten_struct_fields(struct_name):
            if field_type.kind == TypeKind.ARRAY:
                leaf = leaf_type(field_type)
                total_width = type_byte_width(field_type, self.struct_registry)
                instructions.append(Push(left_base))
                instructions.append(Push(right_base))
                left_field_addr = Register('r14')
                right_field_addr = Register('r15')
                instructions.append(MovQ(src=left_base, dst=left_field_addr))
                if offset:
                    instructions.append(AddQ(src=Imm(offset), dst=left_field_addr))
                instructions.append(MovQ(src=right_base, dst=right_field_addr))
                if offset:
                    instructions.append(AddQ(src=Imm(offset), dst=right_field_addr))
                if leaf == Type.STR:
                    instructions.extend(self._gen_array_str_equality_loop(
                        left_field_addr, right_field_addr, total_width // 8, mismatch_label
                    ))
                elif leaf.kind == TypeKind.STRUCT:
                    struct_width = type_byte_width(leaf, self.struct_registry)
                    instructions.extend(self._gen_array_struct_equality_loop(
                        left_field_addr, right_field_addr, leaf.struct_name, total_width // struct_width, mismatch_label
                    ))
                else:
                    instructions.extend(self._gen_array_flat_byte_equality_loop(
                        left_field_addr, right_field_addr, total_width, mismatch_label
                    ))
                instructions.append(Pop(right_base))
                instructions.append(Pop(left_base))
            elif field_type == Type.STR:
                instructions.append(Push(left_base))
                instructions.append(Push(right_base))
                instructions.append(MovQ(src=Memory(left_base.name, offset), dst=Register('rdi')))
                instructions.append(MovQ(src=Memory(right_base.name, offset), dst=Register('rsi')))
                instructions.append(CallInstr('strcmp'))
                instructions.append(Cmp(src=Imm(0), dst=Register('eax')))
                instructions.append(Jne(mismatch_label))
                instructions.append(Pop(right_base))
                instructions.append(Pop(left_base))
            else:
                # int or bool -- a plain 4-byte compare, no call
                # involved, so no protection needed at all.
                instructions.append(Mov(src=Memory(left_base.name, offset), dst=Register('eax')))
                instructions.append(Cmp(src=Memory(right_base.name, offset), dst=Register('eax')))
                instructions.append(Jne(mismatch_label))
        return instructions

    def gen_struct_equality_into(self, expr: Binary, dst: Register) -> list[Instruction]:
        """`left == right` / `left != right` for two values of the
        same struct type (semantic.py already guarantees every field,
        at any nesting depth, is itself comparable).

        `left`/`right` must each have a real address (Variable, Field,
        or Index) -- a struct literal or struct-returning call used
        directly as an operand isn't supported; assign it to a
        variable first.

        Computes both addresses (protecting the first while evaluating
        the second), delegates the actual comparison to
        _gen_struct_fields_equality_at_addresses, and wraps the result
        in the same mismatch/done label shape gen_array_equality_into
        uses."""
        struct_type = type_of(expr.left)
        left_base = Register('r10')
        right_base = Register('r11')
        instructions = self.gen_struct_address_into(expr.left, left_base)
        instructions.append(Push(left_base))
        instructions.extend(self.gen_struct_address_into(expr.right, right_base))
        instructions.append(Pop(left_base))

        mismatch_label = self.new_label("struct_eq_mismatch")
        done_label = self.new_label("struct_eq_done")
        instructions.extend(self._gen_struct_fields_equality_at_addresses(
            struct_type.struct_name, left_base, right_base, mismatch_label
        ))

        instructions.append(Mov(src=Imm(1 if expr.op == BinaryOp.EQUAL else 0), dst=dst))
        instructions.append(Jmp(done_label))
        instructions.append(Label(mismatch_label))
        instructions.append(Mov(src=Imm(0 if expr.op == BinaryOp.EQUAL else 1), dst=dst))
        instructions.append(Label(done_label))
        return instructions
