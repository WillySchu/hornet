"""Structs. TODO"""

from codegen.assembly_ast import (
    Register,
    Instruction,
    MovQ,
    Memory,
    LeaQFrame,
    AddQ,
    Imm,
    Push,
    Pop,
    Mov,
    CallInstr,
    Cmp,
    Jne,
    Jmp,
    Label,
)
from codegen.errors import CodegenError
from codegen.utils import type_byte_width, type_of, leaf_type, gen_protecting_dst_across
from parser import Node, Variable, Field, Index, Call, NoneLiteral, Binary, BinaryOp
from semantic import TypeKind, Type


class StructsMixin:
    """TODO"""
    def _field_offset(self, struct_name: str, field_name: str) -> int:
        """Returns the byte offset of `field_name` within a value of
        the named struct type -- the sum of every PRECEDING field's
        own width, in declaration order (StructInfo.fields is an
        ordinary dict, which already preserves insertion order -- see
        its own docstring in semantic.py). No padding or alignment is
        ever inserted between fields: x86-64 doesn't require aligned
        access the way some architectures do, the same reasoning
        _frame_size's own docstring already gives for why a stack
        frame's own locals need none either, so this is exactly the
        same "sum of what came before" computation type_byte_width
        itself already does for a struct's own TOTAL width, just
        stopping partway through instead of summing everything."""
        offset = 0
        for name, field_type in self.struct_registry[struct_name].fields.items():
            if name == field_name:
                return offset
            offset += type_byte_width(field_type, self.struct_registry)
        raise CodegenError(f"Struct '{struct_name}' has no field '{field_name}'")

    def gen_struct_address_into(self, expr: Node, dst: Register) -> list[Instruction]:
        """Computes the ADDRESS of a struct-typed expression -- a
        Variable referring to a struct-typed local or parameter, a
        Field node that itself resolves to a nested struct (`a.inner`,
        the outer field of a chain like `a.inner.v`), or an Index node
        that resolves to a struct-typed array element (`rows[0]`,
        where `rows` is an array of structs) -- into the 64-bit
        register `dst`. Mirrors gen_array_address_into exactly, one
        level over -- see its own docstring for why a heap-allocated
        Variable needs a genuinely different instruction (a movq to
        LOAD the pointer its own slot holds) rather than just a
        different offset, which applies identically here.

        A struct-returning Call as a field's own base (`makePoint(1,
        2).x`) is deliberately NOT supported yet -- matching this
        file's own established restriction on other unnamed-expression
        bases (see gen_slice_arg_into's identical one for a slice-typed
        call argument): materializing an arbitrary struct-returning
        expression into a scratch slot just to immediately read one
        field back out isn't taken on for what both cases judge to be
        a comparatively rare shape. Assign it to a named variable
        first."""
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
        """Computes the address of `expr.base.expr.name` into `dst` (a
        64-bit register) -- the shared foundation for reading a field
        (gen_expr_into's Field case) and writing one (gen_field_
        assign), exactly mirroring gen_index_address_into's own role
        for Index one level over. expr.base's own address is computed
        first (via gen_struct_address_into, which recurses through
        however many further Field/Index links precede this one --
        `a.b.c`, `rows[0].f`, ... -- with no depth limit, the same way
        gen_array_address_into's own Index recursion has none), then
        the field's own byte offset (see _field_offset) is added on
        top -- skipped entirely when it's zero (the field is first in
        its own struct's declaration order), since `addq $0, dst`
        would be correct but pointlessly wasteful."""
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

    def gen_struct_call_into(self, dst_mem: Memory, expr: Call) -> list[Instruction]:
        """Calls a function that returns a struct, writing its result
        directly into dst_mem via the exact same hidden-pointer
        convention gen_array_call_into already uses -- see its own
        docstring for the full reasoning, unchanged in every respect:
        neither method's own body actually reads its (array- or
        struct-typed) destination's width at all, since the callee is
        the one that knows how many bytes its own return type needs
        and writes exactly that many through the pointer it receives
        -- so the exact same mechanism serves both without needing its
        own struct-specific variant beyond this thin, separately-named
        entry point (kept distinct from gen_array_call_into itself
        purely so gen_struct_value_into's own Call case reads the same
        way its Variable/Field/Index cases do -- one clearly-named
        method per expression kind -- not because the underlying code
        needs to differ at all)."""
        return self.gen_array_call_into(dst_mem, expr, None)

    def gen_struct_literal_into(self, expr: Call, dst_mem: Memory, struct_type: Type) -> list[Instruction]:
        """Writes a struct literal's own fields directly into dst_mem,
        one at a time -- already validated by semantic.py's own check_
        struct_literal/_check_named_struct_literal, so this never
        needs to guard against a missing, extra, unknown, or duplicate
        field itself. Mirrors gen_array_literal_into's own per-element
        writing, one level over: each field's value is routed to
        whichever value-producing method its OWN declared type needs
        -- gen_array_value_into/gen_slice_value_into/gen_struct_value_
        into for a composite field, or a plain gen_expr_into for a
        scalar (int/bool/str) field.

        dst_mem's own base is protected via push/pop around EVERY
        field's own write, when it isn't 'rbp' -- not just the scalar
        case, and not just the ones that happen to need it. This used
        to be scoped to the scalar case only, on the reasoning that
        gen_array_value_into/gen_slice_value_into/gen_struct_value_
        into "already protect an arbitrary destination base
        internally" -- true for THEIR OWN write, but not for what
        happens to dst_mem.base's own physical register as a SIDE
        EFFECT of writing that one field, which the NEXT field then
        silently inherits. A composite field's own value can involve a
        real function call (an array-, slice-, or struct-returning
        Call) that clobbers dst_mem.base's own register as an ordinary
        caller-saved side effect of that call -- verified directly: a
        struct literal with an array-returning-call field FOLLOWED BY
        any other field, on a heap-allocated destination, silently
        corrupted that later field's own write (it computed the
        field's address from whatever garbage the call left behind,
        not the struct's real base) before this fix. Push-before-the-
        field's-own-write, pop-after fixes it the same way the scalar
        case's own push/compute/restore/write ordering already
        avoided the identical hazard for a scalar-typed value's own
        call: the call's own clobbering only matters for whether
        dst_mem.base survives for the NEXT field, not for whether
        THIS field's own write is correct (composite fields write
        through a hidden pointer computed and pushed onto the real
        stack BEFORE the call runs, so the call clobbering registers
        afterward never affects a write that already happened).

        A slice-typed field's value can be `none` -- checked here via
        isinstance, not the field's own resolved type (which for a
        NoneLiteral is always Type.NONE, never equal to the field's
        declared SLICE type) -- exactly the same none-flowing-into-a-
        slice-typed-slot short-circuit gen_var_decl/gen_assign/gen_
        field_assign each already need of their own, for the identical
        reason (see gen_none_into's own docstring).

        A struct-typed field's value can itself be a NESTED struct
        literal (`Outer(inner=Inner(1), b=2)`, or the positional
        `Outer(Inner(1), 2)`) -- recursing through gen_struct_value_
        into exactly like any other struct-typed value, which already
        detects a Call-is-a-struct-name and dispatches back into this
        same method one level deeper; no special-casing needed here
        for that, since it's just an ordinary struct-typed expression
        as far as this method is concerned.

        NAMED construction (expr.kwargs populated instead of
        expr.args -- see Call's own docstring in parser.py) is
        normalized into the same (field_name, value_expr, field_type)
        shape the positional form already iterates, via a dict lookup
        by name instead of a zip-by-position -- but unlike the
        positional form (always exhaustive), this walks EVERY one of
        the struct's own fields, in declaration order, not just the
        ones expr.kwargs actually provided: an omitted field's own
        `value_expr` comes back as None from that lookup, which the
        per-field loop recognizes as its own distinct case (see just
        below) rather than being silently absent from the loop
        entirely the way it used to be.

        PARTIAL construction (a named literal omitting a field
        entirely, `arg_expr is None` in the loop below) now gets that
        field's own implicit zero value -- the exact same _gen_zero_
        value_into a `T x` VarDecl with no initializer at all already
        uses -- rather than being skipped and left as whatever dst_mem
        already held. This closes the deliberate, temporary
        inconsistency this docstring used to describe: partial
        construction was left as a separate follow-up when zero-init
        first shipped specifically so it could be revisited once that
        feature existed to reuse, and this is that follow-up."""
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

    def gen_struct_value_into(self, expr: Node, dst_mem: Memory, struct_type: Type) -> list[Instruction]:
        """Stores a struct-typed expression's VALUE into dst_mem,
        matching struct_type's own shape -- the struct counterpart to
        gen_array_value_into just above, one level over, dispatched on
        what kind of expression is producing the value:
          - Call, where expr.name names a STRUCT (a struct literal,
            `A a = A(6, 'hello')`): each argument written directly into
            its own field -- see gen_struct_literal_into. Checked FIRST
            and separately from the ordinary Call case below, via
            struct_registry membership, exactly mirroring semantic.py's
            own check_struct_literal/check_call split -- the two are
            never ambiguous, since a name can never be both a struct
            and a function (see semantic.py's own collision check in
            analyze()).
          - Variable: a copy from wherever the source's data actually
            lives (via gen_array_copy, which already handles ANY
            value's own flat byte copy correctly -- struct included,
            see its own docstring for why no field-by-field recursion
            is needed) into dst_mem, mirroring the array case's own
            heap-vs-stack handling exactly.
          - Field (a nested struct field, e.g. `Point p = outer.
            inner`): its SOURCE address is computed first (gen_field_
            address_into), then copied from that computed address --
            the struct-specific case the array version doesn't need at
            all, since Index already plays this same role for arrays
            (an array can't itself be "a field" the way this specific
            expression shape can be a struct).
          - Index (a struct-typed array element, e.g. `Point p =
            rows[i]`): mirrors the array case's own Index handling
            exactly, via gen_index_address_into.
          - Call, otherwise (an ordinary function returning a struct):
            calls through the hidden-output-pointer convention, writing
            directly into dst_mem -- see gen_struct_call_into."""
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
        """Returns field_name's own declared type within base_expr's
        own struct type -- shared by gen_field_assign and gen_field_
        address_into's own callers wherever the field's type (not just
        its address) is needed, mirroring semantic.py's own _check_
        struct_and_field, just returning a Type instead of raising on
        an invalid access (already validated by the time codegen ever
        runs -- see compile_to_asm)."""
        base_type = type_of(base_expr)
        return self.struct_registry[base_type.struct_name].fields[field_name]

    def _flatten_struct_fields(self, struct_name: str, base_offset: int = 0):
        """Recursively walks struct_name's own fields (and, for any
        field that's itself a STRUCT, that struct's own fields, and so
        on, arbitrarily deep), yielding (leaf_field_type, absolute_
        byte_offset_from_the_OUTERMOST_struct's_own_base) pairs in
        declaration order -- entirely at compile time, in Python, with
        no runtime recursion or register-protection complexity
        involved at all: a nested struct's own fields are just as
        directly addressable via a single constant offset from the
        outermost struct's base as any of its own top-level fields
        are (Memory operands already support an arbitrary base
        register plus a constant offset natively), so flattening this
        way up front is what lets _gen_struct_fields_equality_at_
        addresses emit one simple, linear sequence of field
        comparisons -- no genuine runtime recursion needed to walk a
        struct's own nested shape, only to walk an ARRAY-typed field's
        own runtime-many elements (see _gen_array_struct_equality_
        loop), which is a fundamentally different kind of "how many"
        (a compile-time-fixed number of fields vs. a size that can be
        arbitrarily large) that this method deliberately doesn't try
        to handle the same way."""
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
        """Compares every one of struct_name's own fields -- including
        those of any nested struct field, at any depth, via _flatten_
        struct_fields's own compile-time flattening -- jumping to
        mismatch_label the moment any one differs. Shared by gen_
        struct_equality_into's own bare struct-vs-struct comparison
        and _gen_array_struct_equality_loop's own per-element one;
        this method itself has no idea (and doesn't need to know)
        which of the two contexts it's being called from, only that
        `left_base`/`right_base` are two registers currently holding
        real addresses of two same-typed struct_name values.

        left_base/right_base are protected on the real stack across
        EVERY field's own comparison, unconditionally, not just the
        ones that happen to need it -- computing that field's own
        address into a FRESH scratch register first (or, for a
        scalar field, simply reading it via Memory(base, offset)
        directly, needing no separate address register at all), so
        that whatever "risky" comparison follows (a strcmp call for a
        str field, or one of the three array-equality loop helpers for
        an array field -- each of which has its OWN internal register
        usage that could otherwise collide with this method's own base
        registers) is completely free to use any register it wants,
        including left_base/right_base's own, without ever needing to
        know or care what this method is doing with them. This is the
        same "protect on the stack, then let the next thing use
        whatever registers it wants" discipline used throughout this
        file, just applied uniformly to every field rather than only
        the ones proven to need it -- correct regardless of how many
        fields there are, at the cost of a few unconditional push/pop
        pairs even for the cheap int/bool case, exactly the kind of
        "simple over maximally efficient" trade this file already
        makes in plenty of other places (e.g. gen_binary_into's own
        single-register stack-spill scheme)."""
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
        """`left == right` / `left != right`, both the exact same
        struct type (already guaranteed by semantic.py's own check_
        binary, including that every one of the struct's own fields --
        at any nesting depth, through further nested structs or array
        fields -- is itself comparable: int, bool, str, a comparable
        array, or another comparable struct; never a slice, which has
        no '==' defined for it at all yet).

        `left`/`right` must each already have a real address (a
        Variable, Field, or Index -- whatever gen_struct_address_into
        already accepts); a struct literal or a struct-returning call
        used directly as an equality operand isn't supported, matching
        this file's established "assign it to a variable first"
        restriction on unnamed struct/array values elsewhere.

        The actual field-by-field comparison is _gen_struct_fields_
        equality_at_addresses's job -- this method just computes the
        two base addresses, protects the first across evaluating the
        second (the same push-before-evaluating-the-other-side pattern
        used throughout this file), and wraps the result in the same
        shared mismatch/done label shape gen_array_equality_into's own
        docstring already explains."""
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
