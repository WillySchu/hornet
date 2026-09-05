"""Arrays are fixed-size, stack-allocated (unless promoted) value
types: `b = a` copies every element into independent storage, and
every index is bounds-checked at runtime. Slices are 24-byte {ptr,
len, cap} descriptors that alias an array's backing storage rather
than copying it -- cap exists to let `append` grow into existing
spare room instead of always reallocating. Heap promotion combines a
pure size threshold with real escape analysis (see
escape_analysis.py): a small array whose slice might outlive the
function it's declared in still needs heap-backed storage regardless
of size."""

from typing import Union

from codegen.assembly_ast import (
    Add,
    AddQ,
    CallInstr,
    Cmp,
    CmpQ,
    Imm,
    IMul,
    Instruction,
    Ja,
    Jae,
    Je,
    Jle,
    Jmp,
    Jne,
    Label,
    LeaQ,
    LeaQFrame,
    Memory,
    Mov,
    MovZX,
    MovB,
    MovQ,
    Operand,
    Pop,
    Push,
    Register,
    SetCC,
    ShiftRightArithmetic,
    Sub,
)
from codegen.errors import CodegenError
from codegen.utils import type_of, type_byte_width, leaf_type, as_byte_register, gen_protecting_dst_across
from parser import Node, ArrayLiteral, Call, Field, Index, Slice, Variable, NoneLiteral, Binary, BinaryOp
from semantic import TypeKind, Type


class ArraysSlicesMixin:
    def gen_array_literal_heap_alloc_into(self, expr: ArrayLiteral) -> list[Instruction]:
        """Mallocs a NEW, heap-allocated array sized to fit expr's
        elements, writes them in via gen_array_literal_into
        (Memory('rax', 0) here matches gen_array_literal_into's
        non-'rbp'-base protection, and %rax is guaranteed to still
        hold the malloc'd address afterward), and leaves the resulting
        pointer in %rax.

        Shared by both ways a slice literal's backing array gets
        created: the general, typed form (`[]int[1, 2, 3]`) via
        gen_indexable_base_into's ArrayLiteral case, and the untyped
        form used directly as a slice-typed VarDecl/Assign value
        (`[]int s = [1, 2, 3]`) via their own ArrayLiteral-as-slice-
        value short-circuit.

        Always allocates at least 1 byte, even for an empty literal
        (`[]int[]`) -- guaranteeing a genuine, non-null, unique pointer
        regardless of malloc(0)'s implementation-defined behavior,
        which is what makes `s == none` correctly false for an
        intentionally empty slice literal (a real, live, zero-length
        slice, not a nil one, same as `arr[5:5]`).

        Every slice literal's backing array is heap-allocated here
        UNCONDITIONALLY, regardless of size -- unlike an ordinary array
        variable, which only heap-promotes past the 16KB stack-size
        threshold: a slice literal's backing array has to outlive the
        statement that creates it, so it needs the same "can safely
        cross frame boundaries" guarantee every other sliced array
        gets, for the same reason.
        """
        array_type = type_of(expr)
        width = max(1, type_byte_width(array_type, self.struct_registry))
        instructions = [
            Mov(src=Imm(width), dst=Register('edi')),
            CallInstr('malloc'),
        ]
        instructions.extend(self.gen_array_literal_into(Memory('rax', 0), expr, array_type))
        return instructions

    def gen_indexable_base_into(
            self,
            expr: Node,
            addr_dst: Register,
            len_dst: Register, cap_dst: Register,
    ) -> tuple[list[Instruction], Union[Imm, Register], Union[Imm, Register]]:
        """Computes the address of `expr`'s data into `addr_dst`, and
        returns (instructions, length_operand, cap_operand): each
        operand is an Imm (the array's declared size, for both len and
        cap -- an array has no separate capacity) when `expr` is
        array-typed, or `len_dst`/`cap_dst` (populated with a runtime
        value from a slice's descriptor) when `expr` is slice-typed.
        cap_dst is always computed, even by callers
        (gen_index_address_into) that never read it back -- cheap
        enough that a single, uniform three-value contract beats making
        it optional.

        Shared by gen_index_address_into (`base[i]`, never needs cap:
        an index equal to len is already out of bounds regardless of
        any spare room), gen_slice_into (`base[low:high]`, needs cap
        for both its bounds check and the result's own capacity), and
        gen_append_call_into (needs all three as genuine input values).

        A slice-typed `expr` that's a Variable is read directly out of
        its %rbp-relative slot. Every other slice-typed shape -- a
        Slice (an unnamed slice expression used directly as a base,
        e.g. `arr[:][0]`), a Call returning a slice, an Index yielding
        a slice (`rows[0][1]`), or a Field yielding a slice
        (`p.values[0]`) -- has no existing descriptor to read, so each
        is materialized into the same shared, per-function scratch
        slot (_unnamed_slice_temp_offset) via its own value-producing
        method (gen_slice_into/gen_slice_call_into/
        gen_slice_value_into), then immediately read back out into
        addr_dst/len_dst/cap_dst.

        Reusing ONE shared scratch slot for every materialization --
        rather than a fresh one per nesting level -- is safe under
        arbitrarily deep chaining (`arr[:][0:2][0]`, `rows[0][1]`)
        because gen_slice_into and gen_index_address_into both compute
        their own base's address/length FIRST, immediately consume it,
        and only ever write their own result as the very LAST step: a
        deeper level's write to the shared slot always happens and
        drains strictly BEFORE the shallower level that triggered it
        writes its own result there -- the same nested-lifetime
        discipline that makes reusing one call stack safe for
        recursion of any depth.

        An ARRAY-typed `expr` can also be an ArrayLiteral directly --
        not an existing Variable/Index, but a freshly-created one, for
        a slice LITERAL's own backing array (see
        gen_array_literal_heap_alloc_into): a genuinely different kind
        of "address" than the ordinary Variable/Index cases, since it
        mallocs a new allocation and writes the literal's elements
        into it rather than computing the address of something that
        already exists.

        `expr` being anything else falls through to the CodegenError
        below.
        """
        base_type = type_of(expr)
        if base_type.kind == TypeKind.ARRAY:
            if isinstance(expr, ArrayLiteral):
                instructions = self.gen_array_literal_heap_alloc_into(expr)
                instructions.append(MovQ(src=Register('rax'), dst=addr_dst))
                return instructions, Imm(base_type.size), Imm(base_type.size)
            instructions = self.gen_array_address_into(expr, addr_dst)
            return instructions, Imm(base_type.size), Imm(base_type.size)
        if base_type.kind == TypeKind.SLICE:
            if isinstance(expr, Variable):
                offset = self._local_offset(expr.name)
                instructions = [
                    MovQ(src=Memory('rbp', offset + 8), dst=len_dst),
                    MovQ(src=Memory('rbp', offset + 16), dst=cap_dst),
                    MovQ(src=Memory('rbp', offset), dst=addr_dst),
                ]
                return instructions, len_dst, cap_dst
            if isinstance(expr, Slice):
                temp = self._unnamed_slice_temp_offset
                instructions = self.gen_slice_into(expr, Memory('rbp', temp))
                instructions.append(MovQ(src=Memory('rbp', temp + 8), dst=len_dst))
                instructions.append(MovQ(src=Memory('rbp', temp + 16), dst=cap_dst))
                instructions.append(MovQ(src=Memory('rbp', temp), dst=addr_dst))
                return instructions, len_dst, cap_dst
            if isinstance(expr, Index):
                # Slice-typed Index result (`rows[0]`) -- same shared
                # scratch slot as the Slice case above, via
                # gen_slice_value_into's Index case.
                temp = self._unnamed_slice_temp_offset
                instructions = self.gen_slice_value_into(expr, Memory('rbp', temp))
                instructions.append(MovQ(src=Memory('rbp', temp + 8), dst=len_dst))
                instructions.append(MovQ(src=Memory('rbp', temp + 16), dst=cap_dst))
                instructions.append(MovQ(src=Memory('rbp', temp), dst=addr_dst))
                return instructions, len_dst, cap_dst
            if isinstance(expr, Field):
                # Slice-typed Field result (`p.values`) -- same shared
                # scratch slot, via gen_slice_value_into's Field case.
                temp = self._unnamed_slice_temp_offset
                instructions = self.gen_slice_value_into(expr, Memory('rbp', temp))
                instructions.append(MovQ(src=Memory('rbp', temp + 8), dst=len_dst))
                instructions.append(MovQ(src=Memory('rbp', temp + 16), dst=cap_dst))
                instructions.append(MovQ(src=Memory('rbp', temp), dst=addr_dst))
                return instructions, len_dst, cap_dst
            if isinstance(expr, Call):
                # Slice-returning Call, via the hidden-pointer
                # convention (gen_slice_call_into) -- same shared
                # scratch slot as the other non-Variable cases.
                temp = self._unnamed_slice_temp_offset
                instructions = self.gen_slice_call_into(Memory('rbp', temp), expr)
                instructions.append(MovQ(src=Memory('rbp', temp + 8), dst=len_dst))
                instructions.append(MovQ(src=Memory('rbp', temp + 16), dst=cap_dst))
                instructions.append(MovQ(src=Memory('rbp', temp), dst=addr_dst))
                return instructions, len_dst, cap_dst
            raise CodegenError(
                f"Cannot use a {type(expr).__name__} directly as the "
                f"base of an index or slice expression when it's "
                f"slice-typed -- assign it to a named variable first"
            )
        raise CodegenError(f"Cannot index or slice a value of type {base_type}")

    def gen_array_address_into(self, expr: Node, dst: Register) -> list[Instruction]:
        """Computes the ADDRESS of an array-typed expression -- a
        Variable, an Index resolving to a sub-array (outer dimensions
        of a multi-dimensional access), or a Field resolving to an
        array-typed struct field -- into the 64-bit register `dst`.
        `dst` must already be a 64-bit register -- addresses are
        always 64-bit, regardless of how wide the array's elements are.

        A heap-allocated Variable needs a genuinely different
        instruction here, not just a different offset: its slot holds
        a POINTER to the array's actual data, not the data itself, so
        getting the address means LOADING that pointer (movq) rather
        than computing the slot's own address (leaq). Every other
        array-address computation in this file goes through this one
        method for a bare Variable, so this is the only place that
        distinction needs to be made.
        """
        if isinstance(expr, Variable):
            offset = self._local_offset(expr.name)
            array_type = self._local_type(expr.name)
            if self._is_heap_allocated(self._local_decl_id(expr.name), array_type):
                return [MovQ(src=Memory('rbp', offset), dst=dst)]
            return [LeaQFrame(offset=offset, dst=dst)]
        if isinstance(expr, Index):
            return self.gen_index_address_into(expr, dst)
        if isinstance(expr, Field):
            return self.gen_field_address_into(expr, dst)
        raise CodegenError(f"Cannot compute an array address for: {expr!r}")

    def gen_index_address_into(self, expr: Index, dst: Register) -> list[Instruction]:
        """Computes the address of `expr.array[expr.index]` into `dst`
        -- the shared foundation for reading an element
        (gen_expr_into's Index case), writing one (gen_index_assign),
        and reading a whole SUB-array for multi-dimensional access
        (this method's own recursive base case, via
        gen_array_address_into, when `expr.array` is itself an Index).

        `expr.array` can be array- OR slice-typed (indexing into a
        slice, `s[i]`, uses this same method) -- see
        gen_indexable_base_into for how the base's address and length
        are computed either way. For an array base, the length is an
        Imm known at compile time; for a slice base, it's a runtime
        value kept alive in `len_reg` across evaluating the index
        expression.

        Includes a runtime bounds check: an out-of-range index prints
        a message and calls abort() rather than silently reading or
        writing adjacent stack memory -- which, given arrays live in
        the same frame as the saved return address and callee-saved
        registers, could otherwise corrupt exactly the state that
        keeps `call`/`ret` working, not just return a wrong value.

        `expr.array`'s address (and, for a slice base, its length) is
        computed first and protected on the stack while the index
        expression is evaluated, the same push-before-recursing
        pattern used everywhere else in this file. This works
        correctly regardless of what register `dst` itself is: every
        value that needs to survive is protected on the stack, and the
        final address is only written into `dst` as the last step.
        """
        array_type = type_of(expr.array)
        element_stride = type_byte_width(array_type.element_type, self.struct_registry)

        # len_reg only matters for a slice base (a runtime length);
        # picked dynamically since dst could be any register a caller
        # passes. cap_reg is never read afterward -- ordinary indexing
        # never needs capacity -- but gen_indexable_base_into's
        # contract always populates it, so this still needs a real,
        # distinct register to receive it into.
        len_reg = Register('rdx' if dst.name != 'rdx' else 'r10')
        cap_reg = next(Register(r) for r in ('rdx', 'r10', 'r11') if r not in (dst.name, len_reg.name))

        instructions, length_operand, _ = self.gen_indexable_base_into(expr.array, dst, len_reg, cap_reg)
        instructions.append(Push(dst))
        is_runtime_length = isinstance(length_operand, Register)
        if is_runtime_length:
            instructions.append(Push(len_reg))

        instructions.extend(self.gen_expr_into(expr.index, Register('eax')))

        # Unsigned comparison: catches index >= length AND index < 0 in
        # one check, since a negative int, reinterpreted unsigned,
        # becomes a huge positive number.
        if is_runtime_length:
            instructions.append(Pop(len_reg))
            len_reg_32 = Register({'rdx': 'edx', 'r10': 'r10d'}[len_reg.name])
            instructions.append(Cmp(src=len_reg_32, dst=Register('eax')))
        else:
            instructions.append(Cmp(src=length_operand, dst=Register('eax')))
        instructions.append(Jae(self._get_bounds_check_fail_label("array index out of bounds")))
        # A plain 32-bit imul is safe here: the bounds check above
        # already guarantees the index is small and non-negative, and
        # a 32-bit write zero-extends into the full 64-bit rax.
        instructions.append(IMul(src=Imm(element_stride), dst=Register('eax')))
        instructions.append(Pop(Register('rcx')))  # restore expr.array's base address
        instructions.append(AddQ(src=Register('rax'), dst=Register('rcx')))
        instructions.append(MovQ(src=Register('rcx'), dst=dst))
        return instructions

    def gen_slice_into(self, expr: Slice, dst_mem: Memory) -> list[Instruction]:
        """Generates `expr.array[expr.low:expr.high]`'s resulting
        {ptr, len, cap} descriptor directly into dst_mem (ptr at
        offset+0, len at offset+8, cap at offset+16) -- the slice
        counterpart to gen_array_value_into, dispatched from
        gen_slice_value_into wherever a slice-typed value needs to be
        produced.

        The base's address and length are computed first (see
        gen_indexable_base_into), then `low` and `high` are resolved
        -- each defaulting to 0 / the base's length when omitted --
        and finally bounds-checked against each other and the base's
        length before the resulting ptr/len/cap are computed and
        written.

        Every intermediate value is protected on the CPU stack across
        evaluating whichever of expr.low/expr.high are present, pushed
        in a specific order (address, then length if runtime, then
        high, then low) and popped in exact reverse -- except
        defaulting `high` to the base's runtime length, which peeks
        the top of the stack (`(%rsp)`, no pop) since nothing else has
        been pushed since the length was.

        Bounds checks use `ja` (strictly "above"), not `jae`: unlike
        ordinary indexing, `low` and `high` are both allowed to equal
        the base's own CAP (`arr[5:5]` on a 5-element array is a
        valid, empty-slice-producing expression) -- checked against
        CAP, not len, matching Go's actual re-slicing rule: `high` may
        reach the base's remaining capacity, not just its current
        length, which is what lets a re-sliced view grow into room a
        prior append (or the base's own construction) already
        reserved. cap is computed as base_cap - low, inheriting the
        base's remaining capacity from the new starting point, rather
        than simply matching the newly-computed len.

        dst_mem.base is protected on the stack too, whenever it isn't
        'rbp', across all of the above -- pushed before even
        gen_indexable_base_into runs and popped back right before the
        final writes, as the OUTERMOST push/pop pair around this
        method's own stack discipline. Needed once a slice-typed value
        can be produced somewhere other than an ordinary local slot: an
        array literal whose own elements are themselves slices writes
        each element by calling this method with dst_mem.base equal to
        the outer array's own base. Found necessary by a real bug in
        this exact area, not assumed defensively -- this method used
        to just assert dst_mem.base == 'rbp' and refuse anything else.
        """
        protect_dst = dst_mem.base != 'rbp'
        instructions = []
        if protect_dst:
            instructions.append(Push(Register(dst_mem.base)))

        base_type = type_of(expr.array)
        element_stride = type_byte_width(base_type.element_type, self.struct_registry)

        addr_reg = Register('rbx')
        len_reg = Register('r11')
        cap_reg = Register('r14')
        base_instructions, length_operand, cap_operand = self.gen_indexable_base_into(expr.array, addr_reg, len_reg, cap_reg)
        instructions.extend(base_instructions)
        is_runtime_length = isinstance(length_operand, Register)

        instructions.append(Push(addr_reg))
        if is_runtime_length:
            # Pushed cap BEFORE len (reverse of the descriptor's own
            # field order) so len ends up on TOP of the stack, for the
            # "peek at (%rsp) for len's default" logic just below.
            instructions.append(Push(cap_reg))
            instructions.append(Push(len_reg))

        # Resolve `high` before `low`, so defaulting it (when the
        # base's length is a runtime value) can safely peek the top of
        # the stack. high still defaults to the base's own LEN here,
        # not its cap -- `arr[3:]` means "from 3 to the current end" --
        # only the upper bound high is allowed to reach when explicitly
        # given has changed.
        if expr.high is not None:
            instructions.extend(self.gen_expr_into(expr.high, Register('eax')))
        elif is_runtime_length:
            instructions.append(Mov(src=Memory('rsp', 0), dst=Register('eax')))
        else:
            instructions.append(Mov(src=Imm(base_type.size), dst=Register('eax')))
        instructions.append(Push(Register('rax')))

        if expr.low is not None:
            instructions.extend(self.gen_expr_into(expr.low, Register('eax')))
        else:
            instructions.append(Mov(src=Imm(0), dst=Register('eax')))
        instructions.append(Push(Register('rax')))

        low_reg = Register('r10')
        high_reg = Register('r9')
        low_32 = Register('r10d')
        high_32 = Register('r9d')

        instructions.append(Pop(low_reg))
        instructions.append(Pop(high_reg))
        if is_runtime_length:
            instructions.append(Pop(len_reg))
            instructions.append(Pop(cap_reg))
        instructions.append(Pop(addr_reg))

        # Bounds check: 0 <= low <= high <= CAP (not len) -- see this
        # method's docstring for why cap, not len, is the bound.
        fail_label = self._get_bounds_check_fail_label("slice bounds out of range")
        cap_op = Register('r14d') if is_runtime_length else cap_operand
        instructions.append(Cmp(src=cap_op, dst=low_32))
        instructions.append(Ja(fail_label))
        instructions.append(Cmp(src=cap_op, dst=high_32))
        instructions.append(Ja(fail_label))
        instructions.append(Cmp(src=high_32, dst=low_32))
        instructions.append(Ja(fail_label))

        # new_cap = cap - low -- the base's remaining capacity from
        # the new starting point (see docstring). Computed into its
        # own register, BEFORE low is scaled below, for the same
        # reason new_len (high - low, just after) has to be: scaling
        # would destroy the unscaled value both still need. Mov (not
        # MovQ) since cap_op may be a 32-bit Imm (an array base) or a
        # 32-bit register (a slice base) alike.
        new_cap_32 = Register('r13d')
        instructions.append(Mov(src=cap_op, dst=new_cap_32))
        instructions.append(Sub(src=low_32, dst=new_cap_32))

        # len = high - low, computed BEFORE low is scaled below --
        # scaling would destroy the unscaled value this still needs.
        instructions.append(Sub(src=low_32, dst=high_32))
        # ptr = addr + low * element_stride. A plain 32-bit imul is
        # safe here: the bounds check above already guarantees low is
        # small and non-negative, and a 32-bit write zero-extends into
        # the full 64-bit low_reg.
        instructions.append(IMul(src=Imm(element_stride), dst=low_32))
        instructions.append(AddQ(src=low_reg, dst=addr_reg))

        # dst_mem.base is only needed again now, for these final
        # writes -- restored after every other computation above
        # (including the bounds check, which never falls through here
        # on failure -- it aborts) has finished.
        if protect_dst:
            instructions.append(Pop(Register(dst_mem.base)))
        instructions.append(MovQ(src=addr_reg, dst=Memory(dst_mem.base, dst_mem.offset)))
        instructions.append(MovQ(src=high_reg, dst=Memory(dst_mem.base, dst_mem.offset + 8)))
        instructions.append(MovQ(src=Register('r13'), dst=Memory(dst_mem.base, dst_mem.offset + 16)))
        return instructions

    def gen_slice_value_into(self, expr: Node, dst_mem: Memory) -> list[Instruction]:
        """Stores a slice-typed expression's VALUE (its {ptr, len,
        cap} descriptor) into dst_mem -- an arbitrary Memory operand,
        not just an ordinary local slot: dst_mem.base is 'rax'
        whenever this writes a slice-typed element into a heap-
        allocated array literal (every slice literal's backing array
        always is), which is what first made every case below need
        real protection rather than assuming 'rbp'. Dispatched on the
        producing expression:
          - Slice (`arr[1:3]`, or a slice literal): computed directly
            via gen_slice_into, which protects dst_mem.base
            internally.
          - Variable (`s2 = s1`): a flat 24-byte copy of an existing
            slice's descriptor. Deliberately NOT routed through
            gen_array_copy, which handles an ARRAY whose ELEMENTS are
            slices, not a bare slice descriptor -- a fixed three-field
            copy is simpler here. Uses %r8/%r9/%r10 as scratch, not
            %rax: dst_mem.base is never %r8/%r9/%r10 (only ever 'rbp'
            or 'rax'), so this case needs no push/pop protection,
            unlike the others -- an earlier version used %rax for this
            and was silently wrong whenever dst_mem.base was 'rax'
            itself.
          - Call (`otherFn()`, also slice-returning): hidden-output-
            pointer convention, via gen_slice_call_into -- structurally
            identical to gen_array_value_into's own Call case.
          - Index (`rows[0]`, a slice-typed array element): the
            element's address is computed first, then its 24-byte
            descriptor is read through it -- the same flat copy the
            Variable case does, from a computed rather than fixed
            address.
          - Field (`p.values`, a slice-typed struct field): identical
            to Index, one level over.
          - ArrayLiteral (an untyped literal, `[]int s = [1, 2, 3]`,
            flowing into a slice-typed target -- the general typed
            form, `[]int[1, 2, 3]`, never reaches this case, since it
            parses as a Slice wrapping an ArrayLiteral): mallocs a
            fresh backing array and writes the literal's elements into
            it. cap is set equal to len -- a fresh literal's backing
            array has no spare room to grow into yet.

        Every case that does real work between "start" and "write the
        result" -- every one except Variable and Call -- protects
        dst_mem.base on the stack across that work whenever it isn't
        'rbp'. This generalizes what gen_array_literal_into's own
        scalar-element case established for a single value -- see its
        docstring for the real (not hypothetical) bug that made it
        necessary there.

        NoneLiteral is NOT handled here -- its resolved type
        (Type.NONE) never matches SLICE, so it can't reach this method
        through _gen_store's ordinary dispatch; see gen_none_into and
        gen_var_decl/gen_assign's NoneLiteral short-circuit for where
        that's handled instead.
        """
        protect_dst = dst_mem.base != 'rbp'

        if isinstance(expr, Slice):
            return self.gen_slice_into(expr, dst_mem)

        if isinstance(expr, Variable):
            src_offset = self._local_offset(expr.name)
            return [
                MovQ(src=Memory('rbp', src_offset), dst=Register('r8')),
                MovQ(src=Register('r8'), dst=Memory(dst_mem.base, dst_mem.offset)),
                MovQ(src=Memory('rbp', src_offset + 8), dst=Register('r9')),
                MovQ(src=Register('r9'), dst=Memory(dst_mem.base, dst_mem.offset + 8)),
                MovQ(src=Memory('rbp', src_offset + 16), dst=Register('r10')),
                MovQ(src=Register('r10'), dst=Memory(dst_mem.base, dst_mem.offset + 16)),
            ]

        if isinstance(expr, Call):
            if expr.name == 'append':
                return self.gen_append_call_into(expr, dst_mem)
            return self.gen_slice_call_into(dst_mem, expr)

        if isinstance(expr, Index):
            instructions = []
            if protect_dst:
                instructions.append(Push(Register(dst_mem.base)))
            addr_reg = Register('r11')
            instructions.extend(self.gen_index_address_into(expr, addr_reg))
            instructions.append(MovQ(src=Memory(addr_reg.name, 0), dst=Register('r8')))
            instructions.append(MovQ(src=Memory(addr_reg.name, 8), dst=Register('r9')))
            instructions.append(MovQ(src=Memory(addr_reg.name, 16), dst=Register('r10')))
            if protect_dst:
                instructions.append(Pop(Register(dst_mem.base)))
            instructions.append(MovQ(src=Register('r8'), dst=Memory(dst_mem.base, dst_mem.offset)))
            instructions.append(MovQ(src=Register('r9'), dst=Memory(dst_mem.base, dst_mem.offset + 8)))
            instructions.append(MovQ(src=Register('r10'), dst=Memory(dst_mem.base, dst_mem.offset + 16)))
            return instructions

        if isinstance(expr, Field):
            instructions = []
            if protect_dst:
                instructions.append(Push(Register(dst_mem.base)))
            addr_reg = Register('r11')
            instructions.extend(self.gen_field_address_into(expr, addr_reg))
            instructions.append(MovQ(src=Memory(addr_reg.name, 0), dst=Register('r8')))
            instructions.append(MovQ(src=Memory(addr_reg.name, 8), dst=Register('r9')))
            instructions.append(MovQ(src=Memory(addr_reg.name, 16), dst=Register('r10')))
            if protect_dst:
                instructions.append(Pop(Register(dst_mem.base)))
            instructions.append(MovQ(src=Register('r8'), dst=Memory(dst_mem.base, dst_mem.offset)))
            instructions.append(MovQ(src=Register('r9'), dst=Memory(dst_mem.base, dst_mem.offset + 8)))
            instructions.append(MovQ(src=Register('r10'), dst=Memory(dst_mem.base, dst_mem.offset + 16)))
            return instructions

        if isinstance(expr, ArrayLiteral):
            instructions = []
            if protect_dst:
                instructions.append(Push(Register(dst_mem.base)))
            instructions.extend(self.gen_array_literal_heap_alloc_into(expr))
            instructions.append(MovQ(src=Register('rax'), dst=Register('r8')))
            element_count = len(expr.elements)
            if protect_dst:
                instructions.append(Pop(Register(dst_mem.base)))
            instructions.append(MovQ(src=Register('r8'), dst=Memory(dst_mem.base, dst_mem.offset)))
            instructions.append(MovQ(src=Imm(element_count), dst=Memory(dst_mem.base, dst_mem.offset + 8)))
            instructions.append(MovQ(src=Imm(element_count), dst=Memory(dst_mem.base, dst_mem.offset + 16)))
            return instructions

        raise CodegenError(f"No codegen rule for a slice-typed value: {expr!r}")

    def gen_array_copy(self, dst_mem: Memory, src_mem: Memory, array_type: Type) -> list[Instruction]:
        """Copies array_type's worth of data from src_mem to dst_mem
        -- both arbitrary Memory operands -- via a flat sequence of
        movq/movl/movb instructions. A multi-dimensional array is just
        one contiguous block of leaf values in row-major order for
        copying purposes, so no per-dimension logic is needed, just
        the total byte width and the leaf element's own width.

        Each leaf-sized chunk is copied as a flat run of 8-byte movqs,
        then one trailing 4-byte movl if at least 4 bytes remain, then
        a trailing run of 1-byte movbs for whatever's left (0 to 3
        bytes) -- correct for ANY leaf width, not just a multiple of 4
        the way this used to assume. int8/uint8's 1-byte storage broke
        that assumption (a bare int8/uint8 leaf has width 1; a struct
        leaf containing one can land on any width at all): the old
        two-tier version silently copied NOTHING for either shape -- a
        real, found bug, not a hypothetical one.

        A raw, flat byte copy is always semantically identical to
        copying a value "as" whatever logical type or fields those
        bytes represent, given this language's value semantics: there's
        no reference counting, copy-constructor, or write barrier
        anywhere that a flat byte copy could get wrong. This is why a
        slice element (24 bytes: ptr, len, cap) already worked before
        struct existed, and why a struct containing a nested
        array/slice/struct needs nothing more than this same flat copy.

        The scratch register shuttling each chunk is picked dynamically
        to differ from BOTH src_mem's and dst_mem's own base register
        -- otherwise loading a value into it would destroy the address
        a later iteration still needs. Found as a real bug: gen_return
        passes Memory('rax', 0) as the destination when writing an
        array through a received hidden return pointer, and using %rax
        as scratch there destroyed that address before it could even
        be written anywhere.
        """
        leaf = leaf_type(array_type)
        used_bases = {src_mem.base, dst_mem.base}
        scratch_64, scratch_32 = next(
            (r64, r32) for r64, r32 in [('rax', 'eax'), ('rcx', 'ecx'), ('rdx', 'edx')]
            if r64 not in used_bases
        )
        leaf_width = type_byte_width(leaf, self.struct_registry)
        total = type_byte_width(array_type, self.struct_registry)
        instructions = []
        off = 0
        while off < total:
            # Copy exactly leaf_width bytes starting at offset `off`:
            # as many 8-byte movq chunks as fit, then one 4-byte movl
            # if at least 4 bytes remain, then a trailing run of
            # 1-byte movbs for whatever's left (0 to 3 bytes, so at
            # most three movb pairs, never a real loop of its own).
            chunk_off = 0
            while leaf_width - chunk_off >= 8:
                field_src = Memory(src_mem.base, src_mem.offset + off + chunk_off)
                field_dst = Memory(dst_mem.base, dst_mem.offset + off + chunk_off)
                instructions.append(MovQ(src=field_src, dst=Register(scratch_64)))
                instructions.append(MovQ(src=Register(scratch_64), dst=field_dst))
                chunk_off += 8
            if leaf_width - chunk_off >= 4:
                field_src = Memory(src_mem.base, src_mem.offset + off + chunk_off)
                field_dst = Memory(dst_mem.base, dst_mem.offset + off + chunk_off)
                instructions.append(Mov(src=field_src, dst=Register(scratch_32)))
                instructions.append(Mov(src=Register(scratch_32), dst=field_dst))
                chunk_off += 4
            scratch_8 = None
            while leaf_width - chunk_off >= 1:
                if scratch_8 is None:
                    scratch_8 = as_byte_register(Register(scratch_32))
                field_src = Memory(src_mem.base, src_mem.offset + off + chunk_off)
                field_dst = Memory(dst_mem.base, dst_mem.offset + off + chunk_off)
                instructions.append(MovB(src=field_src, dst=scratch_8))
                instructions.append(MovB(src=scratch_8, dst=field_dst))
                chunk_off += 1
            off += leaf_width
        return instructions

    def gen_array_arg_address_into(self, expr: Node, dst: Register) -> list[Instruction]:
        """Computes the address to pass for an array-typed function-
        call argument, into `dst`. A Variable or an Index yielding a
        sub-array already has a real address (see
        gen_array_address_into); an ArrayLiteral or a call returning
        an array used DIRECTLY as an argument has no home of its own,
        so it's materialized first (see
        _gen_materialize_argument_temp_into).

        The callee copies from this address into its own local slot on
        entry, so what's passed here only needs to stay valid for that
        one copy -- the caller's own array is never mutated through it:
        the callee's copy is independent, preserving value semantics
        across the call the same way an ordinary `arr2 = arr1` does
        within a single function."""
        if isinstance(expr, (Variable, Index)):
            return self.gen_array_address_into(expr, dst)
        if isinstance(expr, (ArrayLiteral, Call)):
            return self._gen_materialize_argument_temp_into(expr, type_of(expr), dst)
        raise CodegenError(
            f"Array-typed call arguments must be a variable, an "
            f"indexing expression, an array literal, or a call, not "
            f"{type(expr).__name__}"
        )

    def gen_slice_arg_into(
            self, expr: Node, ptr_dst: Register, len_dst: Register, cap_dst: Register) -> list[Instruction]:
        """Computes a slice-typed call ARGUMENT's ptr/len/cap directly
        into ptr_dst/len_dst/cap_dst. A Variable or NoneLiteral
        (`none`) already has its descriptor sitting somewhere real (a
        local slot, or nowhere at all -- `none` is just three
        immediate zeros); anything else -- a slice literal or
        re-slice, a slice-returning Call, or a slice-typed Field/Index
        -- has no pre-existing descriptor, and is materialized first
        via gen_slice_value_into into the same shared, per-function
        scratch slot gen_indexable_base_into's analogous cases use
        (_unnamed_slice_temp_offset), then read straight back out.

        Reusing that one shared slot here -- rather than needing its
        own per-occurrence storage the way an array/struct-typed
        argument does -- is safe for a different reason than
        gen_indexable_base_into's own "fully drained before anything
        else can reuse it" argument: a slice argument is passed BY
        VALUE -- three register values, immediately pushed onto the
        stack right after this method returns -- not by address, so
        nothing about the call ever needs this scratch slot's contents
        to still be valid afterward; only the pushed register values
        do, and those already live on the stack by then. That's also
        what makes two slice-typed arguments to the same call safe with
        only one shared slot: _gen_call_arguments_into evaluates
        arguments strictly one at a time, so each is fully read out of
        the slot and pushed before the next argument's materialization
        touches the slot again -- and the same strictly-nested
        reasoning covers a slice-returning call whose own argument is
        itself another unnamed slice."""
        if isinstance(expr, NoneLiteral):
            return [
                MovQ(src=Imm(0), dst=ptr_dst),
                MovQ(src=Imm(0), dst=len_dst),
                MovQ(src=Imm(0), dst=cap_dst),
            ]
        if isinstance(expr, Variable):
            offset = self._local_offset(expr.name)
            return [
                MovQ(src=Memory('rbp', offset), dst=ptr_dst),
                MovQ(src=Memory('rbp', offset + 8), dst=len_dst),
                MovQ(src=Memory('rbp', offset + 16), dst=cap_dst),
            ]
        temp = self._unnamed_slice_temp_offset
        instructions = self.gen_slice_value_into(expr, Memory('rbp', temp))
        instructions.append(MovQ(src=Memory('rbp', temp), dst=ptr_dst))
        instructions.append(MovQ(src=Memory('rbp', temp + 8), dst=len_dst))
        instructions.append(MovQ(src=Memory('rbp', temp + 16), dst=cap_dst))
        return instructions

    def gen_array_call_into(self, dst_mem: Memory, expr: Call, array_type: Type) -> list[Instruction]:
        """Calls a function that returns an array, writing its result
        directly into dst_mem via the hidden-pointer convention: the
        callee receives a pointer to where its result should go as an
        extra, FIRST argument (in %rdi), with every genuine argument
        shifted one register position later. The callee writes its
        return value directly through that pointer -- there's nothing
        for the CALLER to copy afterward. This is also what makes
        forwarding one array-returning call's result straight out of
        another free (`return bar()`): the same destination address
        just gets passed one level deeper, with no intermediate copy.

        dst_mem's address is computed and pushed onto the stack FIRST,
        before any argument is evaluated, so it survives regardless of
        what an argument expression does internally -- the same push-
        before-evaluating-something-else discipline used everywhere
        else in this file a value needs to survive past a sub-
        expression. Every other argument is handled by
        _gen_call_arguments_into (reg_shift=1, since the hidden pointer
        already occupies the first register slot).
        """
        total_slots = 1 + self._total_arg_slots(expr.args)  # +1: the hidden pointer itself
        if total_slots > 6:
            raise CodegenError(
                f"Call to '{expr.name}' needs {total_slots} argument "
                f"register(s) (the hidden output pointer uses one, "
                f"and a slice-typed argument needs 3) -- this compiler "
                f"only supports up to 6"
            )
        instructions = self._gen_address_of_memory_into(dst_mem, Register('rax'))
        instructions.append(Push(Register('rax')))
        instructions.extend(self._gen_call_arguments_into(expr.args, reg_shift=1))
        instructions.append(Pop(Register('rdi')))
        instructions.append(CallInstr(expr.name))
        return instructions

    def gen_slice_call_into(self, dst_mem: Memory, expr: Call) -> list[Instruction]:
        """Calls a function that returns a slice, writing its result
        directly into dst_mem via the same hidden-pointer convention
        gen_array_call_into uses, except this writes 24 bytes (a
        slice's {ptr, len, cap} descriptor) rather than an array's
        type-dependent width. Slices used to return via a dedicated
        %rax:%rdx two-register convention; that stopped fitting once a
        slice's descriptor grew a third field (cap), so slice returns
        now share the same mechanism arrays already had. This is also
        what makes forwarding one slice-returning call's result
        straight out of another free (`return otherFn()`): the same
        destination address just gets passed one level deeper, with no
        intermediate copy.
        """
        total_slots = 1 + self._total_arg_slots(expr.args)  # +1: the hidden pointer itself
        if total_slots > 6:
            raise CodegenError(
                f"Call to '{expr.name}' needs {total_slots} argument "
                f"register(s) (the hidden output pointer uses one, "
                f"and a slice-typed argument needs 3) -- this compiler "
                f"only supports up to 6"
            )
        instructions = self._gen_address_of_memory_into(dst_mem, Register('rax'))
        instructions.append(Push(Register('rax')))
        instructions.extend(self._gen_call_arguments_into(expr.args, reg_shift=1))
        instructions.append(Pop(Register('rdi')))
        instructions.append(CallInstr(expr.name))
        return instructions

    def gen_array_literal_into(self, dst_mem: Memory, expr: ArrayLiteral, array_type: Type) -> list[Instruction]:
        """Stores an array literal's elements directly into consecutive
        memory locations starting at dst_mem -- almost always a fixed
        local slot, but see gen_array_value_into for why this takes a
        general Memory operand rather than a bare offset. Each element
        is evaluated via ordinary gen_expr_into, except when the
        element type is itself an ARRAY (a multi-dimensional literal's
        "elements" are ArrayLiterals, handled by recursing through
        gen_array_value_into), a SLICE (an array whose elements are
        slices, e.g. the synthesized outer literal of `[][]int[[1, 2],
        [3, 4]]` -- handled by gen_slice_value_into), or a STRUCT (an
        array of structs -- handled by gen_struct_value_into, which
        covers every shape a struct-typed element can take, including
        a struct literal directly).

        This STRUCT case used to not exist: an array-of-structs literal
        would fall through to the scalar path below, whose
        gen_expr_into call flatly rejects any struct-typed read --
        this failed even for the simplest case, `[p1, p2]` with
        p1/p2 ordinary struct variables. Every OTHER operation on an
        array of structs already worked, since each routes through
        gen_array_copy or gen_index_assign/gen_field_address_into
        rather than through this method's per-element construction
        path -- literal construction specifically was the one gap.

        dst_mem's base register is protected on the stack across each
        element's value computation whenever it isn't 'rbp' -- found
        necessary by a real bug during development: 'rbp' is never
        clobbered by gen_expr_into, so no protection is needed there,
        but a computed or received address held in a general-purpose
        register (e.g. Memory('rax', 0), a hidden return pointer or a
        slice literal's freshly-mallocd backing array) is exactly the
        kind of register gen_expr_into's value computation, which
        always targets %eax/%rax, can and did clobber -- silently
        overwriting the destination address before a single element
        was written through it. Neither the SLICE nor the STRUCT case
        needs protection of its own here, unlike the scalar case
        below: gen_slice_value_into/gen_struct_value_into both already
        protect dst_mem.base internally, so by the time either
        returns, dst_mem.base is guaranteed correct again."""
        element_type = array_type.element_type
        element_width = type_byte_width(element_type, self.struct_registry)
        protect_dst = dst_mem.base != 'rbp'
        instructions = []
        for i, elem_expr in enumerate(expr.elements):
            elem_mem = Memory(dst_mem.base, dst_mem.offset + i * element_width)
            if element_type.kind == TypeKind.ARRAY:
                instructions.extend(self.gen_array_value_into(elem_expr, elem_mem, element_type))
                continue
            if element_type.kind == TypeKind.SLICE:
                instructions.extend(self.gen_slice_value_into(elem_expr, elem_mem))
                continue
            if element_type.kind == TypeKind.STRUCT:
                instructions.extend(self.gen_struct_value_into(elem_expr, elem_mem, element_type))
                continue
            if protect_dst:
                instructions.append(Push(Register(dst_mem.base)))
            instructions.extend(self.gen_expr_into(elem_expr, Register('eax')))
            if element_type == Type.STR:
                if protect_dst:
                    instructions.append(MovQ(src=Register('rax'), dst=Register('r8')))
                    instructions.append(Pop(Register(dst_mem.base)))
                    instructions.append(MovQ(src=Register('r8'), dst=elem_mem))
                else:
                    instructions.append(MovQ(src=Register('rax'), dst=elem_mem))
            else:
                if protect_dst:
                    if element_type == Type.INT64:
                        # A full 64-bit shuttle -- an ordinary 32-bit
                        # Mov here would discard int64's high 32 bits
                        # before _gen_write_scalar_from gets a chance
                        # to write them correctly.
                        instructions.append(MovQ(src=Register('rax'), dst=Register('r8')))
                    else:
                        instructions.append(Mov(src=Register('eax'), dst=Register('r8d')))
                    instructions.append(Pop(Register(dst_mem.base)))
                    instructions.extend(self._gen_write_scalar_from(Register('r8d'), element_type, elem_mem))
                else:
                    instructions.extend(self._gen_write_scalar_from(Register('eax'), element_type, elem_mem))
        return instructions

    def gen_array_value_into(self, expr: Node, dst_mem: Memory, array_type: Type) -> list[Instruction]:
        """Stores an array-typed expression's VALUE into dst_mem,
        matching array_type's shape. This is the array counterpart to
        _gen_store's scalar path -- an array can't fit into a single
        register the way an int/bool/str value can, dispatched on the
        producing expression:
          - ArrayLiteral: each element stored directly (see
            gen_array_literal_into).
          - Variable: a copy from wherever the source's data actually
            lives (see gen_array_copy) into dst_mem. For a stack-
            allocated source, that's a flat offset-to-offset copy. For
            a heap-allocated source, the slot holds a POINTER rather
            than the data itself, so that pointer is loaded first
            (protecting dst_mem's base register across the load, via
            _gen_protecting_dst_across), and the copy reads through it
            instead. Either way, this is what makes `arr2 = arr1` a
            real, independent copy rather than a pointer alias --
            heap-backed storage doesn't change that guarantee, only
            where the bytes being copied live.
          - Index (a sub-array, `[3]int row = matrix[i]`): its SOURCE
            address is computed first (gen_array_address_into), since
            it depends on a runtime index, then copied from that
            address. dst_mem is protected across that computation,
            since it includes bounds-checking and index arithmetic
            that freely uses %rax/%rcx internally.
          - Call (a function returning an array): calls through the
            hidden-output-pointer convention -- see gen_array_call_into.

        dst_mem is a general Memory operand, not always a fixed local
        slot: it's Memory('rbp', offset) for an ordinary local, but
        Memory(some_reg, 0) when the destination is itself a computed
        or received address -- e.g. gen_return uses this to write an
        array-typed return value directly through the hidden pointer
        it received, without materializing an intermediate local copy.
        """
        if isinstance(expr, ArrayLiteral):
            return self.gen_array_literal_into(dst_mem, expr, array_type)
        if isinstance(expr, Variable):
            src_offset = self._local_offset(expr.name)
            src_type = self._local_type(expr.name)
            if self._is_heap_allocated(self._local_decl_id(expr.name), src_type):
                load_ptr = gen_protecting_dst_across(
                    dst_mem, [MovQ(src=Memory('rbp', src_offset), dst=Register('rbx'))]
                )
                return load_ptr + self.gen_array_copy(dst_mem, Memory('rbx', 0), array_type)
            return self.gen_array_copy(dst_mem, Memory('rbp', src_offset), array_type)
        if isinstance(expr, Index):
            addr_instructions = gen_protecting_dst_across(
                dst_mem, self.gen_array_address_into(expr, Register('rbx'))
            )
            return addr_instructions + self.gen_array_copy(dst_mem, Memory('rbx', 0), array_type)
        if isinstance(expr, Call):
            return self.gen_array_call_into(dst_mem, expr, array_type)
        raise CodegenError(f"No codegen rule for an array-typed value: {expr!r}")

    def gen_array_literal_side_effects_only(self, expr: ArrayLiteral) -> list[Instruction]:
        """A bare array-literal statement (`[3]int[1, 2, 3]` alone,
        with no assignment) never needs its VALUE materialized
        anywhere -- nothing ever reads it as a coherent array -- so
        rather than reserving a scratch slot sized to fit it (an array
        literal has no natural upper bound the way a slice's fixed
        24-byte descriptor does), this just evaluates each of the
        literal's directly-written elements for whatever side effects
        it might have (e.g. a function call), discarding every result
        -- like any other bare expression statement (see
        gen_expr_stmt).

        Recurses for a nested ArrayLiteral element (a multi-dimensional
        literal used bare). An element that's itself some other, non-
        literal array-, slice-, or struct-typed expression (a Variable,
        an indexed sub-array, an array/struct-returning Call, ...) is a
        real, deliberately out-of-scope gap: reading a bare array-typed
        Variable has no side effect worth preserving, but an array-
        returning Call might, and correctly distinguishing the two
        isn't implemented here. Raises a clear error rather than
        silently skipping (which could drop a real side effect)."""
        instructions = []
        for element in expr.elements:
            if isinstance(element, ArrayLiteral):
                instructions.extend(self.gen_array_literal_side_effects_only(element))
                continue
            element_type = type_of(element)
            if element_type.kind in (TypeKind.ARRAY, TypeKind.SLICE, TypeKind.STRUCT):
                raise CodegenError(
                    f"A bare array-literal statement can't have a "
                    f"{type(element).__name__} element of type "
                    f"{element_type} -- assign the literal to a "
                    f"variable first if you need this element's value "
                    f"or side effect evaluated"
                )
            instructions.extend(self.gen_expr_into(element, Register('eax')))
        return instructions

    def gen_array_equality_into(self, expr: Binary, dst: Register) -> list[Instruction]:
        """`left == right` / `left != right`, both the same array type
        (already guaranteed comparable by semantic.py: the array's
        leaf type is int, bool, str, or a comparable struct -- never a
        slice, or a struct with a slice buried in it, neither of which
        has '==' defined yet).

        `left`/`right` must each already have a real address (a
        Variable, Index, or Field); an array literal or array-
        returning call used directly as an operand isn't supported --
        assign it to a variable first.

        Dispatches on the array's leaf type into one of three
        different comparison strategies, each its own loop helper:
          - int/bool leaf: _gen_array_flat_byte_equality_loop. Neither
            type is a pointer, so the whole array -- however many
            elements, however deeply nested -- can be compared as one
            flat run of bytes, the same "treat a nested array as one
            flat block" trick gen_array_copy uses for copying.
          - str leaf: _gen_array_str_equality_loop. A str element IS a
            pointer, so raw byte-for-byte pointer equality would be
            wrong -- two equal strings can live at different
            addresses. Calls strcmp on each corresponding pair
            instead, like gen_string_compare_into's own str == str
            comparison, minus its concatenation-freeing logic (array
            elements are always fixed, already-allocated storage).
          - struct leaf: _gen_array_struct_equality_loop. A struct's
            fields can be a mix of types, so no single flat-byte-or-
            strcmp strategy covers it -- reuses
            _gen_struct_fields_equality_at_addresses once per element.

        All three loops jump to a shared `mismatch_label` the moment
        any element differs; falling all the way through means every
        element matched, and the final result is two immediate moves
        -- 1/0 for EQUAL, 0/1 for NOT_EQUAL -- the same "compute the
        boolean the long way, then pick the right immediate" shape
        gen_short_circuit uses for AND/OR."""
        array_type = type_of(expr.left)
        leaf = leaf_type(array_type)
        total_width = type_byte_width(array_type, self.struct_registry)

        left_addr = Register('r10')
        right_addr = Register('r11')
        instructions = self.gen_array_address_into(expr.left, left_addr)
        instructions.append(Push(left_addr))
        instructions.extend(self.gen_array_address_into(expr.right, right_addr))
        instructions.append(Pop(left_addr))

        mismatch_label = self.new_label("array_eq_mismatch")
        done_label = self.new_label("array_eq_done")

        if leaf == Type.STR:
            instructions.extend(self._gen_array_str_equality_loop(
                left_addr, right_addr, total_width // 8, mismatch_label
            ))
        elif leaf.kind == TypeKind.STRUCT:
            struct_width = type_byte_width(leaf, self.struct_registry)
            instructions.extend(self._gen_array_struct_equality_loop(
                left_addr, right_addr, leaf.struct_name, total_width // struct_width, mismatch_label
            ))
        else:
            # int8/uint8 need a 1-byte step (a 4-byte step would be a
            # real, out-of-bounds bug for either); int/bool/slice stay
            # at 4 bytes, since type_byte_width guarantees their
            # total_width is a multiple of 4 regardless of nesting.
            step = 1 if leaf in (Type.INT8, Type.UINT8) else 4
            instructions.extend(self._gen_array_flat_byte_equality_loop(
                left_addr, right_addr, total_width, mismatch_label, step=step
            ))

        # Fell all the way through: every element matched.
        instructions.append(Mov(src=Imm(1 if expr.op == BinaryOp.EQUAL else 0), dst=dst))
        instructions.append(Jmp(done_label))
        instructions.append(Label(mismatch_label))
        instructions.append(Mov(src=Imm(0 if expr.op == BinaryOp.EQUAL else 1), dst=dst))
        instructions.append(Label(done_label))
        return instructions

    def _gen_array_flat_byte_equality_loop(
            self,
            left_addr: Register,
            right_addr: Register,
            total_width: int,
            mismatch_label: str,
            step: int = 4) -> list[Instruction]:
        """Compares `total_width` bytes at left_addr/right_addr, `step`
        bytes at a time -- 4 for an int/bool/slice leaf (type_byte_
        width guarantees total_width is a multiple of 4 for any of
        those, however deeply nested), or 1 for an int8/uint8 leaf.

        The 1-byte case is a real, found bug's fix: this loop used to
        always step 4 bytes at a time, which was fine while every leaf
        was 4-or-a-multiple-of-4 bytes wide -- but int8/uint8's 1-byte
        storage means total_width isn't generally a multiple of 4 (a
        [3]int8 array is 3 bytes total). Stepping 4 bytes regardless
        read one byte past the end on every comparison, silently
        comparing adjacent stack memory instead of correctly reporting
        equality.

        The 1-byte step reads each side via MovZX rather than a plain
        4-byte Mov, needing a second register to hold the right side's
        zero-extended value before comparing -- correct regardless of
        whether the actual leaf is signed (int8) or unsigned (uint8):
        byte equality never depends on interpretation, only on whether
        the bits are identical.

        Jumps to mismatch_label the moment any chunk differs, or falls
        through once every chunk has matched. No calls happen in this
        loop, so nothing needs a callee-saved register the way the
        str-leaf loop below does."""
        index_32 = Register('ecx')
        index_64 = Register('rcx')
        loop_start = self.new_label("array_eq_flat_loop")
        loop_done = self.new_label("array_eq_flat_done")
        left_word_addr = Register('r8')
        right_word_addr = Register('r9')
        instructions = [
            Mov(src=Imm(0), dst=index_32),
            Label(loop_start),
            Cmp(src=Imm(total_width), dst=index_32),
            Jae(loop_done),
            MovQ(src=left_addr, dst=left_word_addr),
            AddQ(src=index_64, dst=left_word_addr),
            MovQ(src=right_addr, dst=right_word_addr),
            AddQ(src=index_64, dst=right_word_addr),
        ]
        if step == 1:
            instructions.append(MovZX(src=Memory(left_word_addr.name, 0), dst=Register('eax')))
            instructions.append(MovZX(src=Memory(right_word_addr.name, 0), dst=Register('edx')))
            instructions.append(Cmp(src=Register('edx'), dst=Register('eax')))
        else:
            instructions.append(Mov(src=Memory(left_word_addr.name, 0), dst=Register('eax')))
            instructions.append(Cmp(src=Memory(right_word_addr.name, 0), dst=Register('eax')))
        instructions.append(Jne(mismatch_label))
        instructions.append(Add(src=Imm(step), dst=index_32))
        instructions.append(Jmp(loop_start))
        instructions.append(Label(loop_done))
        return instructions

    def _gen_array_str_equality_loop(
            self,
            left_addr: Register,
            right_addr: Register,
            element_count: int,
            mismatch_label: str) -> list[Instruction]:
        """The str-leaf counterpart to _gen_array_flat_byte_equality_
        loop: each of the array's `element_count` str elements is a
        POINTER, so this calls strcmp on each corresponding pair
        rather than comparing raw pointer bytes.

        strcmp is a real external call, free to clobber any CALLER-
        saved register -- so unlike the flat-byte loop, the two base
        addresses and the loop index all have to live in CALLEE-saved
        registers (%rbx/%r12/%r13) to survive it: every function's own
        prologue/epilogue already saves and restores these four
        unconditionally, so using them as scratch across a call, in
        any function, is always safe."""
        left_base = Register('rbx')
        right_base = Register('r12')
        index_32 = Register('r13d')
        index_64 = Register('r13')
        offset_32 = Register('r14d')
        offset_64 = Register('r14')

        loop_start = self.new_label("array_eq_str_loop")
        loop_done = self.new_label("array_eq_str_done")

        instructions = [
            MovQ(src=left_addr, dst=left_base),
            MovQ(src=right_addr, dst=right_base),
            Mov(src=Imm(0), dst=index_32),
            Label(loop_start),
            Cmp(src=Imm(element_count), dst=index_32),
            Jae(loop_done),
        ]
        # byte offset = index * 8 (each str element is one 8-byte
        # pointer), recomputed fresh each iteration before the call.
        instructions.append(Mov(src=index_32, dst=offset_32))
        instructions.append(IMul(src=Imm(8), dst=offset_32))
        left_elem_addr = Register('r8')
        right_elem_addr = Register('r9')
        instructions.append(MovQ(src=left_base, dst=left_elem_addr))
        instructions.append(AddQ(src=offset_64, dst=left_elem_addr))
        instructions.append(MovQ(src=right_base, dst=right_elem_addr))
        instructions.append(AddQ(src=offset_64, dst=right_elem_addr))
        # Load the string POINTERS at these element addresses straight
        # into strcmp's own argument registers.
        instructions.append(MovQ(src=Memory(left_elem_addr.name, 0), dst=Register('rdi')))
        instructions.append(MovQ(src=Memory(right_elem_addr.name, 0), dst=Register('rsi')))
        instructions.append(CallInstr('strcmp'))
        instructions.append(Cmp(src=Imm(0), dst=Register('eax')))
        instructions.append(Jne(mismatch_label))
        instructions.append(Add(src=Imm(1), dst=index_32))
        instructions.append(Jmp(loop_start))
        instructions.append(Label(loop_done))
        return instructions

    def _gen_array_struct_equality_loop(
            self,
            left_addr: Register,
            right_addr: Register,
            struct_name: str,
            element_count: int,
            mismatch_label: str) -> list[Instruction]:
        """The struct-leaf counterpart to the two array-equality loops
        above: unlike int/bool (one flat byte comparison) or str (one
        strcmp per element), a struct element's fields can be a MIX of
        types, so each element is compared via
        _gen_struct_fields_equality_at_addresses -- the same field-by-
        field machinery gen_struct_equality_into's bare struct-vs-
        struct case uses.

        Uses the same CALLEE-saved register discipline
        _gen_array_str_equality_loop establishes, for the same reason
        (a struct element being compared might itself contain a str
        field, needing a real strcmp call inside
        _gen_struct_fields_equality_at_addresses).

        Beyond that, this loop's left_base/right_base/index
        (%rbx/%r12/%r13) need one MORE layer of protection neither
        sibling loop does: _gen_struct_fields_equality_at_addresses can
        itself recurse back into ONE of these same three array-
        equality loops, for a struct field that's itself an array
        (including, recursively, another array of structs). Any such
        nested loop reuses the exact same fixed register names this
        one does (there's no way to allocate a fresh, distinct set per
        nesting depth at codegen time) -- so without explicitly saving
        this loop's own %rbx/%r12/%r13 across the per-element
        comparison call, a struct field needing one of those nested
        loops would silently corrupt this OUTER loop's base addresses
        and index. Protecting them via an ordinary push/pop around
        that one call is what makes this correct at any nesting depth:
        at every level, whatever's about to run might reuse these
        registers, so whatever's already relying on them saves its own
        values first. (%r14, the per-iteration byte offset, needs no
        such protection: it's always freshly recomputed at the start
        of each iteration, never read again afterward.)"""
        struct_width = type_byte_width(Type(TypeKind.STRUCT, struct_name=struct_name), self.struct_registry)
        left_base = Register('rbx')
        right_base = Register('r12')
        index_32 = Register('r13d')
        index_64 = Register('r13')
        offset_32 = Register('r14d')
        offset_64 = Register('r14')

        loop_start = self.new_label("array_eq_struct_loop")
        loop_done = self.new_label("array_eq_struct_done")

        instructions = [
            MovQ(src=left_addr, dst=left_base),
            MovQ(src=right_addr, dst=right_base),
            Mov(src=Imm(0), dst=index_32),
            Label(loop_start),
            Cmp(src=Imm(element_count), dst=index_32),
            Jae(loop_done),
        ]
        instructions.append(Mov(src=index_32, dst=offset_32))
        instructions.append(IMul(src=Imm(struct_width), dst=offset_32))
        left_elem_addr = Register('r8')
        right_elem_addr = Register('r9')
        instructions.append(MovQ(src=left_base, dst=left_elem_addr))
        instructions.append(AddQ(src=offset_64, dst=left_elem_addr))
        instructions.append(MovQ(src=right_base, dst=right_elem_addr))
        instructions.append(AddQ(src=offset_64, dst=right_elem_addr))

        instructions.append(Push(left_base))
        instructions.append(Push(right_base))
        instructions.append(Push(index_64))
        instructions.extend(self._gen_struct_fields_equality_at_addresses(
            struct_name, left_elem_addr, right_elem_addr, mismatch_label
        ))
        instructions.append(Pop(index_64))
        instructions.append(Pop(right_base))
        instructions.append(Pop(left_base))

        instructions.append(Add(src=Imm(1), dst=index_32))
        instructions.append(Jmp(loop_start))
        instructions.append(Label(loop_done))
        return instructions

    def _gen_zero_array_into(self, array_type: Type, dst_mem: Memory) -> list[Instruction]:
        """Zeroes a whole array -- dispatching on the array's LEAF type
        into one of three strategies, mirroring array equality's own
        three-way split for the same underlying reason: a leaf's zero-
        value representation determines whether the whole array can be
        zeroed as one flat run of raw bytes, or needs a real
        per-element write.
          - int8/uint8, int, bool, OR SLICE leaf: _gen_array_flat_zero_
            loop. All four types' zero value is ALL RAW ZERO BYTES
            with no pointer or special representation (a slice's none-
            shaped {0, 0, 0} descriptor IS 24 zero bytes), so the whole
            array is zeroed as one flat run. Array equality couldn't
            offer slice this treatment (a slice-typed element isn't
            comparable yet), but zeroing has no such restriction --
            there's nothing to compare, only a zero value to write.
            int8/uint8 need a 1-byte step through that flat run rather
            than a 4-byte one, for the same reason the equality loop's
            step parameter does.
          - str leaf: _gen_array_str_zero_loop -- a str's zero value is
            a POINTER, so each element needs that address written
            individually, not raw zero bytes.
          - struct leaf: _gen_array_struct_zero_loop -- a struct's
            fields can be a mix of types, so each element is zeroed via
            a recursive call back into _gen_zero_value_into itself."""
        leaf = leaf_type(array_type)
        total_width = type_byte_width(array_type, self.struct_registry)
        if leaf == Type.STR:
            return self._gen_array_str_zero_loop(dst_mem, total_width // 8)
        if leaf.kind == TypeKind.STRUCT:
            struct_width = type_byte_width(leaf, self.struct_registry)
            return self._gen_array_struct_zero_loop(dst_mem, leaf.struct_name, total_width // struct_width)
        step = 1 if leaf in (Type.INT8, Type.UINT8) else 4
        return self._gen_array_flat_zero_loop(dst_mem, total_width, step=step)

    def _gen_array_flat_zero_loop(self, dst_mem: Memory, total_width: int, step: int = 4) -> list[Instruction]:
        """Zeroes `total_width` bytes at dst_mem, `step` bytes at a
        time -- 4 (a plain 4-byte Mov of Imm(0)) for an int, bool, or
        slice leaf at any nesting depth.

        1 is correct instead for an int8/uint8 leaf (via a 1-byte MovB
        of that same Imm(0)), for the identical reason
        _gen_array_flat_byte_equality_loop's step parameter exists:
        type_byte_width no longer guarantees total_width is a multiple
        of 4 once a 1-byte-wide leaf exists -- stepping 4 bytes
        regardless would write one byte past the end of the array.

        No calls happen in this loop, so every register is ordinary
        caller-saved scratch.

        Computes dst_mem's starting address into a FIXED register
        (%r10) via _gen_address_of_memory_into, rather than assuming
        dst_mem.base remains valid to keep reading from directly.
        This method never relies on dst_mem's own base surviving its
        execution; any caller that needs dst_mem.base to still be
        valid AFTERWARD (see _gen_zero_value_into's struct case) is
        responsible for protecting it externally."""
        base_reg = Register('r10')
        instructions = self._gen_address_of_memory_into(dst_mem, base_reg)
        index_32 = Register('ecx')
        index_64 = Register('rcx')
        write_addr = Register('r11')
        loop_start = self.new_label("array_zero_flat_loop")
        loop_done = self.new_label("array_zero_flat_done")
        instructions.append(Mov(src=Imm(0), dst=index_32))
        instructions.append(Label(loop_start))
        instructions.append(Cmp(src=Imm(total_width), dst=index_32))
        instructions.append(Jae(loop_done))
        instructions.append(MovQ(src=base_reg, dst=write_addr))
        instructions.append(AddQ(src=index_64, dst=write_addr))
        if step == 1:
            instructions.append(MovB(src=Imm(0), dst=Memory(write_addr.name, 0)))
        else:
            instructions.append(Mov(src=Imm(0), dst=Memory(write_addr.name, 0)))
        instructions.append(Add(src=Imm(step), dst=index_32))
        instructions.append(Jmp(loop_start))
        instructions.append(Label(loop_done))
        return instructions

    def _gen_array_str_zero_loop(self, dst_mem: Memory, element_count: int) -> list[Instruction]:
        """The str-leaf counterpart to _gen_array_flat_zero_loop:
        computes the shared empty-string address ONCE, then writes
        that same 8-byte value into each of `element_count` consecutive
        element slots. No calls happen here either, so every register
        is ordinary caller-saved scratch."""
        base_reg = Register('r10')
        instructions = self._gen_address_of_memory_into(dst_mem, base_reg)
        empty_str_reg = Register('r11')
        instructions.append(LeaQ(label=self._get_empty_str_label(), dst=empty_str_reg))
        total_width = element_count * 8
        offset_32 = Register('ecx')
        offset_64 = Register('rcx')
        write_addr = Register('r8')
        loop_start = self.new_label("array_zero_str_loop")
        loop_done = self.new_label("array_zero_str_done")
        instructions.append(Mov(src=Imm(0), dst=offset_32))
        instructions.append(Label(loop_start))
        instructions.append(Cmp(src=Imm(total_width), dst=offset_32))
        instructions.append(Jae(loop_done))
        instructions.append(MovQ(src=base_reg, dst=write_addr))
        instructions.append(AddQ(src=offset_64, dst=write_addr))
        instructions.append(MovQ(src=empty_str_reg, dst=Memory(write_addr.name, 0)))
        instructions.append(Add(src=Imm(8), dst=offset_32))
        instructions.append(Jmp(loop_start))
        instructions.append(Label(loop_done))
        return instructions

    def _gen_array_struct_zero_loop(self, dst_mem: Memory, struct_name: str, element_count: int) -> list[Instruction]:
        """The struct-leaf counterpart to the two loops above: each
        element is zeroed via a recursive call back into
        _gen_zero_value_into, since a struct's fields can be a mix of
        types with no single flat-bytes-or-repeated-value strategy.

        Unlike its two siblings, this loop's base address (%r12) and
        index (%r13) DO need protecting across that recursive call:
        the struct being zeroed could itself have an array-typed
        field, which would dispatch back into one of THESE SAME THREE
        loops (reusing the identical fixed register names, since
        there's no way to hand out a fresh set per nesting depth) --
        silently corrupting this outer loop's %r12/%r13 if they
        weren't saved first. Protected via an ordinary push/pop around
        the one recursive call, exactly like
        _gen_array_struct_equality_loop's identical situation. %r14
        (the per-iteration byte offset) needs no such protection: it's
        always freshly recomputed at the start of an iteration."""
        struct_width = type_byte_width(Type(TypeKind.STRUCT, struct_name=struct_name), self.struct_registry)
        base_reg = Register('r12')
        instructions = self._gen_address_of_memory_into(dst_mem, base_reg)
        index_32 = Register('r13d')
        index_64 = Register('r13')
        offset_32 = Register('r14d')
        offset_64 = Register('r14')
        elem_addr = Register('r10')
        loop_start = self.new_label("array_zero_struct_loop")
        loop_done = self.new_label("array_zero_struct_done")
        instructions.append(Mov(src=Imm(0), dst=index_32))
        instructions.append(Label(loop_start))
        instructions.append(Cmp(src=Imm(element_count), dst=index_32))
        instructions.append(Jae(loop_done))
        instructions.append(Mov(src=index_32, dst=offset_32))
        instructions.append(IMul(src=Imm(struct_width), dst=offset_32))
        instructions.append(MovQ(src=base_reg, dst=elem_addr))
        instructions.append(AddQ(src=offset_64, dst=elem_addr))
        instructions.append(Push(base_reg))
        instructions.append(Push(index_64))
        instructions.extend(self._gen_zero_value_into(
            Type(TypeKind.STRUCT, struct_name=struct_name), Memory(elem_addr.name, 0)
        ))
        instructions.append(Pop(index_64))
        instructions.append(Pop(base_reg))
        instructions.append(Add(src=Imm(1), dst=index_32))
        instructions.append(Jmp(loop_start))
        instructions.append(Label(loop_done))
        return instructions

    def gen_slice_none_comparison_into(self, expr: Binary, dst: Register) -> list[Instruction]:
        """Computes `slice_expr == none` or `slice_expr != none` (in
        either operand order) into dst -- checking specifically
        whether the slice's `ptr` field is null, matching Go's own
        nil-vs-empty-slice distinction: a real, zero-length slice
        sliced from a real array (`arr[5:5]`) has a non-null pointer
        and is NOT `== none`, even though both are equally safe,
        equally zero-length slices for every other purpose.

        semantic.py's check_binary already guarantees exactly one
        operand is slice-typed and the other none-typed by the time
        this is reached.

        Reuses gen_indexable_base_into for the slice's address even
        though only the address, not the length/capacity it also
        computes, is needed here -- two harmless unused extra loads
        rather than a second, narrower helper duplicating its
        Variable-vs-Index and array-vs-slice handling for one call
        site.

        Uses CmpQ (64-bit), not the ordinary 32-bit Cmp every other
        comparison in this file uses -- checking only a pointer's low
        32 bits against zero could, in principle, miss a real, non-null
        pointer whose low 32 bits happen to be zero.
        """
        slice_expr = expr.left if type_of(expr.left).kind == TypeKind.SLICE else expr.right
        addr_reg = Register('rbx')
        len_reg = Register('r12')  # unused here; gen_indexable_base_into always computes it
        cap_reg = Register('r13')  # unused here too
        instructions, _, _ = self.gen_indexable_base_into(slice_expr, addr_reg, len_reg, cap_reg)
        instructions.append(CmpQ(src=Imm(0), dst=addr_reg))
        cc = 'e' if expr.op == BinaryOp.EQUAL else 'ne'
        byte_dst = as_byte_register(dst)
        instructions.append(SetCC(cc=cc, operand=byte_dst))
        instructions.append(MovZX(src=byte_dst, dst=dst))
        return instructions

    def _gen_grow_and_append_one_into(
            self,
            r_ptr: Register,
            r_len: Register,
            r_len_32: Register,
            r_cap: Register,
            r_cap_32: Register,
            element_width: int,
            copy_one_element,
            write_new_value_at,
    ) -> list[Instruction]:
        """The reuse-vs-reallocate growth core shared by
        gen_append_call_into and the print buffer's single-byte append
        (see gen_buffer_append_byte_into) -- factored out so the growth
        arithmetic (new_cap = cap*2 if cap < 256, else cap + cap//4,
        with a cap==0 floor of 1; see gen_append_call_into's docstring
        for the reuse-vs-reallocate reasoning) never has to exist twice.

        Operates entirely on registers the caller has already loaded
        with a live {ptr, len, cap} triple (both the 64-bit register
        and its 32-bit view, for each of len/cap) -- never on a Memory
        location directly. Loading the initial triple and writing the
        final one back out is the caller's responsibility.

        Internally fixed scratch: %r8/%r8d, %r9d, %r10/%r10d, %r11/
        %r11d, %eax/%ecx, and %r14. Callers must choose their own
        ptr/len/cap registers to avoid these.

        `copy_one_element(dst_addr, src_addr)` is called once per
        existing element, ONLY on the reallocate path, to move it into
        the new backing array -- gen_append_call_into passes one that
        calls gen_array_copy; the print buffer passes one that just
        moves a single byte.

        `write_new_value_at(target_addr)` is called exactly once, at
        the correct final address for the newly appended element (in
        whichever backing array ends up live) -- gen_append_call_into
        passes one that evaluates an arbitrary Hornet expression via
        _gen_write_value_at_address_into; the print buffer passes one
        that writes an already-computed raw byte directly.

        On return, r_ptr/r_len/r_cap (and their 32-bit views) hold the
        FINAL triple: len always incremented by exactly one, ptr
        repointed to a fresh block if reallocation happened, unchanged
        otherwise."""
        instructions = []
        realloc_label = self.new_label("append_realloc")
        end_label = self.new_label("append_end")

        # len >= cap (equivalently, given len <= cap always, len ==
        # cap exactly) means no spare room -- must reallocate.
        instructions.append(Cmp(src=r_cap_32, dst=r_len_32))
        instructions.append(Jae(realloc_label))

        # REUSE PATH: len < cap. target = ptr + len*element_width.
        target_addr = Register('r10')
        instructions.append(MovQ(src=r_ptr, dst=target_addr))
        instructions.append(Mov(src=r_len_32, dst=Register('r11d')))
        instructions.append(IMul(src=Imm(element_width), dst=Register('r11d')))
        instructions.append(AddQ(src=Register('r11'), dst=target_addr))
        instructions.extend(write_new_value_at(target_addr))
        instructions.append(Add(src=Imm(1), dst=r_len_32))
        instructions.append(Jmp(end_label))

        # REALLOCATE PATH: len == cap. new_cap computed from cap alone,
        # in place -- the old cap value is never needed again once
        # this decides new_cap, so overwriting r_cap_32 here is safe.
        instructions.append(Label(realloc_label))
        zero_label = self.new_label("append_cap_zero")
        quarter_label = self.new_label("append_cap_quarter")
        growth_done_label = self.new_label("append_growth_done")

        instructions.append(Cmp(src=Imm(0), dst=r_cap_32))
        instructions.append(Je(zero_label))
        instructions.append(Cmp(src=Imm(256), dst=r_cap_32))
        instructions.append(Jae(quarter_label))
        instructions.append(IMul(src=Imm(2), dst=r_cap_32))
        instructions.append(Jmp(growth_done_label))
        instructions.append(Label(zero_label))
        instructions.append(Mov(src=Imm(1), dst=r_cap_32))
        instructions.append(Jmp(growth_done_label))
        instructions.append(Label(quarter_label))
        instructions.append(Mov(src=r_cap_32, dst=Register('eax')))
        instructions.append(Mov(src=Imm(2), dst=Register('ecx')))
        instructions.append(ShiftRightArithmetic(dst=r_cap_32))
        instructions.append(Add(src=Register('eax'), dst=r_cap_32))
        instructions.append(Label(growth_done_label))
        # r_cap_32 (and, via the zero-extension a 32-bit write always
        # gives its own 64-bit register, r_cap itself) now holds
        # new_cap.

        instructions.append(Mov(src=r_cap_32, dst=Register('edi')))
        instructions.append(IMul(src=Imm(element_width), dst=Register('edi')))
        instructions.append(CallInstr('malloc'))
        r_new_ptr = Register('r14')
        instructions.append(MovQ(src=Register('rax'), dst=r_new_ptr))

        # Copy the existing len elements from the OLD array (r_ptr)
        # into the NEW one (r_new_ptr) via copy_one_element, a genuine
        # RUNTIME loop since len is a runtime value here.
        loop_start_label = self.new_label("append_copy_loop")
        loop_done_label = self.new_label("append_copy_done")
        i_32 = Register('r9d')
        instructions.append(Mov(src=Imm(0), dst=i_32))
        instructions.append(Label(loop_start_label))
        instructions.append(Cmp(src=r_len_32, dst=i_32))
        instructions.append(Jae(loop_done_label))
        instructions.append(Mov(src=i_32, dst=Register('r11d')))
        instructions.append(IMul(src=Imm(element_width), dst=Register('r11d')))
        instructions.append(MovQ(src=r_ptr, dst=Register('r10')))
        instructions.append(AddQ(src=Register('r11'), dst=Register('r10')))
        instructions.append(MovQ(src=r_new_ptr, dst=Register('r8')))
        instructions.append(AddQ(src=Register('r11'), dst=Register('r8')))
        instructions.extend(copy_one_element(Register('r8'), Register('r10')))
        instructions.append(Add(src=Imm(1), dst=i_32))
        instructions.append(Jmp(loop_start_label))
        instructions.append(Label(loop_done_label))

        # Write the new element at new_ptr + len*element_width -- the
        # one slot the copy loop above deliberately left untouched.
        target_addr2 = Register('r10')
        instructions.append(MovQ(src=r_new_ptr, dst=target_addr2))
        instructions.append(Mov(src=r_len_32, dst=Register('r11d')))
        instructions.append(IMul(src=Imm(element_width), dst=Register('r11d')))
        instructions.append(AddQ(src=Register('r11'), dst=target_addr2))
        instructions.extend(write_new_value_at(target_addr2))

        instructions.append(Add(src=Imm(1), dst=r_len_32))
        instructions.append(MovQ(src=r_new_ptr, dst=r_ptr))

        instructions.append(Label(end_label))
        return instructions

    def gen_append_call_into(self, expr: Call, dst_mem: Memory) -> list[Instruction]:
        """`append(s, value)` -- Go-style: writes a NEW {ptr, len, cap}
        descriptor into dst_mem, never mutating s's own three fields
        (s keeps pointing at exactly what it always did, with its
        original len and cap).

        s (expr.args[0]) can be any slice-typed expression: a bare
        Variable or `none` (materialized inline, no scratch slot
        needed), or anything else (a slice literal, a re-slice, an
        Index, a slice-returning Call), materialized into the shared
        per-function scratch slot (_unnamed_slice_temp_offset) via
        gen_slice_value_into, then read back out like a Variable's own
        slot would be.

        s's three fields are loaded into CALLEE-SAVED registers
        (%rbx/%r12/%r13 for ptr/len/cap), not caller-saved ones,
        specifically because the REALLOCATE path (inside
        _gen_grow_and_append_one_into) calls malloc, which is free to
        clobber any caller-saved register but obligated to preserve
        callee-saved ones.

        The actual growth policy -- reuse-vs-reallocate, the growth
        arithmetic, the copy-existing-elements loop -- lives entirely
        in _gen_grow_and_append_one_into now, shared with the print
        buffer's single-byte append; this method's remaining job is
        just materializing s into registers beforehand, and writing
        the final triple to dst_mem afterward, protecting dst_mem.base
        (whenever it isn't 'rbp') across the whole thing.
        """
        slice_arg, value_arg = expr.args
        slice_type = type_of(slice_arg)
        element_type = slice_type.element_type
        element_width = type_byte_width(element_type, self.struct_registry)

        protect_dst = dst_mem.base != 'rbp'
        instructions = []
        if protect_dst:
            instructions.append(Push(Register(dst_mem.base)))

        r_ptr = Register('rbx')
        r_len = Register('r12')
        r_cap = Register('r13')
        r_len_32 = Register('r12d')
        r_cap_32 = Register('r13d')

        if isinstance(slice_arg, NoneLiteral):
            instructions.append(MovQ(src=Imm(0), dst=r_ptr))
            instructions.append(MovQ(src=Imm(0), dst=r_len))
            instructions.append(MovQ(src=Imm(0), dst=r_cap))
        elif isinstance(slice_arg, Variable):
            offset = self._local_offset(slice_arg.name)
            instructions.append(MovQ(src=Memory('rbp', offset), dst=r_ptr))
            instructions.append(MovQ(src=Memory('rbp', offset + 8), dst=r_len))
            instructions.append(MovQ(src=Memory('rbp', offset + 16), dst=r_cap))
        else:
            # Any other slice-typed expression: build its {ptr, len,
            # cap} descriptor into the shared, per-function unnamed-
            # slice scratch slot first, then read it back out.
            scratch = self._unnamed_slice_temp_offset
            instructions.extend(self.gen_slice_value_into(slice_arg, Memory('rbp', scratch)))
            instructions.append(MovQ(src=Memory('rbp', scratch), dst=r_ptr))
            instructions.append(MovQ(src=Memory('rbp', scratch + 8), dst=r_len))
            instructions.append(MovQ(src=Memory('rbp', scratch + 16), dst=r_cap))

        instructions.extend(self._gen_grow_and_append_one_into(
            r_ptr, r_len, r_len_32, r_cap, r_cap_32, element_width,
            copy_one_element=lambda dst, src: self.gen_array_copy(
                Memory(dst.name, 0), Memory(src.name, 0), element_type
            ),
            write_new_value_at=lambda target: self._gen_write_value_at_address_into(
                value_arg, element_type, target
            ),
        ))

        if protect_dst:
            instructions.append(Pop(Register(dst_mem.base)))
        instructions.append(MovQ(src=r_ptr, dst=Memory(dst_mem.base, dst_mem.offset)))
        instructions.append(MovQ(src=r_len, dst=Memory(dst_mem.base, dst_mem.offset + 8)))
        instructions.append(MovQ(src=r_cap, dst=Memory(dst_mem.base, dst_mem.offset + 16)))
        return instructions

    def gen_buffer_append_byte_into(
            self,
            r_ptr: Register,
            r_len: Register,
            r_len_32: Register,
            r_cap: Register,
            r_cap_32: Register,
            byte_value: Operand,
    ) -> list[Instruction]:
        """Appends exactly ONE byte to a growable byte buffer already
        held in r_ptr/r_len/r_cap (and their 32-bit views) -- the
        print machinery's single-character append, sharing the
        identical growth policy `append()` itself uses (see
        _gen_grow_and_append_one_into) with element_width=1 and a
        byte-sized write/copy in place of a general Hornet element
        type's gen_array_copy/_gen_write_value_at_address_into.

        `byte_value` is whatever operand already holds the byte to
        append -- typically an Imm or an 8-bit register alias (via
        as_byte_register) if the byte was computed into a register
        first. Written via MovB -- the first place this compiler has
        needed a genuine single-byte memory write, as opposed to a
        4-byte int/bool or an 8-byte pointer.

        Unlike gen_append_call_into, this never touches a Memory
        destination: r_ptr/r_len/r_cap are expected to stay live in
        registers across many further appends while a single value's
        representation is being built up, not written back out after
        every byte. Callers that DO need the current triple durably
        persisted are responsible for spilling it themselves."""
        def copy_one_byte(dst_addr: Register, src_addr: Register) -> list[Instruction]:
            scratch = as_byte_register(Register('eax'))
            return [
                MovB(src=Memory(src_addr.name, 0), dst=scratch),
                MovB(src=scratch, dst=Memory(dst_addr.name, 0)),
            ]

        def write_byte_at(target_addr: Register) -> list[Instruction]:
            return [MovB(src=byte_value, dst=Memory(target_addr.name, 0))]

        return self._gen_grow_and_append_one_into(
            r_ptr, r_len, r_len_32, r_cap, r_cap_32,
            element_width=1,
            copy_one_element=copy_one_byte,
            write_new_value_at=write_byte_at,
        )

    def gen_buffer_append_bytes_into(
            self,
            r_ptr: Register,
            r_len: Register,
            r_len_32: Register,
            r_cap: Register,
            r_cap_32: Register,
            source_addr: Register,
            count: Operand,
    ) -> list[Instruction]:
        """Appends `count` bytes in one bulk operation, copied from
        source_addr -- the print machinery's multi-byte append, as
        opposed to gen_buffer_append_byte_into's single-character one.
        This is a genuinely DIFFERENT growth calculation, not just a
        loop calling the single-byte version count times: the single-
        element growth formula (see gen_append_call_into's docstring)
        is only correct because it assumes reallocation happens
        exactly when len == cap, one element at a time -- appending a
        40-byte chunk when only 4 bytes of spare capacity remain needs
        `needed` (len + count) to enter the decision directly, which
        that simplified arithmetic can't do.

        GROWTH: needed = len + count. If needed <= cap, no
        reallocation -- just copy directly into the existing backing
        array at ptr + len. Only when needed > cap does this
        reallocate, to new_cap = max(needed, cap*2 if cap < 256 else
        cap + cap//4) -- the FULL formula, not the single-element
        simplification: needed can be arbitrarily larger than one
        doubling would produce, so the max has to be computed for
        real. This also means the single-element formula's explicit
        cap==0 floor of 1 isn't needed here: when cap is 0, max(needed,
        0) already resolves to needed on its own (needed is always
        positive).

        Both the reallocate path's copy-existing-bytes step and the
        final copy-the-new-bytes-in step move bytes one at a time via
        MovB, in a runtime loop -- there's no bulk memory-move
        instruction in this compiler's Instruction vocabulary yet;
        correctness came first.

        Internally fixed scratch, distinct from
        _gen_grow_and_append_one_into's own set so this can be called
        independently: %rax/%eax, %rcx/%ecx, %rdx/%edx, %rdi, %r10,
        %r11, %r14, %r15. Callers must choose r_ptr/r_len/r_cap, and
        whatever register source_addr lives in, to avoid all of
        these."""
        instructions = []
        no_grow_label = self.new_label("bulk_append_no_grow")
        copy_new_label = self.new_label("bulk_append_copy_new")

        # Move source_addr (always a register in practice) -- and
        # count, if it's a register rather than a compile-time Imm --
        # into callee-saved registers (%r14/%r15) BEFORE any of the
        # growth/malloc logic below runs. The reallocate path calls
        # malloc, which is free to clobber whatever CALLER-saved
        # register the caller passed in for these, corrupting them by
        # the time the copy-new loop below needs them. Found as a real
        # bug: missing this protection was correct on the no-grow path
        # (no malloc call to clobber anything) but silently wrong on
        # the reallocate path -- surfacing as heap corruption several
        # appends downstream rather than an obviously-wrong value at
        # the call site itself. An Imm count needs no such protection:
        # it's baked directly into the instructions that use it.
        instructions.append(MovQ(src=source_addr, dst=Register('r14')))
        source_addr = Register('r14')
        if not isinstance(count, Imm):
            instructions.append(Mov(src=count, dst=Register('r15d')))
            count = Register('r15d')

        # needed = len + count, in %eax.
        if isinstance(count, Imm):
            instructions.append(Mov(src=Imm(count.value), dst=Register('eax')))
        else:
            instructions.append(Mov(src=count, dst=Register('eax')))
        instructions.append(Add(src=r_len_32, dst=Register('eax')))

        instructions.append(Cmp(src=r_cap_32, dst=Register('eax')))
        instructions.append(Jle(no_grow_label))

        # REALLOCATE: new_cap = max(needed, doubled-or-quartered(cap)).
        # %eax already holds `needed`; %ecx becomes the doubled-or-
        # quartered candidate, then the larger of the two wins.
        quarter_label = self.new_label("bulk_append_quarter")
        candidate_done_label = self.new_label("bulk_append_candidate_done")
        instructions.append(Mov(src=r_cap_32, dst=Register('ecx')))
        instructions.append(Cmp(src=Imm(256), dst=Register('ecx')))
        instructions.append(Jae(quarter_label))
        instructions.append(IMul(src=Imm(2), dst=Register('ecx')))
        instructions.append(Jmp(candidate_done_label))
        instructions.append(Label(quarter_label))
        instructions.append(Mov(src=r_cap_32, dst=Register('edx')))
        instructions.append(Mov(src=Imm(2), dst=Register('ecx')))
        # Shift the CANDIDATE (starting from cap, in %edx) right by 2,
        # then add cap back -- ShiftRightArithmetic's fixed %cl-sourced
        # count means %ecx holds the shift amount (2) here, not the
        # candidate itself, unlike the plain-doubling branch above
        # where %ecx directly holds the result.
        instructions.append(ShiftRightArithmetic(dst=Register('edx')))
        instructions.append(Mov(src=r_cap_32, dst=Register('ecx')))
        instructions.append(Add(src=Register('edx'), dst=Register('ecx')))
        instructions.append(Label(candidate_done_label))
        # %eax = needed, %ecx = candidate. new_cap = max of the two,
        # left in %ecx (needed's value in %eax is still required below
        # for how many NEW bytes to copy in, so %eax is never
        # overwritten by this comparison).
        instructions.append(Cmp(src=Register('ecx'), dst=Register('eax')))
        max_done_label = self.new_label("bulk_append_max_done")
        instructions.append(Jle(max_done_label))
        instructions.append(Mov(src=Register('eax'), dst=Register('ecx')))
        instructions.append(Label(max_done_label))

        # new_cap (in %ecx, caller-saved) MUST move into r_cap_32
        # (callee-saved) before calling malloc, not after: malloc is
        # free to clobber %ecx during its own execution, and is only
        # OBLIGATED to preserve callee-saved registers. Keeping the
        # computed value in %ecx across the call and reading it back
        # afterward is a real, silent bug found only by directly
        # checking the resulting cap against a hand-worked-out
        # expected value -- the visible symptom (a buffer sized
        # smaller than what was actually written into it) doesn't
        # reliably crash for a small overrun like this one.
        instructions.append(Mov(src=Register('ecx'), dst=r_cap_32))

        instructions.append(Mov(src=r_cap_32, dst=Register('edi')))
        instructions.append(CallInstr('malloc'))
        r_new_ptr = Register('r10')
        instructions.append(MovQ(src=Register('rax'), dst=r_new_ptr))

        # Copy the existing len bytes from the OLD array into the NEW
        # one, one byte at a time -- a genuine runtime loop, since len
        # is a runtime value.
        copy_old_loop = self.new_label("bulk_append_copy_old_loop")
        copy_old_done = self.new_label("bulk_append_copy_old_done")
        i_reg = Register('edx')
        i_reg_64 = Register('rdx')  # same physical register, 64-bit view for AddQ's own address arithmetic below
        instructions.append(Mov(src=Imm(0), dst=i_reg))
        instructions.append(Label(copy_old_loop))
        instructions.append(Cmp(src=r_len_32, dst=i_reg))
        instructions.append(Jae(copy_old_done))
        # %eax/%ecx are both already free by this point in the
        # reallocate path -- needed's value in %eax was last needed to
        # compute new_cap, already consumed into %ecx and then
        # malloc'd (whose own return value has already been copied out
        # into r_new_ptr) -- so nothing here needs preserving across a
        # single byte move.
        old_byte = as_byte_register(Register('eax'))
        instructions.append(MovQ(src=r_ptr, dst=Register('r11')))
        instructions.append(AddQ(src=i_reg_64, dst=Register('r11')))
        instructions.append(MovB(src=Memory('r11', 0), dst=old_byte))
        instructions.append(MovQ(src=r_new_ptr, dst=Register('r11')))
        instructions.append(AddQ(src=i_reg_64, dst=Register('r11')))
        instructions.append(MovB(src=old_byte, dst=Memory('r11', 0)))
        instructions.append(Add(src=Imm(1), dst=i_reg))
        instructions.append(Jmp(copy_old_loop))
        instructions.append(Label(copy_old_done))

        instructions.append(MovQ(src=r_new_ptr, dst=r_ptr))
        # r_cap_32 already holds new_cap, written BEFORE the malloc
        # call above so it survives that call correctly.
        instructions.append(Jmp(copy_new_label))

        # NO-GROW: needed <= cap, existing backing array already has
        # enough spare room.
        instructions.append(Label(no_grow_label))

        # COPY-NEW: copy `count` bytes from source_addr into ptr+len,
        # one byte at a time, then len += count. By this point r_ptr
        # is already correct either way (untouched on the no-grow
        # path, repointed to the fresh block on the reallocate path).
        instructions.append(Label(copy_new_label))
        if isinstance(count, Imm):
            count_reg = Register('ecx')
            instructions.append(Mov(src=Imm(count.value), dst=count_reg))
        else:
            count_reg = Register('ecx')
            instructions.append(Mov(src=count, dst=count_reg))
        copy_new_loop = self.new_label("bulk_append_copy_new_loop")
        copy_new_done = self.new_label("bulk_append_copy_new_done")
        j_reg = Register('edx')
        j_reg_64 = Register('rdx')  # same physical register, 64-bit view for AddQ's own address arithmetic below
        instructions.append(Mov(src=Imm(0), dst=j_reg))
        instructions.append(Label(copy_new_loop))
        instructions.append(Cmp(src=count_reg, dst=j_reg))
        instructions.append(Jae(copy_new_done))
        new_byte = as_byte_register(Register('eax'))
        instructions.append(MovQ(src=source_addr, dst=Register('r11')))
        instructions.append(AddQ(src=j_reg_64, dst=Register('r11')))
        instructions.append(MovB(src=Memory('r11', 0), dst=new_byte))
        instructions.append(MovQ(src=r_ptr, dst=Register('r11')))
        instructions.append(AddQ(src=r_len, dst=Register('r11')))
        instructions.append(AddQ(src=j_reg_64, dst=Register('r11')))
        instructions.append(MovB(src=new_byte, dst=Memory('r11', 0)))
        instructions.append(Add(src=Imm(1), dst=j_reg))
        instructions.append(Jmp(copy_new_loop))
        instructions.append(Label(copy_new_done))

        instructions.append(Add(src=count_reg, dst=r_len_32))
        return instructions

    def _gen_write_value_at_address_into(
            self, value_expr: Node, element_type: Type, addr_reg: Register) -> list[Instruction]:
        """Writes value_expr (evaluated as element_type) into
        Memory(addr_reg, 0) -- shared by gen_append_call_into's reuse
        and reallocate paths, both of which need to write the newly-
        appended element at a computed (not fixed-offset) address,
        with the element's type possibly being scalar, array, slice,
        or struct.

        For an ARRAY, SLICE, or STRUCT element type, this just hands
        addr_reg straight to gen_array_value_into/gen_slice_value_into/
        gen_struct_value_into as an ordinary Memory destination -- all
        three already protect an arbitrary base internally. For a
        scalar (int/bool/str), addr_reg is protected manually, matching
        gen_array_literal_into's own scalar-element pattern: push
        addr_reg, compute the value (which could itself involve a call
        that clobbers addr_reg), stash the result in %r8/%r8d, pop
        addr_reg back, then write from %r8/%r8d -- never straight from
        %eax/%rax, which popping addr_reg back into would otherwise
        clobber."""
        if element_type.kind == TypeKind.SLICE:
            return self.gen_slice_value_into(value_expr, Memory(addr_reg.name, 0))
        if element_type.kind == TypeKind.ARRAY:
            return self.gen_array_value_into(value_expr, Memory(addr_reg.name, 0), element_type)
        if element_type.kind == TypeKind.STRUCT:
            return self.gen_struct_value_into(value_expr, Memory(addr_reg.name, 0), element_type)
        instructions = [Push(addr_reg)]
        instructions.extend(self.gen_expr_into(value_expr, Register('eax')))
        if element_type == Type.STR:
            instructions.append(MovQ(src=Register('rax'), dst=Register('r8')))
            instructions.append(Pop(addr_reg))
            instructions.append(MovQ(src=Register('r8'), dst=Memory(addr_reg.name, 0)))
        else:
            # A full 64-bit shuttle for int64 -- an ordinary 32-bit Mov
            # would discard its high 32 bits. The final write goes
            # through _gen_write_scalar_from (not a bare Mov) so
            # int8/uint8 get their own correct, narrow, 1-byte write
            # too -- a bare 4-byte Mov here would be a latent bug for
            # them: writing 4 bytes at a 1-byte element's offset can
            # write past a freshly-grown backing array's allocated
            # capacity, not just harmlessly into extra headroom.
            if element_type == Type.INT64:
                instructions.append(MovQ(src=Register('rax'), dst=Register('r8')))
            else:
                instructions.append(Mov(src=Register('eax'), dst=Register('r8d')))
            instructions.append(Pop(addr_reg))
            instructions.extend(self._gen_write_scalar_from(Register('r8d'), element_type, Memory(addr_reg.name, 0)))
        return instructions

    def _get_bounds_check_fail_label(self, message: str) -> str:
        """Lazily creates a per-function, per-message label that every
        bounds check using this exact `message` jumps to on failure --
        reused across however many checks in this function share the
        same message, rather than duplicating the panic sequence (see
        _gen_bounds_check_panic_block) at every check site. A single
        function can use more than one message (e.g. "array index out
        of bounds" vs "slice bounds out of range"), each getting its
        own fail label, all reset together at the start of every
        function -- unlike the message labels below, these are purely
        LOCAL jump targets, meaningless outside the function they're
        generated for."""
        if message not in self._bounds_check_fail_labels:
            self._bounds_check_fail_labels[message] = self.new_label("bounds_check_fail")
        return self._bounds_check_fail_labels[message]

    def _get_bounds_check_message_label(self, message: str) -> str:
        """Lazily creates and caches (for the rest of the whole
        compilation, unlike the per-function fail labels above -- a
        plain static string, safely shared by every function that
        needs it) a label for this exact `message` string."""
        if message not in self._bounds_check_message_labels:
            label = self.new_label("bounds_msg")
            self._bounds_check_message_labels[message] = label
            self.string_literals.append((label, message))
        return self._bounds_check_message_labels[message]

    def _gen_bounds_check_panic_block(self) -> list[Instruction]:
        """Appended once at the end of a function's instructions (see
        gen_function) for every distinct message that function's
        bounds checks actually used -- none at all if it never
        triggered any. Each block prints its message, then calls
        abort() (SIGABRT) rather than a plain exit() -- an out-of-
        bounds access is a genuine program bug, not a normal
        termination condition, the same "abnormal termination"
        character division by zero's hardware-trapped SIGFPE already
        has. Never reached via ordinary fall-through from the
        function's body -- every return already leaves via
        `leave; ret`, and abort() itself never returns -- so appending
        these at the very end is always safe.

        Explicitly calls fflush(NULL) between puts() and abort() --
        found necessary by testing: abort() terminates via a raw
        signal, bypassing the normal exit() path that would otherwise
        flush libc's buffered stdio. Without this, the message prints
        reliably when stdout is line-buffered (an interactive
        terminal) but is silently LOST whenever stdout is redirected
        or piped -- the case for most non-interactively run programs.
        """
        instructions = []
        for message, fail_label in self._bounds_check_fail_labels.items():
            msg_label = self._get_bounds_check_message_label(message)
            instructions.extend([
                Label(fail_label),
                LeaQ(label=msg_label, dst=Register('rdi')),
                CallInstr('puts'),
                Mov(src=Imm(0), dst=Register('edi')),
                CallInstr('fflush'),
                CallInstr('abort'),
            ])
        return instructions

    def gen_len_call_into(self, expr: Call, dst: Operand) -> list[Instruction]:
        """`len(x)`: reuses gen_indexable_base_into directly -- the
        same "address plus length, however each is represented"
        abstraction indexing and slicing already share -- rather than
        a narrower restriction of its own: whatever
        gen_indexable_base_into accepts as a base (a Variable, an
        Index, a Slice expression, a slice-returning Call, or an
        ArrayLiteral) is automatically valid here too.

        x's address is computed and then discarded -- len only needs
        the LENGTH half of gen_indexable_base_into's return value --
        but computing it isn't wasted: x is still fully evaluated
        regardless (any bounds-check or side effect buried in it
        genuinely runs). This means `len(arr[i])` still aborts if i is
        out of range, and `len([]int[1, 2, 3])` still performs a real,
        if wasted, heap allocation -- both deliberate.

        For an ARRAY base, length_operand comes back as an Imm (the
        array's declared size, never read out of x at runtime); for a
        SLICE base, as the 64-bit len_dst register holding a runtime
        value from the slice's descriptor -- narrowed through its
        32-bit alias here, matching how every other reader of a
        slice's length field does, since Hornet's int is always 32
        bits even though the descriptor's len field is a full 8-byte
        slot."""
        if dst != Register('eax'):
            raise CodegenError(f"Call codegen requires dst == %eax, got: {dst!r}")
        arg = expr.args[0]
        len_reg = Register('r12')
        cap_reg = Register('r13')
        instructions, length_operand, _ = self.gen_indexable_base_into(
            arg, Register('rbx'), len_reg, cap_reg
        )
        if isinstance(length_operand, Register):
            instructions.append(Mov(src=Register('r12d'), dst=dst))
        else:
            instructions.append(Mov(src=length_operand, dst=dst))
        return instructions

    def _gen_address_of_memory_into(self, mem: Memory, dst: Register) -> list[Instruction]:
        """Computes the ADDRESS a Memory operand refers to, into `dst`.
        Memory('rbp', offset) needs a real leaq -- the address is
        offset-from-frame-pointer, not stored anywhere as a value in
        its own right; Memory(some_reg, offset) already HAS its
        address sitting directly in some_reg, with `offset` (if
        non-zero) added on top via a single AddQ. Used specifically
        for passing a Memory destination on as a POINTER argument --
        the hidden output pointer for an array-returning call
        (gen_array_call_into) or a struct-returning one -- everywhere
        else, a Memory operand is read from or written to directly
        rather than having its own address taken.

        The offset(some_reg) case used to assume offset was always 0
        whenever base wasn't 'rbp' -- true at the time, since nothing
        computed a destination this way except for the WHOLE of a
        Memory destination, offset already folded in or genuinely
        zero. That stopped being true once a struct literal's own
        array-typed FIELD could be populated directly by an array-
        returning call (`Big(1, makeArr())`, where `data` sits at a
        non-zero offset on a heap-allocated Big): the field's own
        field_mem is Memory('rax', 4), say, and the OLD version of this
        method silently discarded that +4, handing makeArr() the
        STRUCT's base address as its hidden return pointer instead of
        the field's -- a real, silent miscompile (verified directly:
        it corrupted the PRECEDING field along with the start of the
        array itself), not a hypothetical one. Adding the AddQ here is
        safe for every existing caller too, since each already only
        ever passed offset=0 for a non-'rbp' base."""
        if mem.base == 'rbp':
            return [LeaQFrame(offset=mem.offset, dst=dst)]
        instructions = [MovQ(src=Register(mem.base), dst=dst)]
        if mem.offset:
            instructions.append(AddQ(src=Imm(mem.offset), dst=dst))
        return instructions

    def gen_none_into(self, dst_mem: Memory, target_type: Type) -> list[Instruction]:
        """Writes `none`'s zero-value representation into dst_mem, for
        whichever nilable type target_type actually is. Only slices
        are nilable so far -- a {ptr: 0, len: 0, cap: 0} descriptor,
        the same shape Go's own nil slice has: a valid, safely-
        indexable-into-nothing slice with no backing array, not a
        special, separately-tracked null flag. Every existing slice
        operation already handles a zero-length slice correctly, so
        this is the ONLY new codegen a none-valued slice needs on the
        producing side; comparing one against `none` again (see
        gen_slice_none_comparison_into) is the only other.

        Called directly from gen_var_decl/gen_assign's NoneLiteral
        short-circuit, rather than being folded into
        gen_slice_value_into's dispatch -- unlike every other kind of
        slice-producing expression there, a NoneLiteral's resolved
        type (Type.NONE) never equals the slice type it's being stored
        into, so the caller has to already know and pass the TARGET
        type.

        Defensively re-checks target_type.kind here even though
        semantic.py already guarantees `none` was only ever allowed
        through for a slice target -- the same "codegen doesn't
        blindly trust its input" posture gen_array_copy's array-of-
        slices handling takes.
        """
        if target_type.kind != TypeKind.SLICE:
            raise CodegenError(
                f"'none' is only supported as a slice's zero value "
                f"right now, not {target_type}"
            )
        return [
            MovQ(src=Imm(0), dst=Memory(dst_mem.base, dst_mem.offset)),
            MovQ(src=Imm(0), dst=Memory(dst_mem.base, dst_mem.offset + 8)),
            MovQ(src=Imm(0), dst=Memory(dst_mem.base, dst_mem.offset + 16)),
        ]

    def _gen_malloc_array(self, array_type: Type) -> list[Instruction]:
        """Calls malloc for array_type's total footprint
        (type_byte_width), a compile-time-known constant, leaving the
        returned pointer in %rax. Used wherever a heap-allocated array
        needs its own, fresh backing allocation: a VarDecl declaring
        one (gen_var_decl) or a parameter receiving one
        (gen_function's parameter loop, which needs its own
        independent copy of the caller's data to preserve value
        semantics across the call -- like a stack-allocated parameter
        already gets via gen_array_copy, just backed by malloc'd
        memory instead of an inline slot)."""
        size = type_byte_width(array_type, self.struct_registry)
        return [Mov(src=Imm(size), dst=Register('edi')), CallInstr('malloc')]
