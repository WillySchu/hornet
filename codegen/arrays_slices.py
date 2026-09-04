"""TODO"""

from typing import Union

from codegen.assembly_ast import (
    Register,
    Instruction,
    Imm,
    Memory,
    Mov,
    CallInstr,
    MovQ,
    LeaQFrame,
    Push,
    Pop,
    Cmp,
    Jae,
    IMul,
    AddQ,
    Ja,
    Sub,
    MovB,
    Jmp,
    Label,
    MovZX,
    Jne,
    Add,
    LeaQ,
    CmpQ,
    SetCC,
    ShiftRightArithmetic,
    Je,
    Operand,
    Jle,
)
from codegen.errors import CodegenError
from codegen.utils import type_of, type_byte_width, leaf_type, as_byte_register, gen_protecting_dst_across
from parser import Node, ArrayLiteral, Call, Field, Index, Slice, Variable, NoneLiteral, Binary, BinaryOp
from semantic import TypeKind, Type


class ArraysSlicesMixin:
    def gen_array_literal_heap_alloc_into(self, expr: ArrayLiteral) -> list[Instruction]:
        """Mallocs a NEW, heap-allocated array sized to fit expr's own
        elements, writes them in (via the ordinary gen_array_literal_
        into, whose own dst_mem-protection logic already handles a
        non-'rbp' base correctly -- Memory('rax', 0) here is exactly
        that case, and by the time gen_array_literal_into returns,
        %rax is guaranteed to still hold the original malloc'd address,
        the same guarantee every other caller of it already relies
        on), and leaves the resulting pointer in %rax.

        Shared by both ways a slice literal's backing array gets
        created: the general, typed expression form (`[]int[1, 2, 3]`,
        parsed as an implicit whole-array Slice -- see parser.py's own
        _parse_bracketed_literal) via gen_indexable_base_into's own
        ArrayLiteral case, and the untyped form used directly as a
        slice-typed VarDecl/Assign value (`[]int s = [1, 2, 3]`) via
        gen_var_decl/gen_assign's own ArrayLiteral-as-slice-value
        short-circuit.

        Always allocates at least 1 byte, even for an empty literal
        (`[]int[]`) -- guaranteeing a genuine, non-null, unique pointer
        regardless of libc's own malloc(0) behavior (implementation-
        defined by POSIX; this doesn't rely on it), which is what makes
        `s == none` correctly false for an intentionally empty slice
        literal, matching the same nil-vs-empty distinction a real,
        non-empty slice already has (see gen_slice_none_comparison_
        into) -- `[]int[]` is a real, live, zero-length slice, not a
        nil one, the same way `arr[5:5]` already is.

        Every slice literal's own backing array is heap-allocated here
        UNCONDITIONALLY, regardless of size -- unlike an ordinary array
        variable, which only heap-promotes past the 16KB stack-size
        threshold (see is_heap_allocated). This isn't a size-based
        decision at all: a slice literal's backing array has to outlive
        the statement that creates it (the whole POINT of a slice is to
        be usable after the expression that produced it), so it needs
        the SAME "can safely cross frame boundaries" guarantee every
        OTHER sliced array already gets, unconditionally, for exactly
        the same reason.
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
        """Computes the address of `expr`'s own data into `addr_dst`,
        and returns (instructions, length_operand, cap_operand): each
        operand is an Imm (a compile-time constant, equal to the
        array's own declared size for BOTH len and cap -- an array has
        no separate capacity concept of its own) when `expr` is
        array-typed, or `len_dst`/`cap_dst` themselves (populated with
        a runtime value read out of a slice's own descriptor) when
        `expr` is slice-typed. cap_dst is computed and populated
        unconditionally, even by callers (gen_index_address_into) that
        never read it back out afterward -- cheap enough (one extra
        Imm, or one extra runtime read alongside the len one already
        happening) that a single, uniform three-value contract beats
        making it optional.

        Shared by gen_index_address_into (indexing, `base[i]`, which
        never needs cap: an index equal to len is already out of
        bounds regardless of any spare room past it) and gen_slice_
        into (slicing, `base[low:high]`, which needs cap for both its
        own bounds check and the result's own capacity -- see its own
        docstring), and now gen_append_call_into too (which needs all
        three fields as genuine input values, not just for a bounds
        check) -- all three need exactly this same "address plus
        length (plus, now, capacity), however each is represented"
        information about whatever's on the left of a `[...]`
        expression or append's own first argument, and each already
        has to branch on which kind of Operand comes back for its own
        use of length.

        A slice-typed `expr` can be a Variable (a named slice, loaded
        directly out of its own %rbp-relative slot), a Slice (an
        UNNAMED slice expression used directly as a base -- e.g.
        `arr[:][0]`, or `matrix[:][0][0]` -- materialized into a
        dedicated, per-function scratch slot, _unnamed_slice_temp_
        offset, via gen_slice_into, then immediately read back out
        into addr_dst/len_dst/cap_dst), a Call to a function that
        itself returns a slice (materialized into that exact same
        scratch slot, via gen_slice_call_into, then read back out the
        same way -- it used to arrive already sitting in %rax/%rdx by
        a dedicated two-register return convention, needing no scratch
        slot at all, back when a slice's own descriptor still fit two
        registers; see the module docstring's SLICE PARAMETERS AND
        RETURNS section for why that's no longer true), an Index
        yielding a slice (`rows[0][1]`, one element of an array OF
        slices used directly as a further base -- materialized into
        that same scratch slot too, via gen_slice_value_into's own
        Index case), or a Field yielding a slice (`p.values[0]`, a
        struct's own slice-typed field used directly as a further
        base -- materialized into that same scratch slot, via gen_
        slice_value_into's own Field case, the identical mechanism
        one level over).

        An ARRAY-typed `expr` can ALSO be an ArrayLiteral directly --
        not an existing Variable/Index at all, but a freshly-created
        one -- for a slice LITERAL's own backing array (`[]int[1, 2,
        3]`, parsed as an implicit whole-array Slice wrapping an
        ArrayLiteral -- see parser.py's own _parse_bracketed_literal).
        See gen_array_literal_heap_alloc_into's own docstring for why
        this is a genuinely different kind of "address" than the
        ordinary Variable/Index cases below: it mallocs a brand new
        allocation and writes the literal's own elements into it,
        rather than computing the address of something that already
        exists.

        Reusing ONE shared scratch slot for every Slice materialization
        -- rather than a fresh one per nesting level -- is safe under
        arbitrarily deep chaining (`arr[:][0:2][0]`, `rows[0][1]`, and
        so on) specifically because of how gen_slice_into and gen_
        index_address_into are both already structured: each computes
        its OWN base's address/length FIRST, immediately consumes it
        (into addr_dst/len_dst, then protects those on the real CPU
        stack before evaluating anything else), and only ever WRITES
        its own result into a destination as the very LAST step. That
        means a deeper level's own write to the shared slot always
        happens (and is always fully drained back into registers)
        strictly BEFORE the shallower level that triggered it writes
        its own result there -- the same strictly-nested lifetime
        discipline that makes reusing one call stack safe for
        recursion of any depth, just applied to one scratch memory
        slot instead of the stack. An Index base (e.g. `rows[0][1]`,
        indexing into an array/slice OF slices) reuses this exact same
        scratch slot too, via gen_slice_value_into's own Index case --
        neither is an array-returning Call as a SLICE base, since it
        can never actually BE slice-typed.

        `expr` being anything else (a Call returning something other
        than an array/slice can't reach here at all, being neither
        array- nor slice-typed) falls through to the final
        CodegenError below.
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
                # A slice-typed Index result (e.g. `rows[0]`, one
                # element of an array OF slices, used directly as the
                # base of a further `[...]`) -- materialized through
                # the exact same scratch slot the Slice case just
                # above uses, via gen_slice_value_into's own Index
                # case, then immediately read back out the same way.
                temp = self._unnamed_slice_temp_offset
                instructions = self.gen_slice_value_into(expr, Memory('rbp', temp))
                instructions.append(MovQ(src=Memory('rbp', temp + 8), dst=len_dst))
                instructions.append(MovQ(src=Memory('rbp', temp + 16), dst=cap_dst))
                instructions.append(MovQ(src=Memory('rbp', temp), dst=addr_dst))
                return instructions, len_dst, cap_dst
            if isinstance(expr, Field):
                # A slice-typed Field result (e.g. `p.values`, a
                # struct's own slice-typed field, used directly as the
                # base of a further `[...]`) -- structurally identical
                # to the Index case just above, one level over:
                # materialized through the exact same shared scratch
                # slot, via gen_slice_value_into's own Field case, then
                # immediately read back out the same way.
                temp = self._unnamed_slice_temp_offset
                instructions = self.gen_slice_value_into(expr, Memory('rbp', temp))
                instructions.append(MovQ(src=Memory('rbp', temp + 8), dst=len_dst))
                instructions.append(MovQ(src=Memory('rbp', temp + 16), dst=cap_dst))
                instructions.append(MovQ(src=Memory('rbp', temp), dst=addr_dst))
                return instructions, len_dst, cap_dst
            if isinstance(expr, Call):
                # A slice-returning Call now writes through the hidden-
                # pointer convention (see gen_slice_call_into), just
                # like an array-returning one -- materialized through
                # the exact same shared scratch slot the Slice/Index
                # cases just above use, then immediately read back out
                # the same way. Used to leave its result directly in
                # %rax/%rdx instead, back when a slice's own descriptor
                # still fit two registers.
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
        Variable referring to an array-typed local, an Index node
        that itself resolves to a sub-array (the outer dimensions of a
        multi-dimensional access), or a Field node that resolves to an
        array-typed struct field (`b.data`, then indexed further as
        `b.data[0]`) -- into the 64-bit register `dst`. `dst` must
        already be a 64-bit register (e.g. Register('rax'), not
        Register('eax')) -- addresses are always 64-bit values,
        regardless of how wide the array's own elements are.

        A heap-allocated Variable (see is_heap_allocated) needs a
        genuinely different instruction here, not just a different
        offset: its own slot holds a POINTER to the array's actual
        data, not the data itself, so getting the array's address
        means LOADING that pointer (movq) rather than computing the
        slot's own address (leaq) the way a stack-allocated array's
        does. Every other array-address computation in this file --
        gen_index_address_into's own recursive base case, and
        everything that calls through it (gen_index_assign, an Index
        read in gen_expr_into, gen_array_arg_address_into) -- goes
        through this one method for a bare Variable, so this is the
        only place that distinction needs to be made at all.
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
        (a 64-bit register) -- the shared foundation for reading an
        element (gen_expr_into's Index case), writing one
        (gen_index_assign), and reading a whole SUB-array for
        multi-dimensional access (this method's own recursive base
        case, via gen_array_address_into above, when `expr.array` is
        itself an Index).

        `expr.array` can now be array- OR slice-typed (indexing into a
        slice, `s[i]`, uses this exact same method) -- see
        gen_indexable_base_into for how the base's address and length
        are computed either way. For an array base, the length is an
        Imm known at compile time; for a slice base, it's a runtime
        value read out of the slice's own descriptor, kept alive in
        `len_reg` (picked dynamically, distinct from `dst`, the same
        way gen_array_copy's own scratch register is) across
        evaluating the index expression.

        Includes a runtime bounds check: an out-of-range index prints a
        message and calls abort() (see _gen_bounds_check_panic_block)
        rather than silently reading or writing adjacent stack memory
        -- which, given arrays live in the same frame as the saved
        return address and the callee-saved registers every function
        call already depends on, could otherwise corrupt exactly the
        state that keeps `call`/`ret` working correctly, not just
        return a wrong value.

        `expr.array`'s own address (and, for a slice base, its length
        too) is computed first and protected on the real CPU stack (not
        a fixed register) while the index expression -- which could be
        arbitrarily complex, including another indexing operation or a
        function call -- is evaluated, the same push-before-recursing
        pattern used everywhere else in this file a value needs to
        survive evaluating something else. This works out correctly no
        matter what register `dst` itself is (including if it happens
        to coincide with the %rax/%rcx this method uses internally): every
        value that needs to survive is protected on the stack, and the
        final address is only ever written into `dst` as the very last
        step.
        """
        array_type = type_of(expr.array)
        element_stride = type_byte_width(array_type.element_type, self.struct_registry)

        # len_reg only matters for a slice base (a runtime length);
        # picked dynamically, distinct from dst, since dst could in
        # principle be any register a caller passes. cap_reg is never
        # actually read afterward -- ordinary indexing never needs
        # capacity, only length -- but gen_indexable_base_into's own
        # contract always populates it, so this still needs a real,
        # distinct register to receive it into, picked the same way.
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
        counterpart to gen_array_value_into, dispatched from gen_slice_
        value_into wherever a slice-typed value needs to be produced.

        The base's own address and length are computed first (see
        gen_indexable_base_into -- an Imm for an array base, a
        runtime value read out of the descriptor for a slice base),
        then `low` and `high` are resolved -- each defaulting to 0 /
        the base's own length respectively when omitted (see Slice's
        own docstring in parser.py for why both stay None at parse
        time rather than one being defaulted earlier) -- and finally
        bounds-checked against each other and the base's length
        before the resulting ptr/len are computed and written.

        Every intermediate value (the base's address, its length,
        high, low) is protected on the real CPU stack across
        evaluating whichever of expr.low/expr.high are present --
        each of which could be an arbitrarily complex expression,
        including a function call -- rather than assumed to survive
        in whatever register initially held it. Pushed in a specific
        order (address, then length if runtime, then high, then low)
        and popped in exact reverse, so nothing ever needs to be read
        out of the middle of the stack -- except for one case:
        defaulting `high` to the base's own RUNTIME length (a slice
        base with no explicit high bound) reads it via a plain peek at
        the top of the stack (`(%rsp)`, no pop), since at that exact
        point nothing else has been pushed since the length was, and
        reading it without popping keeps it protected for the bounds
        check that comes later.

        Bounds checks use `ja` (strictly "above", unsigned), not `jae`
        -- unlike ordinary indexing (see gen_index_address_into),
        where an index equal to the array's own size is already out
        of bounds, `low` and `high` are both allowed to equal the
        base's own CAP (`arr[5:5]` on a 5-element array, or `s[5:5]`
        on a slice whose own cap is 5 even if its len is smaller, is a
        valid, empty-slice-producing expression) -- so the boundary
        itself genuinely differs here, not just the label it jumps to.

        Checked against CAP, not len -- `high` may reach all the way
        to the base's own remaining CAPACITY, matching Go's actual
        re-slicing rule, not just its current length. This is what
        lets a re-sliced view grow into room a PRIOR append (or the
        base's own construction) already reserved: cap is computed as
        base_cap - low below, inheriting the base's own remaining
        capacity from the new starting point, rather than simply
        matching the newly-computed len (high - low) the way it used
        to before cap-aware re-slicing existed -- see the module
        docstring's APPEND BUILTIN section for why this and `append`
        itself landed together rather than as two separate,
        sequential pieces of work: with every other slice-producing
        site setting cap equal to len, there was no way to observe or
        test a genuinely cap-aware re-slice until append existed to
        first produce a slice whose cap differs from its len at all.

        dst_mem.base is protected on the stack too, whenever it isn't
        'rbp', across ALL of the above -- pushed before even
        gen_indexable_base_into runs (the earliest point it could be
        clobbered: an ArrayLiteral base mallocs, via
        gen_array_literal_heap_alloc_into, and any evaluated low/high
        expression always targets %eax/%rax like every other
        expression in this file does) and popped back right before the
        final two writes, nested as the OUTERMOST push/pop pair around
        this method's own existing stack discipline -- everything else
        this method already pushes and pops happens strictly between
        the two, so nothing about their own relative ordering changes.
        Needed once a slice-typed value can be produced somewhere other
        than an ordinary local slot: an array literal whose OWN
        elements are themselves slices (`[][]int rows = [][]int[[1,
        2], [3, 4]]`) writes each element by calling this method with
        dst_mem.base equal to the OUTER array's own base -- 'rbp' if
        it's stack-allocated, or 'rax' if it's heap-allocated (every
        slice literal's own backing array always is -- see gen_array_
        literal_heap_alloc_into). Found necessary by the same class of
        real bug this file has hit before in this exact area (see
        gen_array_literal_into's own docstring), not assumed
        defensively -- this method used to just assert dst_mem.base ==
        'rbp' and refuse anything else, rather than risk it silently.
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
            # Pushed cap BEFORE len (the reverse of the field order in
            # the descriptor itself) specifically so len ends up on
            # TOP of the stack -- preserving, unchanged, the existing
            # "peek at (%rsp) for len's own default" logic just below,
            # which predates cap existing at all.
            instructions.append(Push(cap_reg))
            instructions.append(Push(len_reg))

        # Resolve `high` before `low`, so that defaulting it (when the
        # base's length is a runtime value) can safely peek the top of
        # the stack -- nothing else has been pushed since the length
        # was, right above. high still defaults to the base's own LEN
        # here, not its cap -- `arr[3:]` means "from 3 to the current
        # end", exactly as before; only the UPPER BOUND high is
        # allowed to reach when explicitly given (checked below) has
        # changed, not what an omitted one defaults to.
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

        # Bounds check: 0 <= low <= high <= CAP -- not len. This is
        # the real, deliberate change from before cap existed: Go's
        # own re-slicing rule allows high to reach all the way to the
        # base's remaining CAPACITY, not just its current length,
        # which is exactly what lets a re-slice grow into room a
        # PRIOR append (or the base's own construction) already
        # reserved. `low` is bounded by cap too, for the same reason
        # (low can equal cap, producing a valid, empty, zero-capacity
        # slice at the very end -- the base case a chain of further
        # re-slices or appends would still handle correctly).
        fail_label = self._get_bounds_check_fail_label("slice bounds out of range")
        cap_op = Register('r14d') if is_runtime_length else cap_operand
        instructions.append(Cmp(src=cap_op, dst=low_32))
        instructions.append(Ja(fail_label))
        instructions.append(Cmp(src=cap_op, dst=high_32))
        instructions.append(Ja(fail_label))
        instructions.append(Cmp(src=high_32, dst=low_32))
        instructions.append(Ja(fail_label))

        # new_cap = cap - low -- the base's own remaining capacity
        # from the new starting point, matching Go's actual re-slicing
        # rule (see this method's own docstring for the growth-policy
        # motivation: this is what lets `append` sometimes grow a
        # re-sliced view into its parent's own backing array instead
        # of always allocating fresh). Computed into its own register,
        # BEFORE low is scaled below, for the same reason new_len
        # (high - low, just after) already has to be: scaling would
        # destroy the unscaled value both of these still need. Mov
        # (not MovQ) here since cap_op may be a 32-bit Imm (an array
        # base) as easily as a 32-bit register (a slice base) -- both
        # are valid Mov sources into a 32-bit destination alike.
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
        # writes -- restored here, after every other computation above
        # (including the bounds check, which never falls through to
        # here on failure at all -- it aborts) has already finished.
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
        whenever this is writing a SLICE-typed element into an array
        literal whose OWN backing storage is heap-allocated (which
        every slice literal's own backing array always is -- see
        gen_array_literal_heap_alloc_into), the case that first made
        every one of the cases below need real protection rather than
        assuming 'rbp'. Dispatched on what kind of expression is
        producing the value:
          - Slice (e.g. `arr[1:3]`, or a slice LITERAL, `[]int[1, 2,
            3]`, parsed as one -- see parser.py's own _parse_
            bracketed_literal): computed directly, protecting
            dst_mem.base internally across its own, considerably more
            involved computation (see gen_slice_into's own docstring).
          - Variable (e.g. `s2 = s1`): a flat 24-byte copy of an
            existing slice's own descriptor. Deliberately NOT routed
            through gen_array_copy, even though that method's own
            flat-copy loop could technically move 24 bytes just as
            well as any other width: gen_array_copy's own handling of
            a slice LEAF type is specifically about an ARRAY whose
            ELEMENTS are slices, not about copying a bare slice
            descriptor itself, which is exactly what this case is --
            and a slice's descriptor is always exactly 24 bytes
            regardless of element type, so a fixed three-field copy
            (no loop needed at all) is both simpler and avoids that
            mismatch entirely. Uses %r8/%r9/%r10 as scratch, not %rax
            -- dst_mem.base is never %r8, %r9, or %r10 anywhere in
            this file (its only two established values are 'rbp' and
            'rax'), so this case needs no push/pop protection at all,
            unlike the others: an EARLIER version of this case used
            %rax as scratch for both (then just two) fields, which was
            silently wrong whenever dst_mem.base happened to BE 'rax'
            -- a later field's write would have used the just-loaded
            VALUE as the base address instead of the real one, since
            %rax no longer held it by then.
          - Call (e.g. `[]int s = otherFn()`, where otherFn also
            returns a slice): calls through the hidden-output-pointer
            convention, writing directly into dst_mem -- see gen_
            slice_call_into. Structurally identical to the ARRAY
            counterpart of this same case (gen_array_value_into's own
            Call case), now that slice returns use the exact same
            mechanism arrays already did.
          - Index (e.g. `[]int r = rows[0]`, reading one slice-typed
            element out of an array OF slices): the element's own
            address is computed first (gen_index_address_into), then
            its 24-byte descriptor is read through it -- structurally
            the same flat copy the Variable case does, just from a
            computed address rather than a fixed local offset.
          - Field (e.g. `[]int r = p.values`, reading a slice-typed
            STRUCT FIELD): structurally identical to the Index case
            just above, one level over -- the field's own address is
            computed first (gen_field_address_into), then its 24-byte
            descriptor is read through it the same way.
          - ArrayLiteral (e.g. `[]int s = [1, 2, 3]`, an UNTYPED
            literal flowing directly into a slice-typed target -- see
            semantic.py's _check_value_flowing_into and check_array_
            literal's own expected_element_type parameter for how this
            gets recognized during type-checking; note the general,
            TYPED form, `[]int[1, 2, 3]`, never reaches this case at
            all, since it parses as a Slice wrapping an ArrayLiteral,
            not a bare one -- see the Slice case above): mallocs a
            fresh backing array and writes the literal's own elements
            into it (gen_array_literal_heap_alloc_into), exactly like
            gen_indexable_base_into's own, separate ArrayLiteral case
            does for the general, typed form -- both ultimately need
            identical work, just reached from different call sites.
            cap is set equal to len here -- a fresh literal's backing
            array is sized to exactly fit its own elements, with no
            spare room to grow into yet.

        Every case that does real work between "start" and "write the
        result into dst_mem" -- every one except Variable and Call, per
        their own notes above -- protects dst_mem.base on the stack
        across that work whenever it isn't 'rbp', computing the result
        into scratch registers (or gen_slice_into's own internal ones)
        first and restoring dst_mem.base only immediately before the
        final writes use it. This generalizes what gen_array_literal_
        into's own scalar-element case already established for a
        single value; see its docstring for the real bug (not a
        hypothetical one) that made it necessary there -- the exact
        same class of bug applies here, just for a wider value produced
        across more instructions instead of a 4-or-8-byte one produced
        by a single gen_expr_into call.

        NoneLiteral is NOT handled here -- its own resolved type
        (Type.NONE) never matches SLICE, so it can't even reach this
        method through _gen_store's ordinary dispatch; see
        gen_none_into and gen_var_decl/gen_assign's own NoneLiteral
        short-circuit for why that's handled one level up instead.
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
        """Copies array_type's worth of data from src_mem to dst_mem --
        both arbitrary Memory operands (e.g. Memory('rbp', -24) for a
        fixed local's own slot, or Memory('rbx', 0) for a computed
        address held in %rbx) -- via a flat sequence of movl/movq
        instructions. A multi-dimensional array is just one contiguous
        block of leaf values in row-major order for copying purposes,
        so no per-dimension logic is needed here at all, just the
        total byte width and the leaf element's own width (see
        type_byte_width/leaf_type).

        Each leaf-sized chunk is copied as a flat run of 8-byte movqs,
        then one trailing 4-byte movl if at least 4 bytes remain, then
        a trailing run of 1-byte movbs for whatever's left after
        that (0 to 3 bytes) -- correct for ANY leaf width at all, not
        just a multiple of 4 the way this used to assume (back when
        every leaf was at least 4 bytes wide: 4 for int/bool, 8 for
        str, 24 for a slice descriptor, or a struct's own width, which
        used to always be a sum of 4-and-8-byte fields and so always
        landed on a multiple of 4 itself). int8/uint8's own genuinely
        1-byte-wide storage broke that assumption two ways at once: a
        BARE int8/uint8 leaf has leaf_width 1 directly, and a STRUCT
        leaf containing an int8/uint8 field can land on any width at
        all (1 int8 + 1 int field is 5, two int8s alone is 2, ...) --
        both were a real, found bug, not a hypothetical one: the old
        two-tier version (8-byte chunks, then EXACTLY one 4-byte
        remainder or none at all) silently copied NOTHING for either
        shape, since neither loop condition (`>= 8` chunks, `== 4`
        exactly) was ever satisfied by a 1-byte or 5-byte leaf_width --
        `b = a` for a `[3]int8` array, or an array of a struct with an
        int8 field, was a complete, silent no-op, not a wrong-but-
        partial copy. That generality is exactly why a STRUCT leaf
        already worked at all before int8/uint8 existed (a struct's
        own width can be any multiple of 4: 12, 20, 28, ... depending
        on its fields), and it needed no field-by-field recursion to
        get there: a raw,
        flat copy of every byte a value occupies is ALWAYS semantically
        identical to copying it "as" whatever logical type or fields
        those bytes represent, given this language's value semantics
        throughout -- there's no reference counting, no copy-
        constructor, and no write barrier anywhere in this language
        that a flat byte copy could possibly get wrong. This is exactly
        why a slice ELEMENT (24 bytes: pointer, then length, then cap)
        already worked before struct existed at all: those three
        sequential 8-byte movqs ARE a flat byte copy of the descriptor,
        which is exactly the shallow, alias-preserving copy slice
        values already get everywhere else (`s2 = s1`, see gen_slice_
        value_into's own Variable case) -- not a special case invented
        for arrays specifically. A struct containing a nested array,
        slice, or another struct needs nothing more than this same
        flat copy, for the identical reason: whatever's nested is
        already just more contiguous bytes within the outer value's
        own footprint.

        The scratch register shuttling each chunk's value between src
        and dst is picked dynamically to differ from BOTH src_mem's and
        dst_mem's own base register -- otherwise loading a value into
        it would destroy the very address a later iteration still needs
        to read from or write to. Found as a real bug during
        development, not a hypothetical one: gen_return passes
        Memory('rax', 0) as the destination when writing an array
        directly through a received hidden return pointer, and
        unconditionally using %eax/%rax as scratch (the very reasonable
        choice everywhere else in this file, since gen_expr_into always
        targets it) destroyed that address the moment the first
        element's value was loaded, before it could even be written
        anywhere. rcx and rdx are never used as a Memory base anywhere
        else in this file, so picking whichever of rax/rcx/rdx isn't
        already one of the two bases here stays correct regardless of
        how many 8- or 4-byte chunks a single leaf's own copy needs."""
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
            # if at least 4 bytes remain after that, then a trailing
            # run of 1-byte movbs (via as_byte_register on the same
            # scratch_32 register the 4-byte case already uses) for
            # whatever's left after THAT -- always 0 to 3 bytes, so at
            # most three movb pairs, never a real loop of its own. See
            # this method's own docstring for why all three tiers are
            # necessary now, not just the first two.
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
        """Computes the address to pass for an array-typed function-call
        argument, into the 64-bit register `dst`. A Variable or an
        Index yielding a sub-array already has a real, existing address
        (see gen_array_address_into); an ArrayLiteral or a call
        returning an array used DIRECTLY as an argument (`foo([1,2,3])`
        or `foo(bar())`) has no home of its own, so it's materialized
        first -- see _gen_materialize_argument_temp_into.

        The callee copies from this address into its own local slot on
        entry (see gen_function's parameter loop) -- so what's passed
        here only needs to stay valid for the duration of that one
        copy, not any longer, and the caller's own array is never
        itself mutated through it: the callee's copy is independent,
        preserving value semantics across the call the same way an
        ordinary `arr2 = arr1` does within a single function (see the
        module docstring's ARRAYS section)."""
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
        """Computes a slice-typed call ARGUMENT's own ptr/len/cap
        directly into ptr_dst/len_dst/cap_dst. A Variable (a named
        slice) or NoneLiteral (`none`) already has its own descriptor
        sitting somewhere real (a local slot, or nowhere at all --
        `none` is just three immediate zeros); anything else -- a
        slice literal or re-slice (`foo([]int[1,2,3])`, `foo(arr[1:
        3])`), an ordinary slice-returning Call (`foo(makeSlice())`),
        or a slice-typed Field/Index (`foo(s.values)`, `foo(rows[0])`)
        -- has no pre-existing descriptor of its own to read, and is
        materialized first via gen_slice_value_into (which already
        handles every one of these shapes) into the SAME shared, per-
        function scratch slot gen_indexable_base_into's own analogous
        cases already use (_unnamed_slice_temp_offset), then read
        straight back out of it.

        Reusing that ONE shared slot here -- rather than needing its
        own per-occurrence storage the way an array/struct-typed
        argument does (see _collect_argument_temps's own docstring for
        why THOSE need one) -- is safe for a genuinely different
        reason than gen_indexable_base_into's own "fully drained
        before anything else can reuse it" argument: a slice argument
        is passed BY VALUE -- three register values, immediately
        pushed onto the real stack right after this method returns
        (see _gen_call_arguments_into) -- not by address the way an
        array/struct argument is, so nothing about the call itself
        ever needs this scratch slot's own contents to still be valid
        afterward; only the pushed register VALUES do, and those
        already live on the real stack by then. That's also what makes
        two slice-typed arguments to the SAME call safe with only one
        shared slot (`foo([]int[1,2], []int[3,4])`): _gen_call_
        arguments_into evaluates arguments strictly one at a time, so
        the first argument's own descriptor is fully read out of the
        slot and pushed onto the stack before the second argument's
        own materialization ever touches the slot again -- and the
        identical strictly-nested reasoning covers a slice-returning
        call whose OWN argument is itself another unnamed slice
        (`foo(makeOuter(makeInner()))`): makeInner()'s own result is
        fully drained out of the slot and pushed as makeOuter's own
        argument, and `call makeOuter` -- the only thing that will
        eventually write THIS level's own result into the slot, via
        whatever hidden pointer it receives -- doesn't even execute
        until after that inner materialization has already completed."""
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

    def gen_slice_call_into(self, dst_mem: Memory, expr: Call) -> list[Instruction]:
        """Calls a function that returns a slice, writing its result
        directly into dst_mem via the exact same hidden-pointer
        convention gen_array_call_into already uses -- see its own
        docstring for the full reasoning, unchanged here in every
        respect except that this writes 24 bytes (a slice's own {ptr,
        len, cap} descriptor -- see gen_return's own Slice case on the
        receiving side) rather than an array's own, type-dependent
        width. Slices used to return via a dedicated %rax:%rdx two-
        register convention instead; that stopped fitting once a
        slice's own descriptor grew a third field (cap), with no
        established three-register return shape to grow into -- so
        slice returns now share the exact same mechanism arrays
        already had, rather than inventing a new one. This is also
        what makes forwarding one slice-returning call's result
        straight out of another free (`return otherFn()`), exactly
        like it already was for arrays: the SAME destination address
        just gets passed one level deeper, with no intermediate copy.
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
        local slot (Memory('rbp', offset)), but see gen_array_value_into
        for why this takes a general Memory operand rather than a bare
        offset. Each element is evaluated via the ordinary
        gen_expr_into (so an element can be any expression, not just a
        constant), except when the element type is ITSELF an array (a
        multi-dimensional literal's "elements" are themselves
        ArrayLiterals, handled by recursing through gen_array_value_into,
        which dispatches straight back here), a SLICE (an array whose
        elements are slices -- `[N][]int` -- e.g. the synthesized outer
        literal a slice-of-slices literal always is, `[][]int[[1, 2],
        [3, 4]]` -- handled by gen_slice_value_into, exactly like any
        other slice-producing expression; each element there might be
        an untyped ArrayLiteral needing a fresh backing allocation of
        its own, a named slice Variable, another Slice expression, or
        anything else that method already covers), or a STRUCT (an
        array of structs, `[N]Point` -- handled by gen_struct_value_
        into, which already covers every shape a struct-typed element
        can take: an ordinary Variable/Field/Index, a struct-returning
        Call, or -- as of the same fix that added struct literals as
        array elements at the semantic layer -- a struct literal
        directly, `[Point(1,2), Point(3,4)]`, via that method's own
        Call-is-a-struct-name dispatch).

        This STRUCT case used to not exist at all: an array-of-structs
        literal (or a struct-typed element written through _gen_write_
        value_at_address_into, append's own counterpart to this method
        -- see its own identical fix) would fall through to the
        scalar path below, whose gen_expr_into call flatly rejects any
        struct-typed read regardless of what expression produced it --
        this failed even for the simplest possible case, `[p1, p2]`
        with p1/p2 ordinary, already-declared struct variables, with
        no literal construction involved at all. Every OTHER operation
        on an array of structs (declaring one, indexing into it and
        reading/writing a FIELD of one element via Index-then-Field,
        whole-array copy via plain assignment, passing one as a
        parameter, returning one) already worked before this fix,
        since each of those routes through gen_array_copy (which
        already handles ANY leaf width as a flat byte copy -- struct
        included, see its own docstring) or gen_index_assign/gen_
        field_address_into (which already had their own STRUCT cases)
        rather than through this method's own per-element construction
        path -- literal construction specifically was the one gap.

        dst_mem's own base register is protected on the stack across
        each element's value computation whenever it isn't 'rbp' --
        found necessary by a real bug during development, not assumed:
        'rbp' (the frame pointer, used for every ordinary local slot)
        is never clobbered by gen_expr_into, so no protection is needed
        there, but a computed or received address held in a general-
        purpose register (e.g. Memory('rax', 0), the hidden return
        pointer for a literal returned directly -- `return [1,2,3]`,
        or a slice literal's own freshly-mallocd backing array) is
        exactly the kind of register gen_expr_into's own value
        computation, which always targets %eax/%rax, can and did
        clobber -- silently overwriting the destination address before
        a single element was ever actually written through it. Neither
        the SLICE nor the STRUCT case needs protection of its own
        here, unlike the scalar case just below: gen_slice_value_into/
        gen_struct_value_into both already protect dst_mem.base
        internally across whatever real work producing their own value
        takes (see their own docstrings), so by the time either
        returns, dst_mem.base is guaranteed correct again -- this loop
        can just call either directly and move on to the next
        element."""
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
                        # Mov here would discard int64's own high 32
                        # bits before _gen_write_scalar_from ever gets
                        # a chance to correctly write them, the exact
                        # same bug already found and fixed in gen_
                        # function's own parameter-binding logic and
                        # gen_print_call_into's own scratch-slot write.
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
        register the way an int/bool/str value can, so it needs its
        own dedicated "store" logic entirely, dispatched on what kind
        of expression is producing the value:
          - ArrayLiteral: each element stored directly (see
            gen_array_literal_into).
          - Variable: a copy from wherever the source's data actually
            lives (see gen_array_copy) into dst_mem. For a stack-
            allocated source, that's a flat offset-to-offset copy --
            no address computation needed at all, since the variable's
            own slot offset is already known at compile time. For a
            heap-allocated source (see is_heap_allocated), the slot
            holds a POINTER rather than the data itself, so that
            pointer is loaded first (protecting dst_mem's own base
            register across the load, via _gen_protecting_dst_across,
            in case they happen to coincide), and the copy reads
            through it instead. Either way, this is what makes
            `arr2 = arr1` a real, independent copy rather than a
            pointer alias -- see the module docstring's ARRAYS section
            on value semantics: heap-backed storage doesn't change
            that guarantee, only where the bytes being copied live.
          - Index (a sub-array, e.g. `[3]int row = matrix[i]`): its
            SOURCE address has to be computed first
            (gen_array_address_into), since it depends on a runtime
            index, then copied from that computed address. dst_mem is
            protected (see _gen_protecting_dst_across) across that
            computation, since it isn't just a simple move -- it
            includes bounds-checking and index arithmetic that freely
            uses %rax/%rcx internally.
          - Call (a function returning an array): calls through the
            hidden-output-pointer convention, writing directly into
            dst_mem -- see gen_array_call_into.

        dst_mem is a general Memory operand, not always a fixed local
        slot: it's Memory('rbp', offset) for an ordinary local variable
        or literal-initialized declaration, but Memory(some_reg, 0) when
        the destination is itself a computed or received address --
        e.g. gen_return uses this to write an array-typed return value
        directly through the hidden pointer it received, without ever
        materializing an intermediate local copy first. gen_array_copy
        and gen_array_literal_into both already work with an arbitrary
        Memory destination for exactly this reason -- nothing about the
        recursive structure here needed to change to support returns,
        only the type of the destination each caller happens to pass.
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
        anywhere at all -- nothing ever reads it as a coherent array
        -- so rather than reserving a scratch slot sized to fit it
        (which, unlike a slice's fixed 24-byte descriptor, an array
        literal has no natural upper bound for), this just evaluates
        each of the literal's own, directly-written elements for
        whatever side effects it might have (e.g. a function call),
        discarding every result -- exactly like any other bare
        expression statement already does (see gen_expr_stmt).

        Recurses for a nested ArrayLiteral element (a multi-dimensional
        literal used bare), the same way check_array_literal's own
        type-checking already does. An element that's ITSELF some
        other, non-literal array-, slice-, or struct-typed expression
        (a Variable, an indexed sub-array, an array/struct-returning
        Call, ...) is a real, deliberately out-of-scope gap: reading a
        bare array-typed Variable has no side effect worth preserving,
        but an array-returning Call might, and correctly distinguishing
        the two -- or materializing either one just to discard it --
        isn't implemented here. Raises a clear error rather than
        silently skipping (which could drop a real side effect) or
        guessing. (A struct-typed element specifically would already
        raise via gen_expr_into's own defensive rejection even without
        being listed here explicitly, since this method falls through
        to it for anything not caught above -- STRUCT is included in
        the tuple below anyway, for the same clearer, statement-
        specific message every other composite kind gets here, rather
        than relying on gen_expr_into's own more generic one.)
        """
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
        """`left == right` / `left != right`, both the exact same
        array type (already guaranteed by semantic.py's own check_
        binary, including that the array is actually comparable --
        see _is_comparable_type -- meaning its own LEAF type -- see
        leaf_type -- is int, bool, str, or a comparable struct; never
        a slice, or a struct with a slice buried in it somewhere,
        neither of which has '==' defined for it at all yet).

        `left`/`right` must each already have a real address (a
        Variable, Index, or Field -- whatever gen_array_address_into
        already accepts); an array literal or an array-returning call
        used directly as an equality operand isn't supported, matching
        this file's established "assign it to a variable first"
        restriction on unnamed array values elsewhere (e.g. gen_array_
        arg_address_into before argument materialization existed).

        Dispatches on the array's own leaf type into one of three
        genuinely different comparison strategies, each factored into
        its own small loop helper:

          - int/bool leaf: _gen_array_flat_byte_equality_loop. Neither
            type is a pointer, so the WHOLE array -- however many
            elements, however deeply nested (`[2][3]int` is just one
            contiguous 24-byte block) -- can be compared as one flat
            run of bytes, exactly the same "treat a nested array as
            one flat block" trick gen_array_copy already relies on for
            copying (via leaf_type/type_byte_width), just applied to
            comparison instead of copying.
          - str leaf: _gen_array_str_equality_loop. A str element IS a
            pointer (see the module docstring's STRINGS section), so
            raw byte-for-byte equality of the pointers themselves
            would be wrong -- two equal strings can easily live at two
            different addresses. This calls strcmp on each
            corresponding pair of elements instead, exactly like
            gen_string_compare_into's own ordinary str == str
            comparison does for a single pair, just without that
            method's own concatenation-freeing logic: array elements
            are always fixed, already-allocated storage, never a
            fresh concatenation result of their own.
          - struct leaf: _gen_array_struct_equality_loop. Neither of
            the two loops above applies: a struct's own fields can be
            a MIX of types, so there's no single flat-byte-or-strcmp
            strategy that covers a whole struct-typed element the way
            there is for a uniformly-int/bool or uniformly-str one --
            this reuses _gen_struct_fields_equality_at_addresses (the
            same field-by-field comparison gen_struct_equality_into's
            own bare struct-vs-struct case uses) once per element.

        All three loops jump to a shared `mismatch_label` the moment
        any element differs (or, for the flat-byte path, any 4-byte
        chunk differs); falling all the way through any of them means
        every element matched. From there, the final result is just
        two immediate moves -- 1/0 for EQUAL, or 0/1 for NOT_EQUAL,
        whichever bytes-matched actually means for this specific
        operator -- exactly the same "compute the boolean the long
        way, then pick the right immediate for this operator" shape
        gen_short_circuit already uses for AND/OR, one level over."""
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
            # int8/uint8 need a 1-byte step (see this loop's own
            # docstring for why 4 bytes at a time would be a real,
            # out-of-bounds bug for either); int/bool/slice all stay
            # the existing 4-byte step, since type_byte_width already
            # guarantees their own total_width is a multiple of 4
            # regardless of nesting depth.
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
        width guarantees total_width is always a multiple of 4 for any
        of those, however deeply nested), or 1 for an int8/uint8 leaf.

        The 1-byte case is a real, found bug's fix, not a defensive
        addition: this loop used to ALWAYS step 4 bytes at a time,
        which was fine as long as every leaf this compiler had was 4
        (or a multiple of 4, for a slice leaf's own 24) bytes wide --
        but int8/uint8's own genuinely 1-byte-wide storage means
        total_width isn't generally a multiple of 4 at all (a [3]int8
        array is 3 bytes total). Stepping 4 bytes at a time regardless
        read one byte past the end of the array on every comparison,
        silently comparing whatever adjacent stack memory happened to
        follow it instead of correctly reporting equality.

        The 1-byte step reads each side via MovZX (zero-extending into
        a 32-bit register) rather than a plain 4-byte Mov, needing a
        second register (%edx, otherwise unused in this loop) to hold
        the right side's own zero-extended value before comparing the
        two directly -- correct regardless of whether the ACTUAL leaf
        is signed (int8) or unsigned (uint8): byte-for-byte equality
        never depends on how those bits are INTERPRETED, only on
        whether they're identical, and zero-extension is a
        deterministic, injective mapping from one byte to 32 bits, so
        two bytes are equal if and only if their zero-extended 32-bit
        versions are.

        Jumps to mismatch_label the moment any chunk differs, or
        simply falls through once every chunk has matched. No calls
        happen anywhere in this loop, so nothing here needs a callee-
        saved register the way the str-leaf loop below does; every
        register used is ordinary caller-saved scratch, freely
        reusable by whatever runs after this method returns."""
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
        loop just above: each of the array's `element_count` str
        elements is a POINTER, so this calls strcmp on each
        corresponding pair rather than comparing raw pointer bytes.

        strcmp is a real external call, free to clobber any CALLER-
        saved register -- so unlike the flat-byte loop, the two base
        addresses and the loop index all have to live in CALLEE-saved
        registers (%rbx/%r12/%r13) to survive it, the exact same
        discipline gen_append_call_into's own malloc-crossing
        registers already follow, for the identical reason: every
        function's own prologue/epilogue already saves and restores
        these four unconditionally (see _CALLEE_SAVED_SCRATCH_
        REGISTERS), so using them as scratch across a call, in ANY
        function, is always safe."""
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
        # pointer) -- computed fresh each iteration, before the call,
        # so it never needs to survive one itself.
        instructions.append(Mov(src=index_32, dst=offset_32))
        instructions.append(IMul(src=Imm(8), dst=offset_32))
        left_elem_addr = Register('r8')
        right_elem_addr = Register('r9')
        instructions.append(MovQ(src=left_base, dst=left_elem_addr))
        instructions.append(AddQ(src=offset_64, dst=left_elem_addr))
        instructions.append(MovQ(src=right_base, dst=right_elem_addr))
        instructions.append(AddQ(src=offset_64, dst=right_elem_addr))
        # Load the actual string POINTERS stored at these element
        # addresses -- straight into the argument registers strcmp
        # itself expects, since nothing else needs them first.
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
        strcmp per element), a struct element's own fields can be a
        MIX of types, so each element is compared via _gen_struct_
        fields_equality_at_addresses -- the same field-by-field
        machinery gen_struct_equality_into's own bare struct-vs-struct
        case uses -- rather than a single uniform per-element
        operation.

        Uses the identical CALLEE-saved register discipline _gen_
        array_str_equality_loop already established, for the same
        reason (a struct element being compared might itself contain
        a str field, which needs a real strcmp CALL somewhere inside
        _gen_struct_fields_equality_at_addresses).

        Beyond that, this loop's own left_base/right_base/index
        (%rbx/%r12/%r13) need one MORE layer of protection that
        neither sibling loop does: _gen_struct_fields_equality_at_
        addresses can itself recurse back into ONE of these same
        three array-equality loops, for a struct field that's itself
        an array (including, recursively, another array of structs --
        e.g. comparing `[M]Outer` where Outer has a `[N]Inner rows`
        field, and Inner is itself a comparable struct). Any such
        nested loop reuses the EXACT SAME fixed register names this
        one does (%rbx/%r12/%r13/%r14, since there's no way to
        allocate a fresh, distinct set per nesting depth at codegen
        time) -- so without explicitly saving THIS loop's own
        %rbx/%r12/%r13 across the per-element comparison call, a
        struct field found to need one of those nested loops would
        silently corrupt this OUTER loop's own base addresses and
        index. Protecting them via an ordinary push/pop pair around
        that one call is what makes this correct at ANY nesting depth,
        by the same induction _gen_struct_fields_equality_at_
        addresses's own per-field protection already relies on: at
        every level, whatever's ABOUT to run might reuse these
        registers for its own purposes, so whatever's ALREADY relying
        on them saves its own values first and restores them
        afterward, regardless of what happened in between. (%r14,
        the per-iteration byte offset, needs no such protection: it's
        always freshly recomputed at the START of each iteration,
        before being used to compute this iteration's own element
        addresses, and never read again afterward.)"""
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
        """Zeroes a whole array -- dispatching on the array's own LEAF
        type (see leaf_type) into one of three genuinely different
        strategies, mirroring array equality's own identical three-way
        split (_gen_array_flat_byte_equality_loop/_gen_array_str_
        equality_loop/_gen_array_struct_equality_loop) for the same
        underlying reason: a leaf's own zero-value representation
        determines whether the whole array can be zeroed as one flat
        run of raw bytes, or needs a real per-element write.

          - int8/uint8, int, bool, OR SLICE leaf: _gen_array_flat_zero_
            loop. All four types' own zero value is ALL RAW ZERO BYTES
            with no pointer or other special representation (a slice's
            own none-shaped {0, 0, 0} descriptor -- see gen_none_
            into -- IS 24 zero bytes, nothing more), so the WHOLE
            array, however many elements and however deeply nested, is
            zeroed as one flat run -- the same "treat a nested array as
            one contiguous block" trick gen_array_copy/array equality's
            own flat-byte loop already rely on. Array equality couldn't
            offer slice this same treatment (a slice-typed array
            element isn't COMPARABLE at all yet, so it never reached
            that dispatch), but zeroing has no such restriction: there
            being nothing to compare, only a zero value to write, is
            exactly what makes slice fit here for free. int8/uint8 need
            a 1-byte STEP through that same flat run rather than
            int/bool/slice's own 4-byte one -- see _gen_array_flat_
            zero_loop's own docstring for why (the identical "total_
            width isn't generally a multiple of 4 for a genuinely
            1-byte-wide leaf" reasoning array equality's own flat loop
            already needed fixing for).
          - str leaf: _gen_array_str_zero_loop -- a str's own zero
            value is a POINTER (see _gen_zero_value_into's own STR
            case), so each element needs that same address written
            individually, not raw zero bytes.
          - struct leaf: _gen_array_struct_zero_loop -- a struct's own
            fields can be a MIX of types, so there's no single flat-
            bytes-or-repeated-pointer strategy that covers a whole
            struct-typed element; each one is zeroed via a recursive
            call back into _gen_zero_value_into itself."""
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
        time -- see _gen_zero_array_into's own docstring for why 4 is
        correct for an int, bool, or slice leaf at any nesting depth,
        via a plain 4-byte Mov of Imm(0).

        1 is correct instead for an int8/uint8 leaf (via a 1-byte
        MovB, rather than a 4-byte Mov, of that same Imm(0)) for the
        identical reason _gen_array_flat_byte_equality_loop's own step
        parameter exists: type_byte_width no longer guarantees total_
        width is a multiple of 4 once a genuinely 1-byte-wide leaf
        exists (a [3]int8 array is 3 bytes total) -- stepping 4 bytes
        at a time regardless would write one byte past the end of the
        array, corrupting whatever stack memory happens to follow it.

        No calls happen anywhere in this loop, so every register here
        is ordinary caller-saved scratch, freely reusable by whatever
        runs after this method returns -- the same posture _gen_array_
        flat_byte_equality_loop's own identical loop shape already
        takes, for the identical reason.

        Computes dst_mem's own starting address into a FIXED register
        (%r10) via _gen_address_of_memory_into, rather than assuming
        dst_mem.base itself remains valid to keep reading from
        directly -- correct even when dst_mem.base happens to BE %r10
        already (that method's own self-copy-then-add shape handles
        that case safely), but this method never relies on dst_mem's
        own base surviving its own execution either way; any caller
        that needs dst_mem.base to still be valid AFTERWARD (see _gen_
        zero_value_into's own struct case) is responsible for
        protecting it externally, via push/pop, before calling this."""
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
        """The str-leaf counterpart to _gen_array_flat_zero_loop just
        above: computes the shared empty-string address ONCE, then
        writes that same 8-byte value into each of `element_count`
        consecutive element slots. No calls happen here either (LeaQ
        computes a RIP-relative address, it doesn't call anything), so
        -- like the flat-zero loop -- every register is ordinary
        caller-saved scratch, with nothing here relying on it surviving
        past this method's own return."""
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
        element is zeroed via a recursive call back into _gen_zero_
        value_into, since a struct's own fields can be a mix of types
        with no single flat-bytes-or-repeated-value strategy.

        Unlike its two siblings, this loop's own base address (%r12)
        and index (%r13) DO need protecting across that recursive
        call: the struct being zeroed could itself have an array-typed
        field, which would dispatch back into one of THESE SAME THREE
        loops (reusing the identical fixed register names, since
        there's no way to hand out a fresh, distinct set per nesting
        depth at codegen time) -- silently corrupting this outer
        loop's own %r12/%r13 if they weren't saved first. Protected
        via an ordinary push/pop pair around the one recursive call,
        exactly like _gen_array_struct_equality_loop's own identical
        situation (see its own docstring for the fuller explanation of
        why this is correct at any nesting depth, by induction: every
        level protects only what IT locally needs to survive, trusting
        nothing else about what runs in between). %r14 (the per-
        iteration byte offset) needs no such protection, for the same
        reason it doesn't there either: always freshly recomputed at
        the start of an iteration, never read again after computing
        this iteration's own element address."""
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
        whether the slice's own `ptr` field is null, matching Go's
        own nil-vs-empty-slice distinction (see NoneLiteral's own
        docstring in parser.py): a real, zero-length slice sliced
        from a real array (e.g. `arr[5:5]`) has a non-null pointer
        and is NOT `== none`, even though both are equally safe,
        equally zero-length slices for every other purpose (indexing,
        printing, re-slicing).

        semantic.py's check_binary already guarantees, by the time
        this is reached, that exactly one operand is slice-typed and
        the other is none-typed -- so this doesn't need to re-derive
        or defensively check which side is which beyond picking out
        whichever one IS slice-typed.

        Reuses gen_indexable_base_into for the slice's own address
        (see its own docstring for why the base must be a bare
        Variable when it's slice-typed) even though only the address,
        not the length or capacity it also computes, is actually
        needed here -- two harmless, unused extra loads rather than a
        second, narrower helper that would duplicate its Variable-vs-
        Index and array-vs-slice handling for a single call site.

        Uses CmpQ (64-bit), not the ordinary 32-bit Cmp every other
        comparison in this file uses -- a pointer is a full 64-bit
        value, and checking only its low 32 bits against zero could,
        in principle, miss a real, non-null pointer whose low 32 bits
        happen to be zero.
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
        """The reuse-vs-reallocate growth core shared by gen_append_
        call_into and (see gen_buffer_append_byte_into) the print
        buffer's own single-byte append -- factored out of what used
        to be gen_append_call_into's own, sole implementation of this
        exact policy, specifically so a second, parallel copy of the
        identical growth arithmetic never has to exist. See gen_
        append_call_into's own docstring for the growth-policy
        arithmetic itself (new_cap = cap*2 if cap < 256, else cap +
        cap//4, with a cap==0 floor of 1) and the reuse-vs-reallocate
        reasoning behind it -- unchanged here, just no longer
        duplicated.

        Operates entirely on registers the caller has already loaded
        with a live {ptr, len, cap} triple (both the 64-bit register
        and its own 32-bit view, passed explicitly as a pair for each
        of len/cap, matching how every other place in this file that
        needs both already keeps them as two separate Register values
        rather than deriving one from the other) -- never on a Memory
        location directly. Loading the initial triple from wherever it
        actually lives, and writing the final one back out afterward
        (or, for the print buffer's own case, simply leaving it live in
        registers across many further appends without ever touching
        Memory at all), is entirely the caller's own responsibility.

        Internally fixed scratch, matching gen_append_call_into's own
        original, unfactored version exactly: %r8/%r8d, %r9d, %r10/
        %r10d, %r11/%r11d, %eax/%ecx, and %r14. Callers must choose
        their own ptr/len/cap registers to avoid these.

        `copy_one_element(dst_addr, src_addr)` is called once per
        existing element, ONLY on the reallocate path, to move it from
        the old backing array into the new one -- gen_append_call_into
        passes one that calls gen_array_copy for element_type; the
        print buffer passes one that just moves a single byte.

        `write_new_value_at(target_addr)` is called exactly once, at
        the correct final address for the newly appended element (in
        whichever backing array ends up live -- the original, on the
        reuse path, or the freshly malloc'd one, on the reallocate
        path) -- gen_append_call_into passes one that evaluates an
        arbitrary Hornet expression via _gen_write_value_at_address_
        into; the print buffer passes one that writes an already-
        computed raw byte operand directly. Neither callback needs to
        know how the other one works, or how growth itself is decided.

        On return, r_ptr/r_len/r_cap (and their 32-bit views) hold the
        FINAL triple: len always incremented by exactly one, ptr
        repointed to a fresh block if reallocation happened,
        unchanged otherwise."""
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
        (s keeps pointing at exactly what it always did, with exactly
        its own original len and cap) -- see the module docstring's
        APPEND BUILTIN section for the full growth-and-aliasing story
        this is built around.

        s (expr.args[0]) can be any slice-typed expression -- a bare
        Variable or `none` (materialized inline, no scratch slot
        needed), or anything else (a slice literal, a re-slice, an
        Index, a slice-returning Call, ...), which gets materialized
        into the shared per-function scratch slot (_unnamed_slice_
        temp_offset -- see gen_indexable_base_into's own Slice-base
        case for the same pattern already used there) via gen_slice_
        value_into, then read back out exactly like a Variable's own
        slot would be. This used to be restricted to just a Variable
        or NoneLiteral, on the theory that append exists specifically
        to feed a reassignment (`x = append(x, v)`) and a bare slice
        expression as its own first argument would be rare enough not
        to justify the extra materialization step -- lifted once that
        turned out to matter in practice (`append([]int[], 1)`, for
        instance, needs exactly this to build a slice from scratch in
        a single expression). The materialization itself needs no
        special handling for what it's protecting: gen_slice_value_
        into already protects an arbitrary destination base
        internally, and the scratch slot here is always 'rbp'-based,
        so there's nothing dst_mem-shaped for it to clobber.

        s's own three fields are loaded into CALLEE-SAVED registers
        (%rbx/%r12/%r13 for ptr/len/cap) -- not caller-saved ones --
        specifically because the REALLOCATE path (inside _gen_grow_
        and_append_one_into) calls malloc, which (like any real, ABI-
        conforming function) is free to clobber any caller-saved
        register but is OBLIGATED to preserve callee-saved ones -- the
        exact same guarantee gen_array_literal_heap_alloc_into and
        gen_function's own heap-allocated-parameter handling already
        rely on.

        The actual growth policy -- the reuse-vs-reallocate decision,
        the growth-policy arithmetic itself, the copy-existing-
        elements loop -- lives entirely in _gen_grow_and_append_one_
        into now, shared with the print buffer's own single-byte
        append; this method's own remaining job is just materializing
        s into registers beforehand, and writing the final triple to
        dst_mem afterward, protecting dst_mem.base (whenever it isn't
        'rbp') across the whole thing the same way every other slice-
        producing case in this file does -- popped back only
        immediately before the final three-field write actually
        needs it.
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
            # Any other slice-typed expression (a slice literal, a
            # re-slice, an Index, a slice-returning Call, ...): build
            # its own {ptr, len, cap} descriptor into the shared,
            # per-function unnamed-slice scratch slot first, then read
            # it back out exactly like a Variable's own slot would be.
            # This scratch slot is always 'rbp'-based, so there's
            # nothing here for gen_slice_value_into's own dst_mem
            # protection to need to guard against beyond what it
            # already does internally.
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
        held in r_ptr/r_len/r_cap (and their own 32-bit views) -- the
        print machinery's own single-character append, sharing the
        identical growth policy `append()` itself uses (see _gen_
        grow_and_append_one_into) with element_width=1 and a byte-
        sized write/copy in place of a general Hornet element type's
        own gen_array_copy/_gen_write_value_at_address_into.

        `byte_value` is whatever operand already holds the byte to
        append -- typically an Imm (a literal ASCII byte, e.g. the
        '[' opening a collection, or a decimal digit already reduced
        to a compile-time or runtime-computed Imm) or an 8-bit
        register alias (via as_byte_register) if the byte was computed
        into a register first. Written via MovB -- the first place
        this compiler has ever needed a genuine single-byte memory
        write, as opposed to a 4-byte int/bool or an 8-byte pointer.

        Unlike gen_append_call_into, this never touches a Memory
        destination at all: r_ptr/r_len/r_cap are expected to stay
        live in registers across many further appends while a single
        value's own representation is being built up (see the
        recursive stringify machinery this exists for), not written
        back out after every single byte -- that would be needless
        memory traffic for something that might get appended to
        hundreds of times while building one struct's own printed
        form. Callers that DO need the current triple durably
        persisted (spanning a call into another function, for
        instance) are responsible for spilling it themselves."""
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
        source_addr -- the print machinery's own multi-byte append
        (a whole literal fragment like '[', a run of decimal digits
        just converted, another already-built piece), as opposed to
        gen_buffer_append_byte_into's single-character one. This is a
        genuinely DIFFERENT growth calculation, not just a loop calling
        the single-byte version count times: that method's own growth
        formula is only correct because it's derived under the
        assumption reallocation happens exactly when len == cap, one
        element at a time (see gen_append_call_into's own docstring)
        -- appending a 40-byte chunk when only, say, 4 bytes of spare
        capacity remain needs `needed` (len + count) to enter the
        decision directly, which the single-element formula's own
        already-simplified arithmetic has no way to do.

        GROWTH: needed = len + count. If needed <= cap, there's
        already enough spare room -- no reallocation at all, just copy
        directly into the existing backing array at ptr + len. Only
        when needed > cap does this reallocate, to new_cap = max(needed,
        cap*2 if cap < 256 else cap + cap//4) -- the FULL, general
        formula gen_append_call_into's own docstring describes and
        then simplifies away (since ITS OWN needed is always exactly
        cap+1, small enough that doubling always already exceeds it).
        Here needed can be arbitrarily larger than a single doubling
        would produce, so the max has to be computed for real, not
        assumed away -- and this also means the single-element
        formula's own explicit cap==0 floor of 1 isn't needed here
        either: when cap is 0, the doubled-or-quartered side of the
        max is just 0, and max(needed, 0) already correctly resolves
        to needed on its own, since needed (len + count, with count
        always at least 1 for any real call) is always positive.

        Both the reallocate path's own copy-existing-bytes step and
        the final copy-the-new-bytes-in step move `len` (or `count`)
        bytes one at a time via MovB, in a genuine runtime loop --
        there's no bulk memory-move instruction in this file's own
        Instruction vocabulary yet (a real `rep movsb`, or SSE-based
        copy, would be the natural next step if this ever needs to be
        fast; correctness came first here, matching this compiler's
        existing posture everywhere else).

        Internally fixed scratch, distinct from gen_grow_and_append_
        one_into's own set so this can be called independently:
        %rax/%eax, %rcx/%ecx, %rdx/%edx, %rdi, %r10, %r11, %r14, %r15
        (the latter two hold protected copies of source_addr/count --
        see the comment where they're introduced below). Callers must
        choose r_ptr/r_len/r_cap, and whatever register source_addr
        itself lives in, to avoid all of these, exactly like
        gen_buffer_append_byte_into's own callers already must."""
        instructions = []
        no_grow_label = self.new_label("bulk_append_no_grow")
        copy_new_label = self.new_label("bulk_append_copy_new")

        # Move source_addr (always a register in practice) -- and
        # count, if it's a register rather than a compile-time Imm --
        # into callee-saved registers (%r14/%r15) BEFORE any of the
        # growth/malloc logic below runs. The reallocate path calls
        # malloc internally, which -- like any real, ABI-conforming
        # function -- is free to clobber whatever CALLER-saved
        # register the caller happened to pass in for these (e.g.
        # %r8/%r9, as _gen_stringify_bulk_append's own callers do),
        # corrupting them by the time the copy-new loop below needs
        # them afterward. This exact class of bug already has one
        # instance fixed below for new_cap (%ecx -> r_cap_32) -- the
        # same protection was missing here, and manifested identically:
        # correct on the no-grow path (no malloc call to clobber
        # anything), silently wrong -- reading and copying garbage as
        # the new bytes' own source -- on the reallocate path
        # specifically, and only there, which is exactly why it first
        # surfaced as heap corruption several appends downstream
        # rather than as an obviously-wrong value at the call site
        # itself. An Imm count needs no such protection: it's baked
        # directly into the instructions that use it, never stored in
        # a register malloc could touch.
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
        # Shift the CANDIDATE (starting from cap, held in %edx) right
        # by 2, then add cap back -- ShiftRightArithmetic's own fixed
        # %cl-sourced count means %ecx has to hold the shift amount
        # (2) here, not the candidate itself, unlike the plain-doubling
        # branch just above where %ecx directly holds the result.
        instructions.append(ShiftRightArithmetic(dst=Register('edx')))
        instructions.append(Mov(src=r_cap_32, dst=Register('ecx')))
        instructions.append(Add(src=Register('edx'), dst=Register('ecx')))
        instructions.append(Label(candidate_done_label))
        # %eax = needed, %ecx = candidate. new_cap = max of the two,
        # left in %ecx (needed's own value in %eax is still required
        # below, for how many NEW bytes to copy in, so %eax itself is
        # never overwritten by this comparison).
        instructions.append(Cmp(src=Register('ecx'), dst=Register('eax')))
        max_done_label = self.new_label("bulk_append_max_done")
        instructions.append(Jle(max_done_label))
        instructions.append(Mov(src=Register('eax'), dst=Register('ecx')))
        instructions.append(Label(max_done_label))

        # new_cap (in %ecx, caller-saved) MUST move into r_cap_32
        # (callee-saved) before calling malloc, not after: malloc, like
        # any real ABI-conforming function, is free to clobber %ecx
        # during its own execution, and is only OBLIGATED to preserve
        # callee-saved registers -- exactly the guarantee gen_append_
        # call_into's own %rbx/%r12/%r13 already rely on. Keeping the
        # computed value in %ecx across the call and reading it back
        # afterward (an earlier version of this method did exactly
        # that) is a real, silent bug: %ecx isn't guaranteed to still
        # hold what was put there once malloc returns, and glibc's own
        # malloc does in practice clobber it -- found only by directly
        # checking the resulting cap against a hand-worked-out
        # expected value, since the visible SYMPTOM (a buffer sized
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
        # reallocate path -- needed's own value in %eax was last
        # needed to compute new_cap, already consumed into %ecx and
        # then malloc'd (whose OWN return value, briefly also in
        # %eax, has already been copied out into r_new_ptr) -- so
        # nothing here needs preserving across a single byte move,
        # unlike the two Push/Pop pairs an earlier draft of this loop
        # had, which protected against nothing actually still live.
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
        # r_cap_32 already holds new_cap -- written BEFORE the malloc
        # call above, precisely so it survives that call correctly
        # (see this method's own comment there); nothing further to do
        # with %ecx here, which may no longer even hold that value.
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
        Memory(addr_reg, 0) -- shared by gen_append_call_into's own
        reuse and reallocate paths, both of which need to write the
        newly-appended element at a computed (not fixed-offset)
        address, with the element's own type possibly being scalar,
        array, slice, or struct.

        For an ARRAY, SLICE, or STRUCT element type, this just hands
        addr_reg straight to gen_array_value_into/gen_slice_value_
        into/gen_struct_value_into as an ordinary Memory destination --
        all three already protect an arbitrary base internally (see
        their own docstrings), so there's nothing extra to do here;
        mirrors gen_array_literal_into's own identical three-way
        dispatch for a literal's per-element writing, one level over
        (append's own newly-appended element, rather than a literal's
        directly-written one). For a scalar (int/bool/str), addr_reg
        is protected manually, matching gen_array_literal_into's own
        scalar-element pattern exactly: push addr_reg, compute the
        value (which could itself involve a function call that
        clobbers addr_reg, if value_expr is arbitrarily complex),
        stash the computed value in %r8/%r8d (a register distinct from
        addr_reg in every actual call site), pop addr_reg back, then
        write from %r8/%r8d -- never straight from %eax/%rax, which
        popping addr_reg back into would otherwise have to clobber."""
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
            # would discard its own high 32 bits before ever reaching
            # the final write below. The final write itself goes
            # through _gen_write_scalar_from (not a bare Mov, which
            # this used to be) so int8/uint8 get their own correct,
            # narrow, 1-byte write too -- a bare 4-byte Mov here was a
            # latent, if not directly observed, bug for them as well:
            # writing 4 bytes at a 1-byte element's own offset can
            # write past a freshly-grown backing array's own allocated
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
        _gen_bounds_check_panic_block) at every individual check site.
        A single function can use more than one message (e.g. "array
        index out of bounds" for ordinary indexing, "slice bounds out
        of range" for a slice expression's own low/high check) --
        each gets its own fail label, all reset together at the start
        of every function (see gen_function) -- unlike the message
        labels below, these are purely LOCAL jump targets, meaningless
        outside the function they're generated for."""
        if message not in self._bounds_check_fail_labels:
            self._bounds_check_fail_labels[message] = self.new_label("bounds_check_fail")
        return self._bounds_check_fail_labels[message]

    def _get_bounds_check_message_label(self, message: str) -> str:
        """Lazily creates and caches (for the rest of the WHOLE
        compilation, unlike the per-function fail labels above -- each
        is just a static string, safely shared by every function that
        needs it, matching the same lazy-cache pattern print's own
        format-string/true/false labels already use) a label for this
        exact `message` string."""
        if message not in self._bounds_check_message_labels:
            label = self.new_label("bounds_msg")
            self._bounds_check_message_labels[message] = label
            self.string_literals.append((label, message))
        return self._bounds_check_message_labels[message]

    def _gen_bounds_check_panic_block(self) -> list[Instruction]:
        """Appended once at the end of a function's own instructions
        (see gen_function) for every distinct message that function's
        own bounds checks actually used (see
        _get_bounds_check_fail_label) -- none at all if it never
        triggered any. Each block prints its own clear message, then
        calls abort() (SIGABRT) rather than a plain exit() -- an out-
        of-bounds access is a genuine program bug, not a normal
        termination condition, the same "abnormal termination"
        character division by zero's hardware-trapped SIGFPE already
        has, just deliberately raised by this compiler's own generated
        code instead of by the CPU. Never reached via ordinary fall-
        through from the function's own body -- every return already
        leaves via `leave; ret` before control could reach this point,
        and abort() itself never returns -- so appending these at the
        very end of the function is always safe.

        Explicitly calls fflush(NULL) between puts() and abort() --
        found necessary by testing, not assumed: abort() terminates
        the process via a raw signal, bypassing the normal exit() path
        that would otherwise flush libc's buffered stdio streams. Without
        this, the message is reliably printed when stdout happens to be
        line-buffered (an interactive terminal) but silently LOST
        whenever stdout is redirected or piped -- exactly the case for
        any program run non-interactively, which is most of them. A
        NULL argument tells fflush to flush every open output stream,
        so this doesn't need to reference libc's `stdout` symbol
        directly (a global variable, not a function -- meaningfully
        more awkward to reference correctly from hand-written assembly
        than another ordinary `call`).
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
        exact same "address plus length, however each is represented"
        abstraction indexing and slicing already share -- rather than
        a narrower restriction of its own like print's Variable-or-
        Index one (see gen_print_call_into's own docstring): whatever
        gen_indexable_base_into currently accepts as a base (a
        Variable, an Index, a Slice expression, a slice-returning
        Call, or an ArrayLiteral) is automatically valid here too,
        with nothing to keep in sync if that set ever grows.

        x's own address is computed and then simply discarded -- len
        only ever needs the LENGTH half of gen_indexable_base_into's
        own return value -- but computing it is not wasted: x is still
        fully evaluated regardless (any bounds-check or side effect
        buried in it genuinely runs), matching how any other function
        argument's evaluation works, whether or not the computed
        address ends up used for anything afterward. This does mean
        `len(arr[i])` still aborts if i is out of range, and
        `len([]int[1, 2, 3])` still performs a real, if wasted, heap
        allocation -- both deliberate, not something a narrower
        special case tries to avoid (see the module docstring's LEN
        BUILTIN section).

        For an ARRAY base, length_operand comes back as an Imm (a
        compile-time constant -- the array's own declared size, never
        actually read out of x at runtime at all); for a SLICE base,
        as the 64-bit len_dst register holding a runtime value read
        out of the slice's own descriptor -- moved through its own
        32-bit alias here, matching how every other reader of a
        slice's length field already narrows it the same way, since
        Hornet's int is always 32 bits even though the descriptor's
        own len field is stored in a full 8-byte slot."""
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
        """Computes the ADDRESS a Memory operand refers to, into `dst`
        (a 64-bit register). Memory('rbp', offset) needs a real leaq --
        the address is offset-from-frame-pointer, not stored anywhere
        as a value in its own right; Memory(some_reg, offset) already
        HAS its address sitting directly in some_reg, with `offset`
        (if non-zero) added on top via a single AddQ -- see gen_array_
        copy's own docstring for how the some_reg shape arises
        elsewhere in this file. Used specifically for passing a Memory
        destination on as a POINTER argument -- the hidden output
        pointer for an array-returning call (gen_array_call_into) or a
        struct-returning one (gen_struct_call_into, which is really
        gen_array_call_into under a different name -- see its own
        docstring) -- everywhere else, a Memory operand is read from
        or written to directly rather than having its own address
        taken.

        The offset(some_reg) case used to assume offset was always 0
        whenever base wasn't 'rbp' -- true at the time, since nothing
        computed a destination this way for anything but the WHOLE of
        a Memory destination, offset already folded in or genuinely
        zero. That stopped being true once a struct literal's own
        array-typed FIELD could be populated directly by an array-
        returning call (`Big(1, makeArr())`, where `data` -- an array
        field -- sits at some non-zero offset on a heap-allocated
        Big): gen_struct_literal_into's own field_mem for that field
        is Memory('rax', 4), say, and the OLD version of this method
        silently discarded that +4, handing makeArr() the STRUCT's own
        base address as its hidden return pointer instead of the
        field's -- a real, silent miscompile (verified directly: it
        corrupted the PRECEDING field along with the start of the
        array itself), not a hypothetical one. Adding the AddQ here is
        safe for every EXISTING caller too: each one already only ever
        passed offset=0 for a non-'rbp' base, so this is a pure
        generalization, not a behavior change for anything already
        working."""
        if mem.base == 'rbp':
            return [LeaQFrame(offset=mem.offset, dst=dst)]
        instructions = [MovQ(src=Register(mem.base), dst=dst)]
        if mem.offset:
            instructions.append(AddQ(src=Imm(mem.offset), dst=dst))
        return instructions

    def gen_none_into(self, dst_mem: Memory, target_type: Type) -> list[Instruction]:
        """Writes `none`'s own zero-value representation into dst_mem,
        for whichever nilable type target_type actually is. Only
        slices are nilable so far (see NoneLiteral's own docstring in
        parser.py) -- a {ptr: 0, len: 0, cap: 0} descriptor, the same
        shape Go's own nil slice has: a valid, safely-indexable-into-
        nothing slice with no backing array, not a special, separately-
        tracked null flag. Every existing slice operation (indexing,
        printing, re-slicing) already handles a zero-length slice
        correctly -- see TestSliceBoundsChecking's own positive
        control for `arr[5:5]` -- so this is the ONLY new codegen a
        none-valued slice needs on the producing side; comparing one
        against `none` again (see gen_slice_none_comparison_into) is
        the only other.

        Called directly from gen_var_decl/gen_assign's own NoneLiteral
        short-circuit, rather than being folded into
        gen_slice_value_into's own dispatch -- unlike every OTHER kind
        of slice-producing expression there (a Slice expression, a
        Variable holding one), a NoneLiteral's own resolved type
        (Type.NONE) never equals the slice type it's being stored
        into, so the caller has to already know and pass the TARGET
        type; gen_slice_value_into's whole existing dispatch, by
        contrast, only ever needs the expression itself, since every
        other case's own type already matches what needs to be stored.

        Defensively re-checks target_type.kind here even though
        semantic.py's own _types_compatible already guarantees `none`
        was only ever allowed through for a slice target -- the same
        "codegen doesn't blindly trust its input" posture
        gen_array_copy's own array-of-slices handling already takes.
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
        """Calls malloc for array_type's own total footprint
        (type_byte_width), a compile-time-known constant, leaving the
        returned pointer in %rax (the ordinary SysV return-value
        register, not chosen specially here). Used wherever a heap-
        allocated array (see is_heap_allocated) needs its own, fresh
        backing allocation: a VarDecl declaring one (gen_var_decl) or a
        parameter receiving one (gen_function's own parameter loop,
        which needs its own independent copy of the caller's data to
        preserve value semantics across the call -- exactly like a
        stack-allocated parameter already gets via gen_array_copy, just
        backed by malloc'd memory instead of an inline slot)."""
        size = type_byte_width(array_type, self.struct_registry)
        return [Mov(src=Imm(size), dst=Register('edi')), CallInstr('malloc')]
