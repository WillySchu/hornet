"""TODO"""

from codegen.assembly_ast import (Operand, Instruction, MovQ, Imm, Mov, Memory, Je, Jne, Register, Push, Pop, AddQ, SubQ,
    IMulQ, Cqto, IDivQ, AndQ, OrQ, XorQ, ShiftLeftQ, ShiftRightArithmeticQ, Add, Sub, IMul, Cdq, IDiv, And, Or, Xor,
    ShiftLeft, ShiftRightArithmetic, CmpQ, Cmp, SetCC, MovZX, NegQ, Neg, NotQ, Not, MovSX, MovSXD, MovB, CallInstr,
    LeaQ, Jmp, Label,
)
from codegen.errors import CodegenError
from codegen.utils import as_qword_register, COMPARISON_CONDITION_CODES, as_byte_register
from parser import Call, Binary, BinaryOp, UnaryOp
from semantic import Type


class ScalarsMixin:
    def gen_call_into(self, expr: Call, dst: Operand) -> list[Instruction]:
        """`name(arg1, arg2, ...)`: evaluates and passes every
        argument via the shared _gen_call_arguments_into (see its own
        docstring for the full push-then-pop-in-reverse discipline,
        and how a slice-typed argument's own three register slots are
        placed correctly among any ordinary scalar/array ones), then
        calls the function.

        The result already ends up exactly where gen_expr_into's
        contract expects it (%rax/%eax, matching `dst`, which is always
        Register('eax') throughout this file), so there's nothing left
        to move once the call returns. This method is never reached at
        all for a callee that returns an array or a slice -- see
        gen_array_call_into and gen_slice_call_into, which now share
        the exact same hidden-pointer return-value convention (see the
        module docstring's SLICE PARAMETERS AND RETURNS section), and
        that convention doesn't fit a single generic `dst` the way an
        ordinary scalar return does.
        """
        total_slots = self._total_arg_slots(expr.args)
        if total_slots > 6:
            raise CodegenError(
                f"Call to '{expr.name}' needs {total_slots} argument "
                f"register(s) (a slice-typed argument needs 3); this "
                f"compiler only supports up to 6 (passed via registers "
                f"per the SysV ABI -- stack-passed arguments aren't "
                f"implemented)"
            )
        if dst != Register('eax'):
            raise CodegenError(f"Call codegen requires dst == %eax, got: {dst!r}")

        instructions = self._gen_call_arguments_into(expr.args)
        instructions.append(CallInstr(expr.name))
        return instructions

    def gen_short_circuit(
            self, expr: Binary,
            dst: Operand, *,
            short_circuit_jump: type,
            short_circuit_value: int,
            fallthrough_value: int,
            label_prefix: str) -> list[Instruction]:
        """Shared codegen for AND and OR -- they're mirror images of each
        other: each evaluates its left side, tests it against 0, and
        jumps straight past the right side entirely (never emitting the
        instructions that would compute it as *executed* code) if that
        test already decides the answer. Only if it doesn't -- left was
        truthy for AND, falsy for OR -- does the right side actually get
        evaluated, and *that* result decides the answer instead.

          AND: jump early (to `short_circuit_value=0`) when left == 0.
          OR:  jump early (to `short_circuit_value=1`) when left != 0.

        This is what makes `0 and (1 / 0)` return 0 instead of crashing:
        the division is real code sitting in the binary, but control
        flow jumps clean over it.
        """
        if not isinstance(dst, Register):
            raise CodegenError(f"Binary codegen requires a register destination, got: {dst!r}")

        short_label = self.new_label(f"{label_prefix}_short")
        end_label = self.new_label(f"{label_prefix}_end")

        instructions = self.gen_expr_into(expr.left, dst)
        instructions.append(Cmp(src=Imm(0), dst=dst))
        instructions.append(short_circuit_jump(short_label))

        instructions.extend(self.gen_expr_into(expr.right, dst))
        instructions.append(Cmp(src=Imm(0), dst=dst))
        instructions.append(short_circuit_jump(short_label))

        instructions.append(Mov(src=Imm(fallthrough_value), dst=dst))
        instructions.append(Jmp(end_label))
        instructions.append(Label(short_label))
        instructions.append(Mov(src=Imm(short_circuit_value), dst=dst))
        instructions.append(Label(end_label))
        return instructions

    def gen_binary_op(
            self, op: BinaryOp, src: Operand, dst: Operand, operand_type: Type = Type.INT) -> list[Instruction]:
        """Emits the actual operator instruction(s) for `op`, given
        that `src`/`dst` already hold the right-hand/left-hand values
        (see gen_binary_into's own orchestration for how they get
        there). `src`/`dst` are always passed as their ordinary
        32-bit-named register (e.g. 'eax'/'ecx'), the SAME convention
        every other caller in this file follows -- this method itself
        decides internally, via `operand_type`, whether to actually
        operate on that register's own 64-bit VIEW (as_qword_register)
        for int64, exactly the same "caller always passes the 32-bit
        name, the callee decides which view to use" pattern _gen_read_
        scalar_into/_gen_write_scalar_from already established for
        int8/uint8/int64's own storage access -- rather than pushing
        that decision out onto gen_binary_into or every other call
        site.

        A COMPARISON's own RESULT, though, is always an ordinary
        32-bit bool regardless of operand_type: SetCC/MovZX always
        target dst's own 32-bit view even when the comparison itself
        (Cmp vs CmpQ) operated on its 64-bit one, since a bool value
        is never itself wider than 4 bytes no matter how wide the two
        values being compared were -- this is why the comparison
        branch converts to a 64-bit view locally, just for the Cmp/
        CmpQ instruction itself, rather than reusing a single dst64
        variable the way every arithmetic branch above it does."""
        is_64bit = operand_type == Type.INT64
        if is_64bit and op not in COMPARISON_CONDITION_CODES:
            src64 = as_qword_register(src)
            dst64 = as_qword_register(dst)
            if op == BinaryOp.ADD:
                return [AddQ(src=src64, dst=dst64)]
            if op == BinaryOp.SUBTRACT:
                return [SubQ(src=src64, dst=dst64)]
            if op == BinaryOp.MULTIPLY:
                return [IMulQ(src=src64, dst=dst64)]
            if op == BinaryOp.DIVIDE:
                # idivq divides %rdx:%rax by its operand, so by the time
                # this runs, the dividend (dst64, i.e. left) must be in
                # %rax and the divisor (src64, i.e. right) must be in a
                # register -- both guaranteed by how gen_binary_into
                # calls this, exactly like the 32-bit case below.
                if dst64 != Register('rax'):
                    raise CodegenError("Division currently requires its destination to be %rax")
                return [Cqto(), IDivQ(src64)]
            if op == BinaryOp.MODULO:
                # Exactly the same Cqto+IDivQ sequence as DIVIDE --
                # idivq always computes both the quotient (%rax) and the
                # remainder (%rdx) in one instruction -- just followed
                # by moving the remainder into dst64 instead of leaving
                # the quotient there.
                if dst64 != Register('rax'):
                    raise CodegenError("Modulo currently requires its destination to be %rax")
                return [Cqto(), IDivQ(src64), MovQ(src=Register('rdx'), dst=Register('rax'))]
            if op == BinaryOp.BITWISE_AND:
                return [AndQ(src=src64, dst=dst64)]
            if op == BinaryOp.BITWISE_OR:
                return [OrQ(src=src64, dst=dst64)]
            if op == BinaryOp.BITWISE_XOR:
                return [XorQ(src=src64, dst=dst64)]
            if op == BinaryOp.SHIFT_LEFT:
                # `src64` is never referenced here -- ShiftLeftQ
                # hardcodes %cl as its count operand, the identical
                # reason ShiftLeft's own docstring already explains one
                # register-width down; the count itself is never wider
                # than a byte regardless of the value being shifted.
                return [ShiftLeftQ(dst=dst64)]
            if op == BinaryOp.SHIFT_RIGHT:
                return [ShiftRightArithmeticQ(dst=dst64)]
            raise CodegenError(f"No codegen rule for binary operator: {op}")

        if op == BinaryOp.ADD:
            return [Add(src=src, dst=dst)]
        if op == BinaryOp.SUBTRACT:
            return [Sub(src=src, dst=dst)]
        if op == BinaryOp.MULTIPLY:
            return [IMul(src=src, dst=dst)]
        if op == BinaryOp.DIVIDE:
            # idivl divides %edx:%eax by its operand, so by the time this
            # runs, the dividend (`dst`, i.e. left) must be in %eax and
            # the divisor (`src`, i.e. right) must be in a register --
            # both guaranteed by how gen_binary_into calls this.
            if dst != Register('eax'):
                raise CodegenError("Division currently requires its destination to be %eax")
            return [Cdq(), IDiv(src)]
        if op == BinaryOp.MODULO:
            # Exactly the same Cdq+IDiv sequence as DIVIDE -- idivl
            # always computes both the quotient (%eax) and the remainder
            # (%edx) in one instruction -- just followed by moving the
            # remainder into dst instead of leaving the quotient there.
            if dst != Register('eax'):
                raise CodegenError("Modulo currently requires its destination to be %eax")
            return [Cdq(), IDiv(src), Mov(src=Register('edx'), dst=Register('eax'))]
        if op == BinaryOp.BITWISE_AND:
            return [And(src=src, dst=dst)]
        if op == BinaryOp.BITWISE_OR:
            return [Or(src=src, dst=dst)]
        if op == BinaryOp.BITWISE_XOR:
            return [Xor(src=src, dst=dst)]
        if op == BinaryOp.SHIFT_LEFT:
            # `src` (== %ecx, per gen_binary_into) is never referenced
            # here -- ShiftLeft hardcodes %cl as its count operand,
            # since that's the only register x86 allows there, and %ecx
            # is already where the right-hand operand ends up by the
            # time gen_binary_op is called for any binary operator.
            return [ShiftLeft(dst=dst)]
        if op == BinaryOp.SHIFT_RIGHT:
            return [ShiftRightArithmetic(dst=dst)]
        if op in COMPARISON_CONDITION_CODES:
            # Cmp(src=right, dst=left) computes (left - right) and sets
            # flags from that; SetCC turns the relevant flag combination
            # into a 0/1 byte; MovZX zero-extends that byte back out to
            # fill the full destination register (same pattern used for
            # NOT -- see gen_unary_op -- just against a computed `right`
            # instead of the literal 0). For a 64-bit operand_type, the
            # comparison ITSELF (CmpQ, against the 64-bit views) needs
            # the full value to compare correctly -- comparing only the
            # low 32 bits could, e.g., call two large int64 values equal
            # when they aren't -- but the RESULT byte/register (byte_dst,
            # dst) stays exactly as it already was: a bool is always
            # 32-bit-or-narrower regardless of what was being compared.
            byte_dst = as_byte_register(dst)
            cmp_instr = CmpQ(
                src=as_qword_register(src), dst=as_qword_register(dst)) if is_64bit else Cmp(src=src, dst=dst)
            return [
                cmp_instr,
                SetCC(cc=COMPARISON_CONDITION_CODES[op], operand=byte_dst),
                MovZX(src=byte_dst, dst=dst),
            ]
        raise CodegenError(f"No codegen rule for binary operator: {op}")

    def gen_unary_op(self, op: UnaryOp, dst: Operand, operand_type: Type = Type.INT) -> list[Instruction]:
        """`operand_type` follows the identical convention gen_binary_
        op's own new parameter does -- `dst` is always passed as its
        ordinary 32-bit-named register, and this method decides
        internally whether to operate on its 64-bit view for int64.
        UnaryOp.NOT never reaches the int64 branch at all: `not`
        requires a bool operand (see check_unary), which int64 can
        never be, so its own Cmp-against-0/SetCC/MovZX sequence stays
        exactly as it always has, unconditionally 32-bit."""
        if op == UnaryOp.NEGATE:
            if operand_type == Type.INT64:
                return [NegQ(as_qword_register(dst))]
            return [Neg(dst)]
        if op == UnaryOp.COMPLEMENT:
            if operand_type == Type.INT64:
                return [NotQ(as_qword_register(dst))]
            return [Not(dst)]
        if op == UnaryOp.NOT:
            # `not x` is "1 if x == 0, else 0" -- the same cmp/setCC/movzx
            # pattern used for comparisons, just always against 0 and
            # always with cc='e'.
            byte_dst = as_byte_register(dst)
            return [
                Cmp(src=Imm(0), dst=dst),
                SetCC(cc='e', operand=byte_dst),
                MovZX(src=byte_dst, dst=dst),
            ]
        raise CodegenError(f"No codegen rule for unary operator: {op}")

    def gen_cast_narrowing_into(self, target_type: Type, dst: Register) -> list[Instruction]:
        """The actual work behind an explicit `TYPE(expr)` cast (see
        gen_expr_into's own Cast case): re-narrows `dst`'s own value
        to genuinely, correctly represent target_type, given that
        gen_expr_into has already computed the SOURCE expression into
        it (already correctly widened, if the source happened to be
        int8/uint8-typed itself -- see _gen_read_scalar_into).

        A target of int needs NOTHING further: the source's own
        already-widened 32-bit value already IS a valid int, whatever
        the source type actually was (an int8/uint8 source is already
        sign/zero-extended; an int source needs no widening at all).

        A target of int8 or uint8 needs exactly ONE more instruction:
        MovSX (int8) or MovZX (uint8) applied to dst's OWN low-byte
        alias, writing the result back into dst itself -- a single,
        purely register-to-register re-widening (MovSX/MovZX's own
        src operand doesn't have to be memory; see MovZX's own
        docstring), no memory round-trip needed at all. This is
        DELIBERATE, not just an optimization: a cast's own RESULT has
        to be a genuinely, correctly narrowed value immediately, not
        merely "correct once eventually written to int8/uint8-typed
        storage" the way _gen_write_scalar_from's own truncation is --
        `int8(300) + int8(5)` needs 300 already wrapped to 44 BEFORE
        the addition happens, or the arithmetic itself would silently
        be wrong, since every later int8/uint8 operation assumes its
        own operands already correctly represent a narrow value, never
        re-validating that itself.

        Correct regardless of what the source type actually was, not
        just for a narrowing int-to-int8/uint8 cast: re-extending
        whatever's already sitting in the low byte is exactly as
        correct for a same-width REINTERPRETATION (int8-to-uint8 or
        back) as it is for genuine narrowing, since both are really
        the same operation -- "take the low byte, reinterpret it under
        a new sign convention" -- differing only in whether the high
        bytes being discarded happened to already be a trivial (int8/
        uint8 source) or a real (int source) sign/zero-extension of
        that byte. Verified against concrete cases during design, not
        just asserted: int(300) as int8 gives 44 (300's own low byte,
        0x2C, has its high bit clear, so sign-extension leaves it
        positive); int(200) as int8 gives -56 (200's own low byte,
        0xC8, has its high bit set, so sign-extension correctly
        produces the negative two's-complement reinterpretation) --
        both match a real 8-bit truncate-then-reinterpret exactly.

        A target of int64 needs exactly one instruction too, in the
        OPPOSITE direction: MovSXD, sign-extending dst's own 32-bit
        view up into its 64-bit one -- correct regardless of whether
        the source was int, int8, or uint8, since all three are
        already read into a genuinely correct, ordinary 32-bit value
        by the time this runs (see _gen_read_scalar_into), and a non-
        negative 32-bit value's own sign bit is already clear, so
        sign-extending it produces the identical result zero-extending
        it would have. NARROWING out of int64 (int64(x) targeting int,
        int8, or uint8) needs NO new instruction here at all: dst's own
        32-bit view is always simply the low half of whatever's in its
        64-bit view, so falling through to the existing int/int8/uint8
        branches above -- which already operate on dst's own 32-bit
        view or its own low-byte alias -- is already exactly correct,
        the same way it already was before int64 existed."""
        if target_type == Type.INT8:
            return [MovSX(src=as_byte_register(dst), dst=dst)]
        if target_type == Type.UINT8:
            return [MovZX(src=as_byte_register(dst), dst=dst)]
        if target_type == Type.INT64:
            return [MovSXD(src=dst, dst=as_qword_register(dst))]
        return []

    def _gen_read_scalar_into(self, mem: Memory, t: Type, dst: Register) -> list[Instruction]:
        """Reads a scalar value of type `t` (int, int8, uint8, int64,
        or bool) from `mem` into `dst` -- the one choke point every
        scalar READ site in this file goes through, so int8/uint8's
        own genuinely narrow (1-byte) storage AND int64's own genuinely
        wide (8-byte) storage (see type_byte_width) only ever needed
        teaching to ONE place, not rediscovering at every Variable/
        Field/Index read site individually.

        int8 needs a SIGN-extending read (MovSX) and uint8 a ZERO-
        extending one (MovZX, already built for SetE's own unrelated
        need) rather than an ordinary 4-byte Mov, which would read
        three bytes of adjacent memory that were never part of this
        value at all -- and for int8 specifically, would also silently
        misinterpret a negative value as a large positive one (int8(-1)
        == 0xFF read as a raw 4-byte int would become 0x000000FF ==
        255, not -1) even if the adjacent bytes happened to be zero.
        Every later arithmetic/comparison instruction in this file
        already assumes it's operating on a genuinely correct 32-bit
        value, so getting the WIDENING right here, once, is what lets
        everything downstream stay completely unaware int8/uint8 are
        narrower than int at all.

        int64 needs a full 8-byte read (MovQ) into `dst`'s own 64-bit
        VIEW (as_qword_register(dst), e.g. %eax -> %rax) -- `dst`
        itself is always passed as a 32-bit-named register by every
        caller in this file (the same convention str's own handling
        already established elsewhere in gen_expr_into), with THIS
        method responsible for deciding which actual view to read
        into, exactly the way int8/uint8's own narrowing decision is
        made here rather than pushed onto every caller. An ordinary
        4-byte Mov here would silently drop int64's own high 32 bits
        entirely, not just read a stale/incorrect value -- reading is
        the one direction narrower-than-needed storage access can
        outright discard real, distinct bits of a value rather than
        merely mis-INTERPRET already-present ones the way int8/uint8's
        own narrow case can.

        int and bool are untouched -- an ordinary 4-byte Mov, exactly
        as before this method existed."""
        if t == Type.INT8:
            return [MovSX(src=mem, dst=dst)]
        if t == Type.UINT8:
            return [MovZX(src=mem, dst=dst)]
        if t == Type.INT64:
            return [MovQ(src=mem, dst=as_qword_register(dst))]
        return [Mov(src=mem, dst=dst)]

    def _gen_write_scalar_from(self, src: Register, t: Type, dst_mem: Memory) -> list[Instruction]:
        """Writes a scalar value of type `t`, already computed into
        `src`, into `dst_mem` -- the WRITE-side counterpart to _gen_
        read_scalar_into, and the other half of the same one-choke-
        point principle: every scalar WRITE site in this file goes
        through this, rather than each one separately remembering that
        int8/uint8 need a narrower store or int64 a wider one.

        int8/uint8 need a 1-byte, TRUNCATING store (MovB, of src's own
        low-byte alias -- see as_byte_register) rather than an ordinary
        4-byte Mov, which would clobber whatever adjacent memory
        happens to immediately follow this value (an adjacent struct
        field, the next array element, ...) -- exactly the kind of
        silent, hard-to-diagnose corruption a narrow type's own
        storage existing at all is supposed to make possible to write
        correctly, not introduce a new way to get wrong.

        int64 needs a full 8-byte store (MovQ, of src's own 64-bit VIEW
        -- as_qword_register(src)) -- CALLERS are responsible for
        having already computed the value into that same 64-bit view
        before reaching this method (every gen_expr_into case that can
        produce an int64 result does exactly this -- see its own
        Constant/Variable/Binary/Unary/Cast cases), not just src's own
        low 32 bits: an ordinary 4-byte Mov here would write only the
        low half of a value whose own high half might be meaningful,
        and reading src's own 64-bit view when only the low 32 bits
        were ever actually computed would write whatever stale garbage
        happened to occupy that register's own high bits, silently
        corrupting the stored value in a way that could be very hard
        to trace back to its actual cause.

        int and bool are untouched -- an ordinary 4-byte Mov, exactly
        as before this method existed."""
        if t == Type.INT8 or t == Type.UINT8:
            return [MovB(src=as_byte_register(src), dst=dst_mem)]
        if t == Type.INT64:
            return [MovQ(src=as_qword_register(src), dst=dst_mem)]
        return [Mov(src=src, dst=dst_mem)]

