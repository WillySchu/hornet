"""Scalar value production and storage -- int, int8, uint8, int64, and
bool. _gen_read_scalar_into/_gen_write_scalar_from are the one choke
point for every scalar memory access in this compiler, so int8/uint8
being genuinely 1 byte and int64 genuinely 8 only ever needed teaching
to these two methods, not rediscovered at each read/write site: every
caller passes a value's ordinary 32-bit-named register, and these
(along with gen_binary_op/gen_unary_op/gen_cast_narrowing_into) decide
internally which actual width to operate on."""

from codegen.assembly_ast import (
    Add,
    AddQ,
    And,
    AndQ,
    Cdq,
    Cmp,
    CmpQ,
    Cqto,
    IDiv,
    IDivQ,
    Imm,
    IMul,
    IMulQ,
    Instruction,
    Memory,
    Mov,
    MovB,
    MovQ,
    MovSX,
    MovSXD,
    MovZX,
    Neg,
    NegQ,
    Not,
    NotQ,
    Operand,
    Or,
    OrQ,
    Register,
    SetCC,
    ShiftLeft,
    ShiftLeftQ,
    ShiftRightArithmetic,
    ShiftRightArithmeticQ,
    Sub,
    SubQ,
    Xor,
    XorQ,
)
from codegen.errors import CodegenError
from codegen.ir import IRRaw, IRBranch, IRJump, IRLabel, IRMove, IRConst, IRCall
from codegen.utils import as_qword_register, COMPARISON_CONDITION_CODES, as_byte_register, type_of
from parser import Call, Binary, BinaryOp, UnaryOp
from semantic import Type


class ScalarsMixin:
    def gen_call_into(self, expr: Call, dst: Operand) -> list[Instruction]:
        """`name(arg1, arg2, ...)`: evaluates and passes every argument
        via the shared _gen_call_arguments_into, then calls the
        function.

        Argument marshaling isn't expressed in IR yet -- _gen_call_
        arguments_into's existing output is spliced in verbatim via
        IRRaw, since folding it in would mean migrating array/slice/
        struct arguments too. IRCall's dst is None for a void call, or
        a fresh Temp for a scalar-returning one, read into `dst`
        afterward.

        Never reached for a callee that returns an array or slice --
        see gen_array_call_into and gen_slice_call_into, which share a
        hidden-pointer return convention that doesn't fit a single
        generic `dst`."""
        if dst != Register('eax'):
            raise CodegenError(f"Call codegen requires dst == %eax, got: {dst!r}")

        ir, t_result = self._ir_call(expr)
        instructions = self.lower_ir(ir)
        if t_result is not None:
            instructions.extend(self._gen_read_temp_into(t_result, dst))
        return instructions

    def _ir_call(self, expr: Call) -> tuple[list, object]:
        """Builds (without lowering) an ordinary function call's IR:
        args are marshaled via the existing calling-convention code,
        not yet expressed in IR itself (see gen_call_into), then a
        genuine IRCall. Returns (ir, t_result), t_result being None
        for a void call."""
        total_slots = self._total_arg_slots(expr.args)
        if total_slots > 6:
            raise CodegenError(
                f"Call to '{expr.name}' needs {total_slots} argument "
                f"register(s) (a slice-typed argument needs 3); this "
                f"compiler only supports up to 6 (passed via registers "
                f"per the SysV ABI -- stack-passed arguments aren't "
                f"implemented)"
            )
        result_type = type_of(expr)
        t_result = None if result_type == Type.VOID else self._new_temp(result_type)
        ir = [
            IRRaw(self._gen_call_arguments_into(expr.args)),
            IRCall(dst=t_result, name=expr.name, args=[]),
        ]
        return ir, t_result
        return instructions

    def gen_short_circuit(
            self, expr: Binary,
            dst: Operand, *,
            short_circuit_value: int,
            label_prefix: str) -> list[Instruction]:
        """Shared codegen for AND and OR -- mirror images of each
        other: each evaluates its left side, tests it against 0, and
        jumps past the right side entirely (never emitting the
        instructions that would compute it) if that test already
        decides the answer. Only if it doesn't -- left was truthy for
        AND, falsy for OR -- does the right side get evaluated, and
        that result decides the answer instead.

          AND: short_circuit_value=0 -- left (or then right) false
               makes the whole thing false without evaluating further.
          OR:  short_circuit_value=1 -- left (or then right) true
               makes the whole thing true without evaluating further.

        This is what makes `0 and (1 / 0)` return 0 instead of
        crashing: the division is real code sitting in the binary, but
        control flow jumps clean over it.

        Built as a small IR fragment: both left and right feed an
        IRBranch on their own truthiness, sharing one `short` label
        whichever one triggers it (`continue` picks between "go
        evaluate right" for left, or "use the canonical fallthrough
        value" for right -- the only difference between the two)."""
        if not isinstance(dst, Register):
            raise CodegenError(f"Binary codegen requires a register destination, got: {dst!r}")

        ir, t_result = self._ir_short_circuit(expr, short_circuit_value=short_circuit_value, label_prefix=label_prefix)
        instructions = self.lower_ir(ir)
        instructions.extend(self._gen_read_scalar_into(self._temp_mem(t_result), Type.BOOL, dst))
        return instructions

    def _ir_short_circuit(self, expr: Binary, *, short_circuit_value: int, label_prefix: str) -> tuple[list, object]:
        """Builds (without lowering) gen_short_circuit's own IR --
        see its docstring for the shared-label branch structure.
        Returns (ir, t_result)."""
        fallthrough_value = 1 - short_circuit_value
        rhs_label = self.new_label(f"{label_prefix}_rhs")
        short_label = self.new_label(f"{label_prefix}_short")
        fallthrough_label = self.new_label(f"{label_prefix}_fallthrough")
        end_label = self.new_label(f"{label_prefix}_end")

        def targets(continue_label: str) -> tuple[str, str]:
            # (true_target, false_target): whichever outcome matches
            # short_circuit_value goes to `short_label`; the other
            # goes to `continue_label`.
            if short_circuit_value == 1:
                return short_label, continue_label
            return continue_label, short_label

        t_result = self._new_temp(Type.BOOL)
        left_true, left_false = targets(rhs_label)
        right_true, right_false = targets(fallthrough_label)

        left_ir, left_value = self.gen_expr_ir(expr.left)
        right_ir, right_value = self.gen_expr_ir(expr.right)
        ir = [
            *left_ir,
            IRBranch(cond=left_value, true_label=left_true, false_label=left_false),
            IRLabel(rhs_label),
            *right_ir,
            IRBranch(cond=right_value, true_label=right_true, false_label=right_false),
            IRLabel(fallthrough_label),
            IRMove(dst=t_result, src=IRConst(fallthrough_value, Type.BOOL)),
            IRJump(end_label),
            IRLabel(short_label),
            IRMove(dst=t_result, src=IRConst(short_circuit_value, Type.BOOL)),
            IRLabel(end_label),
        ]
        return ir, t_result

    def gen_binary_op(
            self, op: BinaryOp, src: Operand, dst: Operand, operand_type: Type = Type.INT) -> list[Instruction]:
        """Emits the actual operator instruction(s) for `op`, given
        that `src`/`dst` already hold the right-hand/left-hand values.
        `src`/`dst` are always passed as their ordinary 32-bit-named
        register -- this method decides internally, via
        `operand_type`, whether to actually operate on that register's
        64-bit view (as_qword_register) for int64, the same
        caller-passes-32-bit/callee-decides-the-view pattern
        _gen_read_scalar_into/_gen_write_scalar_from use for
        int8/uint8/int64 storage access.

        A COMPARISON's result, though, is always an ordinary 32-bit
        bool regardless of operand_type: SetCC/MovZX always target
        dst's 32-bit view even when the comparison itself (Cmp vs
        CmpQ) operated on the 64-bit one, since a bool is never wider
        than 4 bytes no matter how wide the compared values were."""
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
                # idivq divides %rdx:%rax by its operand, so the
                # dividend (dst64, left) must be in %rax and the
                # divisor (src64, right) in a register -- both
                # guaranteed by how gen_binary_into calls this.
                if dst64 != Register('rax'):
                    raise CodegenError("Division currently requires its destination to be %rax")
                return [Cqto(), IDivQ(src64)]
            if op == BinaryOp.MODULO:
                # Same Cqto+IDivQ sequence as DIVIDE -- idivq computes
                # both quotient (%rax) and remainder (%rdx) in one
                # instruction -- followed by moving the remainder into
                # dst64 instead of leaving the quotient there.
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
                # `src64` is never referenced -- ShiftLeftQ hardcodes
                # %cl as its count operand, same as ShiftLeft one
                # register-width down; the count is never wider than a
                # byte regardless of the value being shifted.
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
            # idivl divides %edx:%eax by its operand, so the dividend
            # (`dst`, left) must be in %eax and the divisor (`src`,
            # right) in a register -- both guaranteed by how
            # gen_binary_into calls this.
            if dst != Register('eax'):
                raise CodegenError("Division currently requires its destination to be %eax")
            return [Cdq(), IDiv(src)]
        if op == BinaryOp.MODULO:
            # Same Cdq+IDiv sequence as DIVIDE -- idivl computes both
            # quotient (%eax) and remainder (%edx) in one instruction
            # -- followed by moving the remainder into dst instead of
            # leaving the quotient there.
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
            # the only register x86 allows there, and %ecx is already
            # where the right-hand operand ends up.
            return [ShiftLeft(dst=dst)]
        if op == BinaryOp.SHIFT_RIGHT:
            return [ShiftRightArithmetic(dst=dst)]
        if op in COMPARISON_CONDITION_CODES:
            # Cmp(src=right, dst=left) computes (left - right) and sets
            # flags from that; SetCC turns the relevant flag
            # combination into a 0/1 byte; MovZX zero-extends that byte
            # back out to fill dst (same pattern as NOT, against a
            # computed `right` instead of literal 0). For a 64-bit
            # operand_type, the comparison itself (CmpQ, against the
            # 64-bit views) needs the full value -- comparing only the
            # low 32 bits could call two large int64 values equal when
            # they aren't -- but the result byte/register stays exactly
            # as it already was: a bool is always 32-bit-or-narrower
            # regardless of what was being compared.
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
        """`operand_type` follows the same convention gen_binary_op's
        parameter does -- `dst` is always passed as its ordinary
        32-bit-named register, and this method decides internally
        whether to operate on its 64-bit view for int64. UnaryOp.NOT
        never reaches the int64 branch: `not` requires a bool operand,
        which int64 can never be, so its Cmp-against-0/SetCC/MovZX
        sequence stays unconditionally 32-bit."""
        if op == UnaryOp.NEGATE:
            if operand_type == Type.INT64:
                return [NegQ(as_qword_register(dst))]
            return [Neg(dst)]
        if op == UnaryOp.COMPLEMENT:
            if operand_type == Type.INT64:
                return [NotQ(as_qword_register(dst))]
            return [Not(dst)]
        if op == UnaryOp.NOT:
            # `not x` is "1 if x == 0, else 0" -- the same cmp/setCC/
            # movzx pattern used for comparisons, always against 0 and
            # always with cc='e'.
            byte_dst = as_byte_register(dst)
            return [
                Cmp(src=Imm(0), dst=dst),
                SetCC(cc='e', operand=byte_dst),
                MovZX(src=byte_dst, dst=dst),
            ]
        raise CodegenError(f"No codegen rule for unary operator: {op}")

    def gen_cast_narrowing_into(self, target_type: Type, dst: Register) -> list[Instruction]:
        """The actual work behind an explicit `TYPE(expr)` cast:
        re-narrows `dst`'s value to correctly represent target_type,
        given that gen_expr_into has already computed the source
        expression into it (already correctly widened if the source
        was int8/uint8-typed -- see _gen_read_scalar_into).

        A target of int needs NOTHING further: the source's already-
        widened 32-bit value already IS a valid int.

        A target of int8 or uint8 needs exactly one more instruction:
        MovSX (int8) or MovZX (uint8) applied to dst's own low-byte
        alias, written back into dst -- a single, register-to-register
        re-widening, no memory round-trip. This is DELIBERATE, not
        just an optimization: a cast's result has to be correctly
        narrowed immediately, not merely "correct once eventually
        written to int8/uint8-typed storage" the way
        _gen_write_scalar_from's truncation is -- `int8(300) + int8(5)`
        needs 300 already wrapped to 44 BEFORE the addition happens,
        since every later int8/uint8 operation assumes its operands
        already correctly represent a narrow value.

        This is correct regardless of the source type, not just for a
        narrowing cast: re-extending whatever's in the low byte is
        exactly as correct for a same-width REINTERPRETATION
        (int8-to-uint8 or back) as for genuine narrowing, since both
        are really "take the low byte, reinterpret it under a new sign
        convention." E.g.: int(300) as int8 gives 44 (300's low byte,
        0x2C, has its high bit clear, so sign-extension leaves it
        positive); int(200) as int8 gives -56 (200's low byte, 0xC8,
        has its high bit set, so sign-extension produces the negative
        two's-complement reinterpretation).

        A target of int64 needs one instruction too, in the opposite
        direction: MovSXD, sign-extending dst's 32-bit view up into its
        64-bit one -- correct regardless of whether the source was int,
        int8, or uint8, since all three are already a correct,
        non-negative 32-bit value by this point, so sign-extending
        produces the same result zero-extending would. NARROWING out of
        int64 (int64(x) targeting int, int8, or uint8) needs no new
        instruction here: dst's 32-bit view is always simply the low
        half of its 64-bit view, so falling through to the existing
        int/int8/uint8 branches above is already correct."""
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
        narrow (1-byte) storage and int64's wide (8-byte) storage only
        ever needed handling in ONE place.

        int8 needs a SIGN-extending read (MovSX) and uint8 a ZERO-
        extending one (MovZX) rather than an ordinary 4-byte Mov,
        which would read adjacent garbage bytes and, for int8, could
        misinterpret a negative value as large and positive (int8(-1)
        == 0xFF read as a raw 4-byte int would become 255, not -1).

        int64 needs a full 8-byte read (MovQ) into `dst`'s 64-bit VIEW
        (as_qword_register(dst)) -- `dst` itself is always passed as a
        32-bit-named register by every caller, with this method
        deciding which actual view to read into. An ordinary 4-byte
        Mov here would silently drop int64's high 32 bits entirely,
        not just read a stale value.

        int and bool are untouched -- an ordinary 4-byte Mov."""
        if t == Type.INT8:
            return [MovSX(src=mem, dst=dst)]
        if t == Type.UINT8:
            return [MovZX(src=mem, dst=dst)]
        if t == Type.INT64:
            return [MovQ(src=mem, dst=as_qword_register(dst))]
        return [Mov(src=mem, dst=dst)]

    def _gen_write_scalar_from(self, src: Register, t: Type, dst_mem: Memory) -> list[Instruction]:
        """Writes a scalar value of type `t`, already computed into
        `src`, into `dst_mem` -- the WRITE-side counterpart to
        _gen_read_scalar_into: every scalar WRITE site in this file
        goes through this, rather than each one separately remembering
        that int8/uint8 need a narrower store or int64 a wider one.

        int8/uint8 need a 1-byte, TRUNCATING store (MovB, of src's low-
        byte alias) rather than an ordinary 4-byte Mov, which would
        clobber adjacent memory (an adjacent struct field, the next
        array element, ...).

        int64 needs a full 8-byte store (MovQ, of src's 64-bit VIEW) --
        CALLERS are responsible for having already computed the value
        into that same 64-bit view before reaching this method (every
        gen_expr_into case that can produce an int64 result does this),
        not just src's low 32 bits: an ordinary 4-byte Mov here would
        write only the low half, and reading src's 64-bit view when
        only the low 32 bits were computed would write whatever stale
        garbage occupied the register's high bits.

        int and bool are untouched -- an ordinary 4-byte Mov."""
        if t == Type.INT8 or t == Type.UINT8:
            return [MovB(src=as_byte_register(src), dst=dst_mem)]
        if t == Type.INT64:
            return [MovQ(src=as_qword_register(src), dst=dst_mem)]
        return [Mov(src=src, dst=dst_mem)]

