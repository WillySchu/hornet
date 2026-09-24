"""The instruction-selection half of scalar codegen -- gen_binary_op/
gen_unary_op/gen_cast_narrowing_into/_gen_read_scalar_into/_gen_write_
scalar_from are ir_lowering.py's own five direct entry points into
this file, each taking already-decided registers or memory locations
and building plain assembly_ast Instructions directly -- none of the
five ever calls back into real IR or anything that builds it. _gen_
read_scalar_into/_gen_write_scalar_from are the one choke point every
scalar memory access in this compiler goes through, so int8/uint8
being genuinely 1 byte and int64 genuinely 8 only ever needed teaching
to these two methods: every caller passes a value's ordinary 32-bit-
named register, and these (along with the other three) decide
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
from codegen.utils import as_qword_register, COMPARISON_CONDITION_CODES, as_byte_register
from ir.utils import is_wide_type
from parser import BinaryOp, UnaryOp
from semantic import Type


class ScalarsLoweringMixin:
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
        is_64bit = is_wide_type(operand_type)
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
                # guaranteed by how ir_lowering.py's own IRBinOp case
                # calls this.
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
            # ir_lowering.py's own IRBinOp case calls this.
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
            # `src` (== %ecx, per ir_lowering.py's own IRBinOp case) is
            # never referenced
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

    def gen_cast_narrowing_into(self, target_type: Type, dst: Register, source_type: Type) -> list[Instruction]:
        """Re-narrows dst (already loaded with the source value) to
        represent target_type. int8/uint8: MovSX/MovZX on dst's low
        byte -- correct for narrowing OR same-width reinterpretation
        alike (int(300)->int8 gives 44; int(200)->int8 gives -56).
        int64: MovSXD sign-extends dst's 32-bit view -- UNLESS
        source_type is already int64 (a same-type, no-op cast), in
        which case dst already holds the full correct value and needs
        no instruction at all. (Bug fix: previously this branch always
        re-derived from the 32-bit view regardless of source_type,
        silently truncating e.g. `int64(1099511628211)` to 435.)
        Narrowing out of int64 needs nothing new: dst's 32-bit view is
        already its low half."""
        if target_type == Type.INT8:
            return [MovSX(src=as_byte_register(dst), dst=dst)]
        if target_type == Type.UINT8:
            return [MovZX(src=as_byte_register(dst), dst=dst)]
        if target_type == Type.INT64:
            if source_type == Type.INT64:
                return []
            return [MovSXD(src=dst, dst=as_qword_register(dst))]
        return []

    def _gen_read_scalar_into(self, mem: Memory, t: Type, dst: Register) -> list[Instruction]:
        """Reads a scalar value of type `t` (int, int8, uint8, int64,
        bool, or pointer) from `mem` into `dst` -- the one choke point
        every scalar READ site in this file goes through, so int8/
        uint8's narrow (1-byte) storage and int64/pointer's wide
        (8-byte) storage only ever needed handling in ONE place.

        int8 needs a SIGN-extending read (MovSX) and uint8 a ZERO-
        extending one (MovZX) rather than an ordinary 4-byte Mov,
        which would read adjacent garbage bytes and, for int8, could
        misinterpret a negative value as large and positive (int8(-1)
        == 0xFF read as a raw 4-byte int would become 255, not -1).

        int64/pointer (is_wide_type, ir/utils.py) need a full 8-byte
        read (MovQ) into `dst`'s 64-bit VIEW (as_qword_register(dst))
        -- `dst` itself is always passed as a 32-bit-named register by
        every caller, with this method deciding which actual view to
        read into. An ordinary 4-byte Mov here would silently drop
        int64's high 32 bits, or -- worse, for a pointer -- corrupt
        the address into something unrelated rather than merely lose
        numeric precision.

        int and bool are untouched -- an ordinary 4-byte Mov."""
        if t == Type.INT8:
            return [MovSX(src=mem, dst=dst)]
        if t == Type.UINT8:
            return [MovZX(src=mem, dst=dst)]
        if is_wide_type(t):
            return [MovQ(src=mem, dst=as_qword_register(dst))]
        return [Mov(src=mem, dst=dst)]

    def _gen_write_scalar_from(self, src: Register, t: Type, dst_mem: Memory) -> list[Instruction]:
        """Writes a scalar value of type `t`, already computed into
        `src`, into `dst_mem` -- the WRITE-side counterpart to
        _gen_read_scalar_into: every scalar WRITE site in this file
        goes through this, rather than each one separately remembering
        that int8/uint8 need a narrower store or int64/pointer a wider
        one.

        int8/uint8 need a 1-byte, TRUNCATING store (MovB, of src's low-
        byte alias) rather than an ordinary 4-byte Mov, which would
        clobber adjacent memory (an adjacent struct field, the next
        array element, ...).

        int64/pointer (is_wide_type, ir/utils.py) need a full 8-byte
        store (MovQ, of src's 64-bit VIEW) -- CALLERS are responsible
        for having already computed the value into that same 64-bit
        view before reaching this method (every real-IR case that can
        produce an int64 or pointer result already does this, via the
        appropriate 64-bit register view), not just src's low 32 bits:
        an ordinary 4-byte Mov here would write only the low half, and
        reading src's 64-bit view when only the low 32 bits were
        computed would write whatever stale garbage occupied the
        register's high bits.

        int and bool are untouched -- an ordinary 4-byte Mov."""
        if t == Type.INT8 or t == Type.UINT8:
            return [MovB(src=as_byte_register(src), dst=dst_mem)]
        if is_wide_type(t):
            return [MovQ(src=as_qword_register(src), dst=dst_mem)]
        return [Mov(src=src, dst=dst_mem)]
