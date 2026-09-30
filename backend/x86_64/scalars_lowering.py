"""Scalar instruction selection: operators, casts, and scalar loads/stores."""

from backend.x86_64.assembly_ast import (
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
from backend.errors import CodegenError
from backend.x86_64.utils import as_qword_register, COMPARISON_CONDITION_CODES, as_byte_register
from typesys import is_wide_type
from ops import BinaryOp, UnaryOp
from typesys import Type


class ScalarsLoweringMixin:
    def gen_binary_op(
            self, op: BinaryOp, src: Operand, dst: Operand, operand_type: Type = Type.INT) -> list[Instruction]:
        """Emit `op` with left in dst and right in src; 64-bit forms by operand_type."""
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
                # idivq: dividend in %rax, divisor elsewhere.
                if dst64 != Register('rax'):
                    raise CodegenError("Division currently requires its destination to be %rax")
                return [Cqto(), IDivQ(src64)]
            if op == BinaryOp.MODULO:
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
            # idivl: dividend in %eax, divisor elsewhere.
            if dst != Register('eax'):
                raise CodegenError("Division currently requires its destination to be %eax")
            return [Cdq(), IDiv(src)]
        if op == BinaryOp.MODULO:
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
            return [ShiftLeft(dst=dst)]
        if op == BinaryOp.SHIFT_RIGHT:
            return [ShiftRightArithmetic(dst=dst)]
        if op in COMPARISON_CONDITION_CODES:
            # cmp; setcc; movzx.
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
        """Emit unary `op` in place on dst."""
        if op == UnaryOp.NEGATE:
            if operand_type == Type.INT:
                return [NegQ(as_qword_register(dst))]
            return [Neg(dst)]
        if op == UnaryOp.COMPLEMENT:
            if operand_type == Type.INT:
                return [NotQ(as_qword_register(dst))]
            return [Not(dst)]
        if op == UnaryOp.NOT:
            # `not x`: x == 0.
            byte_dst = as_byte_register(dst)
            return [
                Cmp(src=Imm(0), dst=dst),
                SetCC(cc='e', operand=byte_dst),
                MovZX(src=byte_dst, dst=dst),
            ]
        raise CodegenError(f"No codegen rule for unary operator: {op}")

    def gen_cast_narrowing_into(self, target_type: Type, dst: Register, source_type: Type) -> list[Instruction]:
        """Re-narrow dst to target_type."""
        if target_type == Type.INT8:
            return [MovSX(src=as_byte_register(dst), dst=dst)]
        if target_type == Type.UINT8:
            return [MovZX(src=as_byte_register(dst), dst=dst)]
        if target_type == Type.INT:
            if source_type == Type.INT:
                return []
            return [MovSXD(src=dst, dst=as_qword_register(dst))]
        if target_type == Type.INT32:
            return [Mov(src=dst, dst=dst)]  # truncate: clear the upper half
        return []

    def _gen_read_scalar_into(self, mem: Memory, t: Type, dst: Register) -> list[Instruction]:
        """Load scalar `t` from `mem` into `dst`."""
        if t == Type.INT8:
            return [MovSX(src=mem, dst=dst)]
        if t == Type.UINT8:
            return [MovZX(src=mem, dst=dst)]
        if is_wide_type(t):
            return [MovQ(src=mem, dst=as_qword_register(dst))]
        return [Mov(src=mem, dst=dst)]

    def _gen_write_scalar_from(self, src: Register, t: Type, dst_mem: Memory) -> list[Instruction]:
        """Store scalar `t` from `src` into `dst_mem`."""
        if t == Type.INT8 or t == Type.UINT8:
            return [MovB(src=as_byte_register(src), dst=dst_mem)]
        if is_wide_type(t):
            return [MovQ(src=as_qword_register(src), dst=dst_mem)]
        return [Mov(src=src, dst=dst_mem)]
