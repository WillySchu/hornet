"""Instruction selection: IR -> assembly AST. Temps not assigned a register get a lazily allocated frame slot,
emitted as a FrameSlot placeholder and resolved after lowering.

Full-register types are selected directly on their registers, slots, and immediates; other
types, and instructions without a direct rule, go through %rax/%rcx. %rax, %rcx, %rdx, %r8
and %r9 are never allocated, so they are always free as scratch.
"""

from typing import Optional

from backend.x86_64.assembly_ast import (
    JCC,
    Add,
    AddQ,
    And,
    AndQ,
    Cdq,
    CmpQ,
    Cqto,
    IDiv,
    IDivQ,
    IMul,
    IMulQ,
    IMulWide,
    MovSX,
    MovSXD,
    ShiftImmQ,
    MovZX,
    Neg,
    NegQ,
    Not,
    NotQ,
    Or,
    OrQ,
    SetCC,
    ShiftLeft,
    ShiftLeftQ,
    ShiftRightArithmetic,
    ShiftRightArithmeticQ,
    Sub,
    SubQ,
    Xor,
    XorQ,
    CallInstr,
    Cmp,
    FrameSlot,
    Imm,
    Instruction,
    Ja,
    Jae,
    Je,
    Jmp,
    Label,
    LeaQ,
    LeaQFrameSlot,
    Memory,
    Mov,
    MovQ,
    Operand,
    Register,
)
from ir.ir import (
    IRBinOp,
    IRBoundsCheck,
    IRBranch,
    IRCall,
    IRCast,
    IRConst,
    IRCopy,
    IRJump,
    IRLabel,
    IRLoad,
    IRLocalAddress,
    IRMove,
    IRReturn,
    IRSliceBoundsCheck,
    IRStaticDataAddress,
    IRStore,
    IRUnOp,
    IRValue,
    Temp,
)
from typesys import is_wide_type
from ir.cfg import uses
from backend.common.frame import Frame
from backend.x86_64.peephole import INVERSE_CC
from backend.common.division import is_power_of_two, magic
from backend.x86_64.utils import as_byte_register, as_qword_register, ARG_REGISTERS_32, ARG_REGISTERS_64, COMPARISON_CONDITION_CODES
from typesys import Type
from ops import BinaryOp, UnaryOp


class InstructionSelector:
    """Lowers one function's IR."""

    def __init__(self, host, ir_fn):
        self.host = host
        self.ir_fn = ir_fn
        if host.frame is None:  # a selector used on its own (tests)
            host.frame = Frame(ir_fn, host.ir_program.struct_registry, host.ir_program.sum_type_registry)

    def _temp_mem(self, temp: Temp) -> Operand:
        """Frame slot for `temp`: its variable's slot, or a spill slot allocated on first use."""
        if temp.id in self.ir_fn.temp_homes:
            return FrameSlot(slot=self.ir_fn.temp_homes[temp.id])
        return FrameSlot(slot=self.host.frame.spill_slot(temp))

    def _gen_load_value(self, value: IRValue, dst: Register) -> list[Instruction]:
        """Load an IRValue into `dst` (32-bit name; widened by type)."""
        if isinstance(value, IRConst):
            if value.type == Type.INT64:
                return [MovQ(src=Imm(value.value), dst=as_qword_register(dst))]
            return [Mov(src=Imm(value.value), dst=dst)]
        return self._gen_read_temp_into(value, dst)

    def _gen_read_temp_into(self, temp: Temp, dst: Register) -> list[Instruction]:
        """Read a Temp from its register or slot into `dst`."""
        reg_name = self.host._register_assignment.get(temp.id)
        if reg_name is not None:
            src = Register(reg_name)
            wide = is_wide_type(temp.type)
            if wide:
                src, dst = as_qword_register(src), as_qword_register(dst)
            if src == dst:
                return []
            return [MovQ(src=src, dst=dst)] if wide else [Mov(src=src, dst=dst)]
        return self.host._gen_read_scalar_into(self._temp_mem(temp), temp.type, dst)

    def _gen_write_temp_from(self, src: Register, temp: Temp) -> list[Instruction]:
        """Write `src` to a Temp's register or slot."""
        reg_name = self.host._register_assignment.get(temp.id)
        if reg_name is not None:
            dst = Register(reg_name)
            if temp.type in (Type.INT8, Type.UINT8):
                # Narrow values live in registers sign/zero-extended from their low byte.
                extend = MovSX if temp.type == Type.INT8 else MovZX
                return [extend(src=as_byte_register(src), dst=dst)]
            wide = is_wide_type(temp.type)
            if wide:
                src, dst = as_qword_register(src), as_qword_register(dst)
            if src == dst:
                return []
            return [MovQ(src=src, dst=dst)] if wide else [Mov(src=src, dst=dst)]
        return self.host._gen_write_scalar_from(src, temp.type, self._temp_mem(temp))

    # -- direct selection ----------------------------------------------------
    # Operands are used where they live (register, frame slot, or immediate).
    # Only full-register types (int, pointers, int32, bool) take this path;
    # anything else falls back to the %rax/%rcx sequences below.

    @staticmethod
    def _width(t: Type):
        """8 or 4 for types selected directly, else None."""
        if is_wide_type(t):
            return 8
        if t in (Type.INT32, Type.BOOL):
            return 4
        return None

    def _same_width(self, *values) -> Optional[int]:
        widths = {self._width(v.type) for v in values}
        return widths.pop() if len(widths) == 1 and None not in widths else None

    @staticmethod
    def _fits_imm(value: int, width: int) -> bool:
        return -2 ** 31 <= value < (2 ** 31 if width == 8 else 2 ** 32)

    def _loc(self, v: IRValue, width: int) -> Operand:
        """Where `v` lives: Imm, register (sized to `width`), or frame slot."""
        if isinstance(v, IRConst):
            return Imm(v.value)
        reg = self.host._register_assignment.get(v.id)
        if reg is not None:
            return as_qword_register(Register(reg)) if width == 8 else Register(reg)
        return self._temp_mem(v)

    @staticmethod
    def _scratch(name: str, width: int) -> Register:
        return as_qword_register(Register(name)) if width == 8 else Register(name)

    @staticmethod
    def _mov(src: Operand, dst: Operand, width: int) -> list:
        if src == dst:
            return []
        return [MovQ(src=src, dst=dst) if width == 8 else Mov(src=src, dst=dst)]

    def _as_source(self, op: Operand, other: Operand, width: int, scratch: str, out: list) -> Operand:
        """`op` usable as a source alongside destination `other`, loading into `scratch` if needed."""
        mem_mem = not isinstance(op, (Register, Imm)) and not isinstance(other, Register)
        big_imm = isinstance(op, Imm) and not self._fits_imm(op.value, width)
        if mem_mem or big_imm:
            reg = self._scratch(scratch, width)
            out.extend(self._mov(op, reg, width))
            return reg
        return op

    _ALU = {
        BinaryOp.ADD: (Add, AddQ), BinaryOp.SUBTRACT: (Sub, SubQ), BinaryOp.MULTIPLY: (IMul, IMulQ),
        BinaryOp.BITWISE_AND: (And, AndQ), BinaryOp.BITWISE_OR: (Or, OrQ), BinaryOp.BITWISE_XOR: (Xor, XorQ),
    }
    _COMMUTATIVE = {BinaryOp.ADD, BinaryOp.MULTIPLY, BinaryOp.BITWISE_AND, BinaryOp.BITWISE_OR, BinaryOp.BITWISE_XOR}

    def _direct_binop(self, instr: IRBinOp) -> Optional[list]:
        if instr.op in COMPARISON_CONDITION_CODES:
            width = self._same_width(instr.left, instr.right)
            if width is None or self._width(instr.dst.type) is None:
                return None
            out = self._direct_cmp(instr.left, instr.right, width)
            dst_width = self._width(instr.dst.type)
            out.append(SetCC(cc=COMPARISON_CONDITION_CODES[instr.op], operand=Register('al')))
            out.append(MovZX(src=Register('al'), dst=Register('eax')))
            out.extend(self._mov(self._scratch('eax', dst_width), self._loc(instr.dst, dst_width), dst_width))
            return out
        width = self._same_width(instr.dst, instr.left, instr.right)
        if width is None:
            return None
        d, a, b = self._loc(instr.dst, width), self._loc(instr.left, width), self._loc(instr.right, width)
        if instr.op in self._ALU:
            return self._direct_alu(instr.op, d, a, b, width)
        if instr.op in (BinaryOp.DIVIDE, BinaryOp.MODULO):
            return self._direct_divmod(instr.op, d, a, b, width)
        if instr.op in (BinaryOp.SHIFT_LEFT, BinaryOp.SHIFT_RIGHT):
            return self._direct_shift(instr.op, d, a, b, width)
        return None

    def _direct_alu(self, op, d: Operand, a: Operand, b: Operand, width: int) -> list:
        cls = self._ALU[op][width == 8]
        out: list = []
        d_is_reg = isinstance(d, Register)
        can_target_d = d_is_reg or op != BinaryOp.MULTIPLY  # imul can't write memory
        if can_target_d and d == a:
            src = self._as_source(b, d, width, 'ecx', out)
            return out + [cls(src=src, dst=d)]
        if can_target_d and d == b and op in self._COMMUTATIVE:
            src = self._as_source(a, d, width, 'ecx', out)
            return out + [cls(src=src, dst=d)]
        if d_is_reg and d != b:
            out.extend(self._mov(a, d, width))
            src = self._as_source(b, d, width, 'ecx', out)
            return out + [cls(src=src, dst=d)]
        acc = self._scratch('eax', width)
        out.extend(self._mov(a, acc, width))
        src = self._as_source(b, acc, width, 'ecx', out)
        out.append(cls(src=src, dst=acc))
        return out + self._mov(acc, d, width)

    def _divmod_by_constant(self, op, d: Operand, a: Operand, divisor: int, width: int) -> list:
        """Division or modulo by a constant (not 0, 1, -1) with multiply-high, shifts, and adds.
        int32 operands are sign-extended and divided as 64-bit; the low half is the result."""
        rax, rcx, rdx = Register('rax'), Register('rcx'), Register('rdx')
        if width == 8:
            out = self._mov(a, rcx, 8)
        else:
            out = self._mov(a, Register('ecx'), 4) + [MovSXD(src=Register('ecx'), dst=rcx)]
        ad = abs(divisor)
        if is_power_of_two(ad):
            k = ad.bit_length() - 1
            out += [MovQ(src=rcx, dst=rdx), ShiftImmQ('sar', 63, rdx), ShiftImmQ('shr', 64 - k, rdx),
                    AddQ(src=rcx, dst=rdx), ShiftImmQ('sar', k, rdx)]
        else:
            m, shift = magic(ad)
            out += [MovQ(src=Imm(m), dst=rax), IMulWide(rcx)]
            if m < 0:
                out.append(AddQ(src=rcx, dst=rdx))
            if shift:
                out.append(ShiftImmQ('sar', shift, rdx))
            out += [MovQ(src=rdx, dst=rax), ShiftImmQ('shr', 63, rax), AddQ(src=rax, dst=rdx)]
        if divisor < 0:
            out.append(NegQ(rdx))
        result = rdx
        if op == BinaryOp.MODULO:
            if self._fits_imm(divisor, 8):
                out.append(IMulQ(src=Imm(divisor), dst=rdx))
            else:
                out += [MovQ(src=Imm(divisor), dst=rax), IMulQ(src=rax, dst=rdx)]
            out.append(SubQ(src=rdx, dst=rcx))
            result = rcx
        if width == 4:
            result = Register({'rdx': 'edx', 'rcx': 'ecx'}[result.name])
        return out + self._mov(result, d, width)

    def _direct_divmod(self, op, d: Operand, a: Operand, b: Operand, width: int) -> list:
        if isinstance(b, Imm) and b.value not in (0, 1, -1):
            return self._divmod_by_constant(op, d, a, b.value, width)
        acc = self._scratch('eax', width)
        out = self._mov(a, acc, width)
        divisor = b
        if isinstance(b, Imm):
            divisor = self._scratch('ecx', width)
            out.extend(self._mov(b, divisor, width))
        out.extend([Cqto(), IDivQ(divisor)] if width == 8 else [Cdq(), IDiv(divisor)])
        result = acc if op == BinaryOp.DIVIDE else self._scratch('edx', width)
        return out + self._mov(result, d, width)

    def _direct_shift(self, op, d: Operand, a: Operand, b: Operand, width: int) -> list:
        cls = {BinaryOp.SHIFT_LEFT: (ShiftLeft, ShiftLeftQ),
               BinaryOp.SHIFT_RIGHT: (ShiftRightArithmetic, ShiftRightArithmeticQ)}[op][width == 8]
        out = self._mov(b, self._scratch('ecx', width), width)  # count first: d may be b's home
        if d != a:
            if not isinstance(d, Register):
                a = self._as_source(a, d, width, 'eax', out)
            out.extend(self._mov(a, d, width))
        return out + [cls(dst=d)]

    def _direct_cmp(self, left: IRValue, right: IRValue, width: int) -> list:
        """Set flags from left - right."""
        out: list = []
        a, b = self._loc(left, width), self._loc(right, width)
        if isinstance(a, Imm) or (not isinstance(a, Register) and not isinstance(b, (Register, Imm))):
            acc = self._scratch('eax', width)
            out.extend(self._mov(a, acc, width))
            a = acc
        b = self._as_source(b, a, width, 'ecx', out)
        return out + [CmpQ(src=b, dst=a) if width == 8 else Cmp(src=b, dst=a)]

    def _direct_move(self, instr: IRMove) -> Optional[list]:
        width = self._same_width(instr.dst, instr.src)
        if width is None:
            return None
        d, s = self._loc(instr.dst, width), self._loc(instr.src, width)
        if isinstance(d, Register):
            return self._mov(s, d, width)
        out: list = []
        s = self._as_source(s, d, width, 'eax', out)
        return out + self._mov(s, d, width)

    def _direct_unop(self, instr: IRUnOp) -> Optional[list]:
        if instr.op not in (UnaryOp.NEGATE, UnaryOp.COMPLEMENT):
            return None
        width = self._same_width(instr.dst, instr.operand)
        if width is None:
            return None
        cls = {UnaryOp.NEGATE: (Neg, NegQ), UnaryOp.COMPLEMENT: (Not, NotQ)}[instr.op][width == 8]
        d, a = self._loc(instr.dst, width), self._loc(instr.operand, width)
        out: list = []
        if d != a:
            a = self._as_source(a, d, width, 'eax', out)
            out.extend(self._mov(a, d, width))
        return out + [cls(d)]

    def _address_operand(self, address: IRValue, scratch: str, out: list, offset: int = 0) -> Memory:
        """Memory operand for `address + offset` (the base in its register if allocated, else via `scratch`)."""
        loc = self._loc(address, 8)
        if isinstance(loc, Register):
            return Memory(loc.name, offset)
        reg = self._scratch(scratch, 8)
        out.extend(self._mov(loc, reg, 8))
        return Memory(reg.name, offset)

    def _direct_load(self, instr: IRLoad) -> Optional[list]:
        width = self._width(instr.dst.type)
        if width is None or not is_wide_type(instr.address.type):
            return None
        out: list = []
        mem = self._address_operand(instr.address, 'eax', out, instr.offset)
        d = self._loc(instr.dst, width)
        if isinstance(d, Register):
            return out + self._mov(mem, d, width)
        acc = self._scratch('ecx', width)
        return out + self._mov(mem, acc, width) + self._mov(acc, d, width)

    def _direct_store(self, instr: IRStore) -> Optional[list]:
        width = self._width(instr.value_type)
        if width is None or self._width(instr.value.type) != width or not is_wide_type(instr.address.type):
            return None
        out: list = []
        mem = self._address_operand(instr.address, 'r9d', out, instr.offset)
        v = self._as_source(self._loc(instr.value, width), mem, width, 'eax', out)
        return out + self._mov(v, mem, width)

    def _direct_branch(self, instr: IRBranch) -> Optional[list]:
        if self._width(instr.cond.type) != 4 or isinstance(instr.cond, IRConst):
            return None
        return [Cmp(src=Imm(0), dst=self._loc(instr.cond, 4)), Je(instr.false_label), Jmp(instr.true_label)]

    def _direct(self, instr) -> Optional[list]:
        """Direct selection for `instr`, or None to use the scratch-register path."""
        if isinstance(instr, IRBinOp):
            return self._direct_binop(instr)
        if isinstance(instr, IRMove):
            return self._direct_move(instr)
        if isinstance(instr, IRUnOp):
            return self._direct_unop(instr)
        if isinstance(instr, IRLoad):
            return self._direct_load(instr)
        if isinstance(instr, IRStore):
            return self._direct_store(instr)
        if isinstance(instr, IRBranch):
            return self._direct_branch(instr)
        if isinstance(instr, IRBoundsCheck):
            width = self._same_width(instr.index, instr.length)
            if width is None:
                return None
            return self._direct_cmp(instr.index, instr.length, width) + [
                Jae(self.host._get_bounds_check_fail_label("array index out of bounds"))]
        if isinstance(instr, IRSliceBoundsCheck):
            width = self._same_width(instr.value, instr.bound)
            if width is None:
                return None
            return self._direct_cmp(instr.value, instr.bound, width) + [
                Ja(self.host._get_bounds_check_fail_label("slice bounds out of range"))]
        if isinstance(instr, IRLocalAddress):
            d = self._loc(instr.dst, 8)
            if isinstance(d, Register):
                return [LeaQFrameSlot(slot=instr.slot, dst=d)]
        if isinstance(instr, IRStaticDataAddress):
            d = self._loc(instr.dst, 8)
            if isinstance(d, Register):
                return [LeaQ(label=instr.label, dst=d)]
        return None

    def _fused_compare_branch(self, instructions: list, i: int, use_sites: dict):
        """`cmp; jNCC false; jmp true` for a comparison read only by the IRBranch right after it, else None."""
        instr = instructions[i]
        if not (isinstance(instr, IRBinOp) and instr.op in COMPARISON_CONDITION_CODES and i + 1 < len(instructions)):
            return None
        branch = instructions[i + 1]
        if not (isinstance(branch, IRBranch) and branch.cond == instr.dst and use_sites.get(instr.dst.id) == [i + 1]):
            return None
        width = self._same_width(instr.left, instr.right)
        if width is not None:
            out = self._direct_cmp(instr.left, instr.right, width)
        else:
            out = self._gen_load_value(instr.left, Register('eax'))
            out.extend(self._gen_load_value(instr.right, Register('ecx')))
            out.append(self._cmp(instr.right, instr.left, wide=is_wide_type(instr.left.type)))
        out.append(JCC(INVERSE_CC[COMPARISON_CONDITION_CODES[instr.op]], branch.false_label))
        out.append(Jmp(branch.true_label))
        return out

    @staticmethod
    def _cmp(src: IRValue, dst: IRValue, wide: bool = None):
        """Compare %eax (dst) against %ecx (src), 64-bit if `wide` (default: either operand is wide)."""
        if wide is None:
            wide = is_wide_type(src.type) or is_wide_type(dst.type)
        if wide:
            return CmpQ(src=Register('rcx'), dst=Register('rax'))
        return Cmp(src=Register('ecx'), dst=Register('eax'))

    def lower_params(self, params: list, body: list) -> list[Instruction]:
        """Move each incoming argument word (SysV: 6 registers, then the caller's stack) into its Temp.
        Words the body never reads are skipped."""
        read = uses(body)
        out: list = []
        for index, temp in enumerate(params):
            if temp.id in read:
                out.extend(self._read_argument(temp, index))
        return out

    def _read_argument(self, dst: Temp, index: int) -> list:
        width = self._width(dst.type)
        if width is not None:
            d = self._loc(dst, width)
            if index < 6:
                return self._mov(Register((ARG_REGISTERS_64 if width == 8 else ARG_REGISTERS_32)[index]), d, width)
            if isinstance(d, Register):
                return self._mov(Memory('rbp', 16 + 8 * (index - 6)), d, width)
        wide = is_wide_type(dst.type)
        src = Register((ARG_REGISTERS_64 if wide else ARG_REGISTERS_32)[index]) if index < 6 else Memory('rbp', 16 + 8 * (index - 6))
        return [MovQ(src=src, dst=Register('rax')) if wide else Mov(src=src, dst=Register('eax'))] + \
            self._gen_write_temp_from(Register('eax'), dst)

    def lower_ir(self, instructions: list) -> list[Instruction]:
        """Lower an IR fragment."""
        out: list[Instruction] = []
        use_sites = uses(instructions)
        skip_next = False
        for i, instr in enumerate(instructions):
            if skip_next:
                skip_next = False
                continue
            fused = self._fused_compare_branch(instructions, i, use_sites)
            if fused is not None:
                out.extend(fused)
                skip_next = True
                continue
            direct = self._direct(instr)
            if direct is not None:
                out.extend(direct)
            elif isinstance(instr, IRMove):
                out.extend(self._gen_load_value(instr.src, Register('eax')))
                out.extend(self._gen_write_temp_from(Register('eax'), instr.dst))
            elif isinstance(instr, IRBinOp):
                out.extend(self._gen_load_value(instr.left, Register('eax')))
                out.extend(self._gen_load_value(instr.right, Register('ecx')))
                out.extend(self.host.gen_binary_op(
                    instr.op, src=Register('ecx'), dst=Register('eax'), operand_type=instr.left.type))
                out.extend(self._gen_write_temp_from(Register('eax'), instr.dst))
            elif isinstance(instr, IRUnOp):
                out.extend(self._gen_load_value(instr.operand, Register('eax')))
                out.extend(self.host.gen_unary_op(instr.op, Register('eax'), operand_type=instr.operand.type))
                out.extend(self._gen_write_temp_from(Register('eax'), instr.dst))
            elif isinstance(instr, IRCast):
                # src.type avoids truncating an already-int64 source.
                out.extend(self._gen_load_value(instr.src, Register('eax')))
                out.extend(self.host.gen_cast_narrowing_into(instr.dst.type, Register('eax'), instr.src.type))
                out.extend(self._gen_write_temp_from(Register('eax'), instr.dst))
            elif isinstance(instr, IRLabel):
                out.append(Label(instr.name))
            elif isinstance(instr, IRJump):
                out.append(Jmp(instr.label))
            elif isinstance(instr, IRBranch):
                out.extend(self._gen_load_value(instr.cond, Register('eax')))
                out.append(Cmp(src=Imm(0), dst=Register('eax')))
                out.append(Je(instr.false_label))
                out.append(Jmp(instr.true_label))
            elif isinstance(instr, IRCall):
                for i, arg_value in enumerate(instr.args):
                    if i < 6:
                        out.extend(self._gen_load_value(arg_value, Register(ARG_REGISTERS_32[i])))
                        continue
                    wide = is_wide_type(arg_value.type)
                    scratch = as_qword_register(Register('eax')) if wide else Register('eax')
                    out.extend(self._gen_load_value(arg_value, Register('eax')))
                    dst = FrameSlot(self.host.frame.outgoing, 8 * (i - 6))
                    out.append(MovQ(src=scratch, dst=dst) if wide else Mov(src=scratch, dst=dst))
                out.append(CallInstr(instr.name))
                if instr.dst is not None:
                    out.extend(self._gen_write_temp_from(Register('eax'), instr.dst))
            elif isinstance(instr, IRReturn):
                if instr.value is not None:
                    out.extend(self._gen_load_value(instr.value, Register('eax')))
                out.extend(self.host._gen_epilogue())
            elif isinstance(instr, IRLoad):
                out.extend(self._gen_load_value(instr.address, Register('eax')))
                if instr.dst.type == Type.STR:
                    out.append(MovQ(src=Memory('rax', instr.offset), dst=Register('rax')))
                else:
                    out.extend(self.host._gen_read_scalar_into(
                        Memory('rax', instr.offset), instr.dst.type, Register('eax')))
                out.extend(self._gen_write_temp_from(Register('eax'), instr.dst))
            elif isinstance(instr, IRStore):
                # address in %r9, value in %eax: both live at the store
                out.extend(self._gen_load_value(instr.address, Register('r9d')))
                out.extend(self._gen_load_value(instr.value, Register('eax')))
                if instr.value_type == Type.STR:
                    out.append(MovQ(src=Register('rax'), dst=Memory('r9', instr.offset)))
                else:
                    out.extend(self.host._gen_write_scalar_from(
                        Register('eax'), instr.value_type, Memory('r9', instr.offset)))
            elif isinstance(instr, IRLocalAddress):
                out.append(LeaQFrameSlot(slot=instr.slot, dst=Register('rax')))
                out.extend(self._gen_write_temp_from(Register('eax'), instr.dst))
            elif isinstance(instr, IRStaticDataAddress):
                out.append(LeaQ(label=instr.label, dst=Register('rax')))
                out.extend(self._gen_write_temp_from(Register('eax'), instr.dst))
            elif isinstance(instr, IRCopy):
                # %r9/%r8: both addresses live at once
                out.extend(self._gen_load_value(instr.dst_address, Register('r9d')))
                out.extend(self._gen_load_value(instr.src_address, Register('r8d')))
                out.extend(self.host.gen_array_copy(
                    Memory('r9', instr.dst_offset), Memory('r8', instr.src_offset), instr.value_type))
            elif isinstance(instr, IRBoundsCheck):
                # Cmp is unsigned; signedness lives in the jump.
                out.extend(self._gen_load_value(instr.length, Register('ecx')))
                out.extend(self._gen_load_value(instr.index, Register('eax')))
                out.append(self._cmp(instr.length, instr.index))
                out.append(Jae(self.host._get_bounds_check_fail_label("array index out of bounds")))
            elif isinstance(instr, IRSliceBoundsCheck):
                out.extend(self._gen_load_value(instr.bound, Register('ecx')))
                out.extend(self._gen_load_value(instr.value, Register('eax')))
                out.append(self._cmp(instr.bound, instr.value))
                out.append(Ja(self.host._get_bounds_check_fail_label("slice bounds out of range")))
            else:
                raise NotImplementedError(f"lower_ir has no rule for: {instr!r}")
        return out
