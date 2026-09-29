"""Instruction selection: IR -> assembly AST. Temps not assigned a register get a lazily allocated frame slot,
emitted as a FrameSlot placeholder and resolved after lowering.

Full-register types are selected directly on their registers, slots, and immediates; other
types, and instructions without a direct rule, go through %rax/%rcx. %rax, %rcx, %rdx, %r8
and %r9 are never allocated, so they are always free as scratch.
"""

from typing import Optional

from codegen.assembly_ast import (
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
    IRReadArgument,
    IRReturn,
    IRSliceBoundsCheck,
    IRSliceGrow,
    IRStaticDataAddress,
    IRStore,
    IRUnOp,
    IRValue,
    Temp,
)
from typesys import is_wide_type, type_byte_width
from ir.cfg import uses
from codegen.peephole import INVERSE_CC
from codegen.utils import as_qword_register, ARG_REGISTERS_32, ARG_REGISTERS_64, COMPARISON_CONDITION_CODES
from typesys import Type
from ops import BinaryOp, UnaryOp


class InstructionSelector:
    """Lowers one function's IR."""

    def __init__(self, host, ir_fn):
        self.host = host
        self.ir_fn = ir_fn

    def _temp_mem(self, temp: Temp) -> Operand:
        """Frame slot for `temp`, allocated on first use."""
        if temp.id in self.host.ir_program.ids._temp_offsets:
            return FrameSlot(slot=self.host.ir_program.ids._temp_offsets[temp.id])
        if temp.id not in self.host.ir_program.ids._temp_slots:
            width = type_byte_width(temp.type, self.host.ir_program.struct_registry, self.host.ir_program.sum_type_registry)
            self.host.ir_program.ids._temp_slots[temp.id] = self.host.ir_program.ids.new_slot(width, f"temp:{temp.id}", self.ir_fn)
        return FrameSlot(slot=self.host.ir_program.ids._temp_slots[temp.id])

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

    def _direct_divmod(self, op, d: Operand, a: Operand, b: Operand, width: int) -> list:
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

    def _address_operand(self, address: IRValue, scratch: str, out: list) -> Memory:
        """Memory operand for the address held in `address` (register if allocated, else via `scratch`)."""
        loc = self._loc(address, 8)
        if isinstance(loc, Register):
            return Memory(loc.name, 0)
        reg = self._scratch(scratch, 8)
        out.extend(self._mov(loc, reg, 8))
        return Memory(reg.name, 0)

    def _direct_load(self, instr: IRLoad) -> Optional[list]:
        width = self._width(instr.dst.type)
        if width is None or not is_wide_type(instr.address.type):
            return None
        out: list = []
        mem = self._address_operand(instr.address, 'eax', out)
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
        mem = self._address_operand(instr.address, 'r9d', out)
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
        if isinstance(instr, IRReadArgument):
            width = self._width(instr.dst.type)
            if width is None:
                return None
            d = self._loc(instr.dst, width)
            if instr.index < 6:
                src = Register((ARG_REGISTERS_64 if width == 8 else ARG_REGISTERS_32)[instr.index])
                return self._mov(src, d, width)
            if isinstance(d, Register):
                return self._mov(Memory('rbp', 16 + 8 * (instr.index - 6)), d, width)
            return None
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
                    dst = FrameSlot(self.ir_fn.outgoing_stack_args_slot, 8 * (i - 6))
                    out.append(MovQ(src=scratch, dst=dst) if wide else Mov(src=scratch, dst=dst))
                out.append(CallInstr(instr.name))
                if instr.dst is not None:
                    out.extend(self._gen_write_temp_from(Register('eax'), instr.dst))
            elif isinstance(instr, IRReadArgument):
                wide = is_wide_type(instr.dst.type)
                if instr.index < 6:
                    src = Register((ARG_REGISTERS_64 if wide else ARG_REGISTERS_32)[instr.index])
                else:
                    src = Memory('rbp', 16 + 8 * (instr.index - 6))
                out.append(MovQ(src=src, dst=Register('rax')) if wide else Mov(src=src, dst=Register('eax')))
                out.extend(self._gen_write_temp_from(Register('eax'), instr.dst))
            elif isinstance(instr, IRReturn):
                if instr.value is not None:
                    out.extend(self._gen_load_value(instr.value, Register('eax')))
                out.extend(self.host._gen_epilogue())
            elif isinstance(instr, IRLoad):
                out.extend(self._gen_load_value(instr.address, Register('eax')))
                if instr.dst.type == Type.STR:
                    out.append(MovQ(src=Memory('rax', 0), dst=Register('rax')))
                else:
                    out.extend(self.host._gen_read_scalar_into(Memory('rax', 0), instr.dst.type, Register('eax')))
                out.extend(self._gen_write_temp_from(Register('eax'), instr.dst))
            elif isinstance(instr, IRStore):
                # address in %r9, value in %eax: both live at the store
                out.extend(self._gen_load_value(instr.address, Register('r9d')))
                out.extend(self._gen_load_value(instr.value, Register('eax')))
                if instr.value_type == Type.STR:
                    out.append(MovQ(src=Register('rax'), dst=Memory('r9', 0)))
                else:
                    out.extend(self.host._gen_write_scalar_from(Register('eax'), instr.value_type, Memory('r9', 0)))
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
                out.extend(self.host.gen_array_copy(Memory('r9', 0), Memory('r8', 0), instr.value_type))
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
            elif isinstance(instr, IRSliceGrow):
                # Fixed callee-saved %rbx/%r12/%r13 survive the malloc inside grow.
                out.extend(self._gen_load_value(instr.ptr, Register('ebx')))
                out.extend(self._gen_load_value(instr.length, Register('r12d')))
                out.extend(self._gen_load_value(instr.cap, Register('r13d')))
                out.extend(self.host._gen_slice_grow_into(
                    Register('rbx'), Register('r12'), Register('r13'), instr.element_width))
                # Order the writes out of %ebx/%r13d so neither clobbers the other's source; swap via %r12 (length is dead).
                dst_ptr_reg = self.host._register_assignment.get(instr.dst_ptr.id)
                dst_cap_reg = self.host._register_assignment.get(instr.dst_cap.id)
                if dst_ptr_reg == 'r13d' and dst_cap_reg == 'ebx':
                    out.append(MovQ(src=Register('rbx'), dst=Register('r12')))
                    out.extend(self._gen_write_temp_from(Register('r13d'), instr.dst_cap))
                    out.extend(self._gen_write_temp_from(Register('r12d'), instr.dst_ptr))
                elif dst_ptr_reg == 'r13d':
                    out.extend(self._gen_write_temp_from(Register('r13d'), instr.dst_cap))
                    out.extend(self._gen_write_temp_from(Register('ebx'), instr.dst_ptr))
                else:
                    out.extend(self._gen_write_temp_from(Register('ebx'), instr.dst_ptr))
                    out.extend(self._gen_write_temp_from(Register('r13d'), instr.dst_cap))
            else:
                raise NotImplementedError(f"lower_ir has no rule for: {instr!r}")
        return out
