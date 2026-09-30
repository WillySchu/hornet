"""Instruction selection: IR -> AArch64 instructions for one function.

Operands are used in their registers; values in stack slots and immediates that don't fit are
first loaded into scratch registers (x16 for the left operand, x17 for the right, x9 for a
result headed to a stack slot, x8 for intermediate values). int8/uint8 values in registers are
kept sign/zero-extended to 32 bits, as on x86-64.
"""

from backend.aarch64.assembly import AddrOf, Call, Cond, FrameSlot, Imm, Instr, LabelDef, LabelRef, Mem, Reg, Shift, SymPage, SymPageOffset, FP
from backend.aarch64.calling_convention import ARG_REGISTERS, MAX_REGISTER_ARGS, SCRATCH_A, SCRATCH_B, SCRATCH_RESULT
from backend.errors import CodegenError
from ir.cfg import uses
from ir.ir import (
    IRBinOp, IRBoundsCheck, IRBranch, IRCall, IRCast, IRConst, IRJump, IRLabel, IRLoad, IRLocalAddress, IRMove, IRReturn,
    IRSliceBoundsCheck, IRStaticDataAddress, IRStore, IRUnOp, Temp,
)
from ops import BinaryOp, UnaryOp
from typesys import Type, is_wide_type, type_byte_width

SCRATCH_Q = Reg('x8')

CONDITIONS = {
    BinaryOp.EQUAL: 'eq', BinaryOp.NOT_EQUAL: 'ne', BinaryOp.LESS_THAN: 'lt',
    BinaryOp.GREATER_THAN: 'gt', BinaryOp.LESS_THAN_OR_EQUAL: 'le', BinaryOp.GREATER_THAN_OR_EQUAL: 'ge',
}
INVERSE = {'eq': 'ne', 'ne': 'eq', 'lt': 'ge', 'ge': 'lt', 'gt': 'le', 'le': 'gt',
           'lo': 'hs', 'hs': 'lo', 'hi': 'ls', 'ls': 'hi'}
_ALU = {BinaryOp.ADD: 'add', BinaryOp.SUBTRACT: 'sub', BinaryOp.MULTIPLY: 'mul', BinaryOp.BITWISE_AND: 'and',
        BinaryOp.BITWISE_OR: 'orr', BinaryOp.BITWISE_XOR: 'eor', BinaryOp.SHIFT_LEFT: 'lsl',
        BinaryOp.SHIFT_RIGHT: 'asr', BinaryOp.DIVIDE: 'sdiv'}


def is_logical_immediate(value: int, bits: int) -> bool:
    """Whether `value` is encodable as an immediate of and/orr/eor on a `bits`-wide register: a
    rotated run of ones in an element of 2..bits bits, repeated across the register."""
    mask = (1 << bits) - 1
    v = value & mask
    if v in (0, mask):
        return False
    size = 2
    while size < bits:
        element = v & ((1 << size) - 1)
        if sum(element << i for i in range(0, bits, size)) == v:
            break
        size *= 2
    element = v & ((1 << size) - 1)
    element_mask = (1 << size) - 1
    for r in range(size):
        rotated = ((element >> r) | (element << (size - r))) & element_mask
        if rotated and rotated & (rotated + 1) == 0:
            return True
    return False


def reg_width(t: Type) -> int:
    """Register width: 8 bytes (x) for int, pointers, and str descriptor addresses; 4 (w) otherwise."""
    return 8 if is_wide_type(t) or t == Type.STR else 4


def sized(reg: Reg, t: Type) -> Reg:
    return reg.x if reg_width(t) == 8 else reg.w


class Selector:
    def __init__(self, host, ir_fn):
        self.host = host
        self.ir_fn = ir_fn
        self.out: list = []

    def emit(self, mnemonic: str, *operands) -> None:
        self.out.append(Instr(mnemonic, operands))

    # -- where values live

    def _mem_size(self, t: Type) -> int:
        if t == Type.STR:
            return 8  # a str-typed Temp holds a descriptor address
        return type_byte_width(t, self.host.ir_program.struct_registry, self.host.ir_program.sum_type_registry)

    def _home(self, temp: Temp):
        """Allocated register (sized) or None."""
        name = self.host.assignment.get(temp.id)
        return sized(Reg(name), temp.type) if name else None

    def _slot(self, temp: Temp) -> FrameSlot:
        homes = self.ir_fn.temp_homes
        return FrameSlot(homes[temp.id] if temp.id in homes else self.host.frame.spill_slot(temp))

    def load_mem(self, t: Type, dst: Reg, mem) -> None:
        size = self._mem_size(t)
        if size == 8:
            self.emit('ldr', dst.x, mem)
        elif size == 4:
            self.emit('ldr', dst.w, mem)
        elif size == 1:
            self.emit('ldrsb' if t == Type.INT8 else 'ldrb', dst.w, mem)
        else:
            raise CodegenError(f"aarch64: no scalar load of {t} ({size} bytes)")

    def store_mem(self, t: Type, src: Reg, mem) -> None:
        size = self._mem_size(t)
        if size == 8:
            self.emit('str', src.x, mem)
        elif size == 4:
            self.emit('str', src.w, mem)
        elif size == 1:
            self.emit('strb', src.w, mem)
        else:
            raise CodegenError(f"aarch64: no scalar store of {t} ({size} bytes)")

    def mov_imm(self, dst: Reg, value: int) -> None:
        bits = 64 if dst.name[0] == 'x' else 32
        value = value & ((1 << bits) - 1)
        signed = value - (1 << bits) if value >> (bits - 1) else value
        if -65536 <= signed < 65536:
            self.emit('mov', dst, Imm(signed))
            return
        chunks = [(value >> s) & 0xFFFF for s in range(0, bits, 16)]
        first = True
        for i, c in enumerate(chunks):
            if c == 0:
                continue
            self.emit('movz' if first else 'movk', dst, Imm(c), Shift('lsl', 16 * i))
            first = False

    def value_in(self, value, scratch: Reg) -> Reg:
        """A register holding `value` (sized by its type): its own register, or `scratch`."""
        t = value.type
        if isinstance(value, IRConst):
            reg = sized(scratch, t)
            self.mov_imm(reg, value.value)
            return reg
        home = self._home(value)
        if home is not None:
            return home
        reg = sized(scratch, t)
        self.load_mem(t, reg, self._slot(value))
        return reg

    def dest(self, temp: Temp) -> Reg:
        """The register to compute `temp` into: its own, or the result scratch."""
        return self._home(temp) or sized(SCRATCH_RESULT, temp.type)

    def finish(self, temp: Temp, reg: Reg) -> None:
        """`reg` holds temp's new value: normalize narrow types, or store it to temp's slot."""
        home = self._home(temp)
        if home is None:
            self.store_mem(temp.type, reg, self._slot(temp))
            return
        if reg != home:
            self.emit('mov', home, sized(reg, temp.type))
        if temp.type == Type.INT8:
            self.emit('sxtb', home.w, home.w)
        elif temp.type == Type.UINT8:
            self.emit('uxtb', home.w, home.w)

    def assign(self, temp: Temp, value) -> None:
        home = self._home(temp)
        if home is not None and isinstance(value, IRConst):
            self.mov_imm(home, value.value)
            self.finish(temp, home)
            return
        self.finish(temp, self.value_in(value, SCRATCH_RESULT if home is None else home))

    # -- instructions

    def lower_params(self, params: list, body: list) -> list:
        read = uses(body)
        for index, temp in enumerate(params):
            if temp.id not in read:
                continue
            if index < MAX_REGISTER_ARGS:
                self.finish(temp, sized(ARG_REGISTERS[index], temp.type))
            else:
                d = self.dest(temp)
                self.load_mem(temp.type, d, Mem(FP, 16 + 8 * (index - MAX_REGISTER_ARGS)))
                self.finish(temp, d)
        return self.out

    def lower(self, body: list) -> list:
        use_sites = uses(body)
        skip = False
        for i, instr in enumerate(body):
            if skip:
                skip = False
                continue
            nxt = body[i + 1] if i + 1 < len(body) else None
            if (isinstance(instr, IRBinOp) and instr.op in CONDITIONS and isinstance(nxt, IRBranch)
                    and nxt.cond == instr.dst and use_sites.get(instr.dst.id) == [i + 1]):
                self._compare(instr.left, instr.right)
                self.emit(f'b.{INVERSE[CONDITIONS[instr.op]]}', LabelRef(nxt.false_label))
                self.emit('b', LabelRef(nxt.true_label))
                skip = True
                continue
            self._lower_one(instr)
        return self.out

    def _compare(self, left, right) -> None:
        a = self.value_in(left, SCRATCH_A)
        if isinstance(right, IRConst) and 0 <= right.value <= 4095:
            self.emit('cmp', a, Imm(right.value))
        elif isinstance(right, IRConst) and -4095 <= right.value < 0:
            self.emit('cmn', a, Imm(-right.value))
        else:
            self.emit('cmp', a, sized(self.value_in(right, SCRATCH_B), left.type))

    def _lower_one(self, instr) -> None:
        if isinstance(instr, IRLabel):
            self.out.append(LabelDef(instr.name))
        elif isinstance(instr, IRJump):
            self.emit('b', LabelRef(instr.label))
        elif isinstance(instr, IRBranch):
            c = self.value_in(instr.cond, SCRATCH_A)
            self.emit('cbz', c.w, LabelRef(instr.false_label))
            self.emit('b', LabelRef(instr.true_label))
        elif isinstance(instr, IRMove):
            self.assign(instr.dst, instr.src)
        elif isinstance(instr, IRBinOp):
            self._binop(instr)
        elif isinstance(instr, IRUnOp):
            d = self.dest(instr.dst)
            a = self.value_in(instr.operand, SCRATCH_A)
            if instr.op == UnaryOp.NEGATE:
                self.emit('neg', d, sized(a, instr.dst.type))
            elif instr.op == UnaryOp.COMPLEMENT:
                self.emit('mvn', d, sized(a, instr.dst.type))
            else:  # not
                self.emit('eor', d.w, a.w, Imm(1))
            self.finish(instr.dst, d)
        elif isinstance(instr, IRCast):
            self._cast(instr)
        elif isinstance(instr, IRLoad):
            addr = self.value_in(instr.address, SCRATCH_A)
            d = self.dest(instr.dst)
            self.load_mem(instr.dst.type, d, Mem(addr.x))
            self.finish(instr.dst, d)
        elif isinstance(instr, IRStore):
            addr = self.value_in(instr.address, SCRATCH_A)
            v = self.value_in(instr.value, SCRATCH_RESULT)
            self.store_mem(instr.value_type, v, Mem(addr.x))
        elif isinstance(instr, IRLocalAddress):
            d = self.dest(instr.dst).x
            self.out.append(AddrOf(d, FrameSlot(instr.slot)))
            self.finish(instr.dst, d)
        elif isinstance(instr, IRStaticDataAddress):
            d = self.dest(instr.dst).x
            self.emit('adrp', d, SymPage(instr.label))
            self.emit('add', d, d, SymPageOffset(instr.label))
            self.finish(instr.dst, d)
        elif isinstance(instr, IRCall):
            self._call(instr)
        elif isinstance(instr, IRBoundsCheck):
            self._compare(instr.index, instr.length)  # unsigned: a negative index is huge
            self.emit('b.hs', LabelRef(self.host.fail_label("array index out of bounds")))
        elif isinstance(instr, IRSliceBoundsCheck):
            self._compare(instr.value, instr.bound)
            self.emit('b.hi', LabelRef(self.host.fail_label("slice bounds out of range")))
        elif isinstance(instr, IRReturn):
            if instr.value is not None:
                reg = sized(ARG_REGISTERS[0], instr.value.type)
                v = self.value_in(instr.value, reg)
                if v != reg:
                    self.emit('mov', reg, v)
            self.out.extend(self.host.epilogue())
        else:
            raise CodegenError(f"aarch64: no lowering yet for {type(instr).__name__}")

    def _binop(self, instr: IRBinOp) -> None:
        op, dst = instr.op, instr.dst
        if op in CONDITIONS:
            self._compare(instr.left, instr.right)
            d = self.dest(dst)
            self.emit('cset', d.w, Cond(CONDITIONS[op]))
            self.finish(dst, d)
            return
        d = self.dest(dst)
        a = self.value_in(instr.left, SCRATCH_A)
        right = instr.right
        bits = reg_width(instr.left.type) * 8
        if op in (BinaryOp.ADD, BinaryOp.SUBTRACT) and isinstance(right, IRConst) and -4095 <= right.value <= 4095:
            mnemonic = _ALU[op] if right.value >= 0 else ('sub' if op == BinaryOp.ADD else 'add')
            self.emit(mnemonic, d, a, Imm(abs(right.value)))
        elif (op in (BinaryOp.BITWISE_AND, BinaryOp.BITWISE_OR, BinaryOp.BITWISE_XOR) and isinstance(right, IRConst)
              and is_logical_immediate(right.value, bits)):
            self.emit(_ALU[op], d, a, Imm(right.value & ((1 << bits) - 1)))
        elif op in (BinaryOp.SHIFT_LEFT, BinaryOp.SHIFT_RIGHT) and isinstance(right, IRConst):
            self.emit(_ALU[op], d, a, Imm(right.value & (bits - 1)))
        elif op == BinaryOp.MODULO:
            b = self.value_in(right, SCRATCH_B)
            q = sized(SCRATCH_Q, instr.left.type)
            self.emit('sdiv', q, a, b)
            self.emit('msub', d, q, b, a)
        elif op in _ALU:
            b = self.value_in(right, SCRATCH_B)
            self.emit(_ALU[op], d, a, b)
        else:
            raise CodegenError(f"aarch64: no lowering for binary operator {op}")
        self.finish(dst, d)

    def _cast(self, instr: IRCast) -> None:
        src, dst = instr.src, instr.dst
        d = self.dest(dst)
        a = self.value_in(src, SCRATCH_A)
        if dst.type == Type.INT8:
            self.emit('sxtb', d.w, a.w)
        elif dst.type == Type.UINT8:
            self.emit('uxtb', d.w, a.w)
        elif reg_width(dst.type) == 8 and reg_width(src.type) == 4:
            if src.type == Type.UINT8:
                self.emit('mov', d.w, a.w)  # already zero-extended; writing a w register clears the top
            else:
                self.emit('sxtw', d.x, a.w)
        elif reg_width(dst.type) == 4:
            self.emit('mov', d.w, a.w)
        else:
            self.emit('mov', d.x, a.x)
        self.finish(dst, d)

    def _call(self, instr: IRCall) -> None:
        for i, arg in enumerate(instr.args):
            if i < MAX_REGISTER_ARGS:
                reg = sized(ARG_REGISTERS[i], arg.type)
                if isinstance(arg, IRConst):
                    self.mov_imm(reg, arg.value)
                else:
                    home = self._home(arg)
                    if home is not None:
                        self.emit('mov', reg, home)
                    else:
                        self.load_mem(arg.type, reg, self._slot(arg))
            else:
                v = self.value_in(arg, SCRATCH_RESULT)
                self.emit('str', v.x, FrameSlot(self.host.frame.outgoing, 8 * (i - MAX_REGISTER_ARGS)))
        self.out.append(Call(instr.name))
        if instr.dst is not None:
            self.finish(instr.dst, sized(ARG_REGISTERS[0], instr.dst.type))
