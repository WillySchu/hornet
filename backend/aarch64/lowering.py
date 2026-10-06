"""Instruction selection: IR -> AArch64 instructions for one function.

Operands are used in their registers; values in stack slots and immediates that don't fit are
first loaded into scratch registers (x16 for the left operand, x17 for the right, x9 for a
result headed to a stack slot, x8 for intermediate values). int8/uint8 values in registers are
kept sign/zero-extended to 32 bits, as on x86-64.
"""

from backend.aarch64.assembly import (
    AddrOf, Call, Cond, FrameSlot, Imm, Instr, LabelDef, LabelRef, Mem, Reg, Shift, SymPage, SymPageOffset, FP,
)
from backend.aarch64.calling_convention import (
    ARG_REGISTERS, MAX_REGISTER_ARGS, SCRATCH_A, SCRATCH_ADDRESS, SCRATCH_B, SCRATCH_RESULT)
from backend.common.division import is_power_of_two, magic
from backend.errors import CodegenError
from ir.cfg import uses
from ir.panics import located
from ir.ir import (
    IRBinOp, IRBoundsCheck, IRBranch, IRCall, IRCast, IRCopy, IRConst, IRJump, IRLabel, IRLoad, IRLocalAddress, IRMove,
    IRReturn, IRSliceBoundsCheck, IRStaticDataAddress, IRStore, IRUnOp, Temp,
)
from ops import BinaryOp, UnaryOp
from typesys import Type, is_wide_type, type_byte_width

SCRATCH_Q = Reg('x8')
UNROLLED_COPY_LIMIT = 128

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

    def address(self, base: Reg, offset: int, scratch: Reg) -> Mem:
        """[base, #offset] when a load or store reaches it directly, else through `scratch`."""
        if -256 <= offset <= 255:
            return Mem(base.x, offset)
        return Mem(self.offset_base(base.x, offset, scratch))

    def offset_base(self, base: Reg, offset: int, scratch: Reg) -> Reg:
        """A register holding base + offset (`base` itself when the offset is 0)."""
        if offset == 0:
            return base
        if 0 < offset <= 4095:
            self.emit('add', scratch.x, base, Imm(offset))
        elif -4095 <= offset < 0:
            self.emit('sub', scratch.x, base, Imm(-offset))
        else:  # beyond an immediate: materialize it (in x17, which holds nothing between instructions)
            self.mov_imm(SCRATCH_ADDRESS, offset)
            self.emit('add', scratch.x, base, SCRATCH_ADDRESS)
        return scratch.x

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
            self.load_mem(instr.dst.type, d, self.address(addr, instr.offset, SCRATCH_A))
            self.finish(instr.dst, d)
        elif isinstance(instr, IRStore):
            addr = self.value_in(instr.address, SCRATCH_A)
            v = self.value_in(instr.value, SCRATCH_RESULT)
            self.store_mem(instr.value_type, v, self.address(addr, instr.offset, SCRATCH_A))
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
        elif isinstance(instr, IRCopy):
            self._copy(instr)
        elif isinstance(instr, IRBoundsCheck):
            self._compare(instr.index, instr.length)  # unsigned: a negative index is huge
            self.emit('b.hs', LabelRef(self.host.fail_label(located("array index out of bounds", instr.where))))
        elif isinstance(instr, IRSliceBoundsCheck):
            self._compare(instr.value, instr.bound)
            self.emit('b.hi', LabelRef(self.host.fail_label(located("slice bounds out of range", instr.where))))
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
        elif op in (BinaryOp.DIVIDE, BinaryOp.MODULO) and isinstance(right, IRConst) and right.value not in (0, 1, -1):
            self._divmod_by_constant(op, d, a, right.value)
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

    def _divmod_by_constant(self, op, d: Reg, a: Reg, divisor: int) -> None:
        """Signed division or modulo by a constant with multiply-high and shifts (backend/common/division.py).
        Narrower operands are sign-extended and divided as 64-bit; the low half is the result."""
        n, q, t = SCRATCH_A, SCRATCH_RESULT, SCRATCH_B
        if a.name[0] == 'w':
            self.emit('sxtw', n, a)
        elif a != n:
            self.emit('mov', n, a)
        ad = abs(divisor)
        if is_power_of_two(ad):
            k = ad.bit_length() - 1
            self.emit('asr', q, n, Imm(63))
            self.emit('add', q, n, q, Shift('lsr', 64 - k))
            self.emit('asr', q, q, Imm(k))
        else:
            m, shift = magic(ad)
            self.mov_imm(t, m)
            self.emit('smulh', q, n, t)
            if m < 0:
                self.emit('add', q, q, n)
            if shift:
                self.emit('asr', q, q, Imm(shift))
            self.emit('add', q, q, q, Shift('lsr', 63))
        if divisor < 0:
            self.emit('neg', q, q)
        if op == BinaryOp.MODULO:
            self.mov_imm(t, divisor)
            self.emit('msub', q, q, t, n)
        result = q if d.name[0] == 'x' else q.w
        if result != d:
            self.emit('mov', d, result)

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

    def _copy(self, instr: IRCopy) -> None:
        """Copy value_type's bytes: unrolled 16/8/4/2/1-byte moves when small, else an 8-byte loop."""
        size = type_byte_width(instr.value_type, self.host.ir_program.struct_registry,
                               self.host.ir_program.sum_type_registry)
        src = self.offset_base(self.value_in(instr.src_address, SCRATCH_A).x, instr.src_offset, SCRATCH_A)
        dst = self.offset_base(self.value_in(instr.dst_address, SCRATCH_Q).x, instr.dst_offset, SCRATCH_Q)
        if size <= UNROLLED_COPY_LIMIT:
            self._copy_tail(dst, src, 0, size)
            return
        # Cursors and a counter in scratch registers: the address registers themselves may be allocated.
        if src != SCRATCH_A:
            self.emit('mov', SCRATCH_A, src)
        if dst != SCRATCH_Q:
            self.emit('mov', SCRATCH_Q, dst)
        counter, data = SCRATCH_B, SCRATCH_RESULT
        self.mov_imm(counter, size // 8)
        loop = self.host.ir_program.ids.new_label("copy_loop")
        self.out.append(LabelDef(loop))
        self.emit('ldr', data, Mem(SCRATCH_A, 8, 'post'))
        self.emit('str', data, Mem(SCRATCH_Q, 8, 'post'))
        self.emit('subs', counter, counter, Imm(1))
        self.emit('b.ne', LabelRef(loop))
        self._copy_tail(SCRATCH_Q, SCRATCH_A, 0, size % 8)

    def _copy_tail(self, dst: Reg, src: Reg, offset: int, size: int) -> None:
        a, b = SCRATCH_RESULT, SCRATCH_B
        while size >= 16:
            self.emit('ldp', a, b, Mem(src, offset))
            self.emit('stp', a, b, Mem(dst, offset))
            offset, size = offset + 16, size - 16
        for width, load, store, reg in ((8, 'ldr', 'str', a), (4, 'ldr', 'str', a.w), (2, 'ldrh', 'strh', a.w),
                                        (1, 'ldrb', 'strb', a.w)):
            while size >= width:
                self.emit(load, reg, Mem(src, offset))
                self.emit(store, reg, Mem(dst, offset))
                offset, size = offset + width, size - width

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
