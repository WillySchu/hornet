"""Direct instruction selection checked on a tiny x86-64 simulator.

Random IR binary ops and moves, with operands in registers, frame slots, or immediates, and
destinations aliasing operands. The simulator rejects operand forms the assembler would
reject, and each case checks the result and that no other temp's home was clobbered.
"""

import random

import pytest

from backend.x86_64.assembly_ast import (
    Add, AddQ, And, AndQ, Cdq, Cmp, CmpQ, Cqto, FrameSlot, IDiv, IDivQ, IMul, IMulQ, IMulWide, Imm, Mov, MovQ,
    MovSXD, MovZX, ShiftImmQ, Neg, NegQ, Not, NotQ, Or, OrQ, Register, SetCC, ShiftLeft, ShiftLeftQ,
    ShiftRightArithmetic, ShiftRightArithmeticQ, Sub, SubQ, Xor, XorQ,
)
from backend.x86_64.codegen import CodeGenerator
from backend.x86_64.ir_lowering import InstructionSelector
from backend.x86_64.utils import _QWORD_REGISTER_ALIASES
from ir.id_allocator import IdAllocator
from ir.ir import IRBinOp, IRConst, IRFunction, IRMove, IRProgram, IRUnOp, Temp
from ops import BinaryOp, UnaryOp
from typesys import Type

POOL = ['r10d', 'r11d', 'r15d', 'ebx', 'r12d', 'r13d', 'r14d']
M64 = (1 << 64) - 1


def _signed(v: int, bits: int) -> int:
    v &= (1 << bits) - 1
    return v - (1 << bits) if v >> (bits - 1) else v


class Sim:
    def __init__(self):
        self.regs = {}
        self.mem = {}
        self.flags = None

    @staticmethod
    def _reg64(r: Register) -> str:
        return _QWORD_REGISTER_ALIASES.get(r.name, r.name)

    def read(self, op, width: int) -> int:
        mask = (1 << (width * 8)) - 1
        if isinstance(op, Imm):
            return op.value & mask
        if isinstance(op, Register):
            if op.name in ('al',):
                return self.regs.get('rax', 0) & 0xFF
            return self.regs.get(self._reg64(op), 0) & mask
        assert isinstance(op, FrameSlot), op
        return self.mem.get(op.slot, 0) & mask

    def write(self, op, value: int, width: int) -> None:
        assert not isinstance(op, Imm), "immediate destination"
        mask = (1 << (width * 8)) - 1
        if isinstance(op, Register):
            if op.name == 'al':
                self.regs['rax'] = (self.regs.get('rax', 0) & ~0xFF) | (value & 0xFF)
                return
            self.regs[self._reg64(op)] = value & mask  # 32-bit writes zero-extend
            return
        old = self.mem.get(op.slot, 0)
        self.mem[op.slot] = (old & ~mask & M64) | (value & mask)

    @staticmethod
    def _check_two(src, dst, width, imul=False, mov=False):
        assert not (isinstance(src, FrameSlot) and isinstance(dst, FrameSlot)), "memory-to-memory"
        if imul:
            assert isinstance(dst, Register), "imul to memory"
        if isinstance(src, Imm) and width == 8 and not (mov and isinstance(dst, Register)):
            assert -2 ** 31 <= src.value < 2 ** 31, "64-bit immediate out of range"

    def run(self, instrs: list) -> None:
        alu = {Add: (4, lambda a, b: a + b), AddQ: (8, lambda a, b: a + b),
               Sub: (4, lambda a, b: a - b), SubQ: (8, lambda a, b: a - b),
               IMul: (4, lambda a, b: a * b), IMulQ: (8, lambda a, b: a * b),
               And: (4, lambda a, b: a & b), AndQ: (8, lambda a, b: a & b),
               Or: (4, lambda a, b: a | b), OrQ: (8, lambda a, b: a | b),
               Xor: (4, lambda a, b: a ^ b), XorQ: (8, lambda a, b: a ^ b)}
        for ins in instrs:
            t = type(ins)
            if t in (Mov, MovQ):
                w = 8 if t is MovQ else 4
                self._check_two(ins.src, ins.dst, w, mov=True)
                self.write(ins.dst, self.read(ins.src, w), w)
            elif t in alu:
                w, fn = alu[t]
                self._check_two(ins.src, ins.dst, w, imul=t in (IMul, IMulQ))
                a, b = _signed(self.read(ins.dst, w), w * 8), _signed(self.read(ins.src, w), w * 8)
                self.write(ins.dst, fn(a, b), w)
            elif t in (Neg, NegQ, Not, NotQ):
                w = 8 if t in (NegQ, NotQ) else 4
                v = _signed(self.read(ins.operand, w), w * 8)
                self.write(ins.operand, -v if t in (Neg, NegQ) else ~v, w)
            elif t in (ShiftLeft, ShiftLeftQ, ShiftRightArithmetic, ShiftRightArithmeticQ):
                w = 8 if t in (ShiftLeftQ, ShiftRightArithmeticQ) else 4
                count = self.regs.get('rcx', 0) & (63 if w == 8 else 31)
                v = _signed(self.read(ins.dst, w), w * 8)
                self.write(ins.dst, v << count if t in (ShiftLeft, ShiftLeftQ) else v >> count, w)
            elif t is Cqto:
                self.regs['rdx'] = M64 if _signed(self.regs.get('rax', 0), 64) < 0 else 0
            elif t is Cdq:
                self.regs['rdx'] = 0xFFFFFFFF if _signed(self.regs.get('rax', 0), 32) < 0 else 0
            elif t in (IDiv, IDivQ):
                w = 8 if t is IDivQ else 4
                assert not isinstance(ins.operand, Imm), "idiv immediate"
                bits = w * 8
                dividend = _signed(((self.regs.get('rdx', 0) & ((1 << bits) - 1)) << bits)
                                   | (self.regs.get('rax', 0) & ((1 << bits) - 1)), 2 * bits)
                divisor = _signed(self.read(ins.operand, w), bits)
                q = abs(dividend) // abs(divisor)
                q = q if (dividend < 0) == (divisor < 0) else -q
                self.write(Register('rax') if w == 8 else Register('eax'), q, w)
                self.write(Register('rdx') if w == 8 else Register('edx'), dividend - q * divisor, w)
            elif t in (Cmp, CmpQ):
                w = 8 if t is CmpQ else 4
                self._check_two(ins.src, ins.dst, w)
                assert not isinstance(ins.dst, Imm), "cmp immediate destination"
                self.flags = (_signed(self.read(ins.dst, w), w * 8), _signed(self.read(ins.src, w), w * 8))
            elif t is SetCC:
                a, b = self.flags
                ok = {'e': a == b, 'ne': a != b, 'l': a < b, 'le': a <= b, 'g': a > b, 'ge': a >= b}[ins.cc]
                self.write(ins.operand, int(ok), 1)
            elif t is MovZX:
                self.write(ins.dst, self.read(ins.src, 1), 4)
            elif t is MovSXD:
                self.write(ins.dst, _signed(self.read(ins.src, 4), 32), 8)
            elif t is IMulWide:
                prod = _signed(self.regs.get('rax', 0), 64) * _signed(self.read(ins.operand, 8), 64)
                self.regs['rax'], self.regs['rdx'] = prod & M64, (prod >> 64) & M64
            elif t is ShiftImmQ:
                v = self.read(ins.dst, 8)
                v = {'sar': _signed(v, 64) >> ins.count, 'shr': v >> ins.count, 'shl': v << ins.count}[ins.kind]
                self.write(ins.dst, v, 8)
            else:
                raise AssertionError(f"simulator has no rule for {ins!r}")


def _model(op, a: int, b: int, bits: int) -> int:
    if op in (BinaryOp.DIVIDE, BinaryOp.MODULO):
        q = abs(a) // abs(b)
        q = q if (a < 0) == (b < 0) else -q
        return _signed(q if op == BinaryOp.DIVIDE else a - q * b, bits)
    if op == BinaryOp.SHIFT_LEFT:
        return _signed(a << (b & (bits - 1)), bits)
    if op == BinaryOp.SHIFT_RIGHT:
        return a >> (b & (bits - 1))
    cmp = {BinaryOp.LESS_THAN: a < b, BinaryOp.LESS_THAN_OR_EQUAL: a <= b, BinaryOp.GREATER_THAN: a > b,
           BinaryOp.GREATER_THAN_OR_EQUAL: a >= b, BinaryOp.EQUAL: a == b, BinaryOp.NOT_EQUAL: a != b}
    if op in cmp:
        return int(cmp[op])
    fn = {BinaryOp.ADD: a + b, BinaryOp.SUBTRACT: a - b, BinaryOp.MULTIPLY: a * b,
          BinaryOp.BITWISE_AND: a & b, BinaryOp.BITWISE_OR: a | b, BinaryOp.BITWISE_XOR: a ^ b}
    return _signed(fn[op], bits)


OPS = [BinaryOp.ADD, BinaryOp.SUBTRACT, BinaryOp.MULTIPLY, BinaryOp.BITWISE_AND, BinaryOp.BITWISE_OR,
       BinaryOp.BITWISE_XOR, BinaryOp.DIVIDE, BinaryOp.MODULO, BinaryOp.SHIFT_LEFT, BinaryOp.SHIFT_RIGHT,
       BinaryOp.LESS_THAN, BinaryOp.GREATER_THAN_OR_EQUAL, BinaryOp.EQUAL, BinaryOp.NOT_EQUAL]


def _case(seed: int):
    r = random.Random(seed)
    bits = r.choice([64, 32])
    t = Type.INT if bits == 64 else Type.INT32
    op = r.choice(OPS)
    kind = r.random()
    temps = [Temp(i, t) for i in range(1, 5)]
    regs = r.sample(POOL, len(POOL))
    assignment = {tmp.id: regs[i] for i, tmp in enumerate(temps) if r.random() < 0.6}

    def operand():
        roll = r.random()
        if roll < 0.2:
            big = r.random() < 0.3 and bits == 64
            v = r.randint(-2 ** 62, 2 ** 62) if big else r.randint(-2 ** 31, 2 ** 31 - 1)
            return IRConst(v if bits == 64 else _signed(v, 32), t)
        return r.choice(temps)

    left, right = operand(), operand()
    is_cmp = op in (BinaryOp.LESS_THAN, BinaryOp.GREATER_THAN_OR_EQUAL, BinaryOp.EQUAL, BinaryOp.NOT_EQUAL)
    if is_cmp:
        dst = Temp(9, Type.BOOL)
        if r.random() < 0.6:
            assignment[9] = next(x for x in POOL if x not in assignment.values())
    else:
        dst = r.choice(temps + [Temp(9, t)])  # often aliases an operand
        if dst.id == 9 and r.random() < 0.6:
            assignment[9] = next(x for x in POOL if x not in assignment.values())
    if kind < 0.1:
        return bits, IRMove(dst=dst if not is_cmp else temps[0], src=left), assignment, None
    if kind < 0.2 and not is_cmp:
        return bits, IRUnOp(dst=dst, op=r.choice([UnaryOp.NEGATE, UnaryOp.COMPLEMENT]), operand=left), assignment, None
    if op in (BinaryOp.DIVIDE, BinaryOp.MODULO) and isinstance(right, IRConst) and right.value in (0, -1):
        right = IRConst(7, t)
    return bits, IRBinOp(dst=dst, op=op, left=left, right=right), assignment, op


@pytest.mark.parametrize('seed', range(600))
def test_direct_selection_matches_model(seed):
    bits, instr, assignment, op = _case(seed)
    host = CodeGenerator()
    host.ir_program = IRProgram(ids=IdAllocator())
    host._register_assignment = assignment
    sel = InstructionSelector(host, IRFunction(name='f'))
    r = random.Random(seed + 10_000)
    values = {}
    for tid in (1, 2, 3, 4, 9):
        v = _signed(r.getrandbits(bits), bits)
        if (
                op in (BinaryOp.DIVIDE, BinaryOp.MODULO)
                and isinstance(instr, IRBinOp)
                and getattr(instr.right, 'id', None) == tid
        ):
            v = r.choice([1, 3, -5, 1000, -2 ** 20]) if tid != getattr(instr.left, 'id', None) else 3
        values[tid] = v
    def home(tid, w):
        t = Temp(
            tid, Type.INT if w == 8 else (Type.BOOL if tid == 9 and op is not None and op in OPS[10:] else Type.INT32)
        )
        return sel._loc(t, w)

    width = bits // 8
    out = sel.lower_ir([instr])
    sim = Sim()
    for tid, v in values.items():
        sim.write(home(tid, width), v, width)
    before = {tid: sim.read(home(tid, width), width) for tid in values}
    sim.run(out)

    def val(v):
        return v.value if isinstance(v, IRConst) else values[v.id]

    if isinstance(instr, (IRMove, IRUnOp)):
        if isinstance(instr, IRMove):
            expected = _signed(val(instr.src), bits)
        else:
            v = val(instr.operand)
            expected = _signed(-v if instr.op == UnaryOp.NEGATE else ~v, bits)
        got = _signed(sim.read(home(instr.dst.id, width), width), bits)
    else:
        expected = _model(op, val(instr.left), val(instr.right), bits)
        dst_width = 4 if instr.dst.type == Type.BOOL else width
        got = _signed(sim.read(home(instr.dst.id, dst_width), dst_width), dst_width * 8)
    assert got == expected, (instr, assignment, out)
    for tid in values:
        if tid != instr.dst.id:
            assert sim.read(home(tid, width), width) == before[tid], f"clobbered temp {tid}: {out}"


DIVISORS = [2, 3, 5, 7, 10, 15, 16, 255, 256, 1000003, 7919, 2 ** 31 - 1, 2 ** 31, 2 ** 40 + 3, 2 ** 62, 2 ** 63 - 1]
DIVISORS += [-x for x in DIVISORS] + [-(2 ** 63)]


@pytest.mark.parametrize('divisor', DIVISORS)
@pytest.mark.parametrize('bits', [64, 32])
def test_division_by_constant_matches_idiv(divisor, bits):
    if bits == 32 and not -2 ** 31 <= divisor < 2 ** 31:
        pytest.skip('not an int32 constant')
    t = Type.INT if bits == 64 else Type.INT32
    width = bits // 8
    r = random.Random(divisor * 7 + bits)
    lo, hi = -2 ** (bits - 1), 2 ** (bits - 1) - 1
    values = [lo, hi, 0, 1, -1, 2, -2]
    values += [_signed(divisor * k + e, bits) for k in (1, 2, 3, -1, -7) for e in (-1, 0, 1)]
    values += [r.randint(lo, hi) for _ in range(40)]
    for op in (BinaryOp.DIVIDE, BinaryOp.MODULO):
        for n in values:
            expected = _model(op, n, divisor, bits)
            for layout in range(4):
                src, dst = Temp(1, t), Temp(2 if layout % 2 else 1, t)
                assignment = {}
                if layout < 2:
                    assignment[1] = 'r12d'
                    if dst.id == 2:
                        assignment[2] = 'r13d'
                host = CodeGenerator()
                host.ir_program = IRProgram(ids=IdAllocator())
                host._register_assignment = assignment
                sel = InstructionSelector(host, IRFunction(name='f'))
                out = sel.lower_ir([IRBinOp(dst=dst, op=op, left=src, right=IRConst(divisor, t))])
                assert not any(isinstance(i, (IDiv, IDivQ)) for i in out)
                sim = Sim()
                sim.write(sel._loc(src, width), n, width)
                sim.write(Register('rbx'), 12345, 8)
                sim.run(out)
                got = _signed(sim.read(sel._loc(dst, width), width), bits)
                assert got == expected, (op, n, divisor, bits, layout, out)
                if dst.id != src.id:
                    assert _signed(sim.read(sel._loc(src, width), width), bits) == _signed(n, bits)
                assert sim.regs['rbx'] == 12345
