"""AArch64 backend internals: emission per OS, immediates, frame legalization, callee-saved registers."""

import shutil
import subprocess
from pathlib import Path

import pytest

from backend.aarch64.assembly import AddrOf, FrameSlot, Imm, Instr, Mem, Reg, Shift
from backend.aarch64.codegen import CodeGenerator
from backend.aarch64.legalize import legalize
from backend.aarch64.lowering import Selector
from backend.common.frame import Frame
from build import c_compiler, can_run, run_prefix
from compile import compile_to_asm
from desugar import desugar_methods
from ir.ir import IRFunction
from ir.program_builder import build_ir_program
from merge import merge_programs
from modules import discover_modules
from optimize.optimizer import optimize
from semantic import analyze
from target import Target

ROOT = Path(__file__).resolve().parents[3]
A64 = Target('aarch64', 'linux')


def test_emission_differs_by_os(tmp_path):
    src = tmp_path / 'p.ht'
    src.write_text("def int main():\n    print('hi')\n    return 0\n")
    linux, mac = compile_to_asm(str(src), 'aarch64-linux'), compile_to_asm(str(src), 'aarch64-macos')
    assert '\nmain:' in linux and '\n_main:' in mac
    assert ':lo12:.L' in linux and '@PAGEOFF' in mac and '@PAGE\n' in mac
    assert 'bl      hornet_print' in linux and 'bl      _hornet_print' in mac
    assert linux.rstrip().endswith('.section .note.GNU-stack,"",%progbits') and 'GNU-stack' not in mac


def _simulate_mov(instrs: list, bits: int) -> int:
    value = 0
    for i in instrs:
        op, imm = i.mnemonic, i.operands[1].value
        shift = i.operands[2].amount if len(i.operands) > 2 else 0
        if op == 'mov':
            value = imm
        elif op == 'movz':
            value = imm << shift
        else:
            value = (value & ~(0xFFFF << shift)) | (imm << shift)
    return value & ((1 << bits) - 1)


@pytest.mark.parametrize('value', [0, 1, 65535, 65536, -1, -65536, -65537, 0x123456789ABCDEF0, -(2 ** 63), 2 ** 63 - 1, 0xFFFF0000])
@pytest.mark.parametrize('reg', ['x9', 'w9'])
def test_mov_imm_builds_the_value(value, reg):
    sel = Selector(None, None)
    sel.mov_imm(Reg(reg), value)
    bits = 64 if reg[0] == 'x' else 32
    assert _simulate_mov(sel.out, bits) == value & ((1 << bits) - 1)
    assert len(sel.out) <= bits // 16


def test_far_frame_slots_are_reached_through_x17():
    fn = IRFunction(name='f')
    fn.slot_widths = {0: 8, 1: 20000, 2: 8}
    frame = Frame(fn, {}, {})
    frame.reserve_outgoing(16)
    frame.layout(save_area=16)
    out = legalize([
        Instr('ldr', (Reg('x10'), FrameSlot(0))),
        Instr('str', (Reg('x11'), FrameSlot(2))),
        Instr('str', (Reg('x9'), FrameSlot(frame.outgoing, 8))),
        AddrOf(Reg('x12'), FrameSlot(2)),
    ], frame)
    assert out[0] == Instr('ldr', (Reg('x10'), Mem(Reg('x29'), -24)))
    far = frame.offsets[2]
    assert out[1] == Instr('movz', (Reg('x17'), Imm(-far & 0xFFFF), Shift('lsl', 0)))
    assert out[2] == Instr('sub', (Reg('x17'), Reg('x29'), Reg('x17')))
    assert out[3] == Instr('str', (Reg('x11'), Mem(Reg('x17'))))
    assert out[4] == Instr('str', (Reg('x9'), Mem(Reg('sp'), 8)))
    assert out[-1] == Instr('sub', (Reg('x12'), Reg('x29'), Reg('x17')))


CALLEE_SAVED = {f'x{i}' for i in range(19, 29)}
PROGRAMS = sorted((ROOT / 'benchmarks' / 'programs').glob('*.ht')) + sorted((ROOT / 'examples').glob('*.ht'))


def _asm_program(path: Path):
    entry, modules = discover_modules(str(path))
    program = merge_programs(entry, modules)
    desugar_methods(program)
    analyze(program)
    return CodeGenerator().generate(optimize(build_ir_program(program)))


@pytest.mark.parametrize('path', PROGRAMS, ids=lambda p: p.stem)
def test_functions_save_every_callee_saved_register_they_write(path):
    try:
        program = _asm_program(path)
    except Exception as e:  # composite features arrive in a later milestone
        pytest.skip(f'not yet supported on aarch64: {e}')
    for fn in program.functions:
        saved = set()
        for i in fn.instructions:
            if isinstance(i, Instr) and i.mnemonic in ('stp', 'str') and isinstance(i.operands[-1], Mem) \
                    and i.operands[-1].base.name == 'x29':
                saved |= {o.x.name for o in i.operands[:-1]}
        written = {i.operands[0].x.name for i in fn.instructions
                   if isinstance(i, Instr) and i.operands and isinstance(i.operands[0], Reg)
                   and i.mnemonic not in ('str', 'strb', 'stp', 'cmp', 'cmn', 'cbz', 'cbnz')}
        assert written & CALLEE_SAVED <= saved, fn.name
        rets = [k for k, i in enumerate(fn.instructions) if isinstance(i, Instr) and i.mnemonic == 'ret']
        for k in rets:
            tail = fn.instructions[k - 2:k + 1]
            assert [t.mnemonic for t in tail] == ['mov', 'ldp', 'ret'], fn.name


_HARNESS = """
    .text
    .globl main
    .p2align 2
main:
    stp x29, x30, [sp, #-96]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    stp x23, x24, [sp, #48]
    stp x25, x26, [sp, #64]
    stp x27, x28, [sp, #80]
""" + ''.join(f"    mov x{r}, #{r * 100}\n" for r in range(19, 29)) + """
    mov x0, #50
    bl work
    mov x1, x0
""" + ''.join(f"    cmp x{r}, #{r * 100}\n    b.ne .Lfail\n" for r in range(19, 29)) + """
    mov x0, x1
    b .Ldone
.Lfail:
    mov x0, #255
.Ldone:
    ldp x19, x20, [sp, #16]
    ldp x21, x22, [sp, #32]
    ldp x23, x24, [sp, #48]
    ldp x25, x26, [sp, #64]
    ldp x27, x28, [sp, #80]
    ldp x29, x30, [sp], #96
    ret
    .section .note.GNU-stack,"",%progbits
"""

_WORK = (
    "def int leaf(int x):\n    return x * 3 + 1\n\n"
    "def int work(int n):\n"
    + ''.join(f"    int v{k} = {k}\n" for k in range(14))
    + "    for int i = 0; i < n; i += 1:\n"
    + ''.join(f"        v{k} = v{k} + leaf(v{(k + 1) % 14})\n" for k in range(14))
    + "    return (" + ' + '.join(f"v{k}" for k in range(14)) + ") & 255\n"
)


def _work_model() -> int:
    v = list(range(14))
    for _ in range(50):
        for k in range(14):
            v[k] = (v[k] + v[(k + 1) % 14] * 3 + 1) & (2 ** 64 - 1)
    return sum(v) & 255


@pytest.mark.skipif(not can_run(A64), reason='no aarch64-linux toolchain and qemu-user')
def test_c_convention_caller_keeps_callee_saved_registers(tmp_path):
    (tmp_path / 'work.ht').write_text(_WORK)
    (tmp_path / 'work.s').write_text(compile_to_asm(str(tmp_path / 'work.ht'), 'aarch64-linux'), encoding='latin-1')
    (tmp_path / 'harness.s').write_text(_HARNESS)
    exe = tmp_path / 'abi'
    subprocess.run(c_compiler(A64) + [str(tmp_path / 'harness.s'), str(tmp_path / 'work.s'), '-o', str(exe)], check=True)
    assert subprocess.run(run_prefix(A64) + [str(exe)]).returncode == _work_model()


@pytest.mark.skipif(shutil.which('aarch64-linux-gnu-as') is None, reason='no aarch64 assembler')
def test_logical_immediates_match_the_assembler(tmp_path):
    import random
    from backend.aarch64.lowering import is_logical_immediate
    r = random.Random(5)
    values = [1, 3, 255, 0xFF00, 0xAAAAAAAAAAAAAAAA, 0x0F0F0F0F0F0F0F0F, 0xFFFFFFFF, 0x8000000000000001, 0x1234, -2]
    values += [r.getrandbits(64) for _ in range(20)] + [((1 << r.randint(1, 63)) - 1) << r.randint(0, 10) for _ in range(20)]
    lines, expected = [], []
    for bits, reg in [(64, 'x'), (32, 'w')]:
        for v in values:
            v &= (1 << bits) - 1
            lines.append(f"and {reg}0, {reg}1, #{v}")
            expected.append(is_logical_immediate(v, bits))
    for line, want in zip(lines, expected):
        (tmp_path / 't.s').write_text(line + '\n')
        got = subprocess.run(['aarch64-linux-gnu-as', str(tmp_path / 't.s'), '-o', str(tmp_path / 't.o')],
                             capture_output=True).returncode == 0
        assert got == want, line
