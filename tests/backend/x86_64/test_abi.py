"""Callee-saved register handling: every function saves what it uses, and a C-convention
caller's callee-saved registers survive a call into Hornet code."""

import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import pytest

from backend.x86_64.assembly_ast import Leave, LeaQFrame, MovQ, Pop, Push, Register, Ret
from backend.x86_64.calling_convention import CALLEE_SAVED_REGISTERS
from backend.x86_64.codegen import CodeGenerator
from backend.x86_64.utils import as_qword_register
from compile import compile_to_asm
from desugar import desugar_methods
from ir.program_builder import build_ir_program
from merge import merge_programs
from modules import discover_modules
from optimize.optimizer import optimize
from semantic import analyze
from tests.test_compiler import GCC_SKIP

ROOT = Path(__file__).resolve().parents[3]
PROGRAMS = sorted((ROOT / 'benchmarks' / 'programs').glob('*.ht')) + sorted((ROOT / 'examples').glob('*.ht'))


def _asm_program(path: Path):
    entry, modules = discover_modules(str(path))
    program = merge_programs(entry, modules)
    desugar_methods(program)
    analyze(program)
    return CodeGenerator().generate(optimize(build_ir_program(program)))


def _qword(op) -> str:
    if not isinstance(op, Register):
        return ''
    try:
        return as_qword_register(op).name
    except Exception:
        return op.name


@pytest.mark.parametrize('path', PROGRAMS, ids=lambda p: p.stem)
def test_functions_save_every_callee_saved_register_they_write(path):
    for fn in _asm_program(path).functions:
        instrs = fn.instructions
        assert instrs[:2] == [Push(Register('rbp')), MovQ(src=Register('rsp'), dst=Register('rbp'))], fn.name
        pushed = []
        for instr in instrs[2:]:
            if not isinstance(instr, Push):
                break
            pushed.append(instr.operand.name if hasattr(instr, 'operand') else instr.reg.name)
        written = set()
        for instr in instrs[2 + len(pushed):]:
            if isinstance(instr, Pop):
                continue
            for f in fields(instr):
                if f.name in ('dst', 'operand'):
                    written.add(_qword(getattr(instr, f.name)))
        assert written & set(CALLEE_SAVED_REGISTERS) <= set(pushed), fn.name
        expected_tail = ([LeaQFrame(offset=-8 * len(pushed), dst=Register('rsp'))] if pushed else []) + \
            [Pop(Register(r)) for r in reversed(pushed)] + [Leave(), Ret()]
        for i, instr in enumerate(instrs):
            if isinstance(instr, Ret):
                assert instrs[i + 1 - len(expected_tail):i + 1] == expected_tail, fn.name


_HORNET = (
    "def int leaf(int x):\n"
    "    return x * 3 + 1\n"
    "\n"
    "def int work(int n):\n"
    "    int a = 1\n"
    "    int b = 2\n"
    "    int c = 3\n"
    "    int d = 4\n"
    "    int e = 5\n"
    "    int f = 6\n"
    "    for int i = 0; i < n; i += 1:\n"
    "        a = a + leaf(b)\n"
    "        b = b ^ leaf(c)\n"
    "        c = c + leaf(d)\n"
    "        d = d - leaf(e)\n"
    "        e = e + leaf(f)\n"
    "        f = f ^ leaf(a)\n"
    "    return (a + b + c + d + e + f) & 255\n"
)

# Sets rbx and r12-r15, calls work(50), exits 255 if any changed, else work's result.
_HARNESS = """    .globl main
main:
    pushq   %rbp
    movq    %rsp, %rbp
    pushq   %rbx
    pushq   %r12
    pushq   %r13
    pushq   %r14
    pushq   %r15
    subq    $8, %rsp
    movabsq $0x1111111111111111, %rbx
    movabsq $0x2222222222222222, %r12
    movabsq $0x3333333333333333, %r13
    movabsq $0x4444444444444444, %r14
    movabsq $0x5555555555555555, %r15
    movq    $50, %rdi
    call    work
    movq    %rax, %rsi
    movabsq $0x1111111111111111, %rax
    cmpq    %rax, %rbx
    jne     .Lfail
    movabsq $0x2222222222222222, %rax
    cmpq    %rax, %r12
    jne     .Lfail
    movabsq $0x3333333333333333, %rax
    cmpq    %rax, %r13
    jne     .Lfail
    movabsq $0x4444444444444444, %rax
    cmpq    %rax, %r14
    jne     .Lfail
    movabsq $0x5555555555555555, %rax
    cmpq    %rax, %r15
    jne     .Lfail
    movq    %rsi, %rax
    jmp     .Ldone
.Lfail:
    movl    $255, %eax
.Ldone:
    addq    $8, %rsp
    popq    %r15
    popq    %r14
    popq    %r13
    popq    %r12
    popq    %rbx
    popq    %rbp
    ret
    .section .note.GNU-stack,"",@progbits
"""


@GCC_SKIP
@pytest.mark.skipif(sys.platform != 'linux', reason='harness is written for Linux symbol names')
def test_c_convention_caller_keeps_callee_saved_registers(tmp_path):
    src = tmp_path / 'work.ht'
    src.write_text(_HORNET)
    (tmp_path / 'work.s').write_text(compile_to_asm(str(src), target='x86_64-linux'), encoding='latin-1')
    (tmp_path / 'harness.s').write_text(_HARNESS)
    exe = tmp_path / 'abi'
    subprocess.run(['gcc', str(tmp_path / 'harness.s'), str(tmp_path / 'work.s'), '-o', str(exe)], check=True)
    assert subprocess.run([str(exe)]).returncode == 114  # computed by a Python model; 255 means clobbered
