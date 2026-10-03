"""x86-64: incoming parameters and outgoing arguments move between registers as parallel moves, since
the argument registers are allocatable too."""

import pytest

from backend.x86_64.assembly_ast import MovQ, Register
from backend.x86_64.ir_lowering import InstructionSelector
from tests.test_compiler import GCC_SKIP, assert_program_stdout


def _simulate(moves: list, start: dict) -> dict:
    regs = dict(start)
    for m in InstructionSelector._parallel_moves(moves):
        assert isinstance(m, MovQ) and isinstance(m.src, Register) and isinstance(m.dst, Register)
        regs[m.dst.name] = regs[m.src.name]
    return regs


@pytest.mark.parametrize('moves', [
    [('rdi', 'rsi'), ('rsi', 'rdi')],  # a swap
    [('rdi', 'rsi'), ('rsi', 'r8'), ('r8', 'rdi')],  # a three-way cycle
    [('r10', 'rdi'), ('rdi', 'rsi'), ('rsi', 'r9')],  # a chain
    [('rdi', 'rsi'), ('rdi', 'r8'), ('rsi', 'rdi'), ('r11', 'r11')],  # a fan-out inside a cycle, and a no-op
])
def test_parallel_moves_give_every_destination_its_sources_old_value(moves):
    start = {r: r for r in ('rdi', 'rsi', 'r8', 'r9', 'r10', 'r11', 'rax')}
    regs = _simulate(moves, start)
    for src, dst in moves:
        assert regs[dst] == src


@GCC_SKIP
def test_arguments_and_parameters_permuted_across_argument_registers():
    """Calls that pass parameters on in other orders (rotations and swaps through the argument
    registers), with more than six arguments so some go on the stack."""
    assert_program_stdout(
        "def int f(int a, int b, int c, int d, int e, int f, int g, int h):\n"
        "    return a * 10000000 + b * 1000000 + c * 100000 + d * 10000 + e * 1000 + f * 100 + g * 10 + h\n"
        "def int rotate(int a, int b, int c, int d, int e, int f, int g, int h):\n"
        "    return f(b, c, d, e, f, a, h, g)\n"
        "def int swap(int a, int b):\n"
        "    return a * 10 + b\n"
        "def int twice(int a, int b):\n"
        "    return swap(b, a) * 100 + swap(a, b)\n"
        "def int main():\n"
        "    print(rotate(1, 2, 3, 4, 5, 6, 7, 8))\n"
        "    print(twice(3, 4))\n"
        "    return 0\n",
        "23456187\n4334\n",
    )
