"""A division or a memory read whose result is never used is not compiled (optimize/dead_code.py),
but the checks before it are: the program still panics where it would have."""

import pytest

from ir.ir import IRBinOp, IRBoundsCheck, IRLoad
from ir.program_builder import build_ir_program
from ops import BinaryOp
from optimize.optimizer import optimize
from tests.test_compiler import GCC_SKIP, _parse, analyze, assert_panics, assert_stdout

SETUP = (
    "    []int xs = [1, 2, 3]\n"
    "    int zero = len(xs) - 3\n"
    "    int five = len(xs) + 2\n"
    "    *int p = none\n"
)


@GCC_SKIP
@pytest.mark.parametrize("statement,message", [
    ("10 / zero", "integer division by zero"),
    ("10 % zero", "integer division by zero"),
    ("int unused = 10 / zero", "integer division by zero"),
    ("int unused = xs[five]", "index out of bounds: index 5, length 3"),
    ("xs[five]", "index out of bounds: index 5, length 3"),
    ("*p", "dereference of none"),
    ("int unused = *p", "dereference of none"),
])
def test_an_unused_result_still_panics(statement, message):
    assert_panics(SETUP + f"    {statement}\n    return 0", message)


@GCC_SKIP
def test_unused_results_change_nothing_else():
    assert_stdout(
        SETUP +
        "    int unused = xs[1] / five\n"
        "    xs[2] % five\n"
        "    print(xs)\n"
        "    return 0",
        "[]int[1, 2, 3]\n",
    )


def _optimized(function: str):
    """(the optimized body of `function`, the program's static strings)."""
    typed_program = analyze(_parse(function + "def int main():\n    return 0\n"))
    program = optimize(build_ir_program(typed_program, keep_unreachable=True))  # (main doesn't call it)
    return program.functions[0].body, "".join(text for _, text in program.string_literals)


def test_the_unused_division_is_gone_and_its_checks_stay():
    body, strings = _optimized("def int f(int a, int d):\n    int unused = a / d\n    return 1\n")
    assert not any(isinstance(i, IRBinOp) and i.op == BinaryOp.DIVIDE for i in body)
    assert "integer division by zero" in strings and "integer overflow in division" in strings


def test_the_unused_load_is_gone_and_its_check_stays():
    body, strings = _optimized("def int f(*int p):\n    int unused = *p\n    return 1\n")
    assert not any(isinstance(i, IRLoad) for i in body)
    assert "dereference of none" in strings


def test_an_unused_element_keeps_its_bounds_check():
    body, _ = _optimized("def int f([]int xs):\n    int unused = xs[1]\n    return 1\n")
    assert any(isinstance(i, IRBoundsCheck) for i in body)
