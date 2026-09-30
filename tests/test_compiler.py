"""End-to-end and semantic tests for the whole compiler pipeline.

Execution tests compile a small program, build it for every E2E target (tests/targets.py), run it,
and check its exit status or output; every target must agree. Semantic-error tests stop after
analysis and need no toolchain.
"""
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytest

from compile import generate_asm
from backend.errors import CodegenError
from ir.errors import IRError
from build import RUNTIME_C_PATH, c_compiler
from target import default_target
from tests.targets import E2E_TARGETS, on_every_target, run_binary
from desugar import desugar_methods
from lexer import lex
from parser import Break, Call, Constant, Continue, For, ForIn, Node, Parser, ParseError
from semantic import SemanticError, analyze as _semantic_analyze


def analyze(program):
    """Semantic analysis after desugaring, as compile_to_asm does."""
    desugar_methods(program)
    _semantic_analyze(program)


GCC_AVAILABLE = shutil.which("gcc") is not None
GCC_SKIP = pytest.mark.skipif(
    not GCC_AVAILABLE,
    reason="gcc not found on PATH; these tests compile and execute real binaries",
)

HOST_IS_MACOS = sys.platform == "darwin"
ASM_TARGET = default_target()

EXECUTION_TIMEOUT = 5


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _parse(source: str):
    """Lex and parse only."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "program.lang"
        src_path.write_text(source)
        tokens = lex(str(src_path))
        return Parser(tokens).parse_program()


def _compile_to_binary(source: str, tmp: Path, target=ASM_TARGET) -> tuple[Path, str]:
    """Compile `source` for `target` and link it with the runtime in `tmp`; returns (binary, assembly)."""
    ast = _parse(source)
    analyze(ast)

    asm_path = tmp / "program.s"
    bin_path = tmp / "program"
    runtime_o_path = tmp / "runtime.o"

    asm = generate_asm(ast, target=target)
    asm_path.write_text(asm, encoding="latin-1")

    runtime_cc_cmd = c_compiler(target) + ["-c", str(RUNTIME_C_PATH), "-o", str(runtime_o_path)]
    runtime_result = subprocess.run(runtime_cc_cmd, capture_output=True, text=True)
    if runtime_result.returncode != 0:
        pytest.fail(
            "gcc failed to compile runtime.c.\n"
            f"command: {' '.join(runtime_cc_cmd)}\n"
            f"--- gcc stdout ---\n{runtime_result.stdout}\n"
            f"--- gcc stderr ---\n{runtime_result.stderr}\n"
        )

    gcc_cmd = c_compiler(target) + [str(asm_path), str(runtime_o_path), "-o", str(bin_path)]

    result = subprocess.run(gcc_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(
            "gcc failed to assemble/link the generated program.\n"
            f"command: {' '.join(gcc_cmd)}\n"
            f"--- gcc stdout ---\n{result.stdout}\n"
            f"--- gcc stderr ---\n{result.stderr}\n"
            f"--- generated assembly ---\n{asm}"
        )
    return bin_path, asm


def _run_binary(bin_path: Path, asm: str, target=ASM_TARGET) -> subprocess.CompletedProcess:
    """Run a built binary (under qemu for a foreign architecture), subject to EXECUTION_TIMEOUT."""
    try:
        return run_binary(
            target, [bin_path], timeout=EXECUTION_TIMEOUT,
            capture_output=True, encoding='latin-1',
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"Compiled program did not exit within {EXECUTION_TIMEOUT}s "
            "-- likely an infinite loop.\n"
            f"--- generated assembly ---\n{asm}"
        )
    except OSError as e:
        if HOST_IS_MACOS:
            pytest.fail(
                f"Failed to execute the compiled x86_64 binary ({e}). "
                "On Apple Silicon this usually means Rosetta 2 isn't "
                "installed -- try `softwareupdate --install-rosetta`."
            )
        raise


def _ir_program(ast):
    """The optimized IR for an analyzed AST."""
    from ir.program_builder import build_ir_program
    from optimize.optimizer import optimize
    return optimize(build_ir_program(ast))


def _heap_allocations(ast) -> list:
    """Sizes passed to malloc anywhere in the program's IR (None where not constant), in order."""
    from ir.ir import IRCall, IRConst
    return [c.args[0].value if isinstance(c.args[0], IRConst) else None
            for fn in _ir_program(ast).functions for c in fn.body
            if isinstance(c, IRCall) and c.name == 'malloc']


def compile_and_run(source: str, agree: bool = True) -> subprocess.CompletedProcess:
    """Build and run `source` for every E2E target; all must agree unless `agree` is False.
    Returns the first result."""
    def build_and_run(target):
        with tempfile.TemporaryDirectory() as tmpdir:
            bin_path, asm = _compile_to_binary(source, Path(tmpdir), target)
            return _run_binary(bin_path, asm, target)
    return on_every_target(build_and_run, source, agree)


def assert_exit_code(body: str, expected: int, return_type: str = "int") -> None:
    """Wrap `body` in `def <return_type> main():` and check the exit status."""
    source = f"def {return_type} main():\n{body}\n"
    result = compile_and_run(source)
    assert result.returncode == expected, (
        f"body:\n{body}\nexpected exit {expected}, got {result.returncode}"
    )


def assert_panics(body: str, message: str, return_type: str = "int") -> None:
    """Like assert_exit_code, but the program must panic (SIGABRT) with `message`."""
    source = f"def {return_type} main():\n{body}\n"
    result = compile_and_run(source)
    assert result.returncode == -signal.SIGABRT and message in result.stdout, (
        f"body:\n{body}\nexpected a panic with {message!r}, got exit {result.returncode}: {result.stdout!r}"
    )


def assert_crashes_with_sigabrt(body: str, return_type: str = "int") -> None:
    """Like assert_exit_code, but the program must die of SIGABRT (a runtime panic)."""
    source = f"def {return_type} main():\n{body}\n"
    result = compile_and_run(source)
    assert result.returncode == -signal.SIGABRT, (
        f"body:\n{body}\nexpected SIGABRT crash, got exit {result.returncode}"
    )


def assert_semantic_error(body: str, return_type: str = "int", match: str = None) -> None:
    """Wrap `body` in `main` and check that semantic analysis rejects it."""
    source = f"def {return_type} main():\n{body}\n"
    ast = _parse(source)
    with pytest.raises(SemanticError, match=match):
        analyze(ast)


def assert_program_exit_code(source: str, expected: int) -> None:
    """Like assert_exit_code, for a complete program."""
    result = compile_and_run(source)
    assert result.returncode == expected, (
        f"program:\n{source}\nexpected exit {expected}, got {result.returncode}"
    )


def assert_program_semantic_error(source: str, match: str = None) -> None:
    """Like assert_semantic_error, for a complete program."""
    ast = _parse(source)
    with pytest.raises(SemanticError, match=match):
        analyze(ast)


def assert_program_codegen_error(source: str, match: str = None) -> None:
    """A well-typed program that the IR builder or backend rejects."""
    ast = _parse(source)
    analyze(ast)
    with pytest.raises(CodegenError, match=match):
        generate_asm(ast, target=ASM_TARGET)


def assert_stdout(body: str, expected_stdout: str, return_type: str = "int") -> None:
    """Like assert_exit_code, but check what the program prints."""
    source = f"def {return_type} main():\n{body}\n"
    result = compile_and_run(source)
    assert result.stdout == expected_stdout, (
        f"body:\n{body}\nexpected stdout {expected_stdout!r}, got {result.stdout!r}"
    )


def assert_program_stdout(source: str, expected_stdout: str) -> None:
    """Like assert_stdout, for a complete program."""
    result = compile_and_run(source)
    assert result.stdout == expected_stdout, (
        f"program:\n{source}\nexpected stdout {expected_stdout!r}, got {result.stdout!r}"
    )


# ---------------------------------------------------------------------------
# Unary operators
# ---------------------------------------------------------------------------

class TestUnaryOperators:
    pytestmark = GCC_SKIP

    @pytest.mark.parametrize("expr,expected", [
        ("-2", 254),   # two's-complement wraparound: -2 -> 254
        ("~2", 253),   # ~2 == -3 == 253
        ("~-2", 1),    # ~(-2) == 1
        ("--2", 2),    # -(-2) == 2
    ])
    def test_int_unary_operator(self, expr, expected):
        assert_exit_code(f"    return {expr}", expected, return_type="int")

    @pytest.mark.parametrize("expr,expected", [
        ("not true", 0),
        ("not false", 1),
        ("not not true", 1),
    ])
    def test_bool_not(self, expr, expected):
        assert_exit_code(f"    return {expr}", expected, return_type="bool")


# ---------------------------------------------------------------------------
# Binary arithmetic
# ---------------------------------------------------------------------------

class TestBinaryArithmetic:
    pytestmark = GCC_SKIP

    @pytest.mark.parametrize("expr,expected", [
        ("1 + 2", 3),
        ("5 - 8", 253),          # -3 -> 253
        ("3 * 4", 12),
        ("20 / 4", 5),
        ("7 / 2", 3),             # integer division truncates
        ("1 + 2 * 3", 7),         # * binds tighter than +
        ("(1 + 2) * 3", 9),       # grouping overrides precedence
        ("10 - 2 - 3", 5),        # left-associative: (10-2)-3
        ("10 - (2 - 3)", 11),     # grouping overrides associativity
        ("20 / 5 / 2", 2),        # left-associative: (20/5)/2
        ("20 / (5 / 2)", 10),     # grouping overrides associativity
        ("2 + 3 * 4 - 5", 9),
        ("-2 + 3", 1),            # unary and binary minus disambiguation
        ("2 + -3", 255),          # 2 + (-3) == -1 -> 255
        ("-(2 + 3)", 251),        # -5 -> 251
        ("~2 + 1", 254),          # ~2 == -3; -3+1 == -2 -> 254
        ("~0 + ~0", 254),         # ~0 == -1; -1 + -1 == -2 -> 254
        ("2 * (3 + 4) - 5", 9),
    ])
    def test_binary_arithmetic(self, expr, expected):
        assert_exit_code(f"    return {expr}", expected, return_type="int")


# ---------------------------------------------------------------------------
# Modulo and the bitwise operators
# ---------------------------------------------------------------------------

class TestBitwiseAndModuloOperators:
    pytestmark = GCC_SKIP

    @pytest.mark.parametrize("expr,expected", [
        ("7 % 3", 1),
        ("17 % 5", 2),
        ("-7 % 3", 255),          # C-style truncating modulo: -7 % 3 == -1 -> 255
        ("12 & 10", 8),
        ("12 | 10", 14),
        ("12 ^ 10", 6),
        ("1 << 4", 16),
        ("256 >> 4", 16),
        ("-8 >> 1", 252),         # arithmetic (sign-preserving) shift: -8 >> 1 == -4 -> 252
        ("10 - 6 % 4", 8),        # % binds as tight as * / -- tighter than -
        ("1 + 1 << 2", 8),        # << is looser than + -- (1+1)<<2, not 1+(1<<2)
        ("5 & 3 | 8", 9),         # & binds tighter than |
        ("1 | 2 ^ 3 & 3", 1),     # & tightest of these three, then ^, then |
    ])
    def test_bitwise_and_modulo(self, expr, expected):
        assert_exit_code(f"    return {expr}", expected, return_type="int")

    def test_modulo_with_variables(self):
        assert_exit_code(
            "    int a = 17\n"
            "    int b = 5\n"
            "    return a % b",
            2,
        )

    def test_modulo_by_zero_panics(self):
        assert_panics("    int a = 5\n    int b = 0\n    return a % b", "integer division by zero")

    @pytest.mark.parametrize("type_,minimum", [("int", "-9223372036854775807 - 1"), ("int32", "int32(-2147483647) - int32(1)")])
    @pytest.mark.parametrize("op", ["/", "%"])
    def test_minimum_divided_by_minus_one_panics(self, type_, minimum, op):
        assert_panics(f"    {type_} a = {minimum}\n    {type_} b = {type_}(-1)\n    print(a {op} b)\n    return 0",
                      "integer overflow in division")

    @pytest.mark.parametrize("type_", ["int8", "uint8", "int32"])
    def test_narrow_division_by_zero_panics(self, type_):
        assert_panics(f"    {type_} a = {type_}(5)\n    {type_} b = {type_}(0)\n    print(a / b)\n    return 0",
                      "integer division by zero")

    def test_int8_minimum_divided_by_minus_one_wraps(self):
        assert_stdout("    int8 a = int8(-128)\n    int8 b = int8(-1)\n    print(a / b)\n    print(a % b)\n    return 0",
                      "-128\n0\n")

    def test_modulo_result_used_as_operand_of_plus(self):
        assert_exit_code(
            "    int x = 5 % 2 + 3\n"
            "    return x",
            4,
        )

    def test_modulo_in_a_loop_condition(self):
        assert_exit_code(
            "    int x = 0\n"
            "    int i = 0\n"
            "    while i < 10:\n"
            "        if i % 2 == 0:\n"
            "            x = x + 1\n"
            "        i = i + 1\n"
            "    return x",
            5,
        )

    def test_bitwise_and_combined_with_logical_and(self):
        assert_exit_code(
            "    int flags = 6\n"
            "    return (flags & 2) == 2 and (flags & 1) == 0",
            1,
            return_type="bool",
        )


# ---------------------------------------------------------------------------
# Comparisons
# ---------------------------------------------------------------------------

class TestComparisons:
    pytestmark = GCC_SKIP

    @pytest.mark.parametrize("expr,expected", [
        ("3 < 5", 1),
        ("5 < 3", 0),
        ("3 <= 3", 1),
        ("4 <= 3", 0),
        ("5 > 3", 1),
        ("3 > 5", 0),
        ("5 >= 5", 1),
        ("3 >= 5", 0),
        ("3 == 3", 1),
        ("3 == 4", 0),
        ("3 != 4", 1),
        ("3 != 3", 0),
    ])
    def test_comparison(self, expr, expected):
        assert_exit_code(f"    return {expr}", expected, return_type="bool")


# ---------------------------------------------------------------------------
# Precedence across every level, and 'and'/'or' chaining
# ---------------------------------------------------------------------------

class TestPrecedenceAndLogicalOperators:
    pytestmark = GCC_SKIP

    @pytest.mark.parametrize("expr,expected", [
        ("1 + 2 * 3 == 7 and 4 < 5", 1),
        ("1 + 2 * 3 == 8 and 4 < 5", 0),
        ("1 < 2 and 3 > 4 or 5 == 5", 1),
        ("1 + 1 == 2 and 3 > 2", 1),   # precedence: == and > both bind tighter than 'and'
        ("false or false or true", 1),
        ("false or false or false", 0),
        ("true and true and false", 0),
        ("true and true and true", 1),
    ])
    def test_precedence(self, expr, expected):
        assert_exit_code(f"    return {expr}", expected, return_type="bool")


# ---------------------------------------------------------------------------
# Short-circuit evaluation of 'and' / 'or'
# ---------------------------------------------------------------------------

class TestShortCircuitEvaluation:
    pytestmark = GCC_SKIP

    @pytest.mark.parametrize("expr,expected", [
        ("false and ((1 / 0) == 1)", 0),   # left is false -> right must be skipped
        ("true or ((1 / 0) == 1)", 1),     # left is true  -> right must be skipped
    ])
    def test_short_circuit_skips_crashing_side(self, expr, expected):
        assert_exit_code(f"    return {expr}", expected, return_type="bool")

    @pytest.mark.parametrize("expr", [
        "true and ((1 / 0) == 1)",   # left doesn't decide -> right MUST run
        "false or ((1 / 0) == 1)",   # left doesn't decide -> right MUST run
    ])
    def test_short_circuit_control_evaluates_when_needed(self, expr):
        assert_panics(f"    return {expr}", "integer division by zero", return_type="bool")


# ---------------------------------------------------------------------------
# Local variables
# ---------------------------------------------------------------------------

class TestVariablesAndStatements:
    pytestmark = GCC_SKIP

    def test_decl_and_assign_same_line(self):
        assert_exit_code(
            "    int a = 1\n"
            "    a = a + 1\n"
            "    return a",
            2,
        )

    def test_decl_and_assign_split_across_lines(self):
        assert_exit_code(
            "    int a\n"
            "    a = 1\n"
            "    a = a + 1\n"
            "    return a",
            2,
        )

    def test_standalone_expression_statement(self):
        assert_exit_code(
            "    2 + 2\n"
            "    return 0",
            0,
        )

    def test_two_variables(self):
        assert_exit_code(
            "    int a = 3\n"
            "    int b = 4\n"
            "    return a + b",
            7,
        )

    def test_variables_in_complex_expression(self):
        assert_exit_code(
            "    int a = 2\n"
            "    int b = 3\n"
            "    return a * b + 1",
            7,
        )

    def test_reassignment_chain(self):
        assert_exit_code(
            "    int a = 1\n"
            "    a = a + 1\n"
            "    a = a + 1\n"
            "    a = a * 10\n"
            "    return a",
            30,
        )

    def test_declare_without_initializer_then_assign(self):
        assert_exit_code(
            "    int a\n"
            "    a = 5\n"
            "    return a",
            5,
        )

    def test_five_variables_frame_size(self):
        assert_exit_code(
            "    int a = 1\n"
            "    int b = 2\n"
            "    int c = 3\n"
            "    int d = 4\n"
            "    int e = 5\n"
            "    return a + b + c + d + e",
            15,
        )

    def test_variable_in_comparison(self):
        assert_exit_code(
            "    int a = 5\n"
            "    int b = 3\n"
            "    return a > b",
            1,
            return_type="bool",
        )

    def test_variable_in_short_circuit(self):
        assert_exit_code(
            "    bool a = false\n"
            "    return a and ((1 / 0) == 1)",
            0,
            return_type="bool",
        )

    def test_standalone_expression_statement_actually_executes(self):
        assert_panics(
            "    1 / 0\n"
            "    return 0",
            "integer division by zero",
        )

    def test_standalone_safe_expression_does_not_crash(self):
        assert_exit_code(
            "    1 + 1\n"
            "    return 42",
            42,
        )


# ---------------------------------------------------------------------------
# Compound assignment
# ---------------------------------------------------------------------------

class TestCompoundAssignment:
    pytestmark = GCC_SKIP

    @pytest.mark.parametrize("body,expected", [
        ("    int x = 5\n    x += 3\n    return x", 8),
        ("    int x = 5\n    x -= 3\n    return x", 2),
        ("    int x = 5\n    x *= 3\n    return x", 15),
        ("    int x = 20\n    x /= 4\n    return x", 5),
        ("    int x = 17\n    x %= 5\n    return x", 2),
        ("    int x = 12\n    x &= 10\n    return x", 8),
        ("    int x = 12\n    x |= 10\n    return x", 14),
        ("    int x = 12\n    x ^= 10\n    return x", 6),
        ("    int x = 1\n    x <<= 4\n    return x", 16),
        ("    int x = 256\n    x >>= 4\n    return x", 16),
    ])
    def test_each_compound_operator(self, body, expected):
        assert_exit_code(body, expected)

    def test_chained_compound_assignments(self):
        assert_exit_code(
            "    int x = 5\n"
            "    x += 3\n"   # 8
            "    x -= 1\n"   # 7
            "    x *= 2\n"   # 14
            "    x /= 2\n"   # 7
            "    x %= 5\n"   # 2
            "    x &= 3\n"   # 2
            "    x |= 4\n"   # 6
            "    x ^= 1\n"   # 7
            "    x <<= 2\n"  # 28
            "    x >>= 1\n"  # 14
            "    return x",
            14,
        )

    def test_string_concat_via_plus_equals(self):
        assert_exit_code(
            "    str s = 'hello'\n"
            "    s += ' world'\n"
            "    return s == 'hello world'",
            1,
            return_type="bool",
        )

    def test_compound_assignment_in_a_loop(self):
        assert_exit_code(
            "    int total = 0\n"
            "    int i = 1\n"
            "    while i <= 5:\n"
            "        total += i\n"
            "        i += 1\n"
            "    return total",
            15,
        )

    def test_compound_assignment_type_mismatch_is_rejected(self):
        assert_semantic_error(
            "    bool b = true\n"
            "    b += 1\n"
            "    return 0",
            match="requires two operands of the same integer type",
        )

    def test_compound_assignment_to_undeclared_variable_is_rejected(self):
        assert_semantic_error(
            "    undeclared_var += 1\n"
            "    return 0",
            match="undeclared variable",
        )

    def test_modulo_assign_by_zero_panics(self):
        assert_panics(
            "    int a = 5\n"
            "    int zero = 0\n"
            "    a %= zero\n"
            "    return a",
            "integer division by zero",
        )


# ---------------------------------------------------------------------------
# if / elif / else
# ---------------------------------------------------------------------------

class TestIfStatements:
    pytestmark = GCC_SKIP

    def test_if_true_branch_taken(self):
        assert_exit_code(
            "    int a = 1\n"
            "    if a == 1:\n"
            "        return true\n"
            "    else:\n"
            "        return false",
            1,
            return_type="bool",
        )

    def test_if_false_branch_taken(self):
        assert_exit_code(
            "    int a = 2\n"
            "    if a == 1:\n"
            "        return true\n"
            "    else:\n"
            "        return false",
            0,
            return_type="bool",
        )

    def test_if_no_else_condition_false_falls_through(self):
        assert_exit_code(
            "    int a = 0\n"
            "    if a == 1:\n"
            "        a = 99\n"
            "    return a",
            0,
        )

    def test_if_no_else_condition_true_body_runs(self):
        assert_exit_code(
            "    int a = 1\n"
            "    if a == 1:\n"
            "        a = 99\n"
            "    return a",
            99,
        )

    @pytest.mark.parametrize("a,expected", [
        (1, 10),   # first branch matches
        (5, 20),   # second branch matches
        (9, 30),   # third branch matches
        (100, 40),  # falls through to else
    ])
    def test_elif_chain(self, a, expected):
        assert_exit_code(
            f"    int a = {a}\n"
            "    if a == 1:\n"
            "        return 10\n"
            "    elif a == 5:\n"
            "        return 20\n"
            "    elif a == 9:\n"
            "        return 30\n"
            "    else:\n"
            "        return 40",
            expected,
        )

    def test_nested_if_in_if_inner_false(self):
        assert_exit_code(
            "    if true:\n"
            "        if false:\n"
            "            return 1\n"
            "        return 2\n"
            "    return 3",
            2,
        )

    def test_nested_if_in_if_outer_false(self):
        assert_exit_code(
            "    if false:\n"
            "        if true:\n"
            "            return 1\n"
            "        return 2\n"
            "    return 3",
            3,
        )

    def test_same_name_in_both_branches_then(self):
        assert_exit_code(
            "    if true:\n"
            "        int a = 1\n"
            "        return a\n"
            "    else:\n"
            "        int a = 2\n"
            "        return a",
            1,
        )

    def test_same_name_in_both_branches_else(self):
        assert_exit_code(
            "    if false:\n"
            "        int a = 1\n"
            "        return a\n"
            "    else:\n"
            "        int a = 2\n"
            "        return a",
            2,
        )

    def test_shadowing_outer_variable_returns_inner_value(self):
        assert_exit_code(
            "    int a = 100\n"
            "    if true:\n"
            "        int a = 5\n"
            "        return a\n"
            "    return a",
            5,
        )

    def test_outer_variable_unaffected_after_shadowing_block_ends(self):
        assert_exit_code(
            "    int a = 100\n"
            "    if true:\n"
            "        int a = 5\n"
            "    return a",
            100,
        )

    def test_assignment_to_outer_variable_from_inside_if(self):
        assert_exit_code(
            "    int a = 1\n"
            "    if true:\n"
            "        a = 2\n"
            "    return a",
            2,
        )

    def test_early_return_from_then_branch(self):
        assert_exit_code(
            "    if true:\n"
            "        return 1\n"
            "    return 2",
            1,
        )

    def test_if_condition_using_and_and_comparisons(self):
        assert_exit_code(
            "    int a = 5\n"
            "    int b = 3\n"
            "    if a > b and b > 0:\n"
            "        return 1\n"
            "    return 0",
            1,
        )

    def test_two_separate_if_blocks_each_declare_their_own_variable(self):
        assert_exit_code(
            "    int x = 1\n"
            "    if true:\n"
            "        int y = 2\n"
            "        x = x + y\n"
            "    if true:\n"
            "        int y = 10\n"
            "        x = x + y\n"
            "    return x",
            13,
        )


# ---------------------------------------------------------------------------
# while / break / continue
# ---------------------------------------------------------------------------

class TestWhileLoops:
    pytestmark = GCC_SKIP

    def test_counts_to_five(self):
        assert_exit_code(
            "    int i = 0\n"
            "    while i < 5:\n"
            "        i = i + 1\n"
            "    return i",
            5,
        )

    def test_condition_false_immediately_zero_iterations(self):
        assert_exit_code(
            "    int i = 10\n"
            "    while i < 5:\n"
            "        i = i + 1\n"
            "    return i",
            10,
        )

    def test_break_exits_immediately(self):
        assert_exit_code(
            "    int i = 0\n"
            "    while true:\n"
            "        i = i + 1\n"
            "        if i == 3:\n"
            "            break\n"
            "    return i",
            3,
        )

    def test_continue_skips_specific_iterations(self):
        assert_exit_code(
            "    int i = 0\n"
            "    int sum = 0\n"
            "    while i < 5:\n"
            "        i = i + 1\n"
            "        if i == 2 or i == 4:\n"
            "            continue\n"
            "        sum = sum + i\n"
            "    return sum",
            9,
        )

    def test_nested_loops_break_only_exits_innermost(self):
        assert_exit_code(
            "    int count = 0\n"
            "    int i = 0\n"
            "    while i < 3:\n"
            "        int j = 0\n"
            "        while j < 3:\n"
            "            if j == 1:\n"
            "                break\n"
            "            count = count + 1\n"
            "            j = j + 1\n"
            "        i = i + 1\n"
            "    return count",
            3,
        )

    def test_nested_loops_continue_only_affects_innermost(self):
        assert_exit_code(
            "    int total = 0\n"
            "    int i = 0\n"
            "    while i < 3:\n"
            "        int j = 0\n"
            "        while j < 3:\n"
            "            j = j + 1\n"
            "            if j == 2:\n"
            "                continue\n"
            "            total = total + 1\n"
            "        i = i + 1\n"
            "    return total",
            6,
        )

    def test_variable_declared_inside_loop_body_reused_each_iteration(self):
        assert_exit_code(
            "    int total = 0\n"
            "    int i = 0\n"
            "    while i < 4:\n"
            "        int doubled = i * 2\n"
            "        total = total + doubled\n"
            "        i = i + 1\n"
            "    return total",
            12,  # 0 + 2 + 4 + 6
        )

    def test_while_condition_using_and(self):
        assert_exit_code(
            "    int i = 0\n"
            "    int j = 10\n"
            "    while i < 5 and j > 0:\n"
            "        i = i + 1\n"
            "        j = j - 1\n"
            "    return i",
            5,
        )

    def test_early_return_from_inside_while(self):
        assert_exit_code(
            "    int i = 0\n"
            "    while true:\n"
            "        i = i + 1\n"
            "        if i == 7:\n"
            "            return i\n"
            "    return 0",
            7,
        )

    def test_if_followed_by_while_mixed_control_flow(self):
        assert_exit_code(
            "    int a = 5\n"
            "    if a > 0:\n"
            "        a = a + 1\n"
            "    int i = 0\n"
            "    while i < a:\n"
            "        i = i + 1\n"
            "    return i",
            6,
        )


class TestForLoops:
    pytestmark = GCC_SKIP

    def test_counts_to_ten(self):
        assert_exit_code(
            "    int total = 0\n"
            "    for int i = 0; i < 10; i += 1:\n"
            "        total = total + i\n"
            "    return total",
            sum(range(10)),
        )

    def test_condition_false_immediately_zero_iterations(self):
        assert_exit_code(
            "    int count = 0\n"
            "    for int i = 10; i < 5; i += 1:\n"
            "        count = count + 1\n"
            "    return count",
            0,
        )

    def test_break_exits_immediately(self):
        assert_exit_code(
            "    int last = -1\n"
            "    for int i = 0; i < 10; i += 1:\n"
            "        if i == 3:\n"
            "            break\n"
            "        last = i\n"
            "    return last",
            2,
        )

    def test_continue_still_runs_the_increment(self):
        assert_exit_code(
            "    int total = 0\n"
            "    for int i = 0; i < 10; i += 1:\n"
            "        if i % 2 == 0:\n"
            "            continue\n"
            "        total = total + i\n"
            "    return total",
            sum(i for i in range(10) if i % 2 != 0),
        )

    def test_nested_for_loops(self):
        assert_exit_code(
            "    int count = 0\n"
            "    for int i = 0; i < 3; i += 1:\n"
            "        for int j = 0; j < 3; j += 1:\n"
            "            count = count + 1\n"
            "    return count",
            9,
        )

    def test_nested_for_loops_break_only_exits_innermost(self):
        assert_exit_code(
            "    int count = 0\n"
            "    for int i = 0; i < 3; i += 1:\n"
            "        for int j = 0; j < 10; j += 1:\n"
            "            if j == 1:\n"
            "                break\n"
            "            count = count + 1\n"
            "    return count",
            3,
        )

    def test_struct_typed_counter(self):
        assert_program_exit_code(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def int main():\n"
            "    int total = 0\n"
            "    for Point p = Point(0, 0); p.x < 3; p = Point(p.x + 1, 0):\n"
            "        total = total + p.x\n"
            "    return total\n",
            0 + 1 + 2,
        )

    def test_increment_accepts_plain_assignment_too(self):
        assert_exit_code(
            "    int total = 0\n"
            "    for int i = 0; i < 5; i = i + 1:\n"
            "        total = total + i\n"
            "    return total",
            sum(range(5)),
        )

    def test_loop_variable_does_not_leak_past_the_loop(self):
        assert_semantic_error(
            "    for int i = 0; i < 10; i += 1:\n"
            "        print(i)\n"
            "    print(i)\n"
            "    return 0",
            match="undeclared variable 'i'",
        )

    def test_non_bool_condition_is_rejected(self):
        assert_semantic_error(
            "    for int i = 0; i; i += 1:\n"
            "        print(i)\n"
            "    return 0",
            match="'for' condition must be bool, got int",
        )

    def test_init_clause_must_be_a_var_decl(self):
        with pytest.raises(ParseError, match="Expected a variable declaration"):
            _parse(
                "def int main():\n"
                "    int i = 0\n"
                "    for i = 0; i < 10; i += 1:\n"
                "        print(i)\n"
                "    return 0\n"
            )

    def test_increment_clause_must_be_an_assignment(self):
        with pytest.raises(ParseError, match="Expected an assignment"):
            _parse(
                "def int main():\n"
                "    [3]int arr = [1, 2, 3]\n"
                "    for int i = 0; i < 3; arr[i] = 5:\n"
                "        print(i)\n"
                "    return 0\n"
            )

    def test_break_and_continue_still_rejected_outside_any_loop(self):
        assert_semantic_error("    break\n    return 0", match="'break' outside of a loop")
        assert_semantic_error("    continue\n    return 0", match="'continue' outside of a loop")


# ---------------------------------------------------------------------------
# str
# ---------------------------------------------------------------------------

class TestStrings:
    pytestmark = GCC_SKIP

    def test_equal_string_literals_compare_equal(self):
        assert_exit_code(
            "    str a = 'hello'\n"
            "    str b = 'hello'\n"
            "    return a == b",
            1,
            return_type="bool",
        )

    def test_different_string_literals_compare_unequal(self):
        assert_exit_code(
            "    str a = 'hello'\n"
            "    str b = 'world'\n"
            "    return a == b",
            0,
            return_type="bool",
        )

    def test_not_equal_on_different_strings(self):
        assert_exit_code(
            "    str a = 'hello'\n"
            "    str b = 'world'\n"
            "    return a != b",
            1,
            return_type="bool",
        )

    def test_basic_concatenation(self):
        assert_exit_code(
            "    str a = 'foo'\n"
            "    str b = 'bar'\n"
            "    return (a + b) == 'foobar'",
            1,
            return_type="bool",
        )

    def test_concatenation_of_two_literals_directly(self):
        assert_exit_code(
            "    return ('foo' + 'bar') == 'foobar'",
            1,
            return_type="bool",
        )

    def test_chained_concatenation_of_three_strings(self):
        assert_exit_code(
            "    str a = 'a'\n"
            "    str b = 'b'\n"
            "    str c = 'c'\n"
            "    return (a + b + c) == 'abc'",
            1,
            return_type="bool",
        )

    def test_concatenation_result_reused_in_later_concatenation(self):
        assert_exit_code(
            "    str a = 'hello'\n"
            "    str greeting = a + ', world'\n"
            "    str full = greeting + '!'\n"
            "    return full == 'hello, world!'",
            1,
            return_type="bool",
        )

    def test_escape_sequence_matches_manual_construction(self):
        assert_exit_code(
            "    str a = 'line1\\nline2'\n"
            "    str b = 'line1' + '\\n' + 'line2'\n"
            "    return a == b",
            1,
            return_type="bool",
        )

    def test_escaped_quote_in_string_literal(self):
        assert_exit_code(
            "    str a = 'it\\'s here'\n"
            "    return a == 'it\\'s here'",
            1,
            return_type="bool",
        )

    def test_string_equality_as_if_condition(self):
        assert_exit_code(
            "    str a = 'hello'\n"
            "    if a == 'hello':\n"
            "        return 1\n"
            "    return 0",
            1,
        )

    def test_string_equality_driving_a_while_loop(self):
        assert_exit_code(
            "    str target = 'stop'\n"
            "    str current = 'go'\n"
            "    int count = 0\n"
            "    while current != target:\n"
            "        count = count + 1\n"
            "        if count == 3:\n"
            "            current = 'stop'\n"
            "        else:\n"
            "            current = current + 'x'\n"
            "    return count",
            3,
        )

    def test_reassigning_a_str_variable(self):
        assert_exit_code(
            "    str a = 'first'\n"
            "    a = 'second'\n"
            "    return a == 'second'",
            1,
            return_type="bool",
        )

    def test_str_int_bool_locals_coexisting(self):
        assert_exit_code(
            "    int x = 5\n"
            "    str s = 'test'\n"
            "    bool b = true\n"
            "    if s == 'test' and b and x == 5:\n"
            "        return 42\n"
            "    return 0",
            42,
        )


# ---------------------------------------------------------------------------
# String concatenation shapes
# ---------------------------------------------------------------------------

class TestStringConcatenationShapes:
    pytestmark = GCC_SKIP

    def test_basic_concat_no_fresh_operands(self):
        assert_exit_code(
            "    str a = 'foo'\n"
            "    str b = 'bar'\n"
            "    str c = a + b\n"
            "    return c == 'foobar'",
            1,
            return_type="bool",
        )

    def test_three_way_chain(self):
        assert_exit_code(
            "    str a = 'x'\n"
            "    str b = 'y'\n"
            "    str c = 'z'\n"
            "    str r = a + b + c\n"
            "    return r == 'xyz'",
            1,
            return_type="bool",
        )

    def test_deep_six_way_chain(self):
        assert_exit_code(
            "    str a = '1'\n"
            "    str b = '2'\n"
            "    str c = '3'\n"
            "    str d = '4'\n"
            "    str e = '5'\n"
            "    str f = '6'\n"
            "    str r = a + b + c + d + e + f\n"
            "    return r == '123456'",
            1,
            return_type="bool",
        )

    def test_named_variable_reused_across_two_concats(self):
        assert_exit_code(
            "    str a = 'shared'\n"
            "    str b = a + '_first'\n"
            "    str c = a + '_second'\n"
            "    return b == 'shared_first' and c == 'shared_second'",
            1,
            return_type="bool",
        )

    def test_named_variable_reused_four_times(self):
        assert_exit_code(
            "    str base = 'X'\n"
            "    str r1 = base + '1'\n"
            "    str r2 = base + '2'\n"
            "    str r3 = base + '3'\n"
            "    str r4 = base + '4'\n"
            "    return r1 == 'X1' and r2 == 'X2' and r3 == 'X3' and r4 == 'X4'",
            1,
            return_type="bool",
        )

    def test_two_literal_operands(self):
        assert_exit_code(
            "    str r = 'lit1' + 'lit2'\n"
            "    return r == 'lit1lit2'",
            1,
            return_type="bool",
        )

    def test_function_call_results_as_concat_operands(self):
        assert_program_exit_code(
            "def str make_a():\n"
            "    return 'aaa'\n"
            "\n"
            "def str make_b():\n"
            "    return 'bbb'\n"
            "\n"
            "def bool main():\n"
            "    str r = make_a() + make_b()\n"
            "    return r == 'aaabbb'\n",
            1,
        )

    def test_fresh_concat_compared_directly(self):
        assert_exit_code(
            "    str a = 'foo'\n"
            "    str b = 'bar'\n"
            "    return (a + b) == 'foobar'",
            1,
            return_type="bool",
        )

    def test_fresh_concat_on_both_sides_of_comparison(self):
        assert_exit_code(
            "    str a = 'x'\n"
            "    str b = 'y'\n"
            "    return (a + b) == ('x' + 'y')",
            1,
            return_type="bool",
        )

    def test_one_fresh_one_named_operand_in_same_concat(self):
        assert_exit_code(
            "    str a = 'p'\n"
            "    str b = 'q'\n"
            "    str c = 'r'\n"
            "    str result = (a + b) + c\n"
            "    return result == 'pqr'",
            1,
            return_type="bool",
        )

    def test_concat_result_stored_in_variable_reused_twice(self):
        assert_exit_code(
            "    str a = 'hello'\n"
            "    str b = 'world'\n"
            "    str combined = a + b\n"
            "    str r1 = combined + '!'\n"
            "    str r2 = combined + '?'\n"
            "    return r1 == 'helloworld!' and r2 == 'helloworld?'",
            1,
            return_type="bool",
        )

    def test_concatenation_with_fresh_intermediate_inside_a_loop(self):
        assert_exit_code(
            "    str result = ''\n"
            "    int i = 0\n"
            "    while i < 5:\n"
            "        result = result + 'a' + 'b'\n"
            "        i = i + 1\n"
            "    return result == 'ababababab'",
            1,
            return_type="bool",
        )


class TestStringRepresentation:
    """str as a {ptr, len} descriptor."""

    pytestmark = GCC_SKIP

    def test_embedded_null_byte_prints_in_full(self):
        assert_program_stdout(
            "def int main():\n"
            "    str s = 'hello\\0world'\n"
            "    print(s)\n"
            "    return 0\n",
            "hello\x00world\n",
        )

    def test_embedded_null_byte_does_not_affect_len(self):
        assert_program_exit_code(
            "def int main():\n"
            "    str s = 'hello\\0world'\n"
            "    return len(s)\n",
            expected=11,
        )

    def test_embedded_null_byte_does_not_affect_equality(self):
        assert_exit_code(
            "    str a = 'hi\\0one'\n"
            "    str b = 'hi\\0two'\n"
            "    return a != b",
            1,
            return_type="bool",
        )

    def test_len_of_a_str_variable(self):
        assert_exit_code(
            "    str s = 'hello'\n"
            "    return len(s)",
            5,
        )

    def test_len_of_a_concatenation(self):
        assert_exit_code(
            "    str a = 'foo'\n"
            "    str b = 'bar'\n"
            "    return len(a + b)",
            6,
        )

    def test_comparison_with_different_lengths_and_a_shared_prefix(self):
        assert_exit_code(
            "    str a = 'ab'\n"
            "    str b = 'abc'\n"
            "    return a != b",
            1,
            return_type="bool",
        )

    def test_comparison_where_the_longer_string_is_on_the_left(self):
        assert_exit_code(
            "    str a = 'abc'\n"
            "    str b = 'ab'\n"
            "    return a != b",
            1,
            return_type="bool",
        )

    def test_pointer_to_str_parameter_mutates_the_callers_variable(self):
        assert_program_exit_code(
            "def setGreeting(*str p):\n"
            "    *p = 'hello'\n"
            "\n"
            "def int main():\n"
            "    str s = 'unset'\n"
            "    setGreeting(&s)\n"
            "    if s == 'hello':\n"
            "        return 1\n"
            "    return 0\n",
            expected=1,
        )

    def test_pointer_to_str_local_outlives_the_function_that_declared_it(self):
        assert_program_exit_code(
            "def *str makeGreeting():\n"
            "    str s = 'hello'\n"
            "    return &s\n"
            "\n"
            "def int main():\n"
            "    *str p = makeGreeting()\n"
            "    if *p == 'hello':\n"
            "        return 1\n"
            "    return 0\n",
            expected=1,
        )

    def test_pointer_to_str_parameter_outlives_the_function(self):
        assert_program_exit_code(
            "def *str identity(str s):\n"
            "    return &s\n"
            "\n"
            "def int main():\n"
            "    *str p = identity('hello')\n"
            "    if *p == 'hello':\n"
            "        return 1\n"
            "    return 0\n",
            expected=1,
        )

    def test_struct_field_equality_with_embedded_null(self):
        assert_program_exit_code(
            "type Holder struct:\n"
            "    str s\n"
            "\n"
            "def int main():\n"
            "    Holder a = Holder('x\\0y')\n"
            "    Holder b = Holder('x\\0y')\n"
            "    Holder c = Holder('x\\0z')\n"
            "    if a == b and a != c:\n"
            "        return 1\n"
            "    return 0\n",
            expected=1,
        )

    def test_assign_an_existing_str_variable_from_an_ordinary_call(self):
        assert_program_exit_code(
            "def str makeGreeting():\n"
            "    return 'hello'\n"
            "\n"
            "def int main():\n"
            "    str s = 'unset'\n"
            "    s = makeGreeting()\n"
            "    if s == 'hello':\n"
            "        return 1\n"
            "    return 0\n",
            expected=1,
        )

    def test_field_assign_a_str_field_from_an_ordinary_call(self):
        assert_program_exit_code(
            "type Holder struct:\n"
            "    str s\n"
            "\n"
            "def str makeGreeting():\n"
            "    return 'hello'\n"
            "\n"
            "def int main():\n"
            "    Holder h = Holder('unset')\n"
            "    h.s = makeGreeting()\n"
            "    if h.s == 'hello':\n"
            "        return 1\n"
            "    return 0\n",
            expected=1,
        )

    def test_index_assign_a_str_element_from_an_ordinary_call(self):
        assert_program_exit_code(
            "def str makeGreeting():\n"
            "    return 'hello'\n"
            "\n"
            "def int main():\n"
            "    [2]str arr = ['a', 'b']\n"
            "    arr[0] = makeGreeting()\n"
            "    if arr[0] == 'hello':\n"
            "        return 1\n"
            "    return 0\n",
            expected=1,
        )


class TestStringSlicing:
    """`s[low:high]` on a str."""

    pytestmark = GCC_SKIP

    def test_basic_slice(self):
        assert_exit_code(
            "    str s = 'hello world'\n"
            "    return s[6:11] == 'world'",
            1,
            return_type="bool",
        )

    def test_omitted_low_bound(self):
        assert_exit_code(
            "    str s = 'hello world'\n"
            "    return s[:5] == 'hello'",
            1,
            return_type="bool",
        )

    def test_omitted_high_bound(self):
        assert_exit_code(
            "    str s = 'hello world'\n"
            "    return s[6:] == 'world'",
            1,
            return_type="bool",
        )

    def test_both_bounds_omitted(self):
        assert_exit_code(
            "    str s = 'hello world'\n"
            "    return s[:] == 'hello world'",
            1,
            return_type="bool",
        )

    def test_empty_slice(self):
        assert_exit_code(
            "    str s = 'hello'\n"
            "    return len(s[2:2])",
            0,
        )

    def test_slice_result_type_is_str_not_slice(self):
        assert_exit_code(
            "    str s = 'hello world'\n"
            "    str sub = s[0:5]\n"
            "    str greeting = sub + '!'\n"
            "    return greeting == 'hello!'",
            1,
            return_type="bool",
        )

    def test_slicing_a_string_literal_directly(self):
        assert_exit_code(
            "    return 'hello world'[6:11] == 'world'",
            1,
            return_type="bool",
        )

    def test_slicing_a_concatenation_result(self):
        assert_exit_code(
            "    return ('foo' + 'bar')[2:5] == 'oba'",
            1,
            return_type="bool",
        )

    def test_re_slicing_a_slice(self):
        assert_exit_code(
            "    str s = 'hello world'\n"
            "    str first = s[0:5]\n"
            "    str second = first[1:3]\n"
            "    return second == 'el'",
            1,
            return_type="bool",
        )

    def test_slicing_preserves_an_embedded_null_byte(self):
        assert_program_stdout(
            "def int main():\n"
            "    str s = 'hello\\0world'\n"
            "    print(s[3:8])\n"
            "    return 0\n",
            "lo\x00wo\n",
        )

    def test_high_exceeding_length_is_rejected_at_runtime(self):
        assert_program_stdout(
            "def int main():\n"
            "    str s = 'hello'\n"
            "    str bad = s[2:10]\n"
            "    print(bad)\n"
            "    return 0\n",
            "slice bounds out of range\n",
        )

    def test_low_greater_than_high_is_rejected_at_runtime(self):
        assert_program_stdout(
            "def int main():\n"
            "    str s = 'hello'\n"
            "    str bad = s[4:2]\n"
            "    print(bad)\n"
            "    return 0\n",
            "slice bounds out of range\n",
        )

    def test_struct_field_constructed_from_a_slice(self):
        assert_program_exit_code(
            "type Holder struct:\n"
            "    str s\n"
            "\n"
            "def int main():\n"
            "    str base = 'hello world'\n"
            "    Holder h = Holder(base[6:11])\n"
            "    if h.s == 'world':\n"
            "        return 1\n"
            "    return 0\n",
            expected=1,
        )

    def test_slicing_a_non_sliceable_type_is_rejected(self):
        assert_semantic_error(
            "    int x = 5\n"
            "    return len(x[0:1])",
            match="only arrays, slices, and str support slicing",
        )

    def test_slice_bound_must_be_int(self):
        assert_semantic_error(
            "    str s = 'hello'\n"
            "    return len(s[true:3])",
            match="Slice low bound must be int",
        )


class TestByteLiterals:
    """Double-quoted byte literals."""

    pytestmark = GCC_SKIP

    def test_basic_byte_literal_value(self):
        assert_program_stdout(
            "def int main():\n"
            "    byte b = \"a\"\n"
            "    print(b)\n"
            "    return 0\n",
            "97\n",
        )

    def test_print_shows_the_number_not_the_character(self):
        assert_program_stdout(
            "def int main():\n"
            "    print(\"A\")\n"
            "    return 0\n",
            "65\n",
        )

    def test_direct_comparison_against_a_byte_literal(self):
        assert_exit_code(
            "    byte b = \"a\"\n"
            "    return b == \"a\"",
            1,
            return_type="bool",
        )

    def test_byte_level_range_check_and_arithmetic(self):
        assert_program_stdout(
            "def int main():\n"
            "    byte a = \"a\"\n"
            "    if a >= \"a\" and a <= \"z\":\n"
            "        print('in range')\n"
            "    byte upper = a - \"a\" + \"A\"\n"
            "    print(upper)\n"
            "    return 0\n",
            "in range\n65\n",
        )

    def test_named_escape_sequences_still_work(self):
        assert_program_stdout(
            "def int main():\n"
            "    byte newline = \"\\n\"\n"
            "    print(newline)\n"
            "    byte tab = \"\\t\"\n"
            "    print(tab)\n"
            "    return 0\n",
            "10\n9\n",
        )

    def test_hex_escape_reaches_every_byte_value(self):
        assert_program_stdout(
            "def int main():\n"
            "    byte lo = \"\\x00\"\n"
            "    print(lo)\n"
            "    byte mid = \"\\x41\"\n"
            "    print(mid)\n"
            "    byte hi = \"\\xff\"\n"
            "    print(hi)\n"
            "    return 0\n",
            "0\n65\n255\n",
        )

    def test_hex_escape_case_insensitive_digits(self):
        assert_exit_code(
            "    byte a = \"\\x4a\"\n"
            "    byte b = \"\\x4A\"\n"
            "    return a == b",
            1,
            return_type="bool",
        )

    def test_empty_byte_literal_is_rejected(self):
        source = (
            "def int main():\n"
            "    byte b = \"\"\n"
            "    return 0\n"
        )
        with pytest.raises(ParseError, match="must resolve to exactly one byte"):
            _parse(source)

    def test_multi_character_byte_literal_is_rejected(self):
        source = (
            "def int main():\n"
            "    byte b = \"ab\"\n"
            "    return 0\n"
        )
        with pytest.raises(ParseError, match="must resolve to exactly one byte"):
            _parse(source)

    def test_hex_escape_in_a_string_literal_matches_len_to_actual_byte_count(self):
        assert_program_stdout(
            "def int main():\n"
            "    str s = 'high byte: \\xc8 end'\n"
            "    print(len(s))\n"
            "    print(s)\n"
            "    return 0\n",
            "16\nhigh byte: \xc8 end\n",
        )

    def test_malformed_hex_escape_falls_back_leniently(self):
        source = (
            "def int main():\n"
            "    byte b = \"\\xg1\"\n"
            "    return 0\n"
        )
        with pytest.raises(ParseError, match="must resolve to exactly one byte"):
            _parse(source)


class TestStringIndexing:
    """`s[i]` on a str."""

    pytestmark = GCC_SKIP

    def test_basic_index(self):
        assert_program_stdout(
            "def int main():\n"
            "    str s = 'hello world'\n"
            "    byte b = s[6]\n"
            "    print(b)\n"
            "    return 0\n",
            "119\n",
        )

    def test_result_compares_directly_against_a_byte_literal(self):
        assert_exit_code(
            "    str s = 'hello world'\n"
            "    return s[6] == \"w\"",
            1,
            return_type="bool",
        )

    def test_indexing_a_string_literal_directly(self):
        assert_program_stdout(
            "def int main():\n"
            "    print('hello'[1])\n"
            "    return 0\n",
            "101\n",
        )

    def test_indexing_a_concatenation_result(self):
        assert_program_stdout(
            "def int main():\n"
            "    print(('foo' + 'bar')[3])\n"
            "    return 0\n",
            "98\n",
        )

    def test_indexing_composes_with_slicing(self):
        assert_program_stdout(
            "def int main():\n"
            "    str s = 'hello world'\n"
            "    print(s[0:5][2])\n"
            "    return 0\n",
            "108\n",
        )

    def test_index_out_of_bounds_is_rejected_at_runtime(self):
        assert_program_stdout(
            "def int main():\n"
            "    str s = 'hello'\n"
            "    byte b = s[10]\n"
            "    print(b)\n"
            "    return 0\n",
            "array index out of bounds\n",
        )

    def test_index_assignment_into_a_str_is_rejected(self):
        assert_semantic_error(
            "    str s = 'hello'\n"
            "    s[0] = \"H\"\n"
            "    return 0",
            match="Cannot assign into a str via indexing",
        )

    def test_index_must_be_int(self):
        assert_semantic_error(
            "    str s = 'hello'\n"
            "    return s[true]",
            match="Index must be int",
        )

    def test_indexing_preserves_an_embedded_null_byte(self):
        assert_program_stdout(
            "def int main():\n"
            "    str s = 'hi\\0there'\n"
            "    print(s[2])\n"
            "    return 0\n",
            "0\n",
        )

    def test_indexing_a_non_str_non_array_non_slice_type_is_rejected(self):
        assert_semantic_error(
            "    int x = 5\n"
            "    return x[0]",
            match="only arrays, slices, str, and dict support indexing",
        )


# ---------------------------------------------------------------------------
# Function calls
# ---------------------------------------------------------------------------

class TestFunctions:
    pytestmark = GCC_SKIP

    def test_no_arg_function_call(self):
        assert_program_exit_code(
            "def int five():\n"
            "    return 5\n"
            "\n"
            "def int main():\n"
            "    return five()\n",
            5,
        )

    def test_two_arg_function_call(self):
        assert_program_exit_code(
            "def int add(int a, int b):\n"
            "    return a + b\n"
            "\n"
            "def int main():\n"
            "    return add(2, 3)\n",
            5,
        )

    def test_nested_calls(self):
        assert_program_exit_code(
            "def int inc(int x):\n"
            "    return x + 1\n"
            "\n"
            "def int main():\n"
            "    return inc(inc(5))\n",
            7,
        )

    def test_recursive_factorial(self):
        assert_program_exit_code(
            "def int fact(int n):\n"
            "    if n == 0:\n"
            "        return 1\n"
            "    return n * fact(n - 1)\n"
            "\n"
            "def int main():\n"
            "    return fact(5)\n",
            120,
        )

    def test_recursive_fibonacci(self):
        assert_program_exit_code(
            "def int fib(int n):\n"
            "    if n < 2:\n"
            "        return n\n"
            "    return fib(n - 1) + fib(n - 2)\n"
            "\n"
            "def int main():\n"
            "    return fib(10)\n",
            55,
        )

    def test_mutual_recursion_with_forward_reference(self):
        assert_program_exit_code(
            "def bool is_even(int n):\n"
            "    if n == 0:\n"
            "        return true\n"
            "    return is_odd(n - 1)\n"
            "\n"
            "def bool is_odd(int n):\n"
            "    if n == 0:\n"
            "        return false\n"
            "    return is_even(n - 1)\n"
            "\n"
            "def int main():\n"
            "    if is_even(10):\n"
            "        return 1\n"
            "    return 0\n",
            1,
        )

    def test_call_result_used_as_argument_to_another_call(self):
        assert_program_exit_code(
            "def int add(int a, int b):\n"
            "    return a + b\n"
            "\n"
            "def int main():\n"
            "    return add(add(1, 2), add(3, 4))\n",
            10,
        )

    def test_str_parameter_and_str_return_type(self):
        assert_program_exit_code(
            "def str greet(str name):\n"
            "    return 'hello, ' + name\n"
            "\n"
            "def bool main():\n"
            "    str result = greet('world')\n"
            "    return result == 'hello, world'\n",
            1,
        )

    def test_function_call_as_bare_statement(self):
        assert_program_exit_code(
            "def int side_effect():\n"
            "    return 99\n"
            "\n"
            "def int main():\n"
            "    side_effect()\n"
            "    return 42\n",
            42,
        )

    def test_register_preservation_across_nested_string_using_call(self):
        assert_program_exit_code(
            "def str inner_concat(str a, str b):\n"
            "    return a + b\n"
            "\n"
            "def bool outer(str x, str y, str z):\n"
            "    str first = x + y\n"
            "    str second = inner_concat(y, z)\n"
            "    return (first + second) == (x + y + y + z)\n"
            "\n"
            "def bool main():\n"
            "    return outer('a', 'b', 'c')\n",
            1,
        )

    def test_recursive_string_concatenation(self):
        assert_program_exit_code(
            "def str repeat(str s, int n):\n"
            "    if n == 0:\n"
            "        return ''\n"
            "    return s + repeat(s, n - 1)\n"
            "\n"
            "def bool main():\n"
            "    str result = repeat('x', 4)\n"
            "    return result == 'xxxx'\n",
            1,
        )

    def test_call_mixing_scalar_and_array_arguments(self):
        assert_program_exit_code(
            "def int sumWithBase([3]int arr, int base):\n"
            "    return arr[0] + arr[1] + arr[2] + base\n"
            "\n"
            "def int main():\n"
            "    [3]int nums = [1, 2, 3]\n"
            "    int base = 100\n"
            "    return sumWithBase(nums, base)\n",
            106,
        )

    def test_call_mixing_scalar_and_struct_arguments(self):
        assert_program_exit_code(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def int distFromOrigin(Point p, int scale):\n"
            "    return (p.x + p.y) * scale\n"
            "\n"
            "def int main():\n"
            "    Point p = Point(3, 4)\n"
            "    return distFromOrigin(p, 2)\n",
            14,
        )

    def test_reassignment_combining_self_reference_and_function_call(self):
        assert_program_exit_code(
            "def int addOne(int x):\n"
            "    return x + 1\n"
            "\n"
            "def int main():\n"
            "    int a = 5\n"
            "    int c = 10\n"
            "    c = c + addOne(a)\n"
            "    return c\n",
            16,
        )

    def test_more_than_six_parameters_now_works_via_the_stack(self):
        assert_program_exit_code(
            "def int seven(int a, int b, int c, int d, int e, int f, int g):\n"
            "    return a + b + c + d + e + f + g\n"
            "\n"
            "def int main():\n"
            "    return seven(1, 2, 3, 4, 5, 6, 7)\n",
            28,
        )

    def test_wide_typed_overflow_arguments(self):
        assert_program_exit_code(
            "def int64 f(int a, int b, int c, int d, int e, int64 g, int64 h):\n"
            "    return g + h\n"
            "\n"
            "def int main():\n"
            "    int64 result = f(1, 2, 3, 4, 5, 5000000000, 6000000000)\n"
            "    return int(result - int64(10999999998))\n",
            2,
        )

    def test_multiple_calls_needing_different_overflow_amounts_share_one_region(self):
        assert_program_exit_code(
            "def int eight(int a, int b, int c, int d, int e, int f, int g, int h):\n"
            "    return a + b + c + d + e + f + g + h\n"
            "\n"
            "def int ten(int a, int b, int c, int d, int e, int f, int g, int h, int i, int j):\n"
            "    return a + b + c + d + e + f + g + h + i + j\n"
            "\n"
            "def int main():\n"
            "    int x = eight(1, 1, 1, 1, 1, 1, 1, 1)\n"
            "    int y = ten(2, 2, 2, 2, 2, 2, 2, 2, 2, 2)\n"
            "    return x + y\n",
            8 + 20,
        )


# ---------------------------------------------------------------------------
# Functions with no declared return type
# ---------------------------------------------------------------------------

class TestFunctionsWithNoDeclaredReturnType:
    pytestmark = GCC_SKIP

    def test_falls_off_the_end_with_no_explicit_return_at_all(self):
        assert_program_stdout(
            "def log(str msg):\n"
            "    print(msg)\n"
            "\n"
            "def int main():\n"
            "    log('hello')\n"
            "    return 0\n",
            "hello\n",
        )

    def test_bare_return_exits_early(self):
        assert_program_stdout(
            "def log(int x):\n"
            "    if x < 0:\n"
            "        return\n"
            "    print(x)\n"
            "\n"
            "def int main():\n"
            "    log(-5)\n"
            "    log(42)\n"
            "    return 0\n",
            "42\n",
        )

    def test_no_parameters_and_no_return_type(self):
        assert_program_stdout(
            "def greet():\n"
            "    print('hi')\n"
            "\n"
            "def int main():\n"
            "    greet()\n"
            "    return 0\n",
            "hi\n",
        )

    def test_mixed_early_return_and_fall_through_paths(self):
        assert_program_stdout(
            "def classify(int x):\n"
            "    if x < 0:\n"
            "        print('negative')\n"
            "        return\n"
            "    if x == 0:\n"
            "        print('zero')\n"
            "        return\n"
            "    print('positive')\n"
            "\n"
            "def int main():\n"
            "    classify(-1)\n"
            "    classify(0)\n"
            "    classify(5)\n"
            "    return 0\n",
            "negative\nzero\npositive\n",
        )

    def test_void_function_calling_another_void_function(self):
        assert_program_stdout(
            "def inner():\n"
            "    print('inner')\n"
            "\n"
            "def outer():\n"
            "    print('outer')\n"
            "    inner()\n"
            "\n"
            "def int main():\n"
            "    outer()\n"
            "    return 0\n",
            "outer\ninner\n",
        )

    def test_recursive_void_function(self):
        assert_program_stdout(
            "def countdown(int n):\n"
            "    if n <= 0:\n"
            "        return\n"
            "    print(n)\n"
            "    countdown(n - 1)\n"
            "\n"
            "def int main():\n"
            "    countdown(3)\n"
            "    return 0\n",
            "3\n2\n1\n",
        )

    def test_while_loop_inside_a_void_function(self):
        assert_program_stdout(
            "def count_up(int n):\n"
            "    int i = 0\n"
            "    while i < n:\n"
            "        print(i)\n"
            "        i = i + 1\n"
            "\n"
            "def int main():\n"
            "    count_up(3)\n"
            "    return 0\n",
            "0\n1\n2\n",
        )

    def test_returning_a_value_from_a_void_function_is_rejected(self):
        assert_program_semantic_error(
            "def log(int x):\n"
            "    return x\n"
            "\n"
            "def int main():\n"
            "    return 0\n",
            match="cannot return a value",
        )

    def test_bare_return_inside_a_non_void_function_is_rejected(self):
        assert_semantic_error(
            "    return",
            match="bare 'return' returns nothing",
        )

    def test_using_a_void_call_result_as_a_value_is_rejected(self):
        source = (
            "def log(int x):\n"
            "    print(x)\n"
            "\n"
            "def int main():\n"
            "    int y = log(5)\n"
            "    return y\n"
        )
        ast = _parse(source)
        with pytest.raises(SemanticError, match="Cannot initialize"):
            analyze(ast)

    def test_non_void_function_still_requires_explicit_returns_on_every_path(self):
        source = (
            "def int classify(int x):\n"
            "    if x < 0:\n"
            "        return -1\n"
            "    print(x)\n"
            "\n"
            "def int main():\n"
            "    return classify(5)\n"
        )
        ast = _parse(source)
        with pytest.raises(SemanticError, match="does not return a value on all code paths"):
            analyze(ast)

    def test_comparing_two_void_call_results_is_rejected(self):
        source = (
            "def log(int x):\n"
            "    print(x)\n"
            "\n"
            "def bool main():\n"
            "    return log(1) == log(2)\n"
        )
        ast = _parse(source)
        with pytest.raises(SemanticError, match="does not support slice, void, sum type, dict, or none operands"):
            analyze(ast)


# ---------------------------------------------------------------------------
# Type annotation
# ---------------------------------------------------------------------------

class TestTypeAnnotation:
    pytestmark = GCC_SKIP

    def test_call_result_directly_as_operand_of_plus(self):
        assert_program_exit_code(
            "def int five():\n"
            "    return 5\n"
            "\n"
            "def int main():\n"
            "    int x = five() + 3\n"
            "    return x\n",
            8,
        )

    def test_modulo_result_directly_as_operand_of_plus(self):
        assert_exit_code(
            "    int x = 5 % 2 + 3\n"
            "    return x",
            4,
        )

    def test_deeply_nested_mixed_expression_annotates_and_executes_correctly(self):
        assert_program_exit_code(
            "def int add(int a, int b):\n"
            "    return a + b\n"
            "\n"
            "def bool main():\n"
            "    int n = add(1, 2) + (3 % 2)\n"
            "    bool b = (n > 0) and not ((5 & 2) == 2)\n"
            "    return b\n",
            1,
        )

    def test_codegen_without_semantic_analysis_raises_clear_error(self):
        ast = _parse("def int main():\n    return 1 + 2\n")
        with pytest.raises(IRError, match="no struct registry"):
            generate_asm(ast, target=ASM_TARGET)


# ---------------------------------------------------------------------------
# print
# ---------------------------------------------------------------------------

class TestPrint:
    pytestmark = GCC_SKIP

    def test_print_int(self):
        assert_stdout(
            "    print(5)\n"
            "    return 0",
            "5\n",
        )

    def test_print_negative_int(self):
        assert_stdout(
            "    print(-42)\n"
            "    return 0",
            "-42\n",
        )

    def test_print_str(self):
        assert_stdout(
            "    print('hello')\n"
            "    return 0",
            "hello\n",
        )

    def test_print_bool_true(self):
        assert_stdout(
            "    print(true)\n"
            "    return 0",
            "true\n",
        )

    def test_print_bool_false(self):
        assert_stdout(
            "    print(false)\n"
            "    return 0",
            "false\n",
        )

    def test_print_expression_results_not_just_literals(self):
        assert_stdout(
            "    int a = 3\n"
            "    int b = 4\n"
            "    print(a + b)\n"
            "    print(a > b)\n"
            "    print('re' + 'sult')\n"
            "    return 0",
            "7\nfalse\nresult\n",
        )

    def test_print_multiple_calls_in_sequence(self):
        assert_stdout(
            "    print(1)\n"
            "    print(2)\n"
            "    print(3)\n"
            "    return 0",
            "1\n2\n3\n",
        )

    def test_print_inside_a_loop(self):
        assert_stdout(
            "    int i = 0\n"
            "    while i < 3:\n"
            "        print(i)\n"
            "        i = i + 1\n"
            "    return 0",
            "0\n1\n2\n",
        )

    def test_print_inside_if_branches(self):
        assert_stdout(
            "    bool flag = true\n"
            "    if flag:\n"
            "        print('yes')\n"
            "    else:\n"
            "        print('no')\n"
            "    return 0",
            "yes\n",
        )

    def test_print_result_not_usable_as_a_value(self):
        assert_semantic_error(
            "    int x = print(5) + 41\n"
            "    return x",
            match="requires two operands of the same integer type .* or two str operands, got void",
        )

    def test_print_inside_a_user_defined_function(self):
        assert_program_stdout(
            "def int announce(int x):\n"
            "    print(x)\n"
            "    return x * 2\n"
            "\n"
            "def int main():\n"
            "    int result = announce(21)\n"
            "    print(result)\n"
            "    return 0\n",
            "21\n42\n",
        )

    def test_print_repeated_calls_in_sequence_share_state_correctly(self):
        assert_stdout(
            "    print(1)\n"
            "    print(22)\n"
            "    print(333)\n"
            "    return 0",
            "1\n22\n333\n",
        )


# ---------------------------------------------------------------------------
# `len`
# ---------------------------------------------------------------------------

class TestLen:
    pytestmark = GCC_SKIP

    def test_len_on_fixed_array(self):
        assert_exit_code(
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    return len(arr)",
            5,
        )

    def test_len_on_named_slice(self):
        assert_exit_code(
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    []int s = arr[1:4]\n"
            "    return len(s)",
            3,
        )

    def test_len_on_unnamed_slice_expression(self):
        assert_exit_code(
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    return len(arr[1:4])",
            3,
        )

    def test_len_on_slice_returning_call_directly(self):
        assert_program_exit_code(
            "def []int f([5]int arr):\n"
            "    return arr[1:4]\n"
            "\n"
            "def int main():\n"
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    return len(f(arr))\n",
            3,
        )

    def test_len_on_array_literal(self):
        assert_exit_code(
            "    return len([3]int[1, 2, 3])",
            3,
        )

    def test_len_on_slice_literal(self):
        assert_exit_code(
            "    return len([]int[1, 2, 3, 4])",
            4,
        )

    def test_len_on_sub_array_row(self):
        assert_exit_code(
            "    [2][3]int m = [[1, 2, 3], [4, 5, 6]]\n"
            "    return len(m[0])",
            3,
        )

    def test_len_usable_as_loop_bound(self):
        assert_exit_code(
            "    [4]int arr = [10, 20, 30, 40]\n"
            "    int total = 0\n"
            "    int i = 0\n"
            "    while i < len(arr):\n"
            "        total = total + arr[i]\n"
            "        i = i + 1\n"
            "    return total",
            100,
        )

    def test_len_still_bounds_checks_out_of_range_argument(self):
        assert_crashes_with_sigabrt(
            "    [3]int arr = [1, 2, 3]\n"
            "    return len(arr[0:10])"
        )

    def test_len_still_aborts_on_out_of_range_array_index_even_though_length_is_compile_time(self):
        assert_crashes_with_sigabrt(
            "    [2][3]int m = [[1, 2, 3], [4, 5, 6]]\n"
            "    return len(m[10])"
        )

    def test_len_argument_side_effect_still_runs(self):
        assert_program_stdout(
            "def int se():\n"
            "    print(77)\n"
            "    return 0\n"
            "\n"
            "def int main():\n"
            "    [3]int arr = [1, 2, 3]\n"
            "    int unused = se()\n"
            "    print(len(arr))\n"
            "    return 0\n",
            "77\n3\n",
        )

    def test_len_on_int_is_rejected(self):
        assert_semantic_error(
            "    return len(5)",
            match="requires an array, slice, str, or dict",
        )

    def test_len_on_bool_is_rejected(self):
        assert_semantic_error(
            "    return len(true)",
            match="requires an array, slice, str, or dict",
        )

    def test_len_of_a_string_literal(self):
        assert_exit_code(
            "    return len('hello')",
            5,
        )

    def test_len_wrong_argument_count_is_rejected(self):
        assert_semantic_error(
            "    [3]int arr = [1, 2, 3]\n"
            "    return len(arr, 5)",
            match="expects exactly 1 argument",
        )

    def test_len_cannot_be_redefined_as_a_function(self):
        source = (
            "def int len([3]int arr):\n"
            "    return 1\n"
            "\n"
            "def int main():\n"
            "    return 0\n"
        )
        with pytest.raises(SemanticError, match="builtin"):
            analyze(_parse(source))

    def test_len_on_array_literal_argument_still_evaluates_elements(self):
        assert_program_stdout(
            "def int se():\n"
            "    print(55)\n"
            "    return 1\n"
            "\n"
            "def int main():\n"
            "    print(len([3]int[se(), 2, 3]))\n"
            "    return 0\n",
            "55\n3\n",
        )


# ---------------------------------------------------------------------------
# All-paths-return checking
# ---------------------------------------------------------------------------

class TestAllPathsReturn:


    def test_simple_trailing_return(self):
        ast = _parse("def int f():\n    return 1\n")
        analyze(ast)  # should not raise

    def test_if_else_both_branches_return(self):
        ast = _parse(
            "def int f(int x):\n"
            "    if x > 0:\n"
            "        return 1\n"
            "    else:\n"
            "        return 2\n"
        )
        analyze(ast)  # should not raise

    def test_if_without_else_followed_by_trailing_return(self):
        ast = _parse(
            "def int f(int x):\n"
            "    if x > 0:\n"
            "        return 1\n"
            "    return 2\n"
        )
        analyze(ast)  # should not raise

    def test_if_elif_else_chain_all_branches_return(self):
        ast = _parse(
            "def int f(int x):\n"
            "    if x > 0:\n"
            "        return 1\n"
            "    elif x < 0:\n"
            "        return 2\n"
            "    else:\n"
            "        return 3\n"
        )
        analyze(ast)  # should not raise

    def test_if_elif_without_final_else_followed_by_trailing_return(self):
        ast = _parse(
            "def int f(int x):\n"
            "    if x > 0:\n"
            "        return 1\n"
            "    elif x < 0:\n"
            "        return 2\n"
            "    return 99\n"
        )
        analyze(ast)  # should not raise

    def test_while_true_with_no_break_needs_no_trailing_return(self):
        ast = _parse(
            "def int f():\n"
            "    while true:\n"
            "        int x = 1\n"
        )
        analyze(ast)  # should not raise

    def test_critical_while_true_with_break_and_trailing_return(self):
        ast = _parse(
            "def int f(bool x):\n"
            "    while true:\n"
            "        if x:\n"
            "            break\n"
            "        int y = 1\n"
            "    return 99\n"
        )
        analyze(ast)  # should not raise

    def test_critical_nested_while_true_inner_break_does_not_satisfy_outer(self):
        ast = _parse(
            "def int f():\n"
            "    while true:\n"
            "        while true:\n"
            "            break\n"
            "        return 1\n"
        )
        analyze(ast)  # should not raise

    def test_finite_while_loop_followed_by_trailing_return(self):
        ast = _parse(
            "def int f():\n"
            "    int i = 0\n"
            "    while i < 10:\n"
            "        i = i + 1\n"
            "    return i\n"
        )
        analyze(ast)  # should not raise

    def test_str_returning_function_with_trailing_return(self):
        ast = _parse(
            "def str f():\n"
            "    str s = 'hello'\n"
            "    return s\n"
        )
        analyze(ast)  # should not raise

    def test_critical_break_inside_elif_chain_inside_while_true_with_trailing_return(self):
        ast = _parse(
            "def int f(int x):\n"
            "    while true:\n"
            "        if x == 1:\n"
            "            int y = 1\n"
            "        elif x == 2:\n"
            "            break\n"
            "        else:\n"
            "            int z = 1\n"
            "        return 1\n"
            "    return 99\n"
        )
        analyze(ast)  # should not raise


    def test_no_return_at_all(self):
        assert_semantic_error(
            "    int x = 1",
            match="does not return a value on all code paths",
        )

    def test_print_only_function_with_no_return(self):
        assert_semantic_error(
            "    print(5)",
            match="does not return a value on all code paths",
        )

    def test_if_without_else_and_nothing_after_it(self):
        assert_semantic_error(
            "    bool x = true\n"
            "    if x:\n"
            "        return 1",
            match="does not return a value on all code paths",
        )

    def test_if_elif_without_final_else_and_nothing_after_it(self):
        assert_semantic_error(
            "    int x = 0\n"
            "    if x == 1:\n"
            "        return 1\n"
            "    elif x == 2:\n"
            "        return 2",
            match="does not return a value on all code paths",
        )

    def test_critical_while_true_with_break_and_no_trailing_return(self):
        assert_semantic_error(
            "    bool x = true\n"
            "    while true:\n"
            "        if x:\n"
            "            break\n"
            "        return 1",
            match="does not return a value on all code paths",
        )

    def test_finite_while_loop_with_nothing_after_it(self):
        assert_semantic_error(
            "    bool x = true\n"
            "    int i = 0\n"
            "    while i < 10:\n"
            "        if x:\n"
            "            return 1\n"
            "        i = i + 1",
            match="does not return a value on all code paths",
        )

    def test_str_returning_function_with_no_return(self):
        assert_semantic_error(
            "    str s = 'hello'",
            match="does not return a value on all code paths",
            return_type="str",
        )


# ---------------------------------------------------------------------------
# Arrays
# ---------------------------------------------------------------------------

class TestArrays:
    pytestmark = GCC_SKIP

    def test_basic_1d_array_read(self):
        assert_exit_code(
            "    [3]int arr = [10, 20, 30]\n"
            "    return arr[0] + arr[1] + arr[2]",
            60,
        )

    def test_index_write(self):
        assert_exit_code(
            "    [3]int arr = [1, 2, 3]\n"
            "    arr[1] = 99\n"
            "    return arr[1]",
            99,
        )

    def test_value_semantics_1d(self):
        assert_exit_code(
            "    [3]int a = [1, 2, 3]\n"
            "    [3]int b = [0, 0, 0]\n"
            "    b = a\n"
            "    b[0] = 99\n"
            "    return a[0] == 1 and b[0] == 99",
            1,
            return_type="bool",
        )

    def test_2d_array_read(self):
        assert_exit_code(
            "    [2][3]int matrix = [[1, 2, 3], [4, 5, 6]]\n"
            "    return matrix[0][1] + matrix[1][2]",
            8,
        )

    def test_2d_array_index_write(self):
        assert_exit_code(
            "    [2][3]int matrix = [[1, 2, 3], [4, 5, 6]]\n"
            "    matrix[1][0] = 99\n"
            "    return matrix[1][0]",
            99,
        )

    def test_value_semantics_2d(self):
        assert_exit_code(
            "    [2][2]int a = [[1, 2], [3, 4]]\n"
            "    [2][2]int b = [[0, 0], [0, 0]]\n"
            "    b = a\n"
            "    b[0][0] = 99\n"
            "    return a[0][0] == 1 and b[0][0] == 99",
            1,
            return_type="bool",
        )

    def test_sub_array_extraction_is_independent(self):
        assert_exit_code(
            "    [2][3]int matrix = [[1, 2, 3], [4, 5, 6]]\n"
            "    [3]int row = matrix[1]\n"
            "    row[0] = 99\n"
            "    return matrix[1][0] == 4 and row[0] == 99",
            1,
            return_type="bool",
        )

    def test_array_of_str_elements(self):
        assert_exit_code(
            "    [3]str names = ['alice', 'bob', 'carol']\n"
            "    return names[0] == 'alice' and names[2] == 'carol'",
            1,
            return_type="bool",
        )

    def test_array_element_in_larger_expression(self):
        assert_exit_code(
            "    [4]int arr = [10, 20, 30, 40]\n"
            "    int i = 2\n"
            "    return arr[i] * 2 + arr[0]",
            70,
        )

    def test_array_element_as_function_argument(self):
        assert_program_exit_code(
            "def int double(int x):\n"
            "    return x * 2\n"
            "\n"
            "def int main():\n"
            "    [3]int arr = [5, 10, 15]\n"
            "    return double(arr[1])\n",
            20,
        )

    def test_array_iteration_via_while_loop(self):
        assert_exit_code(
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    int sum = 0\n"
            "    int i = 0\n"
            "    while i < 5:\n"
            "        sum = sum + arr[i]\n"
            "        i = i + 1\n"
            "    return sum",
            15,
        )

    def test_index_assignment_with_computed_index(self):
        assert_exit_code(
            "    [5]int arr = [0, 0, 0, 0, 0]\n"
            "    int i = 1\n"
            "    arr[i + 1] = 42\n"
            "    return arr[2]",
            42,
        )

    def test_array_parameter_basic(self):
        assert_program_exit_code(
            "def int sum_array([3]int arr):\n"
            "    return arr[0] + arr[1] + arr[2]\n"
            "\n"
            "def int main():\n"
            "    [3]int a = [1, 2, 3]\n"
            "    return sum_array(a)\n",
            6,
        )

    def test_array_parameter_value_semantics(self):
        assert_program_exit_code(
            "def int mutate([3]int arr):\n"
            "    arr[0] = 999\n"
            "    return arr[0]\n"
            "\n"
            "def bool main():\n"
            "    [3]int a = [1, 2, 3]\n"
            "    int result = mutate(a)\n"
            "    return result == 999 and a[0] == 1\n",
            1,
        )

    def test_array_return_basic(self):
        assert_program_exit_code(
            "def [3]int make():\n"
            "    [3]int r = [10, 20, 30]\n"
            "    return r\n"
            "\n"
            "def int main():\n"
            "    [3]int x = make()\n"
            "    return x[0] + x[1] + x[2]\n",
            60,
        )

    def test_array_return_direct_literal(self):
        assert_program_exit_code(
            "def [3]int make():\n"
            "    return [10, 20, 30]\n"
            "\n"
            "def int main():\n"
            "    [3]int x = make()\n"
            "    return x[0] + x[1] + x[2]\n",
            60,
        )

    def test_array_return_via_sub_array_index(self):
        assert_program_exit_code(
            "def [3]int get_row([2][3]int matrix, int i):\n"
            "    return matrix[i]\n"
            "\n"
            "def int main():\n"
            "    [2][3]int m = [[1, 2, 3], [4, 5, 6]]\n"
            "    [3]int row = get_row(m, 1)\n"
            "    return row[0] + row[1] + row[2]\n",
            15,
        )

    def test_nested_array_returning_call_forwarding(self):
        assert_program_exit_code(
            "def [3]int inner():\n"
            "    return [7, 8, 9]\n"
            "\n"
            "def [3]int outer():\n"
            "    return inner()\n"
            "\n"
            "def int main():\n"
            "    [3]int x = outer()\n"
            "    return x[0] + x[1] + x[2]\n",
            24,
        )

    def test_2d_array_as_parameter_and_return_type(self):
        assert_program_exit_code(
            "def [2][2]int double_all([2][2]int m):\n"
            "    [2][2]int result = [[0, 0], [0, 0]]\n"
            "    int i = 0\n"
            "    while i < 2:\n"
            "        int j = 0\n"
            "        while j < 2:\n"
            "            result[i][j] = m[i][j] * 2\n"
            "            j = j + 1\n"
            "        i = i + 1\n"
            "    return result\n"
            "\n"
            "def int main():\n"
            "    [2][2]int a = [[1, 2], [3, 4]]\n"
            "    [2][2]int b = double_all(a)\n"
            "    return b[0][0] + b[0][1] + b[1][0] + b[1][1]\n",
            20,
        )

    def test_multiple_array_parameters(self):
        assert_program_exit_code(
            "def int dot_product([3]int a, [3]int b):\n"
            "    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]\n"
            "\n"
            "def int main():\n"
            "    [3]int x = [1, 2, 3]\n"
            "    [3]int y = [4, 5, 6]\n"
            "    return dot_product(x, y)\n",
            32,
        )

    def test_returned_array_independent_across_separate_calls(self):
        assert_program_exit_code(
            "def [3]int make_and_mutate():\n"
            "    [3]int local = [1, 2, 3]\n"
            "    local[0] = 100\n"
            "    return local\n"
            "\n"
            "def bool main():\n"
            "    [3]int a = make_and_mutate()\n"
            "    [3]int b = make_and_mutate()\n"
            "    a[1] = 999\n"
            "    return a[0] == 100 and a[1] == 999 and b[0] == 100 and b[1] == 2\n",
            1,
        )

    def test_array_argument_as_index_expression(self):
        assert_program_exit_code(
            "def int sum3([3]int arr):\n"
            "    return arr[0] + arr[1] + arr[2]\n"
            "\n"
            "def int main():\n"
            "    [2][3]int matrix = [[1, 2, 3], [4, 5, 6]]\n"
            "    return sum3(matrix[1])\n",
            15,
        )

    def test_str_element_array_as_parameter_and_return_type(self):
        assert_program_exit_code(
            "def bool first_is([3]str names, str target):\n"
            "    return names[0] == target\n"
            "\n"
            "def [3]str make_names():\n"
            "    return ['x', 'y', 'z']\n"
            "\n"
            "def bool main():\n"
            "    [3]str n = make_names()\n"
            "    return first_is(n, 'x') and n[2] == 'z'\n",
            1,
        )

    def test_five_real_params_on_array_returning_function(self):
        assert_program_exit_code(
            "def [2]int make5(int a, int b, int c, int d, int e):\n"
            "    return [a + b, c + d + e]\n"
            "\n"
            "def int main():\n"
            "    [2]int r = make5(1, 2, 3, 4, 5)\n"
            "    return r[0] + r[1]\n",
            15,
        )

    def test_seven_real_params_on_array_returning_function_works_via_the_stack(self):
        assert_program_exit_code(
            "def [2]int make6(int a, int b, int c, int d, int e, int f):\n"
            "    return [a, f]\n"
            "\n"
            "def int main():\n"
            "    [2]int r = make6(1, 2, 3, 4, 5, 6)\n"
            "    return r[0] + r[1]\n",
            7,
        )

    def test_array_literal_as_direct_call_argument(self):
        assert_program_exit_code(
            "def int sum3([3]int arr):\n"
            "    return arr[0] + arr[1] + arr[2]\n"
            "\n"
            "def int main():\n"
            "    return sum3([1, 2, 3])\n",
            6,
        )


# ---------------------------------------------------------------------------
# Typed array literals
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Array equality
# ---------------------------------------------------------------------------

class TestArrayEquality:
    pytestmark = GCC_SKIP

    def test_equal_int_arrays(self):
        assert_exit_code(
            "    [3]int a = [1, 2, 3]\n"
            "    [3]int b = [1, 2, 3]\n"
            "    if a == b:\n"
            "        return 1\n"
            "    return 0",
            1,
        )

    def test_unequal_int_arrays(self):
        assert_exit_code(
            "    [3]int a = [1, 2, 3]\n"
            "    [3]int b = [1, 2, 4]\n"
            "    if a == b:\n"
            "        return 1\n"
            "    return 0",
            0,
        )

    def test_not_equal_operator(self):
        assert_exit_code(
            "    [3]int a = [1, 2, 3]\n"
            "    [3]int b = [1, 2, 4]\n"
            "    if a != b:\n"
            "        return 1\n"
            "    return 0",
            1,
        )

    def test_equal_bool_arrays(self):
        assert_exit_code(
            "    [2]bool a = [true, false]\n"
            "    [2]bool b = [true, false]\n"
            "    if a == b:\n"
            "        return 1\n"
            "    return 0",
            1,
        )

    def test_nested_int_arrays_equal(self):
        assert_exit_code(
            "    [2][3]int a = [[1, 2, 3], [4, 5, 6]]\n"
            "    [2][3]int b = [[1, 2, 3], [4, 5, 6]]\n"
            "    if a == b:\n"
            "        return 1\n"
            "    return 0",
            1,
        )

    def test_nested_int_arrays_differ_in_last_element(self):
        assert_exit_code(
            "    [2][3]int a = [[1, 2, 3], [4, 5, 6]]\n"
            "    [2][3]int b = [[1, 2, 3], [4, 5, 7]]\n"
            "    if a == b:\n"
            "        return 1\n"
            "    return 0",
            0,
        )

    def test_str_arrays_equal_different_pointers(self):
        assert_exit_code(
            "    str prefix = 'hel'\n"
            "    [2]str a = ['hello', 'world']\n"
            "    [2]str b = [prefix + 'lo', 'wor' + 'ld']\n"
            "    if a == b:\n"
            "        return 1\n"
            "    return 0",
            1,
        )

    def test_str_arrays_not_equal(self):
        assert_exit_code(
            "    [2]str a = ['hello', 'world']\n"
            "    [2]str b = ['hello', 'there']\n"
            "    if a == b:\n"
            "        return 1\n"
            "    return 0",
            0,
        )

    def test_single_element_arrays(self):
        assert_exit_code(
            "    [1]int a = [5]\n"
            "    [1]int b = [5]\n"
            "    if a == b:\n"
            "        return 1\n"
            "    return 0",
            1,
        )

    def test_subarray_comparison_via_index(self):
        assert_exit_code(
            "    [2][3]int m = [[1, 2, 3], [1, 2, 3]]\n"
            "    if m[0] == m[1]:\n"
            "        return 1\n"
            "    return 0",
            1,
        )

    def test_array_field_comparison(self):
        assert_program_exit_code(
            "type Row struct:\n"
            "    [3]int values\n"
            "\n"
            "def int main():\n"
            "    Row r1 = Row([1, 2, 3])\n"
            "    Row r2 = Row([1, 2, 3])\n"
            "    if r1.values == r2.values:\n"
            "        return 1\n"
            "    return 0\n",
            1,
        )

    def test_large_array_comparison(self):
        assert_exit_code(
            "    [2000]int a\n"
            "    [2000]int b\n"
            "    int i = 0\n"
            "    while i < 2000:\n"
            "        a[i] = i\n"
            "        b[i] = i\n"
            "        i = i + 1\n"
            "    if a == b:\n"
            "        return 1\n"
            "    return 0",
            1,
        )

    def test_two_bare_array_literals_compared_directly(self):
        assert_program_exit_code(
            "def int main():\n"
            "    if [1, 2, 3] == [1, 2, 3]:\n"
            "        return 1\n"
            "    return 0\n",
            1,
        )

    def test_two_bare_array_literals_compared_directly_not_equal(self):
        assert_program_exit_code(
            "def int main():\n"
            "    if [1, 2, 3] != [1, 2, 4]:\n"
            "        return 1\n"
            "    return 0\n",
            1,
        )

    def test_two_array_returning_calls_compared_directly(self):
        assert_program_exit_code(
            "def [3]int makeA():\n"
            "    return [1, 2, 3]\n"
            "\n"
            "def [3]int makeB():\n"
            "    return [1, 2, 3]\n"
            "\n"
            "def int main():\n"
            "    if makeA() == makeB():\n"
            "        return 1\n"
            "    return 0\n",
            1,
        )

    def test_array_literal_compared_to_array_returning_call(self):
        assert_program_exit_code(
            "def [3]int makeArr():\n"
            "    return [1, 2, 3]\n"
            "\n"
            "def int main():\n"
            "    if [1, 2, 3] == makeArr():\n"
            "        return 1\n"
            "    return 0\n",
            1,
        )


# ---------------------------------------------------------------------------
# Struct equality
# ---------------------------------------------------------------------------

class TestStructEquality:
    pytestmark = GCC_SKIP

    def test_equal_structs(self):
        assert_program_exit_code(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def int main():\n"
            "    Point a = Point(1, 2)\n"
            "    Point b = Point(1, 2)\n"
            "    if a == b:\n"
            "        return 1\n"
            "    return 0\n",
            1,
        )

    def test_unequal_structs(self):
        assert_program_exit_code(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def int main():\n"
            "    Point a = Point(1, 2)\n"
            "    Point b = Point(1, 3)\n"
            "    if a == b:\n"
            "        return 1\n"
            "    return 0\n",
            0,
        )

    def test_not_equal_operator(self):
        assert_program_exit_code(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def int main():\n"
            "    Point a = Point(1, 2)\n"
            "    Point b = Point(1, 3)\n"
            "    if a != b:\n"
            "        return 1\n"
            "    return 0\n",
            1,
        )

    def test_mismatch_in_first_field_is_detected(self):
        assert_program_exit_code(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def int main():\n"
            "    Point a = Point(1, 2)\n"
            "    Point b = Point(9, 2)\n"
            "    if a == b:\n"
            "        return 1\n"
            "    return 0\n",
            0,
        )

    def test_bool_field(self):
        assert_program_exit_code(
            "type Flag struct:\n"
            "    bool on\n"
            "\n"
            "def int main():\n"
            "    Flag a = Flag(true)\n"
            "    Flag b = Flag(true)\n"
            "    if a == b:\n"
            "        return 1\n"
            "    return 0\n",
            1,
        )

    def test_str_field_equal_different_pointers(self):
        assert_program_exit_code(
            "type Person struct:\n"
            "    int age\n"
            "    str name\n"
            "\n"
            "def int main():\n"
            "    str prefix = 'Al'\n"
            "    Person a = Person(30, 'Alice')\n"
            "    Person b = Person(30, prefix + 'ice')\n"
            "    if a == b:\n"
            "        return 1\n"
            "    return 0\n",
            1,
        )

    def test_str_field_not_equal(self):
        assert_program_exit_code(
            "type Person struct:\n"
            "    int age\n"
            "    str name\n"
            "\n"
            "def int main():\n"
            "    Person a = Person(30, 'Alice')\n"
            "    Person b = Person(30, 'Bob')\n"
            "    if a == b:\n"
            "        return 1\n"
            "    return 0\n",
            0,
        )

    def test_nested_struct_field_equal(self):
        assert_program_exit_code(
            "type Inner struct:\n"
            "    int v\n"
            "type Outer struct:\n"
            "    Inner i\n"
            "    int b\n"
            "\n"
            "def int main():\n"
            "    Outer a = Outer(Inner(1), 2)\n"
            "    Outer b = Outer(Inner(1), 2)\n"
            "    if a == b:\n"
            "        return 1\n"
            "    return 0\n",
            1,
        )

    def test_nested_struct_field_mismatch(self):
        assert_program_exit_code(
            "type Inner struct:\n"
            "    int v\n"
            "type Outer struct:\n"
            "    Inner i\n"
            "    int b\n"
            "\n"
            "def int main():\n"
            "    Outer a = Outer(Inner(1), 2)\n"
            "    Outer b = Outer(Inner(9), 2)\n"
            "    if a == b:\n"
            "        return 1\n"
            "    return 0\n",
            0,
        )

    def test_array_field_equal(self):
        assert_program_exit_code(
            "type Row struct:\n"
            "    [3]int values\n"
            "\n"
            "def int main():\n"
            "    Row a = Row([1, 2, 3])\n"
            "    Row b = Row([1, 2, 3])\n"
            "    if a == b:\n"
            "        return 1\n"
            "    return 0\n",
            1,
        )

    def test_array_field_not_equal(self):
        assert_program_exit_code(
            "type Row struct:\n"
            "    [3]int values\n"
            "\n"
            "def int main():\n"
            "    Row a = Row([1, 2, 3])\n"
            "    Row b = Row([1, 2, 4])\n"
            "    if a == b:\n"
            "        return 1\n"
            "    return 0\n",
            0,
        )

    def test_array_of_str_field_equal_different_pointers(self):
        assert_program_exit_code(
            "type Words struct:\n"
            "    [2]str ws\n"
            "\n"
            "def int main():\n"
            "    str prefix = 'wor'\n"
            "    Words a = Words(['hello', 'world'])\n"
            "    Words b = Words(['hello', prefix + 'ld'])\n"
            "    if a == b:\n"
            "        return 1\n"
            "    return 0\n",
            1,
        )

    def test_array_of_comparable_structs(self):
        assert_program_exit_code(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def int main():\n"
            "    [2]Point a = [Point(1, 2), Point(3, 4)]\n"
            "    [2]Point b = [Point(1, 2), Point(3, 4)]\n"
            "    if a == b:\n"
            "        return 1\n"
            "    return 0\n",
            1,
        )

    def test_array_of_comparable_structs_not_equal(self):
        assert_program_exit_code(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def int main():\n"
            "    [2]Point a = [Point(1, 2), Point(3, 4)]\n"
            "    [2]Point b = [Point(1, 2), Point(3, 5)]\n"
            "    if a == b:\n"
            "        return 1\n"
            "    return 0\n",
            0,
        )

    def test_array_of_structs_with_array_of_str_field(self):
        assert_program_exit_code(
            "type Bag struct:\n"
            "    [2]str items\n"
            "\n"
            "def int main():\n"
            "    str prefix = 'ap'\n"
            "    [2]Bag a = [Bag(['apple', 'banana']), Bag(['cherry', 'date'])]\n"
            "    [2]Bag b = [Bag([prefix + 'ple', 'banana']), Bag(['cherry', 'date'])]\n"
            "    if a == b:\n"
            "        return 1\n"
            "    return 0\n",
            1,
        )

    def test_array_of_structs_with_array_of_str_field_mismatch_in_second_element(self):
        assert_program_exit_code(
            "type Bag struct:\n"
            "    [2]str items\n"
            "\n"
            "def int main():\n"
            "    [2]Bag a = [Bag(['apple', 'banana']), Bag(['cherry', 'date'])]\n"
            "    [2]Bag b = [Bag(['apple', 'banana']), Bag(['cherry', 'WRONG'])]\n"
            "    if a == b:\n"
            "        return 1\n"
            "    return 0\n",
            0,
        )

    def test_doubly_nested_struct_with_array_of_structs_with_str_field(self):
        assert_program_exit_code(
            "type Item struct:\n"
            "    str name\n"
            "    int qty\n"
            "type Container struct:\n"
            "    [2]Item items\n"
            "    int tag\n"
            "\n"
            "def int main():\n"
            "    str prefix = 'wid'\n"
            "    Container a = Container([Item('widget', 5), Item('gadget', 3)], 99)\n"
            "    Container b = Container([Item(prefix + 'get', 5), Item('gadget', 3)], 99)\n"
            "    if a == b:\n"
            "        return 1\n"
            "    return 0\n",
            1,
        )

    def test_doubly_nested_mismatch(self):
        assert_program_exit_code(
            "type Item struct:\n"
            "    str name\n"
            "    int qty\n"
            "type Container struct:\n"
            "    [2]Item items\n"
            "    int tag\n"
            "\n"
            "def int main():\n"
            "    Container a = Container([Item('widget', 5), Item('gadget', 3)], 99)\n"
            "    Container b = Container([Item('widget', 5), Item('gadget', 4)], 99)\n"
            "    if a == b:\n"
            "        return 1\n"
            "    return 0\n",
            0,
        )

    def test_comparison_does_not_mutate_either_operand(self):
        assert_program_exit_code(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def int main():\n"
            "    Point a = Point(1, 2)\n"
            "    Point b = Point(1, 2)\n"
            "    bool eq = a == b\n"
            "    return a.x + a.y + b.x + b.y\n",
            6,
        )

    def test_int8_struct_field_equality_equal(self):
        assert_program_exit_code(
            "type Small struct:\n"
            "    int8 a\n"
            "    int8 b\n"
            "\n"
            "def int main():\n"
            "    Small x\n"
            "    x.a = 5\n"
            "    x.b = 6\n"
            "    Small y\n"
            "    y.a = 5\n"
            "    y.b = 6\n"
            "    if x == y:\n"
            "        return 1\n"
            "    return 0\n",
            1,
        )

    def test_int8_struct_field_equality_not_equal(self):
        assert_program_exit_code(
            "type Small struct:\n"
            "    int8 a\n"
            "    int8 b\n"
            "\n"
            "def int main():\n"
            "    Small x\n"
            "    x.a = 5\n"
            "    x.b = 6\n"
            "    Small z\n"
            "    z.a = 5\n"
            "    z.b = 7\n"
            "    if x != z:\n"
            "        return 1\n"
            "    return 0\n",
            1,
        )

    def test_uint8_struct_field_equality_equal(self):
        assert_program_exit_code(
            "type Small struct:\n"
            "    uint8 a\n"
            "    uint8 b\n"
            "\n"
            "def int main():\n"
            "    Small x\n"
            "    x.a = 200\n"
            "    x.b = 6\n"
            "    Small y\n"
            "    y.a = 200\n"
            "    y.b = 6\n"
            "    if x == y:\n"
            "        return 1\n"
            "    return 0\n",
            1,
        )

    def test_uint8_struct_field_equality_not_equal(self):
        assert_program_exit_code(
            "type Small struct:\n"
            "    uint8 a\n"
            "    uint8 b\n"
            "\n"
            "def int main():\n"
            "    Small x\n"
            "    x.a = 200\n"
            "    x.b = 6\n"
            "    Small z\n"
            "    z.a = 200\n"
            "    z.b = 7\n"
            "    if x != z:\n"
            "        return 1\n"
            "    return 0\n",
            1,
        )

    def test_two_struct_returning_calls_compared_directly(self):
        assert_program_exit_code(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def Point makeP1():\n"
            "    return Point(5, 6)\n"
            "\n"
            "def Point makeP2():\n"
            "    return Point(5, 6)\n"
            "\n"
            "def int main():\n"
            "    if makeP1() == makeP2():\n"
            "        return 1\n"
            "    return 0\n",
            1,
        )


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------

class TestMethods:
    pytestmark = GCC_SKIP

    def test_basic_method_call(self):
        assert_program_exit_code(
            "type A struct:\n"
            "    int a\n"
            "    def int add_b(s, int b):\n"
            "        return s.a + b\n"
            "\n"
            "def int main():\n"
            "    A a = A(a=1)\n"
            "    return a.add_b(5)\n",
            6,
        )

    def test_method_with_no_declared_return_type_as_a_bare_statement(self):
        assert_program_stdout(
            "type A struct:\n"
            "    int a\n"
            "    def add_b(s, int b):\n"
            "        print(s.a + b)\n"
            "\n"
            "def int main():\n"
            "    A a = A(a=1)\n"
            "    a.add_b(5)\n"
            "    return 0\n",
            "6\n",
        )

    def test_value_semantics_receiver_is_not_mutated(self):
        assert_program_exit_code(
            "type Counter struct:\n"
            "    int n\n"
            "    def mutate(s):\n"
            "        s.n = 999\n"
            "\n"
            "def int main():\n"
            "    Counter c = Counter(n=1)\n"
            "    c.mutate()\n"
            "    return c.n\n",
            1,
        )

    def test_method_returning_a_struct(self):
        assert_program_exit_code(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "    def Point doubled(s):\n"
            "        return Point(s.x * 2, s.y * 2)\n"
            "\n"
            "def int main():\n"
            "    Point p = Point(3, 4)\n"
            "    Point d = p.doubled()\n"
            "    return d.x + d.y\n",
            14,
        )

    def test_method_returning_a_slice(self):
        assert_program_exit_code(
            "type Row struct:\n"
            "    [3]int values\n"
            "    def []int asSlice(s):\n"
            "        return s.values[:]\n"
            "\n"
            "def int main():\n"
            "    Row r = Row([1, 2, 3])\n"
            "    []int sl = r.asSlice()\n"
            "    return sl[0] + sl[1] + sl[2]\n",
            6,
        )

    def test_method_with_array_field_and_array_param(self):
        assert_program_exit_code(
            "type Row struct:\n"
            "    [3]int values\n"
            "    def int sumPlus(s, [3]int other):\n"
            "        return s.values[0] + s.values[1] + s.values[2] + other[0] + other[1] + other[2]\n"
            "\n"
            "def int main():\n"
            "    Row r = Row([1, 2, 3])\n"
            "    return r.sumPlus([10, 20, 30])\n",
            66,
        )

    def test_method_calling_another_method_on_the_same_struct(self):
        assert_program_exit_code(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "    def int getX(s):\n"
            "        return s.x\n"
            "    def int sum(s):\n"
            "        return s.getX() + s.y\n"
            "\n"
            "def int main():\n"
            "    Point p = Point(3, 4)\n"
            "    return p.sum()\n",
            7,
        )

    def test_method_and_free_function_sharing_a_name(self):
        assert_program_exit_code(
            "type Point struct:\n"
            "    int x\n"
            "    def int getX(s):\n"
            "        return s.x\n"
            "\n"
            "def int getX(int v):\n"
            "    return v + 100\n"
            "\n"
            "def int main():\n"
            "    Point p = Point(5)\n"
            "    return p.getX() + getX(1)\n",
            106,
        )

    def test_two_different_structs_with_same_named_methods(self):
        assert_program_exit_code(
            "type A struct:\n"
            "    int v\n"
            "    def int getV(s):\n"
            "        return s.v\n"
            "type B struct:\n"
            "    int v\n"
            "    def int getV(s):\n"
            "        return s.v * 10\n"
            "\n"
            "def int main():\n"
            "    A a = A(3)\n"
            "    B b = B(3)\n"
            "    return a.getV() + b.getV()\n",
            33,
        )

    def test_method_name_matching_a_field_name(self):
        assert_program_exit_code(
            "type Weird struct:\n"
            "    int x\n"
            "    def int x(s):\n"
            "        return 42\n"
            "\n"
            "def int main():\n"
            "    Weird w = Weird(1)\n"
            "    return w.x() + w.x\n",
            43,
        )

    def test_receiver_via_a_field_access_chain(self):
        assert_program_exit_code(
            "type Inner struct:\n"
            "    int v\n"
            "    def int doubled(s):\n"
            "        return s.v * 2\n"
            "type Outer struct:\n"
            "    Inner i\n"
            "\n"
            "def int main():\n"
            "    Outer o = Outer(Inner(5))\n"
            "    return o.i.doubled()\n",
            10,
        )

    def test_method_call_as_a_function_argument(self):
        assert_program_exit_code(
            "type Point struct:\n"
            "    int x\n"
            "    def int getX(s):\n"
            "        return s.x\n"
            "\n"
            "def int addOne(int v):\n"
            "    return v + 1\n"
            "\n"
            "def int main():\n"
            "    Point p = Point(9)\n"
            "    return addOne(p.getX())\n",
            10,
        )

    def test_method_call_as_an_array_literal_element(self):
        assert_program_exit_code(
            "type Point struct:\n"
            "    int x\n"
            "    def int getX(s):\n"
            "        return s.x\n"
            "\n"
            "def int main():\n"
            "    Point p1 = Point(1)\n"
            "    Point p2 = Point(2)\n"
            "    [2]int xs = [p1.getX(), p2.getX()]\n"
            "    return xs[0] + xs[1]\n",
            3,
        )

    def test_method_call_as_index_assign_value(self):
        assert_program_exit_code(
            "type Point struct:\n"
            "    int x\n"
            "    def int getX(s):\n"
            "        return s.x\n"
            "\n"
            "def int main():\n"
            "    Point p = Point(7)\n"
            "    [1]int arr\n"
            "    arr[0] = p.getX()\n"
            "    return arr[0]\n",
            7,
        )

    def test_method_call_as_field_assign_value(self):
        assert_program_exit_code(
            "type Point struct:\n"
            "    int x\n"
            "    def int getX(s):\n"
            "        return s.x\n"
            "type Holder struct:\n"
            "    int v\n"
            "\n"
            "def int main():\n"
            "    Point p = Point(8)\n"
            "    Holder h\n"
            "    h.v = p.getX()\n"
            "    return h.v\n",
            8,
        )

    def test_chained_method_calls(self):
        assert_program_exit_code(
            "type Inner struct:\n"
            "    int v\n"
            "    def int doubled(s):\n"
            "        return s.v * 2\n"
            "type Outer struct:\n"
            "    Inner i\n"
            "    def Inner getInner(s):\n"
            "        return s.i\n"
            "\n"
            "def int main():\n"
            "    Outer o = Outer(Inner(7))\n"
            "    return o.getInner().doubled()\n",
            14,
        )

    def test_receiver_via_struct_returning_call(self):
        assert_program_exit_code(
            "type A struct:\n"
            "    int v\n"
            "    def int foo(s):\n"
            "        return s.v\n"
            "\n"
            "def A makeA():\n"
            "    return A(9)\n"
            "\n"
            "def int main():\n"
            "    return makeA().foo()\n",
            9,
        )

    def test_struct_literal_as_receiver_works(self):
        assert_program_exit_code(
            "type A struct:\n"
            "    int v\n"
            "    def int foo(s):\n"
            "        return s.v\n"
            "\n"
            "def int main():\n"
            "    return A(1).foo()\n",
            1,
        )

    def test_duplicate_method_name_on_the_same_struct_is_rejected(self):
        assert_program_semantic_error(
            "type A struct:\n"
            "    int v\n"
            "    def int foo(s):\n"
            "        return 1\n"
            "    def int foo(s):\n"
            "        return 2\n"
            "\n"
            "def int main():\n"
            "    return 0\n",
            match="already declared on struct",
        )

    def test_undefined_method_is_rejected(self):
        assert_program_semantic_error(
            "type A struct:\n"
            "    int v\n"
            "\n"
            "def int main():\n"
            "    A a = A(1)\n"
            "    return a.bar()\n",
            match="has no method 'bar'",
        )

    def test_method_call_on_a_non_struct_value_is_rejected(self):
        assert_program_semantic_error(
            "def int main():\n"
            "    int x = 5\n"
            "    return x.foo()\n",
            match="methods are only defined on structs",
        )

    def test_wrong_argument_count_is_rejected(self):
        assert_program_semantic_error(
            "type A struct:\n"
            "    int v\n"
            "    def int foo(s, int b):\n"
            "        return b\n"
            "\n"
            "def int main():\n"
            "    A a = A(1)\n"
            "    return a.foo()\n",
            match="expects 1 argument",
        )

    def test_wrong_argument_type_is_rejected(self):
        assert_program_semantic_error(
            "type A struct:\n"
            "    int v\n"
            "    def int foo(s, int b):\n"
            "        return b\n"
            "\n"
            "def int main():\n"
            "    A a = A(1)\n"
            "    return a.foo('nope')\n",
            match="should be int",
        )

    def test_struct_literal_as_method_argument(self):
        assert_program_exit_code(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "type Line struct:\n"
            "    int id\n"
            "    def int sumWith(s, Point p):\n"
            "        return p.x + p.y + s.id\n"
            "\n"
            "def int main():\n"
            "    Line l = Line(1)\n"
            "    return l.sumWith(Point(2, 3))\n",
            6,
        )

    def test_receiver_plus_six_params_works_via_the_stack(self):
        assert_program_exit_code(
            "type A struct:\n"
            "    int v\n"
            "    def int sum6(s, int a, int b, int c, int d, int e, int f):\n"
            "        return s.v + a + b + c + d + e + f\n"
            "\n"
            "def int main():\n"
            "    A obj = A(v=1)\n"
            "    return obj.sum6(2, 3, 4, 5, 6, 7)\n",
            28,
        )

    def test_five_explicit_params_plus_receiver_fits_exactly(self):
        assert_program_exit_code(
            "type A struct:\n"
            "    int v\n"
            "    def int sum5(s, int a, int b, int c, int d, int e):\n"
            "        return a + b + c + d + e\n"
            "\n"
            "def int main():\n"
            "    A x = A(1)\n"
            "    return x.sum5(1, 2, 3, 4, 5)\n",
            15,
        )


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

class TestTypeAliases:
    pytestmark = GCC_SKIP

    def test_alias_as_vardecl_type(self):
        assert_program_exit_code(
            "type MyInt = int\n"
            "\n"
            "def int main():\n"
            "    MyInt x = 5\n"
            "    return x\n",
            5,
        )

    def test_alias_as_param_and_return_type(self):
        assert_program_exit_code(
            "type MyInt = int\n"
            "\n"
            "def MyInt double(MyInt x):\n"
            "    return x * 2\n"
            "\n"
            "def int main():\n"
            "    return double(21)\n",
            42,
        )

    def test_alias_as_struct_field_type(self):
        assert_program_exit_code(
            "type MyInt = int\n"
            "\n"
            "type Point struct:\n"
            "    MyInt x\n"
            "    MyInt y\n"
            "\n"
            "def int main():\n"
            "    Point p = Point(3, 4)\n"
            "    return p.x + p.y\n",
            7,
        )

    def test_alias_of_alias(self):
        assert_program_exit_code(
            "type A = int\n"
            "type B = A\n"
            "\n"
            "def int main():\n"
            "    B x = 10\n"
            "    return x\n",
            10,
        )

    def test_forward_referenced_alias(self):
        assert_program_exit_code(
            "type B = A\n"
            "type A = int\n"
            "\n"
            "def int main():\n"
            "    B x = 7\n"
            "    return x\n",
            7,
        )

    def test_alias_to_bool(self):
        assert_program_exit_code(
            "type Flag = bool\n"
            "\n"
            "def int main():\n"
            "    Flag f = true\n"
            "    if f:\n"
            "        return 1\n"
            "    return 0\n",
            1,
        )

    def test_alias_to_str(self):
        assert_program_exit_code(
            "type Text = str\n"
            "\n"
            "def int main():\n"
            "    Text t = 'hi'\n"
            "    print(t)\n"
            "    return 0\n",
            0,
        )

    def test_alias_typed_vardecl_zero_init(self):
        assert_program_exit_code(
            "type MyInt = int\n"
            "\n"
            "def int main():\n"
            "    MyInt x\n"
            "    return x\n",
            0,
        )

    def test_alias_as_array_element_type(self):
        assert_program_exit_code(
            "type MyInt = int\n"
            "\n"
            "def int main():\n"
            "    [3]MyInt arr = [1, 2, 3]\n"
            "    return arr[0] + arr[1] + arr[2]\n",
            6,
        )

    def test_alias_as_slice_element_type(self):
        assert_program_exit_code(
            "type MyInt = int\n"
            "\n"
            "def int main():\n"
            "    []MyInt s = []MyInt[1, 2, 3]\n"
            "    return s[0] + s[1] + s[2]\n",
            6,
        )

    def test_alias_in_method_signature(self):
        assert_program_exit_code(
            "type MyInt = int\n"
            "\n"
            "type Adder struct:\n"
            "    int base\n"
            "    def MyInt addTo(s, MyInt x):\n"
            "        return s.base + x\n"
            "\n"
            "def int main():\n"
            "    Adder a = Adder(10)\n"
            "    return a.addTo(5)\n",
            15,
        )

    def test_print_alias_typed_variable(self):
        assert_program_stdout(
            "type MyInt = int\n"
            "\n"
            "def int main():\n"
            "    MyInt x = 42\n"
            "    print(x)\n"
            "    return 0\n",
            "42\n",
        )

    def test_duplicate_alias_is_rejected(self):
        assert_program_semantic_error(
            "type A = int\n"
            "type A = bool\n"
            "\n"
            "def int main():\n"
            "    return 0\n",
            match="already declared",
        )

    def test_direct_cycle_is_rejected(self):
        assert_program_semantic_error(
            "type A = A\n"
            "\n"
            "def int main():\n"
            "    return 0\n",
            match="cycle",
        )

    def test_indirect_cycle_is_rejected(self):
        assert_program_semantic_error(
            "type A = B\n"
            "type B = A\n"
            "\n"
            "def int main():\n"
            "    return 0\n",
            match="cycle",
        )

    def test_alias_to_an_array_type(self):
        assert_program_exit_code(
            "type IntArray = [3]int\n"
            "\n"
            "def int main():\n"
            "    IntArray arr = [1, 2, 3]\n"
            "    return arr[0] + arr[1] + arr[2]\n",
            6,
        )

    def test_alias_to_a_slice_type(self):
        assert_program_exit_code(
            "type IntSlice = []int\n"
            "\n"
            "def int main():\n"
            "    IntSlice s = []int[1, 2, 3]\n"
            "    return s[0] + s[1] + s[2]\n",
            6,
        )

    def test_alias_to_a_multidimensional_array_type(self):
        assert_program_exit_code(
            "type Matrix = [2][3]int\n"
            "\n"
            "def int main():\n"
            "    Matrix m = [[1, 2, 3], [4, 5, 6]]\n"
            "    return m[0][0] + m[1][2]\n",
            7,
        )

    def test_alias_to_an_array_of_a_struct_type(self):
        assert_program_exit_code(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "type Points = [2]Point\n"
            "\n"
            "def int main():\n"
            "    Points pts = [Point(1, 2), Point(3, 4)]\n"
            "    return pts[0].x + pts[1].y\n",
            5,
        )

    def test_alias_to_a_slice_of_another_alias(self):
        assert_program_exit_code(
            "type MyInt = int\n"
            "type MyIntSlice = []MyInt\n"
            "\n"
            "def int main():\n"
            "    MyIntSlice s = []int[7, 8, 9]\n"
            "    return s[0] + s[1] + s[2]\n",
            24,
        )

    def test_array_alias_cycle_through_wrapping_is_rejected(self):
        assert_program_semantic_error(
            "type A = []B\n"
            "type B = []A\n"
            "\n"
            "def int main():\n"
            "    return 0\n",
            match="cycle",
        )

    def test_unknown_target_is_rejected(self):
        assert_program_semantic_error(
            "type Foo = NotAThing\n"
            "\n"
            "def int main():\n"
            "    return 0\n",
            match="expected int, bool, str, a struct name, or another type alias",
        )

    def test_unknown_array_element_target_is_rejected(self):
        assert_program_semantic_error(
            "type Foo = [3]NotAThing\n"
            "\n"
            "def int main():\n"
            "    return 0\n",
            match="expected int, bool, str, a struct name, or another type alias",
        )

    def test_array_alias_as_param_and_return_type(self):
        assert_program_exit_code(
            "type IntArray = [3]int\n"
            "\n"
            "def int sum(IntArray arr):\n"
            "    return arr[0] + arr[1] + arr[2]\n"
            "\n"
            "def int main():\n"
            "    IntArray a = [1, 2, 3]\n"
            "    return sum(a)\n",
            6,
        )

    def test_array_alias_as_struct_field_type(self):
        assert_program_exit_code(
            "type IntArray = [3]int\n"
            "\n"
            "type Row struct:\n"
            "    IntArray values\n"
            "\n"
            "def int main():\n"
            "    Row r = Row([1, 2, 3])\n"
            "    return r.values[0] + r.values[1] + r.values[2]\n",
            6,
        )

    def test_array_alias_zero_init(self):
        assert_program_exit_code(
            "type IntArray = [3]int\n"
            "\n"
            "def int main():\n"
            "    IntArray arr\n"
            "    return arr[0] + arr[1] + arr[2]\n",
            0,
        )

    def test_array_alias_equality(self):
        assert_program_exit_code(
            "type IntArray = [3]int\n"
            "\n"
            "def int main():\n"
            "    IntArray a = [1, 2, 3]\n"
            "    IntArray b = [1, 2, 3]\n"
            "    if a == b:\n"
            "        return 1\n"
            "    return 0\n",
            1,
        )

    def test_large_array_alias_is_still_heap_allocated(self):
        elements = ", ".join(str(1 if i == 4999 else 0) for i in range(5000))
        assert_program_exit_code(
            "type BigArray = [5000]int\n"
            "\n"
            f"def int main():\n"
            f"    BigArray arr = [{elements}]\n"
            f"    return arr[4999]\n",
            1,
        )

    def test_print_array_alias_typed_variable(self):
        assert_program_stdout(
            "type IntArray = [3]int\n"
            "\n"
            "def int main():\n"
            "    IntArray arr = [1, 2, 3]\n"
            "    print(arr)\n"
            "    return 0\n",
            "[3]int[1, 2, 3]\n",
        )

    def test_alias_to_a_struct_name(self):
        assert_program_exit_code(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "type PointAlias = Point\n"
            "\n"
            "def int main():\n"
            "    PointAlias p = Point(3, 4)\n"
            "    return p.x + p.y\n",
            7,
        )

    def test_alias_of_alias_to_a_struct_name(self):
        assert_program_exit_code(
            "type Point struct:\n"
            "    int x\n"
            "\n"
            "type A = Point\n"
            "type B = A\n"
            "\n"
            "def int main():\n"
            "    B p = Point(9)\n"
            "    return p.x\n",
            9,
        )

    def test_alias_to_a_struct_name_as_a_field_type(self):
        assert_program_exit_code(
            "type Inner struct:\n"
            "    int v\n"
            "\n"
            "type InnerAlias = Inner\n"
            "\n"
            "type Outer struct:\n"
            "    InnerAlias inner\n"
            "\n"
            "def int main():\n"
            "    Outer o = Outer(Inner(5))\n"
            "    return o.inner.v\n",
            5,
        )

    def test_constructing_via_the_alias_name_is_not_supported(self):
        assert_program_semantic_error(
            "type Point struct:\n"
            "    int x\n"
            "\n"
            "type PointAlias = Point\n"
            "\n"
            "def int main():\n"
            "    PointAlias p = PointAlias(1)\n"
            "    return p.x\n",
            match="undeclared",
        )

    def test_alias_colliding_with_a_struct_name_is_rejected(self):
        assert_program_semantic_error(
            "type Point = int\n"
            "\n"
            "type Point struct:\n"
            "    int x\n"
            "\n"
            "def int main():\n"
            "    return 0\n",
            match="collides with a struct",
        )

    def test_struct_declared_before_its_colliding_alias_is_still_rejected(self):
        assert_program_semantic_error(
            "type Point struct:\n"
            "    int x\n"
            "\n"
            "type Point = int\n"
            "\n"
            "def int main():\n"
            "    return 0\n",
            match="collides with a struct",
        )

    def test_alias_colliding_with_a_function_name_is_rejected(self):
        assert_program_semantic_error(
            "type foo = int\n"
            "\n"
            "def int foo():\n"
            "    return 0\n",
            match="collides with a type alias",
        )

    def test_alias_named_after_a_builtin_is_rejected(self):
        assert_program_semantic_error(
            "type print = int\n"
            "\n"
            "def int main():\n"
            "    return 0\n",
            match="builtin",
        )


# ---------------------------------------------------------------------------
# Sum types
# ---------------------------------------------------------------------------

class TestSumTypes:


    def test_widening_a_variant_into_a_sum_type_via_var_decl(self):
        ast = _parse(
            "type Circle struct:\n"
            "    int radius\n"
            "\n"
            "type Square struct:\n"
            "    int side\n"
            "\n"
            "type Shape is Circle | Square\n"
            "\n"
            "def int main():\n"
            "    Shape s = Circle(5)\n"
            "    return 0\n"
        )
        analyze(ast)  # should not raise

    def test_widening_a_variant_into_a_sum_type_via_assign(self):
        ast = _parse(
            "type Circle struct:\n"
            "    int radius\n"
            "\n"
            "type Square struct:\n"
            "    int side\n"
            "\n"
            "type Shape is Circle | Square\n"
            "\n"
            "def int main():\n"
            "    Shape s = Circle(5)\n"
            "    s = Square(3)\n"
            "    return 0\n"
        )
        analyze(ast)  # should not raise

    def test_widening_an_already_typed_variable_not_just_a_bare_literal(self):
        ast = _parse(
            "type Circle struct:\n"
            "    int radius\n"
            "\n"
            "type Square struct:\n"
            "    int side\n"
            "\n"
            "type Shape is Circle | Square\n"
            "\n"
            "def int main():\n"
            "    Circle c = Circle(5)\n"
            "    Shape s = c\n"
            "    return 0\n"
        )
        analyze(ast)  # should not raise

    def test_sum_type_as_function_parameter_and_widening_the_argument(self):
        ast = _parse(
            "type Circle struct:\n"
            "    int radius\n"
            "\n"
            "type Square struct:\n"
            "    int side\n"
            "\n"
            "type Shape is Circle | Square\n"
            "\n"
            "def int takesShape(Shape s):\n"
            "    return 0\n"
            "\n"
            "def int main():\n"
            "    return takesShape(Circle(5))\n"
        )
        analyze(ast)  # should not raise

    def test_sum_type_as_return_type_and_widening_the_return_value(self):
        ast = _parse(
            "type Circle struct:\n"
            "    int radius\n"
            "\n"
            "type Square struct:\n"
            "    int side\n"
            "\n"
            "type Shape is Circle | Square\n"
            "\n"
            "def Shape makeShape():\n"
            "    return Circle(5)\n"
            "\n"
            "def int main():\n"
            "    Shape s = makeShape()\n"
            "    return 0\n"
        )
        analyze(ast)  # should not raise

    def test_array_of_sum_type_as_a_local_variable(self):
        ast = _parse(
            "type Circle struct:\n"
            "    int radius\n"
            "\n"
            "type Square struct:\n"
            "    int side\n"
            "\n"
            "type Shape is Circle | Square\n"
            "\n"
            "def int main([3]Shape shapes):\n"
            "    return 0\n"
        )
        analyze(ast)  # should not raise

    def test_array_of_sum_type_with_no_initializer_is_rejected(self):
        assert_program_semantic_error(
            "type Circle struct:\n"
            "    int radius\n"
            "\n"
            "type Square struct:\n"
            "    int side\n"
            "\n"
            "type Shape is Circle | Square\n"
            "\n"
            "def int main():\n"
            "    [3]Shape shapes\n"
            "    return 0\n",
            match="has no initializer",
        )

    def test_more_than_two_variants(self):
        ast = _parse(
            "type Circle struct:\n"
            "    int radius\n"
            "\n"
            "type Square struct:\n"
            "    int side\n"
            "\n"
            "type Triangle struct:\n"
            "    int base\n"
            "\n"
            "type Shape is Circle | Square | Triangle\n"
            "\n"
            "def int main():\n"
            "    Shape s = Triangle(3)\n"
            "    return 0\n"
        )
        analyze(ast)  # should not raise


    def test_widening_an_unrelated_struct_is_rejected(self):
        assert_program_semantic_error(
            "type Circle struct:\n"
            "    int radius\n"
            "\n"
            "type Square struct:\n"
            "    int side\n"
            "\n"
            "type Triangle struct:\n"
            "    int base\n"
            "\n"
            "type Shape is Circle | Square\n"
            "\n"
            "def int main():\n"
            "    Shape s = Triangle(3)\n"
            "    return 0\n",
            match="Cannot initialize 's'",
        )

    def test_variant_naming_an_alias_that_resolves_to_a_scalar_is_accepted(self):
        assert_program_exit_code(
            "type Circle struct:\n"
            "    int radius\n"
            "\n"
            "type MyInt = int\n"
            "\n"
            "type Shape is Circle | MyInt\n"
            "\n"
            "def int main():\n"
            "    return 0\n",
            0,
        )

    def test_unknown_variant_name_is_rejected(self):
        assert_program_semantic_error(
            "type Circle struct:\n"
            "    int radius\n"
            "\n"
            "type Shape is Circle | Nonexistent\n"
            "\n"
            "def int main():\n"
            "    return 0\n",
            match="isn't a declared struct or a valid scalar/str/array/slice/pointer type",
        )

    def test_duplicate_variant_is_rejected(self):
        assert_program_semantic_error(
            "type Circle struct:\n"
            "    int radius\n"
            "\n"
            "type Square struct:\n"
            "    int side\n"
            "\n"
            "type Shape is Circle | Square | Circle\n"
            "\n"
            "def int main():\n"
            "    return 0\n",
            match="more than once",
        )

    def test_no_initializer_is_rejected(self):
        assert_program_semantic_error(
            "type Circle struct:\n"
            "    int radius\n"
            "\n"
            "type Square struct:\n"
            "    int side\n"
            "\n"
            "type Shape is Circle | Square\n"
            "\n"
            "def int main():\n"
            "    Shape s\n"
            "    return 0\n",
            match="has no initializer",
        )

    def test_sum_type_as_a_struct_field_is_rejected(self):
        assert_program_semantic_error(
            "type Circle struct:\n"
            "    int radius\n"
            "\n"
            "type Square struct:\n"
            "    int side\n"
            "\n"
            "type Shape is Circle | Square\n"
            "\n"
            "type Container struct:\n"
            "    Shape s\n"
            "\n"
            "def int main():\n"
            "    return 0\n",
            match="Unknown type 'Shape'",
        )

    def test_array_of_sum_type_as_a_struct_field_is_also_rejected(self):
        assert_program_semantic_error(
            "type Circle struct:\n"
            "    int radius\n"
            "\n"
            "type Square struct:\n"
            "    int side\n"
            "\n"
            "type Shape is Circle | Square\n"
            "\n"
            "type Container struct:\n"
            "    [3]Shape shapes\n"
            "\n"
            "def int main():\n"
            "    return 0\n",
            match="Unknown type 'Shape'",
        )

    def test_equality_between_sum_types_is_rejected(self):
        assert_program_semantic_error(
            "type Circle struct:\n"
            "    int radius\n"
            "\n"
            "type Square struct:\n"
            "    int side\n"
            "\n"
            "type Shape is Circle | Square\n"
            "\n"
            "def int main():\n"
            "    Shape s = Circle(5)\n"
            "    Shape t = Circle(5)\n"
            "    if s == t:\n"
            "        return 1\n"
            "    return 0\n",
            match="does not support slice, void, sum type, dict, or none operands",
        )

    def test_sum_type_name_colliding_with_a_struct_is_rejected(self):
        assert_program_semantic_error(
            "type Circle struct:\n"
            "    int radius\n"
            "\n"
            "type Square struct:\n"
            "    int side\n"
            "\n"
            "type Circle is Circle | Square\n"
            "\n"
            "def int main():\n"
            "    return 0\n",
            match="collides with a struct",
        )

    def test_sum_type_name_colliding_with_an_alias_is_rejected(self):
        assert_program_semantic_error(
            "type Circle struct:\n"
            "    int radius\n"
            "\n"
            "type Square struct:\n"
            "    int side\n"
            "\n"
            "type Shape = int\n"
            "\n"
            "type Shape is Circle | Square\n"
            "\n"
            "def int main():\n"
            "    return 0\n",
            match="collides with a type alias",
        )

    def test_function_name_colliding_with_a_sum_type_is_rejected(self):
        assert_program_semantic_error(
            "type Circle struct:\n"
            "    int radius\n"
            "\n"
            "type Square struct:\n"
            "    int side\n"
            "\n"
            "type Shape is Circle | Square\n"
            "\n"
            "def int Shape():\n"
            "    return 0\n"
            "\n"
            "def int main():\n"
            "    return 0\n",
            match="collides with a sum type",
        )

    def test_sum_type_named_after_a_builtin_is_rejected(self):
        assert_program_semantic_error(
            "type Circle struct:\n"
            "    int radius\n"
            "\n"
            "type Square struct:\n"
            "    int side\n"
            "\n"
            "type print is Circle | Square\n"
            "\n"
            "def int main():\n"
            "    return 0\n",
            match="builtin",
        )

    def test_duplicate_sum_type_declaration_is_rejected(self):
        assert_program_semantic_error(
            "type Circle struct:\n"
            "    int radius\n"
            "\n"
            "type Square struct:\n"
            "    int side\n"
            "\n"
            "type Shape is Circle | Square\n"
            "\n"
            "type Shape is Circle | Square\n"
            "\n"
            "def int main():\n"
            "    return 0\n",
            match="is already declared",
        )


# ---------------------------------------------------------------------------
# Sum types
# ---------------------------------------------------------------------------

class TestSumTypesCodegen:

    _SHAPE_DECLS = (
        "type Circle struct:\n"
        "    int radius\n"
        "\n"
        "type Square struct:\n"
        "    int64 side\n"
        "\n"
        "type Shape is Circle | Square\n"
        "\n"
    )

    def test_var_decl_literal_widening_runs(self):
        assert_program_exit_code(
            self._SHAPE_DECLS +
            "def int main():\n"
            "    Shape s = Circle(5)\n"
            "    return 0\n",
            expected=0,
        )

    def test_var_decl_from_already_sum_typed_variable_runs(self):
        assert_program_exit_code(
            self._SHAPE_DECLS +
            "def int main():\n"
            "    Shape s = Circle(5)\n"
            "    Shape t = s\n"
            "    return 0\n",
            expected=0,
        )

    def test_assign_from_already_sum_typed_variable_runs(self):
        assert_program_exit_code(
            self._SHAPE_DECLS +
            "def int main():\n"
            "    Shape s = Circle(5)\n"
            "    Shape t = Square(9)\n"
            "    t = s\n"
            "    return 0\n",
            expected=0,
        )

    def test_return_forwarding_an_already_sum_typed_value_runs(self):
        assert_program_exit_code(
            self._SHAPE_DECLS +
            "def Shape identity(Shape s):\n"
            "    return s\n"
            "\n"
            "def int main():\n"
            "    Shape s = identity(Circle(5))\n"
            "    return 0\n",
            expected=0,
        )

    def test_array_literal_of_already_sum_typed_elements_runs(self):
        assert_program_exit_code(
            self._SHAPE_DECLS +
            "def int main():\n"
            "    Shape a = Circle(5)\n"
            "    Shape b = Square(9)\n"
            "    [2]Shape shapes = [a, b]\n"
            "    return 0\n",
            expected=0,
        )

    def test_index_assign_from_already_sum_typed_variable_runs(self):
        assert_program_exit_code(
            self._SHAPE_DECLS +
            "def int main():\n"
            "    [2]Shape shapes = [Circle(1), Square(2)]\n"
            "    Shape s = Circle(5)\n"
            "    shapes[0] = s\n"
            "    return 0\n",
            expected=0,
        )

    def test_function_argument_already_sum_typed_runs(self):
        assert_program_exit_code(
            self._SHAPE_DECLS +
            "def int takesShape(Shape s):\n"
            "    return 0\n"
            "\n"
            "def int main():\n"
            "    Shape s = Circle(5)\n"
            "    return takesShape(s)\n",
            expected=0,
        )

    def test_everything_together_runs(self):
        assert_program_exit_code(
            self._SHAPE_DECLS +
            "def int takesShape(Shape s):\n"
            "    return 0\n"
            "\n"
            "def int main():\n"
            "    Shape s = Circle(5)\n"
            "    Shape t = s\n"
            "    t = Square(9)\n"
            "    [2]Shape shapes = [Circle(1), Square(2)]\n"
            "    shapes[0] = Square(3)\n"
            "    []Shape dynShapes\n"
            "    dynShapes = append(dynShapes, Circle(7))\n"
            "    return takesShape(t)\n",
            expected=0,
        )


# ---------------------------------------------------------------------------
# Sum types: print
# ---------------------------------------------------------------------------

class TestSumTypesPrint:

    _SHAPE_DECLS = (
        "type Circle struct:\n"
        "    int radius\n"
        "\n"
        "type Square struct:\n"
        "    int64 side\n"
        "\n"
        "type Shape is Circle | Square\n"
        "\n"
    )

    def test_bare_struct_baseline(self):
        assert_program_stdout(
            self._SHAPE_DECLS +
            "def int main():\n"
            "    Circle c = Circle(5)\n"
            "    print(c)\n"
            "    return 0\n",
            "Circle(radius: 5)\n",
        )

    def test_sum_typed_value_prints_identically_to_the_bare_struct(self):
        assert_program_stdout(
            self._SHAPE_DECLS +
            "def int main():\n"
            "    Shape s = Circle(5)\n"
            "    print(s)\n"
            "    return 0\n",
            "Circle(radius: 5)\n",
        )

    def test_second_variant_prints_its_own_discriminant_correctly(self):
        assert_program_stdout(
            self._SHAPE_DECLS +
            "def int main():\n"
            "    Shape s = Square(9)\n"
            "    print(s)\n"
            "    return 0\n",
            "Square(side: 9)\n",
        )

    def test_three_variants_each_print_correctly(self):
        assert_program_stdout(
            "type Circle struct:\n"
            "    int radius\n"
            "\n"
            "type Square struct:\n"
            "    int64 side\n"
            "\n"
            "type Triangle struct:\n"
            "    [3]int sides\n"
            "\n"
            "type Shape is Circle | Square | Triangle\n"
            "\n"
            "def int main():\n"
            "    Shape a = Circle(1)\n"
            "    Shape b = Square(2)\n"
            "    Shape c = Triangle([3, 4, 5])\n"
            "    print(a)\n"
            "    print(b)\n"
            "    print(c)\n"
            "    return 0\n",
            "Circle(radius: 1)\nSquare(side: 2)\nTriangle(sides: [3]int[3, 4, 5])\n",
        )

    def test_array_of_shapes_prints_each_element_independently(self):
        assert_program_stdout(
            self._SHAPE_DECLS +
            "def int main():\n"
            "    [2]Shape shapes = [Circle(1), Square(2)]\n"
            "    print(shapes)\n"
            "    return 0\n",
            "[2]Shape[Circle(radius: 1), Square(side: 2)]\n",
        )

    def test_variant_with_a_string_field_quotes_correctly(self):
        assert_program_stdout(
            "type Label struct:\n"
            "    str text\n"
            "\n"
            "type Circle struct:\n"
            "    int radius\n"
            "\n"
            "type Shape is Label | Circle\n"
            "\n"
            "def int main():\n"
            "    Label l = Label('hi')\n"
            "    print(l)\n"
            "    Shape s = Label('hi')\n"
            "    print(s)\n"
            "    return 0\n",
            "Label(text: 'hi')\nLabel(text: 'hi')\n",
        )

    def test_sum_typed_parameter_prints_correctly(self):
        assert_program_stdout(
            self._SHAPE_DECLS +
            "def int printIt(Shape s):\n"
            "    print(s)\n"
            "    return 0\n"
            "\n"
            "def int main():\n"
            "    printIt(Circle(1))\n"
            "    printIt(Square(2))\n"
            "    return 0\n",
            "Circle(radius: 1)\nSquare(side: 2)\n",
        )

    def test_sum_typed_parameter_passed_through_another_function(self):
        assert_program_stdout(
            self._SHAPE_DECLS +
            "def int printIt(Shape s):\n"
            "    print(s)\n"
            "    return 0\n"
            "\n"
            "def int forwardIt(Shape s):\n"
            "    return printIt(s)\n"
            "\n"
            "def int main():\n"
            "    forwardIt(Square(7))\n"
            "    return 0\n",
            "Square(side: 7)\n",
        )


# ---------------------------------------------------------------------------
# Sum type narrowing
# ---------------------------------------------------------------------------

class TestNarrowing:

    _SHAPE_DECLS = (
        "type Circle struct:\n"
        "    int radius\n"
        "\n"
        "type Square struct:\n"
        "    int side\n"
        "\n"
        "type Shape is Circle | Square\n"
        "\n"
    )


    def test_narrowed_field_access(self):
        ast = _parse(
            self._SHAPE_DECLS +
            "def int main():\n"
            "    Shape s = Circle(5)\n"
            "    if s is Circle:\n"
            "        return s.radius\n"
            "    return 0\n"
        )
        analyze(ast)  # should not raise

    def test_elif_narrows_to_its_own_variant(self):
        ast = _parse(
            self._SHAPE_DECLS +
            "def int main():\n"
            "    Shape s = Square(9)\n"
            "    if s is Circle:\n"
            "        return s.radius\n"
            "    elif s is Square:\n"
            "        return s.side\n"
            "    return 0\n"
        )
        analyze(ast)  # should not raise

    def test_nested_block_inside_narrowed_branch_still_sees_it(self):
        ast = _parse(
            self._SHAPE_DECLS +
            "def int main():\n"
            "    Shape s = Circle(5)\n"
            "    if s is Circle:\n"
            "        if true:\n"
            "            return s.radius\n"
            "    return 0\n"
        )
        analyze(ast)  # should not raise

    def test_reassigning_an_unrelated_variable_inside_narrowed_branch_is_fine(self):
        ast = _parse(
            self._SHAPE_DECLS +
            "def int main():\n"
            "    Shape s = Circle(5)\n"
            "    if s is Circle:\n"
            "        int x = 5\n"
            "        x = 10\n"
            "        return s.radius\n"
            "    return 0\n"
        )
        analyze(ast)  # should not raise

    def test_two_different_variables_narrowed_simultaneously(self):
        ast = _parse(
            self._SHAPE_DECLS +
            "def int main():\n"
            "    Shape a = Circle(5)\n"
            "    Shape b = Square(9)\n"
            "    if a is Circle:\n"
            "        if b is Square:\n"
            "            return a.radius + b.side\n"
            "    return 0\n"
        )
        analyze(ast)  # should not raise


    def test_reassigning_the_narrowed_variable_is_rejected(self):
        assert_program_semantic_error(
            self._SHAPE_DECLS +
            "def int main():\n"
            "    Shape s = Circle(5)\n"
            "    if s is Circle:\n"
            "        s = Square(9)\n"
            "    return 0\n",
            match="Cannot reassign 's'",
        )

    def test_reassigning_the_narrowed_variable_in_a_nested_block_is_also_rejected(self):
        assert_program_semantic_error(
            self._SHAPE_DECLS +
            "def int main():\n"
            "    Shape s = Circle(5)\n"
            "    if s is Circle:\n"
            "        if true:\n"
            "            s = Square(9)\n"
            "    return 0\n",
            match="Cannot reassign 's'",
        )

    def test_unrelated_struct_is_rejected(self):
        assert_program_semantic_error(
            self._SHAPE_DECLS +
            "type Triangle struct:\n"
            "    int base\n"
            "\n"
            "def int main():\n"
            "    Shape s = Circle(5)\n"
            "    if s is Triangle:\n"
            "        return 0\n"
            "    return 0\n",
            match="is not one of Shape's own declared variants",
        )

    def test_undeclared_type_name_is_rejected(self):
        assert_program_semantic_error(
            self._SHAPE_DECLS +
            "def int main():\n"
            "    Shape s = Circle(5)\n"
            "    if s is Nonexistent:\n"
            "        return 0\n"
            "    return 0\n",
            match="Unknown type 'Nonexistent'",
        )

    def test_non_sum_typed_left_side_is_rejected(self):
        assert_program_semantic_error(
            self._SHAPE_DECLS +
            "def int main():\n"
            "    Circle c = Circle(5)\n"
            "    if c is Circle:\n"
            "        return 0\n"
            "    return 0\n",
            match="is not a sum type",
        )

    def test_undeclared_variable_name_is_rejected(self):
        assert_program_semantic_error(
            self._SHAPE_DECLS +
            "def int main():\n"
            "    if nonexistent is Circle:\n"
            "        return 0\n"
            "    return 0\n",
            match="Reference to undeclared variable",
        )

    def test_narrowing_does_not_leak_outside_the_if(self):
        assert_program_semantic_error(
            self._SHAPE_DECLS +
            "def int main():\n"
            "    Shape s = Circle(5)\n"
            "    if s is Circle:\n"
            "        return s.radius\n"
            "    return s.radius\n",
            match="Cannot access field 'radius' on non-struct type Shape",
        )

    def test_else_body_does_not_narrow(self):
        assert_program_semantic_error(
            self._SHAPE_DECLS +
            "def int main():\n"
            "    Shape s = Circle(5)\n"
            "    if s is Circle:\n"
            "        return 0\n"
            "    else:\n"
            "        return s.radius\n",
            match="Cannot access field 'radius' on non-struct type Shape",
        )


# ---------------------------------------------------------------------------
# Sum type narrowing
# ---------------------------------------------------------------------------

class TestNarrowingCodegen:

    _SHAPE_DECLS = (
        "type Circle struct:\n"
        "    int radius\n"
        "\n"
        "type Square struct:\n"
        "    int width\n"
        "    int height\n"
        "\n"
        "type Shape is Circle | Square\n"
        "\n"
    )

    def test_narrowed_field_read_returns_the_correct_value(self):
        assert_program_exit_code(
            self._SHAPE_DECLS +
            "def int main():\n"
            "    Shape s = Circle(5)\n"
            "    if s is Circle:\n"
            "        return s.radius\n"
            "    return 0\n",
            expected=5,
        )

    def test_false_branch_is_taken_when_the_variant_does_not_match(self):
        assert_program_exit_code(
            self._SHAPE_DECLS +
            "def int main():\n"
            "    Shape s = Square(3, 4)\n"
            "    if s is Circle:\n"
            "        return s.radius\n"
            "    return 99\n",
            expected=99,
        )

    def test_second_field_of_the_narrowed_variant_reads_correctly(self):
        assert_program_exit_code(
            self._SHAPE_DECLS +
            "def int main():\n"
            "    Shape s = Square(3, 4)\n"
            "    if s is Square:\n"
            "        return s.width + s.height\n"
            "    return 0\n",
            expected=7,
        )

    def test_elif_chain_across_three_variants(self):
        assert_program_exit_code(
            "type Circle struct:\n"
            "    int radius\n"
            "\n"
            "type Square struct:\n"
            "    int width\n"
            "    int height\n"
            "\n"
            "type Triangle struct:\n"
            "    int base\n"
            "\n"
            "type Shape is Circle | Square | Triangle\n"
            "\n"
            "def int describe(Shape s):\n"
            "    if s is Circle:\n"
            "        return s.radius * 10\n"
            "    elif s is Square:\n"
            "        return s.width + s.height\n"
            "    elif s is Triangle:\n"
            "        return s.base * 100\n"
            "    return -1\n"
            "\n"
            "def int main():\n"
            "    int total = 0\n"
            "    total = total + describe(Circle(5))\n"
            "    total = total + describe(Square(3, 4))\n"
            "    total = total + describe(Triangle(2))\n"
            "    return total\n",
            expected=(50 + 7 + 200) % 256,
        )

    def test_narrowing_a_function_parameter(self):
        assert_program_exit_code(
            self._SHAPE_DECLS +
            "def int describe(Shape s):\n"
            "    if s is Circle:\n"
            "        return s.radius\n"
            "    return -1\n"
            "\n"
            "def int main():\n"
            "    return describe(Circle(7))\n",
            expected=7,
        )

    def test_nested_block_inside_narrowed_branch_reads_correctly(self):
        assert_program_exit_code(
            self._SHAPE_DECLS +
            "def int main():\n"
            "    Shape s = Circle(9)\n"
            "    if s is Circle:\n"
            "        if true:\n"
            "            return s.radius\n"
            "    return 0\n",
            expected=9,
        )

    def test_two_different_variables_narrowed_simultaneously(self):
        assert_program_exit_code(
            self._SHAPE_DECLS +
            "def int main():\n"
            "    Shape a = Circle(3)\n"
            "    Shape b = Square(4, 5)\n"
            "    if a is Circle:\n"
            "        if b is Square:\n"
            "            return a.radius + b.width + b.height\n"
            "    return 0\n",
            expected=12,
        )


    def test_printing_a_narrowed_variable_prints_as_its_own_variant(self):
        assert_program_stdout(
            self._SHAPE_DECLS +
            "def int main():\n"
            "    Shape s = Circle(5)\n"
            "    if s is Circle:\n"
            "        print(s)\n"
            "    return 0\n",
            "Circle(radius: 5)\n",
        )

    def test_extracting_a_narrowed_variable_into_a_plain_struct_variable(self):
        assert_program_exit_code(
            self._SHAPE_DECLS +
            "def int main():\n"
            "    Shape s = Circle(5)\n"
            "    if s is Circle:\n"
            "        Circle c = s\n"
            "        return c.radius\n"
            "    return 0\n",
            expected=5,
        )

    def test_passing_a_narrowed_variable_as_a_struct_typed_argument(self):
        assert_program_exit_code(
            self._SHAPE_DECLS +
            "def int takesCircle(Circle c):\n"
            "    return c.radius\n"
            "\n"
            "def int main():\n"
            "    Shape s = Circle(5)\n"
            "    if s is Circle:\n"
            "        return takesCircle(s)\n"
            "    return 0\n",
            expected=5,
        )

    def test_returning_a_narrowed_variable_as_a_struct_typed_return_value(self):
        assert_program_exit_code(
            self._SHAPE_DECLS +
            "def Circle asCircle(Shape s):\n"
            "    if s is Circle:\n"
            "        return s\n"
            "    return Circle(-1)\n"
            "\n"
            "def int main():\n"
            "    Circle c = asCircle(Circle(9))\n"
            "    return c.radius\n",
            expected=9,
        )

    def test_equality_between_a_narrowed_variable_and_a_plain_struct(self):
        assert_program_exit_code(
            self._SHAPE_DECLS +
            "def int main():\n"
            "    Shape s = Circle(5)\n"
            "    Circle other = Circle(5)\n"
            "    if s is Circle:\n"
            "        if s == other:\n"
            "            return 1\n"
            "        return 2\n"
            "    return 0\n",
            expected=1,
        )


# ---------------------------------------------------------------------------
# Exhaustive matching
# ---------------------------------------------------------------------------

class TestExhaustiveMatching:

    _SHAPE_DECLS = (
        "type Circle struct:\n"
        "    int radius\n"
        "\n"
        "type Square struct:\n"
        "    int side\n"
        "\n"
        "type Shape is Circle | Square\n"
        "\n"
    )


    def test_exhaustive_two_arm_match(self):
        ast = _parse(
            self._SHAPE_DECLS +
            "def int main():\n"
            "    Shape s = Circle(5)\n"
            "    match s:\n"
            "        is Circle:\n"
            "            return s.radius\n"
            "        is Square:\n"
            "            return s.side\n"
            "    return 0\n"
        )
        analyze(ast)  # should not raise

    def test_exhaustive_three_arm_match(self):
        ast = _parse(
            "type Circle struct:\n"
            "    int radius\n"
            "\n"
            "type Square struct:\n"
            "    int side\n"
            "\n"
            "type Triangle struct:\n"
            "    int base\n"
            "\n"
            "type Shape is Circle | Square | Triangle\n"
            "\n"
            "def int main():\n"
            "    Shape s = Circle(5)\n"
            "    match s:\n"
            "        is Circle:\n"
            "            return s.radius\n"
            "        is Square:\n"
            "            return s.side\n"
            "        is Triangle:\n"
            "            return s.base\n"
            "    return 0\n"
        )
        analyze(ast)  # should not raise

    def test_explicit_else_covers_the_rest_without_testing_every_variant(self):
        ast = _parse(
            self._SHAPE_DECLS +
            "def int main():\n"
            "    Shape s = Circle(5)\n"
            "    match s:\n"
            "        is Circle:\n"
            "            return s.radius\n"
            "        else:\n"
            "            return -1\n"
            "    return 0\n"
        )
        analyze(ast)  # should not raise

    def test_ambiguous_ordinary_if_inside_explicit_else_is_not_mistaken_for_another_arm(self):
        ast = _parse(
            self._SHAPE_DECLS +
            "def int main():\n"
            "    Shape s = Circle(5)\n"
            "    match s:\n"
            "        is Circle:\n"
            "            return s.radius\n"
            "        else:\n"
            "            if s is Square:\n"
            "                return s.side\n"
            "            return -1\n"
            "    return 0\n"
        )
        analyze(ast)  # should not raise

    def test_exhaustive_match_with_no_else_and_no_trailing_return_satisfies_all_paths_return(self):
        ast = _parse(
            self._SHAPE_DECLS +
            "def int describe(Shape s):\n"
            "    match s:\n"
            "        is Circle:\n"
            "            return s.radius\n"
            "        is Square:\n"
            "            return s.side\n"
        )
        analyze(ast)  # should not raise


    def test_missing_variant_is_rejected_and_named(self):
        assert_program_semantic_error(
            self._SHAPE_DECLS +
            "def int main():\n"
            "    Shape s = Circle(5)\n"
            "    match s:\n"
            "        is Circle:\n"
            "            return s.radius\n"
            "    return 0\n",
            match="missing: Square",
        )

    def test_missing_multiple_variants_are_all_named(self):
        assert_program_semantic_error(
            "type Circle struct:\n"
            "    int radius\n"
            "\n"
            "type Square struct:\n"
            "    int side\n"
            "\n"
            "type Triangle struct:\n"
            "    int base\n"
            "\n"
            "type Shape is Circle | Square | Triangle\n"
            "\n"
            "def int main():\n"
            "    Shape s = Circle(5)\n"
            "    match s:\n"
            "        is Circle:\n"
            "            return s.radius\n"
            "    return 0\n",
            match="missing: Square, Triangle",
        )

    def test_duplicate_arm_is_rejected(self):
        assert_program_semantic_error(
            self._SHAPE_DECLS +
            "def int main():\n"
            "    Shape s = Circle(5)\n"
            "    match s:\n"
            "        is Circle:\n"
            "            return s.radius\n"
            "        is Circle:\n"
            "            return 0\n"
            "        is Square:\n"
            "            return s.side\n"
            "    return 0\n",
            match="'Circle' is tested more than once",
        )

    def test_one_arm_not_returning_still_fails_all_paths_return(self):
        assert_program_semantic_error(
            self._SHAPE_DECLS +
            "def int describe(Shape s):\n"
            "    match s:\n"
            "        is Circle:\n"
            "            return s.radius\n"
            "        is Square:\n"
            "            int x = s.side\n",
            match="does not return a value on all code paths",
        )

    def test_ordinary_if_elif_with_no_else_is_still_rejected(self):
        assert_program_semantic_error(
            "def int describe(int x):\n"
            "    if x > 0:\n"
            "        return 1\n"
            "    elif x < 0:\n"
            "        return -1\n",
            match="does not return a value on all code paths",
        )


class TestExhaustiveMatchingCodegen:

    _SHAPE_DECLS = (
        "type Circle struct:\n"
        "    int radius\n"
        "\n"
        "type Square struct:\n"
        "    int side\n"
        "\n"
        "type Shape is Circle | Square\n"
        "\n"
    )

    def test_match_narrows_and_reads_the_correct_field_per_arm(self):
        assert_program_exit_code(
            self._SHAPE_DECLS +
            "def int describe(Shape s):\n"
            "    match s:\n"
            "        is Circle:\n"
            "            return s.radius\n"
            "        is Square:\n"
            "            return s.side\n"
            "\n"
            "def int main():\n"
            "    return describe(Square(9))\n",
            expected=9,
        )

    def test_match_with_explicit_else_runs_correctly(self):
        assert_program_exit_code(
            self._SHAPE_DECLS +
            "def int describe(Shape s):\n"
            "    match s:\n"
            "        is Circle:\n"
            "            return s.radius\n"
            "        else:\n"
            "            return -1\n"
            "\n"
            "def int main():\n"
            "    return describe(Square(9))\n",
            expected=256 - 1,
        )


class TestNarrowingNonBareVariable:
    """`is`/`match` on a subject that is not a bare variable."""

    _SHAPE_DECLS = (
        "type Circle struct:\n"
        "    int radius\n"
        "\n"
        "type Square struct:\n"
        "    int side\n"
        "\n"
        "type Shape is Circle | Square\n"
        "\n"
    )

    def test_array_element_subject(self):
        assert_program_exit_code(
            self._SHAPE_DECLS +
            "def int main():\n"
            "    [2]Shape shapes = [Circle(5), Square(3)]\n"
            "    if shapes[0] is Circle as c:\n"
            "        return c.radius\n"
            "    return -1\n",
            expected=5,
        )

    def test_array_element_subject_false_branch(self):
        assert_program_exit_code(
            self._SHAPE_DECLS +
            "def int main():\n"
            "    [2]Shape shapes = [Square(9), Circle(2)]\n"
            "    if shapes[0] is Circle as c:\n"
            "        return c.radius\n"
            "    return 99\n",
            expected=99,
        )

    def test_call_return_value_subject(self):
        assert_program_exit_code(
            self._SHAPE_DECLS +
            "def Shape makeShape(int flag):\n"
            "    if flag == 1:\n"
            "        return Circle(7)\n"
            "    return Square(4)\n"
            "\n"
            "def int main():\n"
            "    if makeShape(1) is Circle as c:\n"
            "        return c.radius\n"
            "    return -1\n",
            expected=7,
        )

    def test_bare_variable_subject_can_still_be_explicitly_renamed(self):
        assert_program_exit_code(
            self._SHAPE_DECLS +
            "def int main():\n"
            "    Shape x = Circle(11)\n"
            "    if x is Circle as y:\n"
            "        return y.radius\n"
            "    return -1\n",
            expected=11,
        )

    def test_binding_stays_in_scope_and_un_narrowed_in_else(self):
        assert_program_exit_code(
            self._SHAPE_DECLS +
            "def int main():\n"
            "    [2]Shape shapes = [Square(9), Circle(2)]\n"
            "    if shapes[0] is Circle as c:\n"
            "        return c.radius\n"
            "    else:\n"
            "        if c is Square as s:\n"
            "            return s.side\n"
            "        return -1\n",
            expected=9,
        )

    def test_array_element_subject_evaluated_exactly_once(self):
        assert_program_exit_code(
            self._SHAPE_DECLS +
            "def int nextIndex(*int counter):\n"
            "    int current = *counter\n"
            "    *counter = current + 1\n"
            "    return current\n"
            "\n"
            "def int main():\n"
            "    [2]Shape shapes = [Circle(6), Square(1)]\n"
            "    int counter = 0\n"
            "    *int p = &counter\n"
            "    if shapes[nextIndex(p)] is Circle as c:\n"
            "        return c.radius * 10 + counter\n"
            "    return -1\n",
            expected=6 * 10 + 1,
        )

    def test_call_subject_evaluated_exactly_once(self):
        assert_program_exit_code(
            self._SHAPE_DECLS +
            "def Shape makeShape(*int counter):\n"
            "    int current = *counter\n"
            "    *counter = current + 1\n"
            "    if current == 0:\n"
            "        return Circle(9)\n"
            "    return Square(9)\n"
            "\n"
            "def int main():\n"
            "    int counter = 0\n"
            "    *int p = &counter\n"
            "    if makeShape(p) is Circle as c:\n"
            "        return c.radius * 10 + counter\n"
            "    return -1\n",
            expected=9 * 10 + 1,
        )

    def test_match_with_call_subject_evaluated_exactly_once_even_across_arms(self):
        assert_program_exit_code(
            "type Circle struct:\n"
            "    int radius\n"
            "\n"
            "type Square struct:\n"
            "    int side\n"
            "\n"
            "type Triangle struct:\n"
            "    int base\n"
            "\n"
            "type Shape is Circle | Square | Triangle\n"
            "\n"
            "def Shape makeShape(*int counter):\n"
            "    int current = *counter\n"
            "    *counter = current + 1\n"
            "    return Triangle(current)\n"
            "\n"
            "def int main():\n"
            "    int counter = 0\n"
            "    *int p = &counter\n"
            "    match makeShape(p) as s:\n"
            "        is Circle:\n"
            "            return -1\n"
            "        is Square:\n"
            "            return -2\n"
            "        is Triangle:\n"
            "            return counter * 100 + s.base\n",
            expected=100,
        )

    def test_match_bare_subject_still_works_unrenamed(self):
        assert_program_exit_code(
            self._SHAPE_DECLS +
            "def int main():\n"
            "    Shape shape = Square(6)\n"
            "    match shape:\n"
            "        is Circle:\n"
            "            return -1\n"
            "        is Square:\n"
            "            return shape.side\n",
            expected=6,
        )

    def test_match_bare_subject_with_explicit_rename(self):
        assert_program_exit_code(
            self._SHAPE_DECLS +
            "def int main():\n"
            "    Shape shape = Circle(13)\n"
            "    match shape as s:\n"
            "        is Circle:\n"
            "            return s.radius\n"
            "        is Square:\n"
            "            return -1\n",
            expected=13,
        )

    def test_match_with_call_subject_exhaustiveness_still_checked(self):
        assert_program_semantic_error(
            self._SHAPE_DECLS +
            "def Shape makeShape():\n"
            "    return Circle(1)\n"
            "\n"
            "def int main():\n"
            "    match makeShape() as s:\n"
            "        is Circle:\n"
            "            return s.radius\n",
            match="doesn't cover every variant",
        )

    def test_non_sum_typed_subject_is_rejected(self):
        assert_program_semantic_error(
            self._SHAPE_DECLS +
            "def int main():\n"
            "    [2]int arr = [1, 2]\n"
            "    if arr[0] is Circle as c:\n"
            "        return c.radius\n"
            "    return 0\n",
            match="The expression bound to 'c'",
        )

    def test_subject_variant_not_in_the_sum_type_is_rejected(self):
        assert_program_semantic_error(
            self._SHAPE_DECLS +
            "type Triangle struct:\n"
            "    int base\n"
            "\n"
            "def int main():\n"
            "    [2]Shape shapes = [Circle(5), Square(3)]\n"
            "    if shapes[0] is Triangle as t:\n"
            "        return 1\n"
            "    return 0\n",
            match="not one of Shape's own",
        )

    def test_sum_typed_struct_field_is_still_rejected(self):
        assert_program_semantic_error(
            self._SHAPE_DECLS +
            "type Holder struct:\n"
            "    Shape s\n"
            "\n"
            "def int main():\n"
            "    return 0\n",
            match="Unknown type 'Shape'",
        )

    def test_two_independent_bindings_coexist(self):
        assert_program_exit_code(
            self._SHAPE_DECLS +
            "def int main():\n"
            "    [2]Shape shapes = [Circle(4), Square(7)]\n"
            "    if shapes[0] is Circle as c:\n"
            "        if shapes[1] is Square as s:\n"
            "            return c.radius * 10 + s.side\n"
            "        return -1\n"
            "    return -2\n",
            expected=4 * 10 + 7,
        )


# ---------------------------------------------------------------------------
# Sum types, scalar/str variants
# ---------------------------------------------------------------------------

class TestScalarAndStrVariants:

    _MIXED_DECLS = (
        "type Circle struct:\n"
        "    int radius\n"
        "\n"
        "type Mixed is Circle | int | str\n"
        "\n"
    )


    def test_two_scalar_variants_parses_and_resolves(self):
        assert_program_exit_code(
            "type Number is int | str\n"
            "\n"
            "def int main():\n"
            "    return 0\n",
            expected=0,
        )

    def test_duplicate_scalar_variant_by_identical_spelling_is_rejected(self):
        assert_program_semantic_error(
            "type Number is int | int\n"
            "\n"
            "def int main():\n"
            "    return 0\n",
            match="more than once",
        )

    def test_duplicate_scalar_variant_by_different_spelling_is_rejected(self):
        assert_program_semantic_error(
            "type Number is byte | uint8\n"
            "\n"
            "def int main():\n"
            "    return 0\n",
            match="more than once",
        )

    def test_every_scalar_kind_is_a_valid_variant(self):
        assert_program_exit_code(
            "type Anything is int | int8 | uint8 | int32 | bool | str\n"
            "\n"
            "def int main():\n"
            "    return 0\n",
            expected=0,
        )

    def test_struct_and_scalar_and_str_mixed_in_one_sum_type(self):
        assert_program_exit_code(
            self._MIXED_DECLS +
            "def int main():\n"
            "    return 0\n",
            expected=0,
        )


    def test_widening_an_int_literal_prints_correctly(self):
        assert_program_stdout(
            self._MIXED_DECLS +
            "def int main():\n"
            "    Mixed m = 42\n"
            "    print(m)\n"
            "    return 0\n",
            "42\n",
        )

    def test_widening_a_str_literal_prints_correctly(self):
        assert_program_stdout(
            self._MIXED_DECLS +
            "def int main():\n"
            "    Mixed m = 'hello'\n"
            "    print(m)\n"
            "    return 0\n",
            "'hello'\n",
        )

    def test_widening_via_assign_not_just_var_decl(self):
        assert_program_stdout(
            self._MIXED_DECLS +
            "def int main():\n"
            "    Mixed m = Circle(1)\n"
            "    m = 7\n"
            "    print(m)\n"
            "    return 0\n",
            "7\n",
        )

    def test_widening_an_array_element(self):
        assert_program_stdout(
            self._MIXED_DECLS +
            "def int main():\n"
            "    [3]Mixed arr = [Circle(1), 2, 'three']\n"
            "    print(arr[1])\n"
            "    return 0\n",
            "2\n",
        )

    def test_widening_a_scalar_function_argument(self):
        assert_program_stdout(
            self._MIXED_DECLS +
            "def int describe(Mixed m):\n"
            "    print(m)\n"
            "    return 0\n"
            "\n"
            "def int main():\n"
            "    describe(42)\n"
            "    return 0\n",
            "42\n",
        )

    def test_widening_a_scalar_return_value(self):
        assert_program_stdout(
            self._MIXED_DECLS +
            "def Mixed makeInt():\n"
            "    return 99\n"
            "\n"
            "def int main():\n"
            "    Mixed m = makeInt()\n"
            "    print(m)\n"
            "    return 0\n",
            "99\n",
        )

    def test_widening_a_str_return_value(self):
        assert_program_stdout(
            self._MIXED_DECLS +
            "def Mixed makeStr():\n"
            "    return 'hi'\n"
            "\n"
            "def int main():\n"
            "    Mixed m = makeStr()\n"
            "    print(m)\n"
            "    return 0\n",
            "'hi'\n",
        )


    def test_narrowing_to_int_reads_back_a_usable_value(self):
        assert_program_exit_code(
            self._MIXED_DECLS +
            "def int main():\n"
            "    Mixed m = 42\n"
            "    if m is int:\n"
            "        return m + 1\n"
            "    return -1\n",
            expected=43,
        )

    def test_narrowing_to_str_reads_back_a_usable_value(self):
        assert_program_exit_code(
            self._MIXED_DECLS +
            "def int main():\n"
            "    Mixed m = 'hello'\n"
            "    if m is str:\n"
            "        return len(m)\n"
            "    return -1\n",
            expected=5,
        )

    def test_false_branch_when_narrowing_to_int_but_actual_variant_differs(self):
        assert_program_exit_code(
            self._MIXED_DECLS +
            "def int main():\n"
            "    Mixed m = Circle(9)\n"
            "    if m is int:\n"
            "        return 1\n"
            "    return 0\n",
            expected=0,
        )

    def test_narrowing_a_function_parameter_to_a_scalar(self):
        assert_program_exit_code(
            self._MIXED_DECLS +
            "def int describe(Mixed m):\n"
            "    if m is int:\n"
            "        return m * 10\n"
            "    return -1\n"
            "\n"
            "def int main():\n"
            "    return describe(6)\n",
            expected=60,
        )

    def test_narrowing_a_non_bare_variable_subject_to_a_scalar(self):
        assert_program_exit_code(
            self._MIXED_DECLS +
            "def int main():\n"
            "    [2]Mixed arr = [1, Circle(2)]\n"
            "    if arr[0] is int as n:\n"
            "        return n\n"
            "    return -1\n",
            expected=1,
        )

    def test_all_three_variant_kinds_narrowed_across_array_elements(self):
        assert_program_stdout(
            self._MIXED_DECLS +
            "def int main():\n"
            "    [3]Mixed arr = [Circle(5), 6, 'seven']\n"
            "    int i = 0\n"
            "    while i < 3:\n"
            "        if arr[i] is Circle as c:\n"
            "            print(c.radius)\n"
            "        if arr[i] is int as n:\n"
            "            print(n)\n"
            "        if arr[i] is str as s:\n"
            "            print(s)\n"
            "        i = i + 1\n"
            "    return 0\n",
            "5\n6\nseven\n",
        )

    def test_equality_between_a_narrowed_int_and_a_plain_int(self):
        assert_program_exit_code(
            self._MIXED_DECLS +
            "def int main():\n"
            "    Mixed m = 99\n"
            "    if m is int:\n"
            "        if m == 99:\n"
            "            return 1\n"
            "    return 0\n",
            expected=1,
        )


    def test_match_with_a_branch_per_variant_kind(self):
        assert_program_stdout(
            self._MIXED_DECLS +
            "def int describe(Mixed m):\n"
            "    match m:\n"
            "        is Circle:\n"
            "            return m.radius\n"
            "        is int:\n"
            "            return m * 10\n"
            "        is str:\n"
            "            return len(m)\n"
            "\n"
            "def int main():\n"
            "    print(describe(Circle(5)))\n"
            "    print(describe(42))\n"
            "    print(describe('hello'))\n"
            "    return 0\n",
            "5\n420\n5\n",
        )

    def test_match_missing_a_scalar_arm_is_rejected_as_non_exhaustive(self):
        assert_program_semantic_error(
            self._MIXED_DECLS +
            "def int f(Mixed m):\n"
            "    match m:\n"
            "        is Circle:\n"
            "            return 1\n"
            "        is int:\n"
            "            return 2\n"
            "    return 0\n",
            match="missing: str",
        )

    def test_match_testing_the_same_scalar_variant_twice_is_rejected(self):
        assert_program_semantic_error(
            self._MIXED_DECLS +
            "def int f(Mixed m):\n"
            "    match m:\n"
            "        is int:\n"
            "            return 1\n"
            "        is int:\n"
            "            return 2\n"
            "        is str:\n"
            "            return 3\n"
            "        else:\n"
            "            return 0\n",
            match="tested more than once",
        )

# ---------------------------------------------------------------------------
# Sum types, array/slice/pointer variants
# ---------------------------------------------------------------------------

class TestArraySliceAndPointerVariants:

    _CIRCLE_MIXED_DECLS = (
        "type Circle struct:\n"
        "    int radius\n"
        "\n"
        "type Everything is Circle | int | str | [2]int | []int | *int\n"
        "\n"
    )


    def test_array_slice_and_pointer_each_parse_and_resolve_as_variants(self):
        assert_program_exit_code(
            "type X is [3]int | []int | *int | str\n"
            "\n"
            "def int main():\n"
            "    return 0\n",
            expected=0,
        )

    def test_all_six_variant_kinds_mixed_in_one_sum_type(self):
        assert_program_exit_code(
            self._CIRCLE_MIXED_DECLS +
            "def int main():\n"
            "    return 0\n",
            expected=0,
        )

    def test_identical_array_type_listed_twice_is_rejected(self):
        assert_program_semantic_error(
            "type X is [3]int | [3]int\n"
            "\n"
            "def int main():\n"
            "    return 0\n",
            match="more than once",
        )

    def test_arrays_of_different_size_are_distinct_variants(self):
        assert_program_exit_code(
            "type X is [3]int | [4]int\n"
            "\n"
            "def int main():\n"
            "    return 0\n",
            expected=0,
        )

    def test_identical_slice_type_listed_twice_is_rejected(self):
        assert_program_semantic_error(
            "type X is []int | []int\n"
            "\n"
            "def int main():\n"
            "    return 0\n",
            match="more than once",
        )

    def test_identical_pointer_type_listed_twice_is_rejected(self):
        assert_program_semantic_error(
            "type X is *int | *int\n"
            "\n"
            "def int main():\n"
            "    return 0\n",
            match="more than once",
        )

    def test_pointers_to_different_types_are_distinct_variants(self):
        assert_program_exit_code(
            "type X is *int | *bool\n"
            "\n"
            "def int main():\n"
            "    return 0\n",
            expected=0,
        )

    def test_array_of_a_sum_type_as_a_variant_is_rejected(self):
        assert_program_semantic_error(
            "type Circle struct:\n"
            "    int radius\n"
            "\n"
            "type Square struct:\n"
            "    int side\n"
            "\n"
            "type Shape is Circle | Square\n"
            "\n"
            "type Nested is [2]Shape | int\n"
            "\n"
            "def int main():\n"
            "    return 0\n",
            match="Unknown type 'Shape'",
        )

    def test_a_sum_type_named_bare_as_a_variant_is_rejected(self):
        assert_program_semantic_error(
            "type Circle struct:\n"
            "    int radius\n"
            "\n"
            "type Square struct:\n"
            "    int side\n"
            "\n"
            "type Shape is Circle | Square\n"
            "\n"
            "type Nested is Shape | int\n"
            "\n"
            "def int main():\n"
            "    return 0\n",
            match="is itself a sum type",
        )


    def test_widening_an_array_literal(self):
        assert_program_exit_code(
            "type Thing is [3]int | str\n"
            "\n"
            "def int main():\n"
            "    Thing t = [1, 2, 3]\n"
            "    if t is [3]int as arr:\n"
            "        return arr[0] + arr[1] + arr[2]\n"
            "    return -1\n",
            expected=6,
        )

    def test_widening_a_slice(self):
        assert_program_exit_code(
            "type Thing is []int | str\n"
            "\n"
            "def int main():\n"
            "    [3]int a = [1, 2, 3]\n"
            "    Thing t = a[0:2]\n"
            "    if t is []int as s:\n"
            "        return s[0] + s[1]\n"
            "    return -1\n",
            expected=3,
        )

    def test_widening_a_pointer(self):
        assert_program_exit_code(
            "type Thing is *int | str\n"
            "\n"
            "def int main():\n"
            "    int x = 42\n"
            "    Thing t = &x\n"
            "    if t is *int as p:\n"
            "        return *p\n"
            "    return -1\n",
            expected=42,
        )

    def test_widening_an_array_function_argument(self):
        assert_program_stdout(
            "type Thing is [2]int | str\n"
            "\n"
            "def int describe(Thing t):\n"
            "    if t is [2]int as arr:\n"
            "        print(arr[0] + arr[1])\n"
            "    return 0\n"
            "\n"
            "def int main():\n"
            "    describe([4, 5])\n"
            "    return 0\n",
            "9\n",
        )

    def test_widening_an_array_return_value(self):
        assert_program_stdout(
            "type Thing is [3]int | str\n"
            "\n"
            "def Thing makeArr():\n"
            "    return [7, 8, 9]\n"
            "\n"
            "def int main():\n"
            "    Thing t = makeArr()\n"
            "    if t is [3]int as arr:\n"
            "        print(arr[0] + arr[1] + arr[2])\n"
            "    return 0\n",
            "24\n",
        )

    def test_widening_a_slice_function_argument_and_return_value(self):
        assert_program_stdout(
            "type Thing is []int | str\n"
            "\n"
            "def int sumIt(Thing t):\n"
            "    if t is []int as s:\n"
            "        int total = 0\n"
            "        int i = 0\n"
            "        while i < len(s):\n"
            "            total = total + s[i]\n"
            "            i = i + 1\n"
            "        return total\n"
            "    return -1\n"
            "\n"
            "def Thing makeSlice():\n"
            "    [3]int a = [7, 8, 9]\n"
            "    return a[0:2]\n"
            "\n"
            "def int main():\n"
            "    [3]int a = [1, 2, 3]\n"
            "    print(sumIt(a[0:3]))\n"
            "    print(sumIt(makeSlice()))\n"
            "    return 0\n",
            "6\n15\n",
        )

    def test_widening_a_pointer_function_argument_and_return_value(self):
        assert_program_stdout(
            "type Thing is *int | str\n"
            "\n"
            "def int readIt(Thing t):\n"
            "    if t is *int as p:\n"
            "        print(*p)\n"
            "    return 0\n"
            "\n"
            "def Thing makePtr():\n"
            "    int x = 55\n"
            "    return &x\n"
            "\n"
            "def int main():\n"
            "    int y = 33\n"
            "    readIt(&y)\n"
            "    readIt(makePtr())\n"
            "    return 0\n",
            "33\n55\n",
        )


    def test_false_branch_when_narrowing_to_array_but_actual_variant_differs(self):
        assert_program_exit_code(
            "type Thing is [3]int | int\n"
            "\n"
            "def int main():\n"
            "    Thing t = 42\n"
            "    if t is [3]int as arr:\n"
            "        return 1\n"
            "    return 0\n",
            expected=0,
        )

    def test_narrowing_a_non_bare_variable_subject_to_an_array(self):
        assert_program_exit_code(
            "type Thing is [2]int | str\n"
            "\n"
            "def Thing makeThing():\n"
            "    return [7, 8]\n"
            "\n"
            "def int main():\n"
            "    if makeThing() is [2]int as arr:\n"
            "        return arr[0] + arr[1]\n"
            "    return -1\n",
            expected=15,
        )

    def test_narrowing_a_non_bare_variable_subject_to_a_slice(self):
        assert_program_exit_code(
            "type Thing is []int | str\n"
            "\n"
            "def Thing makeThing():\n"
            "    [3]int a = [4, 5, 6]\n"
            "    return a[0:2]\n"
            "\n"
            "def int main():\n"
            "    if makeThing() is []int as s:\n"
            "        return s[0] + s[1]\n"
            "    return -1\n",
            expected=9,
        )

    def test_narrowing_a_non_bare_variable_subject_to_a_pointer(self):
        assert_program_exit_code(
            "type Thing is *int | str\n"
            "\n"
            "def Thing makeThing():\n"
            "    int x = 77\n"
            "    return &x\n"
            "\n"
            "def int main():\n"
            "    if makeThing() is *int as p:\n"
            "        return *p\n"
            "    return -1\n",
            expected=77,
        )

    def test_narrowing_to_slice_when_the_sum_type_is_heap_promoted(self):
        assert_program_exit_code(
            "type Thing is [5000]int | []int\n"
            "\n"
            "def int main():\n"
            "    [3]int a = [1, 2, 3]\n"
            "    Thing t = a[0:2]\n"
            "    if t is []int as s:\n"
            "        return s[0] + s[1]\n"
            "    return -1\n",
            expected=3,
        )

    def test_address_of_a_narrowed_scalar_binding(self):
        assert_program_exit_code(
            "type Mixed is int | str\n"
            "\n"
            "def int main():\n"
            "    Mixed m = 42\n"
            "    if m is int as n:\n"
            "        int p = *&n\n"
            "        return p\n"
            "    return -1\n",
            expected=42,
        )

    def test_address_of_a_narrowed_array_binding(self):
        assert_program_exit_code(
            "type Thing is [3]int | str\n"
            "\n"
            "def int main():\n"
            "    Thing t = [10, 20, 30]\n"
            "    if t is [3]int as arr:\n"
            "        *[3]int p = &arr\n"
            "        return (*p)[1]\n"
            "    return -1\n",
            expected=20,
        )


    def test_match_with_all_six_variant_kinds(self):
        assert_program_stdout(
            self._CIRCLE_MIXED_DECLS +
            "def int describe(Everything e):\n"
            "    match e:\n"
            "        is Circle:\n"
            "            return e.radius\n"
            "        is int:\n"
            "            return e * 10\n"
            "        is str:\n"
            "            return len(e)\n"
            "        is [2]int:\n"
            "            return e[0] + e[1]\n"
            "        is []int:\n"
            "            return len(e)\n"
            "        is *int:\n"
            "            return *e\n"
            "\n"
            "def int main():\n"
            "    print(describe(Circle(3)))\n"
            "    print(describe(42))\n"
            "    print(describe('hello'))\n"
            "    print(describe([5, 6]))\n"
            "    [4]int a = [1, 2, 3, 4]\n"
            "    print(describe(a[0:3]))\n"
            "    int x = 99\n"
            "    print(describe(&x))\n"
            "    return 0\n",
            "3\n420\n5\n11\n3\n99\n",
        )

    def test_match_missing_a_pointer_arm_is_rejected_as_non_exhaustive(self):
        assert_program_semantic_error(
            "type Thing is [3]int | []int | *int\n"
            "\n"
            "def int f(Thing t):\n"
            "    match t:\n"
            "        is [3]int:\n"
            "            return 1\n"
            "        is []int:\n"
            "            return 2\n"
            "    return 0\n",
            match=r"missing: \*int",
        )

# ---------------------------------------------------------------------------
# Dict
# ---------------------------------------------------------------------------

class TestDicts:

    def test_dict_type_and_literal_parse_and_resolve(self):
        assert_program_exit_code(
            "def int main():\n"
            "    dict[str]int ages = dict[str]int{\n"
            "        'alice': 25,\n"
            "        'bob': 17,\n"
            "    }\n"
            "    return 0\n",
            expected=0,
        )

    def test_dict_literal_with_no_trailing_comma_still_parses(self):
        assert_program_exit_code(
            "def int main():\n"
            "    dict[str]int ages = dict[str]int{\n"
            "        'alice': 25,\n"
            "        'bob': 17\n"
            "    }\n"
            "    return 0\n",
            expected=0,
        )

    def test_single_line_dict_literal(self):
        assert_program_exit_code(
            "def int main():\n"
            "    dict[str]int ages = dict[str]int{'alice': 25, 'bob': 17}\n"
            "    return 0\n",
            expected=0,
        )

    def test_empty_dict_literal(self):
        assert_program_exit_code(
            "def int main():\n"
            "    dict[str]int ages = dict[str]int{}\n"
            "    return 0\n",
            expected=0,
        )


    def test_value_type_mismatch_is_rejected(self):
        assert_program_semantic_error(
            "def int main():\n"
            "    dict[str]int ages = dict[str]int{'alice': 'oops'}\n"
            "    return 0\n",
            match="declares value type int, but a value is str",
        )

    def test_key_type_mismatch_is_rejected(self):
        assert_program_semantic_error(
            "def int main():\n"
            "    dict[str]int ages = dict[str]int{42: 25}\n"
            "    return 0\n",
            match="declares key type str, but a key is int",
        )

    def test_duplicate_literal_key_is_rejected(self):
        assert_program_semantic_error(
            "def int main():\n"
            "    dict[str]int ages = dict[str]int{'alice': 25, 'alice': 30}\n"
            "    return 0\n",
            match="lists the key 'alice' more than once",
        )

    def test_slice_key_type_is_rejected(self):
        assert_program_semantic_error(
            "def int main():\n"
            "    dict[[]int]str x\n"
            "    return 0\n",
            match="can't be a dict's own key type",
        )

    def test_struct_key_type_is_rejected(self):
        assert_program_semantic_error(
            "type Circle struct:\n"
            "    int radius\n"
            "\n"
            "def int main():\n"
            "    dict[Circle]str x\n"
            "    return 0\n",
            match="can't be a dict's own key type",
        )

    def test_every_valid_key_kind_is_accepted(self):
        assert_program_exit_code(
            "def int main():\n"
            "    dict[int]int a = dict[int]int{1: 1}\n"
            "    dict[int8]int b = dict[int8]int{1: 1}\n"
            "    dict[uint8]int c = dict[uint8]int{1: 1}\n"
            "    dict[int64]int d = dict[int64]int{1: 1}\n"
            "    dict[bool]int e = dict[bool]int{true: 1}\n"
            "    dict[str]int f = dict[str]int{'x': 1}\n"
            "    return 0\n",
            expected=0,
        )


    def test_str_keyed_dict_prints_correctly(self):
        assert_program_stdout(
            "def int main():\n"
            "    dict[str]int ages = dict[str]int{'alice': 25}\n"
            "    print(ages)\n"
            "    return 0\n",
            "dict[str]int{'alice': 25}\n",
        )

    def test_int_keyed_dict_prints_correctly(self):
        assert_program_stdout(
            "def int main():\n"
            "    dict[int]str codes = dict[int]str{1: 'one'}\n"
            "    print(codes)\n"
            "    return 0\n",
            "dict[int]str{1: 'one'}\n",
        )

    def test_bool_keyed_dict_prints_correctly(self):
        assert_program_stdout(
            "def int main():\n"
            "    dict[bool]int flags = dict[bool]int{true: 100, false: 200}\n"
            "    print(flags)\n"
            "    return 0\n",
            "dict[bool]int{true: 100, false: 200}\n",
        )

    def test_struct_valued_dict_prints_correctly(self):
        assert_program_stdout(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def int main():\n"
            "    dict[str]Point points = dict[str]Point{'origin': Point(0, 0)}\n"
            "    print(points)\n"
            "    return 0\n",
            "dict[str]Point{'origin': Point(x: 0, y: 0)}\n",
        )

    def test_multiple_entries_all_present_regardless_of_print_order(self):
        result = compile_and_run(
            "def int main():\n"
            "    dict[str]int ages = dict[str]int{\n"
            "        'alice': 25,\n"
            "        'bob': 17,\n"
            "    }\n"
            "    print(ages)\n"
            "    return 0\n"
        )
        assert "'alice': 25" in result.stdout
        assert "'bob': 17" in result.stdout

    def test_escaping_dict_variable_survives_past_its_own_function(self):
        assert_program_stdout(
            "def *dict[str]int makeDictPtr():\n"
            "    dict[str]int d = dict[str]int{'x': 1}\n"
            "    return &d\n"
            "\n"
            "def int main():\n"
            "    *dict[str]int p = makeDictPtr()\n"
            "    print(*p)\n"
            "    return 0\n",
            "dict[str]int{'x': 1}\n",
        )


    def test_read_and_write_and_overwrite(self):
        assert_program_stdout(
            "def int main():\n"
            "    dict[str]int ages = dict[str]int{'alice': 25, 'bob': 17}\n"
            "    print(ages['alice'])\n"
            "    print(ages['bob'])\n"
            "    ages['carol'] = 30\n"
            "    print(ages['carol'])\n"
            "    ages['alice'] = 26\n"
            "    print(ages['alice'])\n"
            "    return 0\n",
            "25\n17\n30\n26\n",
        )

    def test_read_a_missing_key_panics(self):
        assert_crashes_with_sigabrt(
            "    dict[str]int ages = dict[str]int{'alice': 25}\n"
            "    print(ages['nonexistent'])\n"
            "    return 0\n"
        )

    def test_int_keyed_read_and_write(self):
        assert_program_stdout(
            "def int main():\n"
            "    dict[int]str codes = dict[int]str{1: 'one'}\n"
            "    codes[2] = 'two'\n"
            "    print(codes[1])\n"
            "    print(codes[2])\n"
            "    return 0\n",
            "one\ntwo\n",
        )

    def test_bool_keyed_read_and_write(self):
        assert_program_stdout(
            "def int main():\n"
            "    dict[bool]str labels = dict[bool]str{true: 'yes'}\n"
            "    labels[false] = 'no'\n"
            "    print(labels[true])\n"
            "    print(labels[false])\n"
            "    return 0\n",
            "yes\nno\n",
        )

    def test_struct_valued_read_and_write(self):
        assert_program_stdout(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def int main():\n"
            "    dict[str]Point points = dict[str]Point{'origin': Point(0, 0)}\n"
            "    points['unit'] = Point(1, 1)\n"
            "    print(points['unit'])\n"
            "    print(points['origin'])\n"
            "    Point p = points['unit']\n"
            "    print(p.x + p.y)\n"
            "    return 0\n",
            "Point(x: 1, y: 1)\nPoint(x: 0, y: 0)\n2\n",
        )

    def test_compound_assignment(self):
        assert_program_stdout(
            "def int main():\n"
            "    dict[str]int counts = dict[str]int{'a': 1}\n"
            "    counts['a'] += 10\n"
            "    print(counts['a'])\n"
            "    return 0\n",
            "11\n",
        )

    def test_compound_assignment_with_a_scalar_key(self):
        assert_program_stdout(
            "def int main():\n"
            "    dict[int]int counts = dict[int]int{1: 5}\n"
            "    counts[1] += 10\n"
            "    print(counts[1])\n"
            "    return 0\n",
            "15\n",
        )

    def test_compound_assignment_on_a_missing_key_panics(self):
        assert_crashes_with_sigabrt(
            "    dict[str]int counts = dict[str]int{'a': 1}\n"
            "    counts['nonexistent'] += 10\n"
            "    return 0\n"
        )

    def test_growth_and_rehash_across_many_insertions(self):
        assert_program_stdout(
            "def int main():\n"
            "    dict[int]int nums = dict[int]int{0: 0}\n"
            "    int i = 1\n"
            "    while i < 100:\n"
            "        nums[i] = i * 10\n"
            "        i = i + 1\n"
            "    i = 0\n"
            "    int total = 0\n"
            "    while i < 100:\n"
            "        total = total + nums[i]\n"
            "        i = i + 1\n"
            "    print(total)\n"
            "    return 0\n",
            f"{sum(i * 10 for i in range(100))}\n",
        )

    def test_non_bare_variable_dict_base(self):
        assert_program_stdout(
            "def int main():\n"
            "    dict[str]int a = dict[str]int{'a': 1}\n"
            "    dict[str]int b = dict[str]int{'b': 2}\n"
            "    [2]dict[str]int arr = [a, b]\n"
            "    print(arr[0]['a'])\n"
            "    print(arr[1]['b'])\n"
            "    arr[0]['a'] = 100\n"
            "    print(arr[0]['a'])\n"
            "    return 0\n",
            "1\n2\n100\n",
        )

    def test_non_bare_field_dict_base(self):
        assert_program_stdout(
            "type Box struct:\n"
            "    dict[str]int contents\n"
            "\n"
            "def int main():\n"
            "    dict[str]int d = dict[str]int{'x': 5}\n"
            "    Box b = Box(d)\n"
            "    print(b.contents['x'])\n"
            "    b.contents['x'] = 99\n"
            "    print(b.contents['x'])\n"
            "    return 0\n",
            "5\n99\n",
        )


    def test_del_removes_only_the_given_key(self):
        assert_program_stdout(
            "def int main():\n"
            "    dict[str]int ages = dict[str]int{'alice': 25, 'bob': 17}\n"
            "    del(ages, 'alice')\n"
            "    print(ages)\n"
            "    return 0\n",
            "dict[str]int{'bob': 17}\n",
        )

    def test_del_on_a_missing_key_panics(self):
        assert_crashes_with_sigabrt(
            "    dict[str]int ages = dict[str]int{'alice': 25}\n"
            "    del(ages, 'nonexistent')\n"
            "    return 0\n"
        )

    def test_lookup_after_del_panics(self):
        assert_crashes_with_sigabrt(
            "    dict[str]int ages = dict[str]int{'alice': 25}\n"
            "    del(ages, 'alice')\n"
            "    print(ages['alice'])\n"
            "    return 0\n"
        )

    def test_del_requires_a_dict_first_argument(self):
        assert_program_semantic_error(
            "def int main():\n"
            "    int x = 5\n"
            "    del(x, 'alice')\n"
            "    return 0\n",
            match="'del' requires a dict as its first argument",
        )

    def test_del_with_wrong_argument_count_is_rejected(self):
        assert_program_semantic_error(
            "def int main():\n"
            "    dict[str]int ages = dict[str]int{'alice': 25}\n"
            "    del(ages)\n"
            "    return 0\n",
            match="'del' expects exactly 2 arguments, got 1",
        )

    def test_del_key_type_mismatch_is_rejected(self):
        assert_program_semantic_error(
            "def int main():\n"
            "    dict[str]int ages = dict[str]int{'alice': 25}\n"
            "    del(ages, 42)\n"
            "    return 0\n",
            match="'del' cannot look up a key of type int",
        )

    def test_tombstone_does_not_break_lookup_for_a_key_that_probed_past_it(self):
        assert_program_stdout(
            "def int main():\n"
            "    dict[int]int nums = dict[int]int{0: 0}\n"
            "    int i = 1\n"
            "    while i < 100:\n"
            "        nums[i] = i * 10\n"
            "        i = i + 1\n"
            "    i = 0\n"
            "    while i < 100:\n"
            "        if i % 2 == 0:\n"
            "            del(nums, i)\n"
            "        i = i + 1\n"
            "    int total = 0\n"
            "    i = 1\n"
            "    while i < 100:\n"
            "        if i % 2 == 1:\n"
            "            total = total + nums[i]\n"
            "        i = i + 2\n"
            "    print(total)\n"
            "    return 0\n",
            f"{sum(i * 10 for i in range(1, 100, 2))}\n",
        )

    def test_repeated_insert_delete_churn_reuses_tombstone_slots(self):
        assert_program_stdout(
            "def int main():\n"
            "    dict[str]int d = dict[str]int{'a': 1}\n"
            "    int i = 0\n"
            "    while i < 20:\n"
            "        d['churn'] = i\n"
            "        del(d, 'churn')\n"
            "        i = i + 1\n"
            "    d['churn'] = 999\n"
            "    print(d['churn'])\n"
            "    print(d['a'])\n"
            "    return 0\n",
            "999\n1\n",
        )

    def test_del_with_a_scalar_key(self):
        assert_program_stdout(
            "def int main():\n"
            "    dict[int]str codes = dict[int]str{1: 'one', 2: 'two'}\n"
            "    del(codes, 1)\n"
            "    print(codes)\n"
            "    return 0\n",
            "dict[int]str{2: 'two'}\n",
        )


    def test_len_reflects_insert_and_delete(self):
        assert_program_stdout(
            "def int main():\n"
            "    dict[str]int ages = dict[str]int{'alice': 25, 'bob': 17}\n"
            "    print(len(ages))\n"
            "    ages['carol'] = 30\n"
            "    print(len(ages))\n"
            "    del(ages, 'alice')\n"
            "    print(len(ages))\n"
            "    return 0\n",
            "2\n3\n2\n",
        )

    def test_len_of_an_empty_dict_literal(self):
        assert_program_stdout(
            "def int main():\n"
            "    dict[str]int empty = dict[str]int{}\n"
            "    print(len(empty))\n"
            "    return 0\n",
            "0\n",
        )

    def test_len_stays_correct_across_growth(self):
        assert_program_stdout(
            "def int main():\n"
            "    dict[int]int nums = dict[int]int{0: 0}\n"
            "    int i = 1\n"
            "    while i < 50:\n"
            "        nums[i] = i\n"
            "        i = i + 1\n"
            "    print(len(nums))\n"
            "    return 0\n",
            "50\n",
        )

    def test_len_on_a_non_bare_variable_dict_base(self):
        assert_program_stdout(
            "def int main():\n"
            "    dict[str]int a = dict[str]int{'x': 1, 'y': 2}\n"
            "    [1]dict[str]int arr = [a]\n"
            "    print(len(arr[0]))\n"
            "    return 0\n",
            "2\n",
        )

    def test_len_requires_an_array_slice_str_or_dict_argument(self):
        assert_program_semantic_error(
            "def int main():\n"
            "    dict[str]int ages = dict[str]int{'alice': 25}\n"
            "    print(len(true))\n"
            "    return 0\n",
            match="requires an array, slice, str, or dict",
        )


    def test_in_reports_present_and_absent_keys(self):
        assert_program_stdout(
            "def int main():\n"
            "    dict[str]int ages = dict[str]int{'alice': 25, 'bob': 17}\n"
            "    if 'alice' in ages:\n"
            "        print('alice found')\n"
            "    if 'carol' in ages:\n"
            "        print('carol found (WRONG)')\n"
            "    return 0\n",
            "alice found\n",
        )

    def test_in_respects_left_operand_precedence(self):
        assert_program_stdout(
            "def int main():\n"
            "    dict[int]int d = dict[int]int{6: 1}\n"
            "    bool x = 2 * 3 not in d\n"
            "    print(x)\n"
            "    return 0\n",
            "false\n",
        )

    def test_not_in_reports_present_and_absent_keys(self):
        assert_program_stdout(
            "def int main():\n"
            "    dict[str]int ages = dict[str]int{'alice': 25}\n"
            "    if 'alice' not in ages:\n"
            "        print('alice missing (WRONG)')\n"
            "    if 'carol' not in ages:\n"
            "        print('carol missing (correct)')\n"
            "    return 0\n",
            "carol missing (correct)\n",
        )

    def test_in_reflects_delete(self):
        assert_program_stdout(
            "def int main():\n"
            "    dict[str]int ages = dict[str]int{'alice': 25}\n"
            "    del(ages, 'alice')\n"
            "    if 'alice' in ages:\n"
            "        print('still there (WRONG)')\n"
            "    if 'alice' not in ages:\n"
            "        print('gone (correct)')\n"
            "    return 0\n",
            "gone (correct)\n",
        )

    def test_in_as_an_ordinary_bool_expression(self):
        assert_program_stdout(
            "def int main():\n"
            "    dict[str]int ages = dict[str]int{'alice': 25}\n"
            "    bool exists = 'alice' in ages\n"
            "    print(exists)\n"
            "    return 0\n",
            "true\n",
        )

    def test_in_with_a_scalar_key(self):
        assert_program_stdout(
            "def int main():\n"
            "    dict[int]str codes = dict[int]str{1: 'one'}\n"
            "    bool exists = 1 in codes\n"
            "    bool missing = 2 in codes\n"
            "    print(exists)\n"
            "    print(missing)\n"
            "    return 0\n",
            "true\nfalse\n",
        )

    def test_in_skips_past_tombstones_rather_than_stopping_at_them(self):
        assert_program_stdout(
            "def int main():\n"
            "    dict[int]int nums = dict[int]int{0: 0}\n"
            "    int i = 1\n"
            "    while i < 100:\n"
            "        nums[i] = i * 10\n"
            "        i = i + 1\n"
            "    i = 0\n"
            "    while i < 100:\n"
            "        if i % 2 == 0:\n"
            "            del(nums, i)\n"
            "        i = i + 1\n"
            "    int found_odd = 0\n"
            "    int found_even = 0\n"
            "    i = 0\n"
            "    while i < 100:\n"
            "        if i in nums:\n"
            "            if i % 2 == 1:\n"
            "                found_odd = found_odd + 1\n"
            "            else:\n"
            "                found_even = found_even + 1\n"
            "        i = i + 1\n"
            "    print(found_odd)\n"
            "    print(found_even)\n"
            "    return 0\n",
            "50\n0\n",
        )

    def test_in_key_type_mismatch_is_rejected(self):
        assert_program_semantic_error(
            "def int main():\n"
            "    dict[str]int ages = dict[str]int{'alice': 25}\n"
            "    bool x = 5 in ages\n"
            "    return 0\n",
            match="Dict declares key type str, but 'in's own left operand is int",
        )

    def test_in_requires_a_dict_array_or_slice_right_operand(self):
        assert_program_semantic_error(
            "def int main():\n"
            "    int y = 5\n"
            "    bool x = 'a' in y\n"
            "    return 0\n",
            match="'in' requires a dict, array, or slice as its right operand",
        )


    def test_in_with_an_array(self):
        assert_program_stdout(
            "def int main():\n"
            "    [5]int arr = [10, 20, 30, 40, 50]\n"
            "    if 30 in arr:\n"
            "        print('found')\n"
            "    if 99 in arr:\n"
            "        print('found (WRONG)')\n"
            "    else:\n"
            "        print('not found (correct)')\n"
            "    return 0\n",
            "found\nnot found (correct)\n",
        )

    def test_in_with_a_slice(self):
        assert_program_stdout(
            "def int main():\n"
            "    []int s = [10, 20, 30, 40, 50]\n"
            "    if 30 in s:\n"
            "        print('found')\n"
            "    if 99 in s:\n"
            "        print('found (WRONG)')\n"
            "    else:\n"
            "        print('not found (correct)')\n"
            "    []int empty = []\n"
            "    if 1 in empty:\n"
            "        print('found in empty (WRONG)')\n"
            "    else:\n"
            "        print('empty correctly has nothing')\n"
            "    return 0\n",
            "found\nnot found (correct)\nempty correctly has nothing\n",
        )

    def test_in_with_str_elements(self):
        assert_program_stdout(
            "def int main():\n"
            "    [3]str names = ['alice', 'bob', 'carol']\n"
            "    if 'bob' in names:\n"
            "        print('found')\n"
            "    if 'dave' in names:\n"
            "        print('found (WRONG)')\n"
            "    else:\n"
            "        print('not found (correct)')\n"
            "    return 0\n",
            "found\nnot found (correct)\n",
        )

    def test_in_with_struct_elements(self):
        assert_program_stdout(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def int main():\n"
            "    [2]Point pts = [Point(1, 2), Point(3, 4)]\n"
            "    Point target = Point(3, 4)\n"
            "    Point missing = Point(5, 6)\n"
            "    if target in pts:\n"
            "        print('found')\n"
            "    if missing in pts:\n"
            "        print('found (WRONG)')\n"
            "    else:\n"
            "        print('not found (correct)')\n"
            "    return 0\n",
            "found\nnot found (correct)\n",
        )

    def test_not_in_with_an_array(self):
        assert_program_stdout(
            "def int main():\n"
            "    [3]int nums = [10, 20, 30]\n"
            "    if 30 not in nums:\n"
            "        print('30 absent (WRONG)')\n"
            "    else:\n"
            "        print('30 present (correct)')\n"
            "    if 99 not in nums:\n"
            "        print('99 absent (correct)')\n"
            "    return 0\n",
            "30 present (correct)\n99 absent (correct)\n",
        )

    def test_in_evaluates_the_needle_exactly_once_for_side_effects(self):
        assert_program_stdout(
            "def int get_target():\n"
            "    print('computing target')\n"
            "    return 30\n"
            "\n"
            "def int main():\n"
            "    [3]int arr = [10, 20, 30]\n"
            "    if get_target() in arr:\n"
            "        print('found')\n"
            "    return 0\n",
            "computing target\nfound\n",
        )

    def test_in_with_an_uncomparable_element_type_is_rejected(self):
        assert_program_semantic_error(
            "def int main():\n"
            "    [][]int nested = [[1, 2], [3, 4]]\n"
            "    []int target = [1, 2]\n"
            "    bool x = target in nested\n"
            "    return 0\n",
            match="'in' does not support an element type of \\[\\]int -- "
                  "membership isn't defined yet when the elements are "
                  "\\(or contain\\) a slice, sum type, or dict",
        )

    def test_in_element_type_mismatch_is_rejected(self):
        assert_program_semantic_error(
            "def int main():\n"
            "    [3]str names = ['alice', 'bob', 'carol']\n"
            "    bool x = 5 in names\n"
            "    return 0\n",
            match="\\[3\\]str declares element type str, but 'in's own left operand is int",
        )


    def test_nil_dict_len_and_membership(self):
        assert_program_stdout(
            "def int main():\n"
            "    dict[str]int d\n"
            "    print(len(d))\n"
            "    print('x' in d)\n"
            "    return 0\n",
            "0\nfalse\n",
        )

    def test_nil_dict_prints_as_empty(self):
        assert_program_stdout(
            "def int main():\n"
            "    dict[str]int d\n"
            "    print(d)\n"
            "    return 0\n",
            "dict[str]int{}\n",
        )

    def test_nil_dict_read_panics(self):
        assert_crashes_with_sigabrt(
            "    dict[str]int d\n"
            "    print(d['missing'])\n"
            "    return 0\n"
        )

    def test_nil_dict_del_panics(self):
        assert_crashes_with_sigabrt(
            "    dict[str]int d\n"
            "    del(d, 'missing')\n"
            "    return 0\n"
        )

    def test_nil_dict_write_bootstraps_a_real_dict(self):
        assert_program_stdout(
            "def int main():\n"
            "    dict[str]int d\n"
            "    d['a'] = 1\n"
            "    print(d)\n"
            "    print(len(d))\n"
            "    print(d['a'])\n"
            "    return 0\n",
            "dict[str]int{'a': 1}\n1\n1\n",
        )

    def test_nil_dict_with_a_scalar_key(self):
        assert_program_stdout(
            "def int main():\n"
            "    dict[int]str d\n"
            "    print(len(d))\n"
            "    print(5 in d)\n"
            "    d[5] = 'hello'\n"
            "    print(d[5])\n"
            "    print(len(d))\n"
            "    return 0\n",
            "0\nfalse\nhello\n1\n",
        )

    def test_nil_dict_grows_correctly_across_many_insertions(self):
        assert_program_stdout(
            "def int main():\n"
            "    dict[int]int d\n"
            "    for int i = 0; i < 100; i += 1:\n"
            "        d[i] = i * 10\n"
            "    int total = 0\n"
            "    for int i = 0; i < 100; i += 1:\n"
            "        total = total + d[i]\n"
            "    print(len(d))\n"
            "    print(total)\n"
            "    return 0\n",
            f"100\n{sum(i * 10 for i in range(100))}\n",
        )

    def test_nil_dict_escaping_its_own_function_still_works(self):
        assert_program_stdout(
            "def *dict[str]int make_nil_dict():\n"
            "    dict[str]int d\n"
            "    return &d\n"
            "\n"
            "def int main():\n"
            "    *dict[str]int p = make_nil_dict()\n"
            "    (*p)['a'] = 1\n"
            "    print(*p)\n"
            "    return 0\n",
            "dict[str]int{'a': 1}\n",
        )


    def test_dict_typed_function_return_parses_and_works(self):
        assert_program_stdout(
            "def dict[str]int make_dict():\n"
            "    dict[str]int d = dict[str]int{'a': 1}\n"
            "    return d\n"
            "\n"
            "def int main():\n"
            "    dict[str]int d = make_dict()\n"
            "    print(d['a'])\n"
            "    return 0\n",
            "1\n",
        )

    def test_dict_typed_function_return_with_parameters(self):
        assert_program_stdout(
            "def dict[str]int make_dict(int value):\n"
            "    dict[str]int d = dict[str]int{'a': value}\n"
            "    return d\n"
            "\n"
            "def int main():\n"
            "    dict[str]int d = make_dict(42)\n"
            "    print(d['a'])\n"
            "    return 0\n",
            "42\n",
        )

    def test_chained_dict_returning_function_calls(self):
        assert_program_stdout(
            "def dict[str]int base_dict():\n"
            "    dict[str]int d = dict[str]int{'a': 1}\n"
            "    return d\n"
            "\n"
            "def dict[str]int extended_dict(int extra):\n"
            "    dict[str]int d = base_dict()\n"
            "    d['b'] = extra\n"
            "    return d\n"
            "\n"
            "def int main():\n"
            "    dict[str]int d = extended_dict(99)\n"
            "    print(d['a'])\n"
            "    print(d['b'])\n"
            "    return 0\n",
            "1\n99\n",
        )

    def test_dict_typed_method_return(self):
        assert_program_stdout(
            "type Counter struct:\n"
            "    int start\n"
            "    def dict[str]int make_dict(c):\n"
            "        dict[str]int d = dict[str]int{'count': c.start}\n"
            "        return d\n"
            "\n"
            "def int main():\n"
            "    Counter c = Counter(start=5)\n"
            "    dict[str]int cd = c.make_dict()\n"
            "    print(cd['count'])\n"
            "    return 0\n",
            "5\n",
        )

    def test_nil_dict_returned_from_a_function(self):
        assert_program_stdout(
            "def dict[int]int make_nil_dict():\n"
            "    dict[int]int d\n"
            "    return d\n"
            "\n"
            "def int main():\n"
            "    dict[int]int b = make_nil_dict()\n"
            "    print(len(b))\n"
            "    b[1] = 100\n"
            "    print(b[1])\n"
            "    return 0\n",
            "0\n100\n",
        )

    def test_returning_a_dict_literal_directly_now_works(self):
        assert_program_stdout(
            "def dict[str]int make_dict():\n"
            "    return dict[str]int{'a': 1, 'b': 2}\n"
            "\n"
            "def int main():\n"
            "    dict[str]int d = make_dict()\n"
            "    print(d['a'])\n"
            "    print(d['b'])\n"
            "    return 0\n",
            "1\n2\n",
        )

    def test_returning_a_nested_dict_literal_directly(self):
        assert_program_stdout(
            "def dict[str]dict[str]int make_nested():\n"
            "    return dict[str]dict[str]int{'outer': dict[str]int{'inner': 42}}\n"
            "\n"
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def dict[str]Point make_points():\n"
            "    return dict[str]Point{'a': Point(x=3, y=4)}\n"
            "\n"
            "def int main():\n"
            "    dict[str]dict[str]int nested = make_nested()\n"
            "    print(nested['outer']['inner'])\n"
            "    dict[str]Point points = make_points()\n"
            "    print(points['a'].x)\n"
            "    print(points['a'].y)\n"
            "    return 0\n",
            "42\n3\n4\n",
        )

    def test_returning_a_dict_literal_directly_from_a_method(self):
        assert_program_stdout(
            "type Factory struct:\n"
            "    int seed\n"
            "    def dict[str]int make(f):\n"
            "        return dict[str]int{'seed': f.seed}\n"
            "\n"
            "def int main():\n"
            "    Factory factory = Factory(seed=7)\n"
            "    dict[str]int made = factory.make()\n"
            "    print(made['seed'])\n"
            "    return 0\n",
            "7\n",
        )

    def test_returning_an_empty_dict_literal_directly(self):
        assert_program_stdout(
            "def dict[str]int make_empty():\n"
            "    return dict[str]int{}\n"
            "\n"
            "def int main():\n"
            "    dict[str]int d = make_empty()\n"
            "    print(len(d))\n"
            "    d['x'] = 1\n"
            "    print(d['x'])\n"
            "    return 0\n",
            "0\n1\n",
        )


    def test_dict_typed_function_argument_works(self):
        assert_program_stdout(
            "def int lookup(dict[str]int d, str key):\n"
            "    return d[key]\n"
            "\n"
            "def int main():\n"
            "    dict[str]int d = dict[str]int{'a': 1}\n"
            "    print(lookup(d, 'a'))\n"
            "    return 0\n",
            "1\n",
        )

    def test_dict_returning_call_used_directly_as_an_argument(self):
        assert_program_stdout(
            "def dict[str]int make_dict():\n"
            "    dict[str]int d = dict[str]int{'a': 1}\n"
            "    return d\n"
            "\n"
            "def int lookup(dict[str]int d, str key):\n"
            "    return d[key]\n"
            "\n"
            "def int main():\n"
            "    print(lookup(make_dict(), 'a'))\n"
            "    return 0\n",
            "1\n",
        )

    def test_multiple_dict_arguments_to_one_function(self):
        assert_program_stdout(
            "def int combine(dict[str]int a, dict[str]int b, str key1, str key2):\n"
            "    return a[key1] + b[key2]\n"
            "\n"
            "def int main():\n"
            "    dict[str]int x = dict[str]int{'p': 10}\n"
            "    dict[str]int y = dict[str]int{'q': 20}\n"
            "    print(combine(x, y, 'p', 'q'))\n"
            "    return 0\n",
            "30\n",
        )

    def test_nil_dict_passed_as_an_argument(self):
        assert_program_stdout(
            "def int lookup_or_default(dict[int]int d, int key):\n"
            "    if key in d:\n"
            "        return d[key]\n"
            "    return -1\n"
            "\n"
            "def int main():\n"
            "    dict[int]int d\n"
            "    print(lookup_or_default(d, 5))\n"
            "    return 0\n",
            "-1\n",
        )

    def test_dict_parameter_whose_address_escapes(self):
        assert_program_stdout(
            "def *dict[str]int get_ref(dict[str]int d):\n"
            "    return &d\n"
            "\n"
            "def int main():\n"
            "    dict[str]int d = dict[str]int{'a': 1}\n"
            "    *dict[str]int p = get_ref(d)\n"
            "    (*p)['b'] = 2\n"
            "    print(*p)\n"
            "    return 0\n",
            "dict[str]int{'a': 1, 'b': 2}\n",
        )

    def test_dict_typed_method_argument(self):
        assert_program_stdout(
            "type Wrapper struct:\n"
            "    int tag\n"
            "    def int lookup(w, dict[str]int d, str key):\n"
            "        return d[key] + w.tag\n"
            "\n"
            "def int main():\n"
            "    Wrapper w = Wrapper(tag=100)\n"
            "    dict[str]int d = dict[str]int{'a': 1}\n"
            "    print(w.lookup(d, 'a'))\n"
            "    return 0\n",
            "101\n",
        )

    def test_dict_literal_as_a_direct_argument_now_works(self):
        assert_program_stdout(
            "def int lookup(dict[str]int d, str key):\n"
            "    return d[key]\n"
            "\n"
            "def int main():\n"
            "    print(lookup(dict[str]int{'a': 1, 'b': 2}, 'a'))\n"
            "    print(lookup(dict[str]int{'a': 1, 'b': 2}, 'b'))\n"
            "    return 0\n",
            "1\n2\n",
        )

    def test_two_dict_literal_arguments_in_one_call(self):
        assert_program_stdout(
            "def int combine(dict[str]int a, dict[str]int b, str key1, str key2):\n"
            "    return a[key1] + b[key2]\n"
            "\n"
            "def int main():\n"
            "    print(combine(dict[str]int{'p': 10}, dict[str]int{'q': 20}, 'p', 'q'))\n"
            "    return 0\n",
            "30\n",
        )

    def test_nested_dict_literal_as_a_direct_argument(self):
        assert_program_stdout(
            "def int deep_lookup(dict[str]dict[str]int d, str outer, str inner):\n"
            "    return d[outer][inner]\n"
            "\n"
            "def int main():\n"
            "    print(deep_lookup(dict[str]dict[str]int{'x': dict[str]int{'y': 42}}, 'x', 'y'))\n"
            "    return 0\n",
            "42\n",
        )

    def test_dict_literal_argument_to_a_method(self):
        assert_program_stdout(
            "type Wrapper struct:\n"
            "    int tag\n"
            "    def int lookup(w, dict[str]int d, str key):\n"
            "        return d[key] + w.tag\n"
            "\n"
            "def int main():\n"
            "    Wrapper w = Wrapper(tag=100)\n"
            "    print(w.lookup(dict[str]int{'a': 1}, 'a'))\n"
            "    return 0\n",
            "101\n",
        )


    def test_nil_dict_equals_none(self):
        assert_program_stdout(
            "def int main():\n"
            "    dict[str]int d\n"
            "    if d == none:\n"
            "        print('equal')\n"
            "    if d != none:\n"
            "        print('not equal (WRONG)')\n"
            "    return 0\n",
            "equal\n",
        )

    def test_non_nil_dict_does_not_equal_none(self):
        assert_program_stdout(
            "def int main():\n"
            "    dict[str]int d = dict[str]int{'a': 1}\n"
            "    if d == none:\n"
            "        print('equal (WRONG)')\n"
            "    if d != none:\n"
            "        print('not equal')\n"
            "    return 0\n",
            "not equal\n",
        )

    def test_dict_vs_dict_equality_is_rejected(self):
        assert_semantic_error(
            "    dict[str]int a = dict[str]int{'x': 1}\n"
            "    dict[str]int b = dict[str]int{'x': 1}\n"
            "    if a == b:\n"
            "        return 1\n"
            "    return 0\n",
            match="does not support slice, void, sum type, dict, or none operands",
        )

    def test_dict_literal_vs_dict_equality_is_also_rejected(self):
        assert_semantic_error(
            "    dict[str]int d = dict[str]int{'a': 1}\n"
            "    if dict[str]int{'a': 1} == d:\n"
            "        return 1\n"
            "    if dict[str]int{'a': 1} == dict[str]int{'a': 1}:\n"
            "        return 1\n"
            "    return 0\n",
            match="does not support slice, void, sum type, dict, or none operands",
        )


    def test_dict_literal_as_a_struct_field_argument(self):
        assert_program_stdout(
            "type Wrapper struct:\n"
            "    dict[str]int d\n"
            "    int tag\n"
            "\n"
            "def int main():\n"
            "    Wrapper w = Wrapper(d=dict[str]int{'a': 1, 'b': 2}, tag=99)\n"
            "    print(w.d['a'])\n"
            "    print(w.d['b'])\n"
            "    print(w.tag)\n"
            "    return 0\n",
            "1\n2\n99\n",
        )

    def test_dict_literal_as_an_array_element(self):
        assert_program_stdout(
            "def int main():\n"
            "    [2]dict[str]int arr = [dict[str]int{'a': 1}, dict[str]int{'b': 2}]\n"
            "    print(arr[0]['a'])\n"
            "    print(arr[1]['b'])\n"
            "    return 0\n",
            "1\n2\n",
        )

    def test_dict_of_dict(self):
        assert_program_stdout(
            "def int main():\n"
            "    dict[str]dict[str]int d = dict[str]dict[str]int{\n"
            "        'outer1': dict[str]int{'inner1': 10, 'inner2': 20},\n"
            "        'outer2': dict[str]int{'inner3': 30},\n"
            "    }\n"
            "    print(d['outer1']['inner1'])\n"
            "    print(d['outer1']['inner2'])\n"
            "    print(d['outer2']['inner3'])\n"
            "    return 0\n",
            "10\n20\n30\n",
        )

    def test_three_level_nesting_struct_array_dict(self):
        assert_program_stdout(
            "type Bundle struct:\n"
            "    [2]dict[str]int items\n"
            "\n"
            "def int main():\n"
            "    Bundle b = Bundle(items=[dict[str]int{'a': 1}, dict[str]int{'b': 2}])\n"
            "    print(b.items[0]['a'])\n"
            "    print(b.items[1]['b'])\n"
            "    return 0\n",
            "1\n2\n",
        )

# ---------------------------------------------------------------------------
# Pointers
# ---------------------------------------------------------------------------

class TestPointers:

    _CIRCLE = "type Circle struct:\n    int radius\n\n"


    def test_basic_pointer_type_resolves(self):
        ast = _parse(
            "def int main():\n"
            "    int x = 5\n"
            "    *int p = &x\n"
            "    return 0\n"
        )
        analyze(ast)  # should not raise

    def test_address_of_produces_a_pointer_to_the_variables_own_type(self):
        ast = _parse(
            self._CIRCLE +
            "def int main():\n"
            "    Circle c = Circle(5)\n"
            "    *Circle p = &c\n"
            "    return 0\n"
        )
        analyze(ast)  # should not raise

    def test_dereference_reads_the_pointee_type(self):
        ast = _parse(
            "def int main():\n"
            "    int x = 5\n"
            "    *int p = &x\n"
            "    int y = *p\n"
            "    return y\n"
        )
        analyze(ast)  # should not raise

    def test_auto_deref_field_read(self):
        ast = _parse(
            self._CIRCLE +
            "def int main():\n"
            "    Circle c = Circle(5)\n"
            "    *Circle p = &c\n"
            "    int r = p.radius\n"
            "    return r\n"
        )
        analyze(ast)  # should not raise

    def test_auto_deref_field_write(self):
        ast = _parse(
            self._CIRCLE +
            "def int main():\n"
            "    Circle c = Circle(5)\n"
            "    *Circle p = &c\n"
            "    p.radius = 9\n"
            "    return c.radius\n"
        )
        analyze(ast)  # should not raise

    def test_auto_deref_method_call(self):
        ast = _parse(
            "type Circle struct:\n"
            "    int radius\n"
            "\n"
            "    def int area(self):\n"
            "        return self.radius * self.radius\n"
            "\n"
            "def int main():\n"
            "    Circle c = Circle(5)\n"
            "    *Circle p = &c\n"
            "    return p.area()\n"
        )
        analyze(ast)  # should not raise

    def test_deref_assign_overwrites_the_whole_pointee(self):
        ast = _parse(
            self._CIRCLE +
            "def int main():\n"
            "    Circle c = Circle(5)\n"
            "    *Circle p = &c\n"
            "    *p = Circle(9)\n"
            "    return c.radius\n"
        )
        analyze(ast)  # should not raise

    def test_pointer_is_compatible_with_none(self):
        ast = _parse(
            self._CIRCLE +
            "def int main():\n"
            "    *Circle p = none\n"
            "    return 0\n"
        )
        analyze(ast)  # should not raise

    def test_same_type_pointers_are_comparable(self):
        ast = _parse(
            self._CIRCLE +
            "def int main():\n"
            "    Circle c = Circle(5)\n"
            "    *Circle p = &c\n"
            "    *Circle q = &c\n"
            "    if p == q:\n"
            "        return 1\n"
            "    return 0\n"
        )
        analyze(ast)  # should not raise

    def test_pointer_comparable_to_none(self):
        ast = _parse(
            self._CIRCLE +
            "def int main():\n"
            "    Circle c = Circle(5)\n"
            "    *Circle p = &c\n"
            "    if p != none:\n"
            "        return 1\n"
            "    return 0\n"
        )
        analyze(ast)  # should not raise

    def test_struct_field_can_be_pointer_typed(self):
        ast = _parse(
            "type Node struct:\n"
            "    int value\n"
            "    *Node next\n"
            "\n"
            "def int main():\n"
            "    Node n = Node(5, none)\n"
            "    return n.value\n"
        )
        analyze(ast)  # should not raise

    def test_array_element_can_be_pointer_typed(self):
        ast = _parse(
            self._CIRCLE +
            "def int main():\n"
            "    Circle c = Circle(5)\n"
            "    [3]*Circle arr = [3]*Circle[&c, none, none]\n"
            "    return 0\n"
        )
        analyze(ast)  # should not raise

    def test_slice_element_can_be_pointer_typed(self):
        ast = _parse(
            self._CIRCLE +
            "def int main():\n"
            "    Circle c = Circle(5)\n"
            "    []*Circle s = []*Circle[&c, none]\n"
            "    return 0\n"
        )
        analyze(ast)  # should not raise


    def test_pointer_to_pointer_is_rejected(self):
        assert_program_semantic_error(
            "def int main():\n"
            "    int x = 5\n"
            "    *int p = &x\n"
            "    **int pp = &p\n"
            "    return 0\n",
            match="Pointer-to-pointer types aren't supported yet",
        )

    def test_address_of_a_field_is_now_accepted(self):
        ast = _parse(
            self._CIRCLE +
            "def int main():\n"
            "    Circle c = Circle(5)\n"
            "    *int p = &c.radius\n"
            "    return *p\n"
        )
        analyze(ast)  # should not raise

    def test_address_of_an_index_is_now_accepted(self):
        ast = _parse(
            "def int main():\n"
            "    [3]int arr = [1, 2, 3]\n"
            "    *int p = &arr[0]\n"
            "    return *p\n"
        )
        analyze(ast)  # should not raise

    def test_address_of_a_field_rooted_in_a_call_is_still_rejected(self):
        assert_program_semantic_error(
            self._CIRCLE +
            "def Circle makeCircle():\n"
            "    return Circle(5)\n"
            "\n"
            "def int main():\n"
            "    *int p = &makeCircle().radius\n"
            "    return 0\n",
            match="'&' can only take the address of a bare variable",
        )

    def test_address_of_an_index_rooted_in_a_call_is_still_rejected(self):
        assert_program_semantic_error(
            "def [3]int makeArray():\n"
            "    return [1, 2, 3]\n"
            "\n"
            "def int main():\n"
            "    *int p = &makeArray()[0]\n"
            "    return 0\n",
            match="'&' can only take the address of a bare variable",
        )

    def test_address_of_a_struct_literal_is_accepted(self):
        ast = _parse(
            self._CIRCLE +
            "def int main():\n"
            "    *Circle p = &Circle(5)\n"
            "    return p.radius\n"
        )
        analyze(ast)  # should not raise

    def test_address_of_a_non_struct_call_is_still_rejected(self):
        assert_program_semantic_error(
            "def int makeFive():\n"
            "    return 5\n"
            "\n"
            "def int main():\n"
            "    *int p = &makeFive()\n"
            "    return 0\n",
            match="'&' can only take the address of a bare variable",
        )

    def test_dereferencing_a_non_pointer_is_rejected(self):
        assert_program_semantic_error(
            "def int main():\n"
            "    int x = 5\n"
            "    int y = *x\n"
            "    return y\n",
            match="'\\*' requires a pointer operand, got int",
        )

    def test_dereferencing_a_pointer_to_struct_as_a_value_is_accepted(self):
        ast = _parse(
            self._CIRCLE +
            "def int main():\n"
            "    Circle c = Circle(5)\n"
            "    *Circle p = &c\n"
            "    Circle copy = *p\n"
            "    return copy.radius\n"
        )
        analyze(ast)  # should not raise

    def test_dereferencing_a_pointer_to_array_as_a_value_is_accepted(self):
        ast = _parse(
            "def int main():\n"
            "    [3]int arr = [1, 2, 3]\n"
            "    *[3]int p = &arr\n"
            "    [3]int copy = *p\n"
            "    return copy[0]\n"
        )
        analyze(ast)  # should not raise

    def test_dereferencing_a_pointer_to_slice_as_a_value_is_accepted(self):
        ast = _parse(
            "def int main():\n"
            "    [3]int arr = [1, 2, 3]\n"
            "    []int s = arr[0:3]\n"
            "    *[]int p = &s\n"
            "    []int copy = *p\n"
            "    return copy[0]\n"
        )
        analyze(ast)  # should not raise

    def test_dereferencing_a_pointer_to_sum_type_as_a_value_is_still_rejected(self):
        assert_program_semantic_error(
            "type Circle struct:\n"
            "    int radius\n"
            "\n"
            "type Square struct:\n"
            "    int side\n"
            "\n"
            "type Shape is Circle | Square\n"
            "\n"
            "def int main():\n"
            "    Shape shape = Circle(5)\n"
            "    *Shape p = &shape\n"
            "    Shape copy = *p\n"
            "    return 0\n",
            match="'\\*' on a pointer to Shape \\(a sum type\\) isn't supported",
        )

    def test_field_access_on_a_non_pointer_non_struct_is_still_rejected(self):
        assert_program_semantic_error(
            "def int main():\n"
            "    int x = 5\n"
            "    int y = x.field\n"
            "    return y\n",
            match="Cannot access field 'field' on non-struct type int",
        )

    def test_deref_assign_to_a_non_pointer_is_rejected(self):
        assert_program_semantic_error(
            self._CIRCLE +
            "def int main():\n"
            "    Circle c = Circle(5)\n"
            "    *c = Circle(9)\n"
            "    return 0\n",
            match="Cannot dereference a value of type Circle for assignment",
        )

    def test_deref_assign_with_an_incompatible_value_is_rejected(self):
        assert_program_semantic_error(
            self._CIRCLE +
            "type Square struct:\n"
            "    int side\n"
            "\n"
            "def int main():\n"
            "    Circle c = Circle(5)\n"
            "    *Circle p = &c\n"
            "    *p = Square(9)\n"
            "    return 0\n",
            match="Cannot assign a value of type Square through a pointer to Circle",
        )

    def test_bare_pointer_vs_incompatible_type_is_still_rejected(self):
        assert_program_semantic_error(
            self._CIRCLE +
            "def int main():\n"
            "    Circle c = Circle(5)\n"
            "    *Circle p = &c\n"
            "    if p == 5:\n"
            "        return 1\n"
            "    return 0\n",
            match="Cannot compare",
        )


# ---------------------------------------------------------------------------
# Pointers
# ---------------------------------------------------------------------------

class TestPointersCodegen:

    _CIRCLE = (
        "type Circle struct:\n"
        "    int radius\n"
        "\n"
    )

    _NODE = (
        "type Node struct:\n"
        "    int value\n"
        "    *Node next\n"
        "\n"
    )

    def test_basic_address_of_and_dereference_round_trip(self):
        assert_program_exit_code(
            "def int main():\n"
            "    int x = 5\n"
            "    *int p = &x\n"
            "    int y = *p\n"
            "    return y\n",
            expected=5,
        )

    def test_address_of_a_struct_literal_purely_local(self):
        assert_program_exit_code(
            self._CIRCLE +
            "def int main():\n"
            "    *Circle p = &Circle(5)\n"
            "    return p.radius\n",
            expected=5,
        )

    def test_address_of_a_struct_literal_with_multiple_fields(self):
        assert_program_exit_code(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def int main():\n"
            "    *Point p = &Point(3, 4)\n"
            "    return p.x + p.y\n",
            expected=7,
        )

    def test_address_of_two_distinct_struct_literals_in_one_function(self):
        assert_program_exit_code(
            self._CIRCLE +
            "def int main():\n"
            "    *Circle p1 = &Circle(5)\n"
            "    *Circle p2 = &Circle(9)\n"
            "    return p1.radius + p2.radius\n",
            expected=14,
        )

    def test_dereferencing_a_pointer_to_struct_as_a_vardecl_initializer(self):
        assert_program_exit_code(
            self._CIRCLE +
            "def int main():\n"
            "    Circle orig = Circle(5)\n"
            "    *Circle p = &orig\n"
            "    Circle copy = *p\n"
            "    return copy.radius\n",
            expected=5,
        )

    def test_dereferencing_a_pointer_to_struct_in_an_assign(self):
        assert_program_exit_code(
            self._CIRCLE +
            "def int main():\n"
            "    Circle orig = Circle(5)\n"
            "    *Circle p = &orig\n"
            "    Circle copy = Circle(0)\n"
            "    copy = *p\n"
            "    return copy.radius\n",
            expected=5,
        )

    def test_dereferencing_a_pointer_to_struct_as_a_function_argument(self):
        assert_program_exit_code(
            self._CIRCLE +
            "def int useCircle(Circle c):\n"
            "    return c.radius\n"
            "\n"
            "def int main():\n"
            "    Circle orig = Circle(5)\n"
            "    *Circle p = &orig\n"
            "    return useCircle(*p)\n",
            expected=5,
        )

    def test_dereferencing_a_pointer_to_struct_as_a_return_value(self):
        assert_program_exit_code(
            self._CIRCLE +
            "def Circle returnDeref(*Circle p):\n"
            "    return *p\n"
            "\n"
            "def int main():\n"
            "    Circle orig = Circle(7)\n"
            "    *Circle p = &orig\n"
            "    Circle result = returnDeref(p)\n"
            "    return result.radius\n",
            expected=7,
        )

    def test_dereferencing_two_pointers_to_struct_in_an_equality(self):
        assert_program_exit_code(
            self._CIRCLE +
            "def int main():\n"
            "    Circle a = Circle(7)\n"
            "    Circle b = Circle(7)\n"
            "    *Circle p = &a\n"
            "    *Circle q = &b\n"
            "    if *p == *q:\n"
            "        return 1\n"
            "    return 0\n",
            expected=1,
        )

    def test_dereferencing_a_pointer_to_struct_in_an_index_assign(self):
        assert_program_exit_code(
            self._CIRCLE +
            "def int main():\n"
            "    Circle orig = Circle(9)\n"
            "    *Circle p = &orig\n"
            "    [2]Circle circles = [Circle(0), Circle(0)]\n"
            "    circles[0] = *p\n"
            "    return circles[0].radius\n",
            expected=9,
        )

    def test_dereferencing_a_pointer_to_struct_in_a_field_assign(self):
        assert_program_exit_code(
            "type Circle struct:\n"
            "    int radius\n"
            "\n"
            "type Holder struct:\n"
            "    Circle c\n"
            "\n"
            "def int main():\n"
            "    Circle orig = Circle(9)\n"
            "    *Circle p = &orig\n"
            "    Holder h = Holder(Circle(0))\n"
            "    h.c = *p\n"
            "    return h.c.radius\n",
            expected=9,
        )

    def test_deref_assign_with_a_dereferenced_source(self):
        assert_program_exit_code(
            self._CIRCLE +
            "def int main():\n"
            "    Circle orig = Circle(9)\n"
            "    *Circle p = &orig\n"
            "    Circle target = Circle(0)\n"
            "    *Circle q = &target\n"
            "    *q = *p\n"
            "    return target.radius\n",
            expected=9,
        )

    def test_dereferencing_a_pointer_to_array_as_a_vardecl_initializer(self):
        assert_program_exit_code(
            "def int main():\n"
            "    [3]int arr = [10, 20, 30]\n"
            "    *[3]int p = &arr\n"
            "    [3]int copy = *p\n"
            "    return copy[1]\n",
            expected=20,
        )

    def test_dereferencing_a_pointer_to_slice_as_a_vardecl_initializer(self):
        assert_program_exit_code(
            "def int main():\n"
            "    [3]int arr = [10, 20, 30]\n"
            "    []int s = arr[0:3]\n"
            "    *[]int p = &s\n"
            "    []int copy = *p\n"
            "    return copy[2]\n",
            expected=30,
        )

    def test_dereferencing_a_pointer_to_slice_as_a_function_argument(self):
        assert_program_exit_code(
            "def int sumIt([]int s):\n"
            "    int total = 0\n"
            "    int i = 0\n"
            "    while i < len(s):\n"
            "        total = total + s[i]\n"
            "        i = i + 1\n"
            "    return total\n"
            "\n"
            "def int main():\n"
            "    [3]int arr = [10, 20, 30]\n"
            "    []int s = arr[0:3]\n"
            "    *[]int p = &s\n"
            "    return sumIt(*p)\n",
            expected=60,
        )

    def test_indexing_directly_into_a_dereferenced_slice_pointer(self):
        assert_program_exit_code(
            "def int main():\n"
            "    [3]int arr = [10, 20, 30]\n"
            "    []int s = arr[0:3]\n"
            "    *[]int p = &s\n"
            "    return (*p)[1]\n",
            expected=20,
        )

    def test_append_with_a_dereferenced_slice_pointer_as_its_first_argument(self):
        assert_program_exit_code(
            "def int main():\n"
            "    [5]int arr = [10, 20, 30, 0, 0]\n"
            "    []int s = arr[0:3]\n"
            "    *[]int p = &s\n"
            "    []int grown = append(*p, 40)\n"
            "    return grown[3]\n",
            expected=40,
        )

    def test_address_of_a_field_purely_local(self):
        assert_program_exit_code(
            self._CIRCLE +
            "def int main():\n"
            "    Circle c = Circle(5)\n"
            "    *int p = &c.radius\n"
            "    return *p\n",
            expected=5,
        )

    def test_address_of_an_element_purely_local(self):
        assert_program_exit_code(
            "def int main():\n"
            "    [3]int arr = [10, 20, 30]\n"
            "    *int p = &arr[1]\n"
            "    return *p\n",
            expected=20,
        )

    def test_address_of_a_field_through_an_auto_dereferenced_pointer(self):
        assert_program_exit_code(
            self._CIRCLE +
            "def *int getFieldThroughPointer(*Circle p):\n"
            "    return &p.radius\n"
            "\n"
            "def int main():\n"
            "    Circle c = Circle(42)\n"
            "    *Circle p = &c\n"
            "    *int q = getFieldThroughPointer(p)\n"
            "    return *q\n",
            expected=42,
        )

    def test_pointer_sees_a_later_mutation_of_the_pointee(self):
        assert_program_exit_code(
            "def int main():\n"
            "    int x = 5\n"
            "    *int p = &x\n"
            "    x = 10\n"
            "    int y = *p\n"
            "    return y\n",
            expected=10,
        )

    def test_scalar_deref_assign_overwrites_the_pointee(self):
        assert_program_exit_code(
            "def int main():\n"
            "    int x = 5\n"
            "    *int p = &x\n"
            "    *p = 10\n"
            "    return x\n",
            expected=10,
        )

    def test_struct_deref_assign_via_struct_literal(self):
        assert_program_exit_code(
            self._CIRCLE +
            "def int main():\n"
            "    Circle c = Circle(5)\n"
            "    *Circle p = &c\n"
            "    *p = Circle(20)\n"
            "    return c.radius\n",
            expected=20,
        )

    def test_struct_deref_assign_via_another_variable(self):
        assert_program_exit_code(
            self._CIRCLE +
            "def int main():\n"
            "    Circle c = Circle(5)\n"
            "    Circle other = Circle(9)\n"
            "    *Circle p = &c\n"
            "    *p = other\n"
            "    return c.radius\n",
            expected=9,
        )

    def test_auto_deref_field_read_and_write(self):
        assert_program_exit_code(
            self._CIRCLE +
            "def int main():\n"
            "    Circle c = Circle(5)\n"
            "    *Circle p = &c\n"
            "    p.radius = 9\n"
            "    return c.radius\n",
            expected=9,
        )

    def test_chained_auto_deref_field_read(self):
        assert_program_exit_code(
            self._NODE +
            "def int main():\n"
            "    Node c = Node(3, none)\n"
            "    Node b = Node(2, &c)\n"
            "    return b.next.value\n",
            expected=3,
        )

    def test_method_call_through_a_pointer_receiver(self):
        assert_program_exit_code(
            "type Circle struct:\n"
            "    int radius\n"
            "\n"
            "    def int area(self):\n"
            "        return self.radius * self.radius\n"
            "\n"
            "def int main():\n"
            "    Circle c = Circle(5)\n"
            "    *Circle p = &c\n"
            "    return p.area()\n",
            expected=25,
        )

    def test_same_type_pointer_equality(self):
        assert_program_exit_code(
            self._CIRCLE +
            "def int main():\n"
            "    Circle c = Circle(5)\n"
            "    *Circle p = &c\n"
            "    *Circle q = &c\n"
            "    if p == q:\n"
            "        return 1\n"
            "    return 0\n",
            expected=1,
        )

    def test_pointer_vs_none_equality(self):
        assert_program_exit_code(
            self._CIRCLE +
            "def int main():\n"
            "    Circle c = Circle(5)\n"
            "    *Circle p = &c\n"
            "    *Circle q = none\n"
            "    if p != none and q == none:\n"
            "        return 1\n"
            "    return 0\n",
            expected=1,
        )

    def test_none_as_a_pointer_assign_target_not_just_var_decl(self):
        assert_program_exit_code(
            self._CIRCLE +
            "def int main():\n"
            "    Circle c = Circle(5)\n"
            "    *Circle q = &c\n"
            "    q = none\n"
            "    if q == none:\n"
            "        return 1\n"
            "    return 0\n",
            expected=1,
        )

    def test_recursive_linked_list_traversal(self):
        assert_program_exit_code(
            self._NODE +
            "def int sumList(*Node head):\n"
            "    if head == none:\n"
            "        return 0\n"
            "    return head.value + sumList(head.next)\n"
            "\n"
            "def int main():\n"
            "    Node c = Node(3, none)\n"
            "    Node b = Node(2, &c)\n"
            "    Node a = Node(1, &b)\n"
            "    return sumList(&a)\n",
            expected=6,
        )

    def test_print_a_bare_pointer_prints_an_address(self):
        result = compile_and_run(
            "def int main():\n"
            "    int x = 5\n"
            "    *int p = &x\n"
            "    print(p)\n"
            "    return 0\n",
            agree=False,
        )
        assert re.match(r"^0x[0-9a-f]+\n$", result.stdout), result.stdout

    def test_print_a_null_pointer_prints_0x0(self):
        assert_program_stdout(
            self._CIRCLE +
            "def int main():\n"
            "    *Circle p = none\n"
            "    print(p)\n"
            "    return 0\n",
            "0x0\n",
        )

    def test_print_a_struct_with_a_pointer_field(self):
        result = compile_and_run(
            self._NODE +
            "def int main():\n"
            "    Node c = Node(3, none)\n"
            "    Node b = Node(2, &c)\n"
            "    print(b)\n"
            "    return 0\n",
            agree=False,
        )
        assert re.match(r"^Node\(value: 2, next: 0x[0-9a-f]+\)\n$", result.stdout), result.stdout

    def test_array_of_pointers(self):
        assert_program_exit_code(
            self._CIRCLE +
            "def int main():\n"
            "    Circle c = Circle(7)\n"
            "    [3]*Circle arr = [3]*Circle[&c, none, none]\n"
            "    *Circle first = arr[0]\n"
            "    return first.radius\n",
            expected=7,
        )


# ---------------------------------------------------------------------------
# Pointer escape analysis
# ---------------------------------------------------------------------------

class TestPointerEscapeAnalysis:

    def test_scalar_wrapped_in_struct_escaping_is_genuinely_heap_safe(self):
        assert_program_exit_code(
            "type Holder struct:\n"
            "    *int p\n"
            "\n"
            "def Holder makeDangling():\n"
            "    int x = 42\n"
            "    return Holder(&x)\n"
            "\n"
            "def int clobber():\n"
            "    int a = 111\n"
            "    int b = 222\n"
            "    int c = 333\n"
            "    int d = 444\n"
            "    return a + b + c + d\n"
            "\n"
            "def int main():\n"
            "    Holder h = makeDangling()\n"
            "    int unused = clobber()\n"
            "    return *h.p\n",
            expected=42,
        )

    def test_scalar_address_returned_directly_is_genuinely_heap_safe(self):
        assert_program_exit_code(
            "def *int makeDangling():\n"
            "    int x = 42\n"
            "    return &x\n"
            "\n"
            "def int clobber():\n"
            "    int a = 111\n"
            "    int b = 222\n"
            "    int c = 333\n"
            "    int d = 444\n"
            "    return a + b + c + d\n"
            "\n"
            "def int main():\n"
            "    *int p = makeDangling()\n"
            "    int unused = clobber()\n"
            "    return *p\n",
            expected=42,
        )

    def test_scalar_address_passed_to_another_function_is_genuinely_heap_safe(self):
        assert_program_exit_code(
            "def int useIt(*int p):\n"
            "    return *p\n"
            "\n"
            "def int caller():\n"
            "    int x = 7\n"
            "    return useIt(&x)\n"
            "\n"
            "def int main():\n"
            "    return caller()\n",
            expected=7,
        )

    def test_reassignment_after_escape_writes_through_the_same_box(self):
        assert_program_exit_code(
            "def *int makeDangling():\n"
            "    int x = 42\n"
            "    *int p = &x\n"
            "    x = 99\n"
            "    return p\n"
            "\n"
            "def int main():\n"
            "    *int p = makeDangling()\n"
            "    return *p\n",
            expected=99,
        )

    def test_reading_the_escaped_variable_by_name_after_promotion(self):
        assert_program_exit_code(
            "def *int makeAndRead():\n"
            "    int x = 42\n"
            "    *int p = &x\n"
            "    x = x + 1\n"
            "    return p\n"
            "\n"
            "def int main():\n"
            "    *int p = makeAndRead()\n"
            "    return *p\n",
            expected=43,
        )

    def test_escaping_str_with_no_initializer_uses_the_empty_string_zero_value(self):
        assert_program_exit_code(
            "def *str makeEmpty():\n"
            "    str s\n"
            "    return &s\n"
            "\n"
            "def int main():\n"
            "    *str p = makeEmpty()\n"
            "    if *p == '':\n"
            "        return 1\n"
            "    return 0\n",
            expected=1,
        )

    def test_address_of_a_field_escaping_is_genuinely_heap_safe(self):
        assert_program_exit_code(
            "type Circle struct:\n"
            "    int radius\n"
            "\n"
            "def *int getFieldAddr():\n"
            "    Circle c = Circle(5)\n"
            "    return &c.radius\n"
            "\n"
            "def int clobber():\n"
            "    int a = 111\n"
            "    int b = 222\n"
            "    int c = 333\n"
            "    int d = 444\n"
            "    return a + b + c + d\n"
            "\n"
            "def int main():\n"
            "    *int p = getFieldAddr()\n"
            "    int unused = clobber()\n"
            "    return *p\n",
            expected=5,
        )

    def test_address_of_an_element_escaping_is_genuinely_heap_safe(self):
        assert_program_exit_code(
            "def *int getElemAddr():\n"
            "    [3]int arr = [10, 20, 30]\n"
            "    return &arr[1]\n"
            "\n"
            "def int clobber():\n"
            "    int a = 111\n"
            "    int b = 222\n"
            "    int c = 333\n"
            "    int d = 444\n"
            "    return a + b + c + d\n"
            "\n"
            "def int main():\n"
            "    *int p = getElemAddr()\n"
            "    int unused = clobber()\n"
            "    return *p\n",
            expected=20,
        )

    def test_escaping_parameter_is_genuinely_heap_safe(self):
        assert_program_exit_code(
            "def *int identity(int x):\n"
            "    return &x\n"
            "\n"
            "def int clobber():\n"
            "    int a = 111\n"
            "    int b = 222\n"
            "    int c = 333\n"
            "    int d = 444\n"
            "    return a + b + c + d\n"
            "\n"
            "def int main():\n"
            "    *int p = identity(17)\n"
            "    int unused = clobber()\n"
            "    return *p\n",
            expected=17,
        )

    def test_struct_address_escaping_is_genuinely_heap_safe(self):
        assert_program_exit_code(
            "type Circle struct:\n"
            "    int radius\n"
            "\n"
            "def *Circle makeCircle():\n"
            "    Circle c = Circle(5)\n"
            "    return &c\n"
            "\n"
            "def int clobber():\n"
            "    int a = 111\n"
            "    int b = 222\n"
            "    int c = 333\n"
            "    int d = 444\n"
            "    return a + b + c + d\n"
            "\n"
            "def int main():\n"
            "    *Circle p = makeCircle()\n"
            "    int unused = clobber()\n"
            "    return p.radius\n",
            expected=5,
        )

    def test_address_of_a_struct_literal_escaping_is_genuinely_heap_safe(self):
        assert_program_exit_code(
            "type Circle struct:\n"
            "    int radius\n"
            "\n"
            "def *Circle makeCircle():\n"
            "    return &Circle(5)\n"
            "\n"
            "def int clobber():\n"
            "    int a = 111\n"
            "    int b = 222\n"
            "    int c = 333\n"
            "    int d = 444\n"
            "    return a + b + c + d\n"
            "\n"
            "def int main():\n"
            "    *Circle p = makeCircle()\n"
            "    int unused = clobber()\n"
            "    return p.radius\n",
            expected=5,
        )

    def test_purely_local_pointer_is_unaffected(self):
        assert_program_exit_code(
            "def int main():\n"
            "    int x = 5\n"
            "    *int p = &x\n"
            "    return *p\n",
            expected=5,
        )


# ---------------------------------------------------------------------------
# extern functions
# ---------------------------------------------------------------------------

class TestExternFunctions:

    def test_basic_extern_call(self):
        ast = _parse(
            "extern int abs(int n)\n"
            "def int main():\n"
            "    return abs(-5)\n"
        )
        analyze(ast)  # should not raise

    def test_extern_call_with_wrong_argument_count_is_rejected(self):
        assert_program_semantic_error(
            "extern int abs(int n)\n"
            "def int main():\n"
            "    return abs(-5, 3)\n",
            match="expects 1 argument",
        )

    def test_extern_call_with_wrong_argument_type_is_rejected(self):
        assert_program_semantic_error(
            "extern int abs(int n)\n"
            "def int main():\n"
            "    return abs(true)\n",
            match="should be int, got bool",
        )

    def test_extern_colliding_with_a_builtin_is_rejected(self):
        assert_program_semantic_error(
            "extern int len(int n)\n"
            "def int main():\n"
            "    return 0\n",
            match="'len' is a builtin and can't be redefined as an extern function",
        )

    def test_extern_colliding_with_a_struct_is_rejected(self):
        assert_program_semantic_error(
            "type Circle struct:\n"
            "    int radius\n"
            "\n"
            "extern int Circle(int n)\n"
            "def int main():\n"
            "    return 0\n",
            match="collides with a struct of the same name",
        )

    def test_extern_colliding_with_a_type_alias_is_rejected(self):
        assert_program_semantic_error(
            "type MyInt = int\n"
            "\n"
            "extern int MyInt(int n)\n"
            "def int main():\n"
            "    return 0\n",
            match="collides with a type alias of the same name",
        )

    def test_extern_colliding_with_a_sum_type_is_rejected(self):
        assert_program_semantic_error(
            "type Circle struct:\n"
            "    int radius\n"
            "\n"
            "type Square struct:\n"
            "    int side\n"
            "\n"
            "type Shape is Circle | Square\n"
            "\n"
            "extern int Shape(int n)\n"
            "def int main():\n"
            "    return 0\n",
            match="collides with a sum type of the same name",
        )

    def test_extern_colliding_with_an_ordinary_function_is_rejected(self):
        assert_program_semantic_error(
            "extern int abs(int n)\n"
            "def int abs():\n"
            "    return 0\n",
            match="'abs' is already declared",
        )

    def test_two_externs_with_the_same_name_are_rejected(self):
        assert_program_semantic_error(
            "extern int abs(int n)\n"
            "extern int abs(int n)\n"
            "def int main():\n"
            "    return 0\n",
            match="'abs' is already declared",
        )

    def test_struct_typed_extern_parameter_is_rejected(self):
        assert_program_semantic_error(
            "type Circle struct:\n"
            "    int radius\n"
            "\n"
            "extern int useCircle(Circle c)\n"
            "def int main():\n"
            "    return 0\n",
            match="only scalar and pointer types are supported in an extern function's signature",
        )

    def test_slice_typed_extern_return_is_rejected(self):
        assert_program_semantic_error(
            "extern []int makeSlice()\n"
            "def int main():\n"
            "    return 0\n",
            match="only scalar and pointer types are supported as an extern function's own return type",
        )

    def test_array_typed_extern_parameter_is_rejected(self):
        assert_program_semantic_error(
            "extern int useArray([3]int arr)\n"
            "def int main():\n"
            "    return 0\n",
            match="only scalar and pointer types are supported in an extern function's signature",
        )

    def test_pointer_param_and_return_are_accepted(self):
        ast = _parse(
            "extern *int8 malloc(int64 size)\n"
            "extern free(*int8 p)\n"
            "def int main():\n"
            "    *int8 p = malloc(8)\n"
            "    free(p)\n"
            "    return 0\n"
        )
        analyze(ast)  # should not raise

    def test_extern_with_no_return_type_is_void(self):
        ast = _parse(
            "extern free(*int8 p)\n"
            "def int main():\n"
            "    return 0\n"
        )
        analyze(ast)  # should not raise

    def test_extern_declared_after_its_own_call_site(self):
        ast = _parse(
            "def int main():\n"
            "    return abs(-5)\n"
            "\n"
            "extern int abs(int n)\n"
        )
        analyze(ast)  # should not raise

    def test_extern_with_more_than_six_parameters(self):
        ast = _parse(
            "extern int sum7(int a, int b, int c, int d, int e, int f, int g)\n"
            "\n"
            "def int main():\n"
            "    return sum7(1, 2, 3, 4, 5, 6, 7)\n"
        )
        analyze(ast)
        generate_asm(ast, target=ASM_TARGET)  # should not raise


class TestExternFunctionsCodegen:

    def test_call_a_fixed_arity_libc_function(self):
        assert_program_exit_code(
            "extern int abs(int n)\n"
            "def int main():\n"
            "    return abs(-42)\n",
            expected=42,
        )

    def test_malloc_and_free_via_extern(self):
        assert_program_exit_code(
            "extern *int malloc(int64 size)\n"
            "extern free(*int p)\n"
            "\n"
            "def int main():\n"
            "    *int p = malloc(8)\n"
            "    *p = 99\n"
            "    int result = *p\n"
            "    free(p)\n"
            "    return result\n",
            expected=99,
        )

    def test_extern_str_parameter_is_rejected(self):
        assert_program_semantic_error(
            "extern int strlen(str s)\n"
            "\n"
            "def int main():\n"
            "    return 0\n",
            match="only scalar and pointer types are supported",
        )

    def test_extern_str_return_is_rejected(self):
        assert_program_semantic_error(
            "extern str getenv(int fd)\n"
            "\n"
            "def int main():\n"
            "    return 0\n",
            match="only scalar and pointer types are supported",
        )

    def test_extern_taking_no_arguments(self):
        assert_program_exit_code(
            "extern int getpid()\n"
            "def int main():\n"
            "    int pid = getpid()\n"
            "    if pid > 0:\n"
            "        return 1\n"
            "    return 0\n",
            expected=1,
        )


class TestInt8Uint8TypeSystem:
    def _check(self, src, expect_error=None):
        """Semantic analysis only."""
        ast = _parse(src)
        if expect_error is None:
            analyze(ast)
            return
        try:
            analyze(ast)
        except SemanticError as e:
            assert expect_error in str(e), f"expected error containing {expect_error!r}, got: {e}"
            return
        assert False, f"expected a SemanticError containing {expect_error!r}, got none"

    def test_int8_literal_in_range_positive(self):
        self._check("def int main():\n    int8 x = 100\n    return 0\n")

    def test_int8_literal_in_range_negative(self):
        self._check("def int main():\n    int8 x = -100\n    return 0\n")

    def test_int8_literal_min_boundary(self):
        self._check("def int main():\n    int8 x = -128\n    return 0\n")

    def test_int8_literal_max_boundary(self):
        self._check("def int main():\n    int8 x = 127\n    return 0\n")

    def test_int8_literal_out_of_range_positive(self):
        self._check(
            "def int main():\n    int8 x = 128\n    return 0\n",
            expect_error="out of range",
        )

    def test_int8_literal_out_of_range_negative(self):
        self._check(
            "def int main():\n    int8 x = -129\n    return 0\n",
            expect_error="out of range",
        )

    def test_uint8_literal_in_range(self):
        self._check("def int main():\n    uint8 x = 200\n    return 0\n")

    def test_uint8_literal_max_boundary(self):
        self._check("def int main():\n    uint8 x = 255\n    return 0\n")

    def test_uint8_literal_negative_is_rejected(self):
        self._check(
            "def int main():\n    uint8 x = -1\n    return 0\n",
            expect_error="out of range",
        )

    def test_uint8_literal_out_of_range_positive(self):
        self._check(
            "def int main():\n    uint8 x = 256\n    return 0\n",
            expect_error="out of range",
        )

    def test_int_variable_does_not_implicitly_narrow_into_int8(self):
        self._check(
            "def int main():\n"
            "    int y = 5\n"
            "    int8 x = y\n"
            "    return 0\n",
            expect_error="Cannot initialize",
        )

    def test_int8_arithmetic_stays_int8(self):
        self._check(
            "def int main():\n"
            "    int8 a = 5\n"
            "    int8 b = 3\n"
            "    int8 c = a + b\n"
            "    return 0\n"
        )

    def test_int8_arithmetic_result_rejected_by_a_wider_target(self):
        self._check(
            "def int main():\n"
            "    int8 a = 5\n"
            "    int8 b = 3\n"
            "    int c = a + b\n"
            "    return 0\n",
            expect_error="Cannot initialize",
        )

    def test_uint8_arithmetic_stays_uint8(self):
        self._check(
            "def int main():\n"
            "    uint8 a = 5\n"
            "    uint8 b = 3\n"
            "    uint8 c = a * b\n"
            "    return 0\n"
        )

    def test_int8_and_uint8_arithmetic_mixing_is_rejected(self):
        self._check(
            "def int main():\n"
            "    int8 a = 5\n"
            "    uint8 b = 3\n"
            "    int8 c = a + b\n"
            "    return 0\n",
            expect_error="requires two operands of the same integer type",
        )

    def test_int8_and_int_arithmetic_mixing_is_rejected(self):
        self._check(
            "def int main():\n"
            "    int8 a = 5\n"
            "    int b = 3\n"
            "    int8 c = a + b\n"
            "    return 0\n",
            expect_error="requires two operands of the same integer type",
        )

    def test_unary_negate_stays_int8(self):
        self._check(
            "def int main():\n"
            "    int8 a = 5\n"
            "    int8 b = -a\n"
            "    return 0\n"
        )

    def test_unary_complement_stays_uint8(self):
        self._check(
            "def int main():\n"
            "    uint8 a = 5\n"
            "    uint8 b = ~a\n"
            "    return 0\n"
        )

    def test_not_still_rejects_int8(self):
        self._check(
            "def int main():\n"
            "    int8 a = 5\n"
            "    bool b = not a\n"
            "    return 0\n",
            expect_error="requires a bool operand",
        )

    def test_int8_ordering_comparison(self):
        self._check(
            "def int main():\n"
            "    int8 a = 5\n"
            "    int8 b = 3\n"
            "    bool result = a > b\n"
            "    return 0\n"
        )

    def test_int8_uint8_ordering_mixing_is_rejected(self):
        self._check(
            "def int main():\n"
            "    int8 a = 5\n"
            "    uint8 b = 3\n"
            "    bool result = a < b\n"
            "    return 0\n",
            expect_error="requires two operands of the same integer type",
        )

    def test_int8_equality(self):
        self._check(
            "def int main():\n"
            "    int8 a = 5\n"
            "    int8 b = 3\n"
            "    bool result = a == b\n"
            "    return 0\n"
        )

    def test_int8_uint8_equality_mixing_is_rejected(self):
        self._check(
            "def int main():\n"
            "    int8 a = 5\n"
            "    uint8 b = 3\n"
            "    bool result = a == b\n"
            "    return 0\n",
            expect_error="both sides must be the same type",
        )

    def test_int8_as_function_parameter(self):
        self._check(
            "def int identity(int8 x):\n"
            "    return 0\n"
            "\n"
            "def int main():\n"
            "    return identity(100)\n"
        )

    def test_int8_function_argument_out_of_range(self):
        self._check(
            "def int identity(int8 x):\n"
            "    return 0\n"
            "\n"
            "def int main():\n"
            "    return identity(200)\n",
            expect_error="out of range",
        )

    def test_int8_as_function_return_type(self):
        self._check(
            "def int8 makeIt():\n"
            "    return 100\n"
            "\n"
            "def int main():\n"
            "    return 0\n"
        )

    def test_int8_return_value_out_of_range(self):
        self._check(
            "def int8 makeIt():\n"
            "    return 200\n"
            "\n"
            "def int main():\n"
            "    return 0\n",
            expect_error="out of range",
        )

    def test_int8_as_method_argument(self):
        self._check(
            "type A struct:\n"
            "    int v\n"
            "    def int useIt(s, int8 x):\n"
            "        return 0\n"
            "\n"
            "def int main():\n"
            "    A a = A(1)\n"
            "    return a.useIt(100)\n"
        )

    def test_int8_struct_field_positional(self):
        self._check(
            "type S struct:\n"
            "    int8 x\n"
            "    uint8 y\n"
            "\n"
            "def int main():\n"
            "    S s = S(1, 2)\n"
            "    return 0\n"
        )

    def test_int8_struct_field_named(self):
        self._check(
            "type S struct:\n"
            "    int8 x\n"
            "    uint8 y\n"
            "\n"
            "def int main():\n"
            "    S s = S(x=1, y=2)\n"
            "    return 0\n"
        )

    def test_int8_struct_field_positional_out_of_range(self):
        self._check(
            "type S struct:\n"
            "    int8 x\n"
            "\n"
            "def int main():\n"
            "    S s = S(200)\n"
            "    return 0\n",
            expect_error="out of range",
        )

    def test_untyped_array_literal_into_int8_array_target(self):
        self._check(
            "def int main():\n"
            "    [3]int8 arr = [1, 2, 3]\n"
            "    return 0\n"
        )

    def test_untyped_array_literal_element_out_of_range(self):
        self._check(
            "def int main():\n"
            "    [3]int8 arr = [1, 200, 3]\n"
            "    return 0\n",
            expect_error="out of range",
        )

    def test_untyped_array_literal_size_mismatch_still_caught(self):
        self._check(
            "def int main():\n"
            "    [3]int8 arr = [1, 2]\n"
            "    return 0\n",
            expect_error="Cannot initialize",
        )

    def test_untyped_array_literal_into_slice_typed_struct_field(self):
        self._check(
            "type Holder struct:\n"
            "    []int xs\n"
            "\n"
            "def int main():\n"
            "    Holder h = Holder([1, 2, 3])\n"
            "    return 0\n"
        )

    def test_ragged_array_literal_still_rejected(self):
        self._check(
            "def int main():\n"
            "    [2][3]int8 matrix = [[1, 2, 3], [4, 5]]\n"
            "    return 0\n",
            expect_error="elements must all be",
        )

    def test_int8_alias(self):
        self._check(
            "type MyByte = int8\n"
            "\n"
            "def int main():\n"
            "    MyByte x = 100\n"
            "    return 0\n"
        )

    def test_int8_alias_out_of_range(self):
        self._check(
            "type MyByte = int8\n"
            "\n"
            "def int main():\n"
            "    MyByte x = 200\n"
            "    return 0\n",
            expect_error="out of range",
        )

    def test_int8_array_element_type(self):
        self._check("def int main():\n    [3]int8 arr\n    return 0\n")

    def test_uint8_array_element_type(self):
        self._check("def int main():\n    [3]uint8 arr\n    return 0\n")


# ---------------------------------------------------------------------------
# int8/uint8
# ---------------------------------------------------------------------------

class TestInt8Uint8Storage:
    pytestmark = GCC_SKIP

    def test_int8_vardecl_and_return(self):
        assert_program_exit_code(
            "def int8 main():\n"
            "    int8 x = 5\n"
            "    return x\n",
            5,
        )

    def test_int8_negative_literal_wraps_as_exit_code(self):
        assert_program_exit_code(
            "def int8 main():\n"
            "    int8 x = -5\n"
            "    return x\n",
            251,
        )

    def test_uint8_vardecl_and_return(self):
        assert_program_exit_code(
            "def uint8 main():\n"
            "    uint8 x = 200\n"
            "    return x\n",
            200,
        )

    def test_two_int8_locals_are_independently_stored(self):
        assert_program_exit_code(
            "def int8 main():\n"
            "    int8 a = 10\n"
            "    int8 b = 20\n"
            "    return a\n",
            10,
        )

    def test_int8_assign(self):
        assert_program_exit_code(
            "def int8 main():\n"
            "    int8 x = 1\n"
            "    x = 42\n"
            "    return x\n",
            42,
        )

    def test_int8_struct_field_read(self):
        assert_program_exit_code(
            "type Point struct:\n"
            "    int8 x\n"
            "    uint8 y\n"
            "\n"
            "def int8 main():\n"
            "    Point p = Point(5, 10)\n"
            "    return p.x\n",
            5,
        )

    def test_int8_struct_field_read_second_field(self):
        assert_program_exit_code(
            "type Point struct:\n"
            "    int8 x\n"
            "    uint8 y\n"
            "\n"
            "def uint8 main():\n"
            "    Point p = Point(5, 10)\n"
            "    return p.y\n",
            10,
        )

    def test_int8_struct_field_assign(self):
        assert_program_exit_code(
            "type S struct:\n"
            "    int8 v\n"
            "\n"
            "def int8 main():\n"
            "    S s = S(0)\n"
            "    s.v = 77\n"
            "    return s.v\n",
            77,
        )

    def test_int_field_after_int8_field_in_struct(self):
        assert_program_exit_code(
            "type S struct:\n"
            "    int8 x\n"
            "    int z\n"
            "\n"
            "def int main():\n"
            "    S s = S(1, 200)\n"
            "    return s.z\n",
            200,
        )

    def test_int8_array_element_read(self):
        assert_program_exit_code(
            "def int8 main():\n"
            "    [3]int8 arr = [1, 2, 3]\n"
            "    return arr[2]\n",
            3,
        )

    def test_int8_array_index_assign(self):
        assert_program_exit_code(
            "def int8 main():\n"
            "    [3]int8 arr = [0, 0, 0]\n"
            "    arr[1] = 99\n"
            "    return arr[1]\n",
            99,
        )

    def test_int8_function_parameter_and_return(self):
        assert_program_exit_code(
            "def int8 identity(int8 x):\n"
            "    return x\n"
            "\n"
            "def int8 main():\n"
            "    return identity(100)\n",
            100,
        )

    def test_int8_negative_function_argument(self):
        assert_program_exit_code(
            "def int8 identity(int8 x):\n"
            "    return x\n"
            "\n"
            "def int8 main():\n"
            "    return identity(-5)\n",
            251,
        )

    def test_uint8_function_parameter(self):
        assert_program_exit_code(
            "def uint8 identity(uint8 x):\n"
            "    return x\n"
            "\n"
            "def uint8 main():\n"
            "    return identity(200)\n",
            200,
        )

    def test_int8_addition_wraps(self):
        assert_program_exit_code(
            "def int8 main():\n"
            "    int8 a = 100\n"
            "    int8 b = 100\n"
            "    int8 c = a + b\n"
            "    return c\n",
            256 - 56,
        )

    def test_uint8_addition_wraps(self):
        assert_program_exit_code(
            "def uint8 main():\n"
            "    uint8 a = 200\n"
            "    uint8 b = 100\n"
            "    uint8 c = a + b\n"
            "    return c\n",
            44,
        )

    def test_int8_multiplication_wraps(self):
        assert_program_exit_code(
            "def int8 main():\n"
            "    int8 a = 20\n"
            "    int8 b = 20\n"
            "    int8 c = a * b\n"
            "    return c\n",
            (20 * 20) % 256,
        )

    def test_int8_negate_boundary_wraps(self):
        assert_program_exit_code(
            "def int8 main():\n"
            "    int8 a = -128\n"
            "    int8 b = -a\n"
            "    return b\n",
            256 - 128,
        )

    def test_int8_subtraction_negative_result(self):
        assert_program_exit_code(
            "def int8 main():\n"
            "    int8 a = 5\n"
            "    int8 b = 10\n"
            "    int8 c = a - b\n"
            "    return c\n",
            256 - 5,
        )

    def test_int8_division(self):
        assert_program_exit_code(
            "def int8 main():\n"
            "    int8 a = 100\n"
            "    int8 b = 7\n"
            "    int8 c = a / b\n"
            "    return c\n",
            100 // 7,
        )

    def test_int8_modulo(self):
        assert_program_exit_code(
            "def int8 main():\n"
            "    int8 a = 100\n"
            "    int8 b = 7\n"
            "    int8 c = a % b\n"
            "    return c\n",
            100 % 7,
        )

    def test_int8_bitwise_and(self):
        assert_program_exit_code(
            "def int8 main():\n"
            "    int8 a = 12\n"
            "    int8 b = 10\n"
            "    int8 c = a & b\n"
            "    return c\n",
            12 & 10,
        )

    def test_int8_complement(self):
        assert_program_exit_code(
            "def int8 main():\n"
            "    int8 a = 5\n"
            "    int8 b = ~a\n"
            "    return b\n",
            256 + (~5),
        )

    def test_int8_comparison_respects_sign(self):
        assert_program_exit_code(
            "def int8 main():\n"
            "    int8 a = -1\n"
            "    int8 b = 1\n"
            "    if a < b:\n"
            "        return 1\n"
            "    return 0\n",
            1,
        )

    def test_uint8_comparison_is_unsigned(self):
        assert_program_exit_code(
            "def uint8 main():\n"
            "    uint8 a = 200\n"
            "    uint8 b = 100\n"
            "    if a > b:\n"
            "        return 1\n"
            "    return 0\n",
            1,
        )

    def test_int8_array_equality_equal(self):
        assert_program_exit_code(
            "def int main():\n"
            "    [3]int8 a = [1, 2, 3]\n"
            "    [3]int8 b = [1, 2, 3]\n"
            "    if a == b:\n"
            "        return 1\n"
            "    return 0\n",
            1,
        )

    def test_int8_array_equality_not_equal(self):
        assert_program_exit_code(
            "def int main():\n"
            "    [3]int8 a = [1, 2, 3]\n"
            "    [3]int8 b = [1, 2, 9]\n"
            "    if a == b:\n"
            "        return 1\n"
            "    return 0\n",
            0,
        )

    def test_uint8_array_equality(self):
        assert_program_exit_code(
            "def int main():\n"
            "    [5]uint8 a = [1, 2, 3, 4, 5]\n"
            "    [5]uint8 b = [1, 2, 3, 4, 5]\n"
            "    if a == b:\n"
            "        return 1\n"
            "    return 0\n",
            1,
        )

    def test_int8_array_zero_init_values(self):
        assert_program_exit_code(
            "def int8 main():\n"
            "    [3]int8 arr\n"
            "    return arr[0] + arr[1] + arr[2]\n",
            0,
        )

    def test_int8_array_zero_init_does_not_corrupt_adjacent_local(self):
        assert_program_exit_code(
            "def int8 main():\n"
            "    int8 guard = 42\n"
            "    [3]int8 arr\n"
            "    return guard\n",
            42,
        )

    def test_struct_with_int8_field_zero_init(self):
        assert_program_exit_code(
            "type S struct:\n"
            "    int8 x\n"
            "    uint8 y\n"
            "    int z\n"
            "\n"
            "def int main():\n"
            "    S s\n"
            "    return s.z\n",
            0,
        )

    def test_struct_with_int8_array_field_zero_init_does_not_corrupt_adjacent_field(self):
        assert_program_exit_code(
            "type S struct:\n"
            "    [3]int8 arr\n"
            "    int guard\n"
            "\n"
            "def int main():\n"
            "    S s\n"
            "    return s.guard\n",
            0,
        )

    def test_plain_array_assignment_with_int8_leaf(self):
        assert_program_exit_code(
            "def int8 main():\n"
            "    [3]int8 a = [1, 2, 3]\n"
            "    [3]int8 b = [0, 0, 0]\n"
            "    b = a\n"
            "    return b[0] + b[1] + b[2]\n",
            6,
        )

    def test_array_of_struct_with_int8_field_assignment(self):
        assert_program_exit_code(
            "type S struct:\n"
            "    int8 x\n"
            "    int y\n"
            "\n"
            "def int main():\n"
            "    [2]S a = [S(1, 10), S(2, 20)]\n"
            "    [2]S b = [S(0, 0), S(0, 0)]\n"
            "    b = a\n"
            "    return b[0].y + b[1].y\n",
            30,
        )

    def test_array_of_struct_with_two_int8_fields_assignment(self):
        assert_program_exit_code(
            "type Pair struct:\n"
            "    int8 a\n"
            "    int8 b\n"
            "\n"
            "def int8 main():\n"
            "    [3]Pair arr = [Pair(1, 2), Pair(3, 4), Pair(5, 6)]\n"
            "    [3]Pair arr2 = [Pair(0, 0), Pair(0, 0), Pair(0, 0)]\n"
            "    arr2 = arr\n"
            "    return arr2[2].b\n",
            6,
        )

    def test_array_of_struct_with_mixed_field_widths_assignment(self):
        assert_program_exit_code(
            "type Mixed struct:\n"
            "    int8 a\n"
            "    int b\n"
            "    uint8 c\n"
            "\n"
            "def uint8 main():\n"
            "    [2]Mixed arr = [Mixed(1, 100, 2), Mixed(3, 200, 4)]\n"
            "    [2]Mixed arr2 = [Mixed(0, 0, 0), Mixed(0, 0, 0)]\n"
            "    arr2 = arr\n"
            "    return arr2[1].c\n",
            4,
        )

    def test_array_parameter_with_int8_leaf(self):
        assert_program_exit_code(
            "def int8 sumFirstTwo([3]int8 arr):\n"
            "    return arr[0] + arr[1]\n"
            "\n"
            "def int8 main():\n"
            "    [3]int8 a = [10, 20, 30]\n"
            "    return sumFirstTwo(a)\n",
            30,
        )

    def test_array_returning_function_with_int8_leaf(self):
        assert_program_exit_code(
            "def [3]int8 makeArr():\n"
            "    return [7, 8, 9]\n"
            "\n"
            "def int8 main():\n"
            "    [3]int8 a = makeArr()\n"
            "    return a[0] + a[1] + a[2]\n",
            24,
        )

    def test_multidimensional_int8_array(self):
        assert_program_exit_code(
            "def int8 main():\n"
            "    [2][3]int8 m = [[1, 2, 3], [4, 5, 6]]\n"
            "    return m[1][2]\n",
            6,
        )

    def test_multidimensional_int8_array_zero_init(self):
        assert_program_exit_code(
            "def int8 main():\n"
            "    [2][3]int8 m\n"
            "    return m[0][0] + m[1][2]\n",
            0,
        )

    def test_multidimensional_int8_array_equality(self):
        assert_program_exit_code(
            "def int main():\n"
            "    [2][3]int8 a = [[1, 2, 3], [4, 5, 6]]\n"
            "    [2][3]int8 b = [[1, 2, 3], [4, 5, 6]]\n"
            "    if a == b:\n"
            "        return 1\n"
            "    return 0\n",
            1,
        )

    def test_slice_of_int8_basic(self):
        assert_program_exit_code(
            "def int8 main():\n"
            "    []int8 s = []int8[1, 2, 3]\n"
            "    return s[0] + s[1] + s[2]\n",
            6,
        )

    def test_slice_of_uint8_basic(self):
        assert_program_exit_code(
            "def uint8 main():\n"
            "    []uint8 s = []uint8[200, 50]\n"
            "    return s[0] + s[1]\n",
            (200 + 50) % 256,
        )

    def test_slice_of_int8_index_assign(self):
        assert_program_exit_code(
            "def int8 main():\n"
            "    []int8 s = []int8[1, 2, 3]\n"
            "    s[1] = 99\n"
            "    return s[1]\n",
            99,
        )

    def test_slice_of_int8_len(self):
        assert_program_exit_code(
            "def int main():\n"
            "    []int8 s = []int8[1, 2, 3, 4, 5]\n"
            "    return len(s)\n",
            5,
        )

    def test_append_int8_to_slice(self):
        assert_program_exit_code(
            "def int8 main():\n"
            "    []int8 s = []int8[1, 2]\n"
            "    s = append(s, 3)\n"
            "    return s[0] + s[1] + s[2]\n",
            6,
        )

    def test_append_uint8_to_slice(self):
        assert_program_exit_code(
            "def uint8 main():\n"
            "    []uint8 s = []uint8[1, 2]\n"
            "    s = append(s, 250)\n"
            "    return s[2]\n",
            250,
        )

    def test_append_int8_across_multiple_growths(self):
        assert_program_exit_code(
            "def int main():\n"
            "    []int8 s = []int8[1]\n"
            "    int guard = 77\n"
            "    int i = 0\n"
            "    while i < 20:\n"
            "        s = append(s, 2)\n"
            "        i = i + 1\n"
            "    return guard\n",
            77,
        )

    def test_append_int8_across_multiple_growths_values(self):
        assert_program_exit_code(
            "def int8 main():\n"
            "    []int8 s = []int8[1]\n"
            "    int i = 0\n"
            "    while i < 20:\n"
            "        s = append(s, 2)\n"
            "        i = i + 1\n"
            "    return s[20]\n",
            2,
        )

    def test_print_int8_now_supported(self):
        assert_program_stdout(
            "def int main():\n    int8 x = 5\n    print(x)\n    return 0\n",
            "5\n",
        )


# ---------------------------------------------------------------------------
# int8/uint8
# ---------------------------------------------------------------------------

class TestInt8Uint8Print:
    pytestmark = GCC_SKIP

    def test_print_positive_int8(self):
        assert_program_stdout(
            "def int main():\n    int8 x = 5\n    print(x)\n    return 0\n",
            "5\n",
        )

    def test_print_negative_int8(self):
        assert_program_stdout(
            "def int main():\n    int8 x = -5\n    print(x)\n    return 0\n",
            "-5\n",
        )

    def test_print_int8_min_boundary(self):
        assert_program_stdout(
            "def int main():\n    int8 x = -128\n    print(x)\n    return 0\n",
            "-128\n",
        )

    def test_print_int8_max_boundary(self):
        assert_program_stdout(
            "def int main():\n    int8 x = 127\n    print(x)\n    return 0\n",
            "127\n",
        )

    def test_print_uint8_basic(self):
        assert_program_stdout(
            "def int main():\n    uint8 x = 200\n    print(x)\n    return 0\n",
            "200\n",
        )

    def test_print_uint8_max_boundary(self):
        assert_program_stdout(
            "def int main():\n    uint8 x = 255\n    print(x)\n    return 0\n",
            "255\n",
        )

    def test_print_uint8_zero(self):
        assert_program_stdout(
            "def int main():\n    uint8 x = 0\n    print(x)\n    return 0\n",
            "0\n",
        )

    def test_print_int8_non_variable_expression(self):
        assert_program_stdout(
            "def int8 main():\n"
            "    int8 a = 5\n"
            "    int8 b = 3\n"
            "    print(a + b)\n"
            "    return 0\n",
            "8\n",
        )

    def test_print_int8_array(self):
        assert_program_stdout(
            "def int main():\n"
            "    [3]int8 arr = [1, -2, 3]\n"
            "    print(arr)\n"
            "    return 0\n",
            "[3]int8[1, -2, 3]\n",
        )

    def test_print_uint8_array(self):
        assert_program_stdout(
            "def int main():\n"
            "    [3]uint8 arr = [1, 200, 3]\n"
            "    print(arr)\n"
            "    return 0\n",
            "[3]uint8[1, 200, 3]\n",
        )

    def test_print_struct_with_int8_and_uint8_fields(self):
        assert_program_stdout(
            "type Point struct:\n"
            "    int8 x\n"
            "    uint8 y\n"
            "\n"
            "def int main():\n"
            "    Point p = Point(-5, 200)\n"
            "    print(p)\n"
            "    return 0\n",
            "Point(x: -5, y: 200)\n",
        )

    def test_print_slice_of_int8(self):
        assert_program_stdout(
            "def int main():\n"
            "    []int8 s = []int8[1, -2, 3]\n"
            "    print(s)\n"
            "    return 0\n",
            "[]int8[1, -2, 3]\n",
        )


# ---------------------------------------------------------------------------
# byte
# ---------------------------------------------------------------------------

class TestByte:
    pytestmark = GCC_SKIP

    def test_byte_basic(self):
        assert_program_exit_code(
            "def byte main():\n    byte x = 200\n    return x\n",
            200,
        )

    def test_byte_and_uint8_interchange_without_a_cast(self):
        assert_program_exit_code(
            "def byte main():\n"
            "    byte x = 5\n"
            "    uint8 y = x\n"
            "    byte z = y\n"
            "    return z\n",
            5,
        )

    def test_byte_literal_out_of_range_error_says_uint8(self):
        assert_program_semantic_error(
            "def int main():\n    byte x = 300\n    return 0\n",
            match="out of range for uint8",
        )

    def test_byte_and_int8_are_still_incompatible(self):
        assert_program_semantic_error(
            "def int main():\n"
            "    byte x = 5\n"
            "    int8 y = 5\n"
            "    byte z = x + y\n"
            "    return 0\n",
            match="requires two operands of the same integer type",
        )

    def test_byte_and_int_are_still_incompatible(self):
        assert_program_semantic_error(
            "def int main():\n"
            "    byte x = 5\n"
            "    int y = 5\n"
            "    byte z = x + y\n"
            "    return 0\n",
            match="requires two operands of the same integer type",
        )

    def test_byte_arithmetic_wraps(self):
        assert_program_exit_code(
            "def byte main():\n"
            "    byte a = 200\n"
            "    byte b = 100\n"
            "    byte c = a + b\n"
            "    return c\n",
            (200 + 100) % 256,
        )

    def test_print_byte_value(self):
        assert_program_stdout(
            "def int main():\n    byte x = 200\n    print(x)\n    return 0\n",
            "200\n",
        )

    def test_print_byte_array_shows_uint8_type_name(self):
        assert_program_stdout(
            "def int main():\n"
            "    [3]byte arr = [1, 2, 3]\n"
            "    print(arr)\n"
            "    return 0\n",
            "[3]uint8[1, 2, 3]\n",
        )

    def test_byte_as_function_parameter_and_return(self):
        assert_program_exit_code(
            "def byte identity(byte x):\n"
            "    return x\n"
            "\n"
            "def byte main():\n"
            "    return identity(250)\n",
            250,
        )

    def test_byte_struct_field(self):
        assert_program_exit_code(
            "type S struct:\n"
            "    byte v\n"
            "\n"
            "def byte main():\n"
            "    S s = S(150)\n"
            "    return s.v\n",
            150,
        )

    def test_byte_array_storage_is_genuinely_dense(self):
        assert_program_exit_code(
            "def byte main():\n"
            "    [5]byte arr = [1, 2, 3, 4, 5]\n"
            "    return arr[0] + arr[4]\n",
            6,
        )

    def test_byte_is_a_reserved_keyword(self):
        with pytest.raises(ParseError, match="Expected a variable name"):
            _parse("def int main():\n    int byte = 5\n    return byte\n")


# ---------------------------------------------------------------------------
# Casting
# ---------------------------------------------------------------------------

class TestCasting:
    def test_int_to_int8_cast_type_checks(self):
        ast = _parse(
            "def int8 main():\n"
            "    int x = 300\n"
            "    return int8(x)\n"
        )
        analyze(ast)

    def test_int8_to_int_cast_type_checks(self):
        ast = _parse(
            "def int main():\n"
            "    int8 x = 5\n"
            "    return int(x)\n"
        )
        analyze(ast)

    def test_int8_to_uint8_cast_type_checks(self):
        ast = _parse(
            "def uint8 main():\n"
            "    int8 x = -5\n"
            "    return uint8(x)\n"
        )
        analyze(ast)

    def test_cast_result_type_matches_target_exactly(self):
        ast = _parse(
            "def int main():\n"
            "    int8 x = 5\n"
            "    int y = int8(x)\n"
            "    return y\n"
        )
        with pytest.raises(SemanticError, match="Cannot initialize"):
            analyze(ast)

    def test_cast_to_bool_is_rejected(self):
        ast = _parse(
            "def int main():\n"
            "    int x = 5\n"
            "    bool b = bool(x)\n"
            "    return 0\n"
        )
        with pytest.raises(SemanticError, match="Cannot cast"):
            analyze(ast)

    def test_cast_int_to_str_is_rejected(self):
        ast = _parse(
            "def int main():\n"
            "    int x = 5\n"
            "    str s = str(x)\n"
            "    return 0\n"
        )
        with pytest.raises(SemanticError, match="takes a byte or"):
            analyze(ast)

    def test_cast_from_bool_is_rejected(self):
        ast = _parse(
            "def int main():\n"
            "    bool b = true\n"
            "    int x = int(b)\n"
            "    return 0\n"
        )
        with pytest.raises(SemanticError, match="Cannot cast"):
            analyze(ast)

    def test_cast_to_a_type_alias_name_is_not_yet_supported(self):
        ast = _parse(
            "type MyByte = int8\n"
            "\n"
            "def int main():\n"
            "    int x = 5\n"
            "    MyByte y = MyByte(x)\n"
            "    return 0\n"
        )
        with pytest.raises(SemanticError, match="undeclared"):
            analyze(ast)

    def test_ordinary_vardecl_still_parses_correctly(self):
        ast = _parse("def int main():\n    int8 x = 5\n    return 0\n")
        analyze(ast)


class TestCastingCodegen:
    pytestmark = GCC_SKIP

    def test_narrowing_cast_int_300_to_int8(self):
        assert_program_exit_code(
            "def int8 main():\n"
            "    int x = 300\n"
            "    return int8(x)\n",
            44,
        )

    def test_narrowing_cast_int_200_to_int8_crosses_sign_boundary(self):
        assert_program_exit_code(
            "def int8 main():\n"
            "    int x = 200\n"
            "    return int8(x)\n",
            256 - 56,
        )

    def test_reinterpreting_cast_uint8_to_int8(self):
        assert_program_exit_code(
            "def int8 main():\n"
            "    uint8 x = 200\n"
            "    return int8(x)\n",
            256 - 56,
        )

    def test_reinterpreting_cast_int8_to_uint8(self):
        assert_program_exit_code(
            "def uint8 main():\n"
            "    int8 x = -5\n"
            "    return uint8(x)\n",
            256 - 5,
        )

    def test_widening_cast_int8_to_int(self):
        assert_program_exit_code(
            "def int main():\n"
            "    int8 x = -5\n"
            "    return int(x)\n",
            256 - 5,
        )

    def test_widening_cast_uint8_to_int(self):
        assert_program_exit_code(
            "def int main():\n"
            "    uint8 x = 200\n"
            "    return int(x)\n",
            200,
        )

    def test_cast_truncates_immediately_not_just_worked_when_written(self):
        assert_program_exit_code(
            "def int8 main():\n"
            "    int x = 300\n"
            "    int8 y = 7\n"
            "    return int8(x) / y\n",
            44 // 7,
        )

    def test_cast_truncation_proven_via_comparison_true_case(self):
        assert_program_exit_code(
            "def int main():\n"
            "    int x = 300\n"
            "    int8 y = 50\n"
            "    if int8(x) < y:\n"
            "        return 1\n"
            "    return 0\n",
            1,
        )

    def test_cast_truncation_proven_via_comparison_false_case(self):
        assert_program_exit_code(
            "def int main():\n"
            "    int x = 60\n"
            "    int8 y = 50\n"
            "    if int8(x) < y:\n"
            "        return 1\n"
            "    return 0\n",
            0,
        )

    def test_cast_as_bare_statement(self):
        assert_program_exit_code(
            "def int main():\n"
            "    int x = 5\n"
            "    int8(x)\n"
            "    return 0\n",
            0,
        )

    def test_nested_casts(self):
        assert_program_exit_code(
            "def int main():\n"
            "    int x = 300\n"
            "    return int(int8(x))\n",
            44,
        )

    def test_cast_of_a_literal(self):
        assert_program_exit_code(
            "def int8 main():\n    return int8(300)\n",
            44,
        )

    def test_cast_of_an_arithmetic_expression(self):
        assert_program_exit_code(
            "def int8 main():\n"
            "    int a = 250\n"
            "    int b = 100\n"
            "    return int8(a + b)\n",
            (250 + 100) % 256,
        )

    def test_print_narrowing_cast_result(self):
        assert_program_stdout(
            "def int main():\n"
            "    int x = 300\n"
            "    print(int8(x))\n"
            "    return 0\n",
            "44\n",
        )

    def test_print_widening_cast_result(self):
        assert_program_stdout(
            "def int main():\n"
            "    int8 x = -5\n"
            "    print(int(x))\n"
            "    return 0\n",
            "-5\n",
        )

    def test_cast_as_function_argument(self):
        assert_program_exit_code(
            "def int8 identity(int8 x):\n"
            "    return x\n"
            "\n"
            "def int8 main():\n"
            "    int x = 300\n"
            "    return identity(int8(x))\n",
            44,
        )

    def test_cast_as_struct_field_value(self):
        assert_program_exit_code(
            "type S struct:\n"
            "    int8 v\n"
            "\n"
            "def int8 main():\n"
            "    int x = 300\n"
            "    S s = S(int8(x))\n"
            "    return s.v\n",
            44,
        )

    def test_widening_cast_of_a_literal_exceeding_int32_range_preserves_the_full_value(self):
        assert_program_stdout(
            "def int main():\n"
            "    print(int64(1099511628211))\n"
            "    return 0\n",
            "1099511628211\n",
        )

    def test_widening_cast_of_a_negative_literal_exceeding_int32_range(self):
        assert_program_stdout(
            "def int main():\n"
            "    print(int64(-1099511628211))\n"
            "    return 0\n",
            "-1099511628211\n",
        )

    def test_repeated_multiplication_by_a_large_int64_literal_constant_in_a_called_function(self):
        assert_program_stdout(
            "def int64 fnvStep(int64 h, byte b):\n"
            "    return (h ^ int64(b)) * int64(1099511628211)\n\n"
            "def int main():\n"
            "    int64 h = -3750763034362895579\n"
            "    int64 i = 0\n"
            "    while i < int64(8):\n"
            "        h = fnvStep(h, byte(i))\n"
            "        i = i + int64(1)\n"
            "    print(h)\n"
            "    return 0\n",
            "-6567292918605886595\n",
        )


# ---------------------------------------------------------------------------
# int64
# ---------------------------------------------------------------------------

class TestInt64TypeSystem:
    def _check(self, src, expect_error=None):
        """Semantic analysis only."""
        ast = _parse(src)
        if expect_error is None:
            analyze(ast)
            return
        try:
            analyze(ast)
        except SemanticError as e:
            assert expect_error in str(e), f"expected error containing {expect_error!r}, got: {e}"
            return
        assert False, f"expected a SemanticError containing {expect_error!r}, got none"

    def test_int64_literal_widening(self):
        self._check("def int64 main():\n    int64 x = 5\n    return 0\n")

    def test_int64_negative_literal_widening(self):
        self._check("def int64 main():\n    int64 x = -5\n    return 0\n")

    def test_int64_large_literal_widening(self):
        self._check("def int64 main():\n    int64 x = 9000000000\n    return 0\n")

    def test_int64_is_int(self):
        assert_program_stdout(
            "def int main():\n"
            "    int y = 5\n"
            "    int64 x = y\n"
            "    int64 c = x + y\n"
            "    print(c)\n"
            "    return 0\n",
            "10\n",
        )

    def test_int32_and_int_mixing_is_rejected(self):
        self._check(
            "def int main():\n"
            "    int32 a = 5\n"
            "    int b = 3\n"
            "    int c = a + b\n"
            "    return 0\n",
            expect_error="requires two operands of the same integer type",
        )

    def test_int64_and_int8_mixing_is_rejected(self):
        self._check(
            "def int main():\n"
            "    int64 a = 5\n"
            "    int8 b = 3\n"
            "    int64 c = a + b\n"
            "    return 0\n",
            expect_error="requires two operands of the same integer type",
        )

    def test_int64_unary_negate(self):
        self._check(
            "def int main():\n"
            "    int64 a = 5\n"
            "    int64 b = -a\n"
            "    return 0\n"
        )

    def test_int64_unary_complement(self):
        self._check(
            "def int main():\n"
            "    int64 a = 5\n"
            "    int64 b = ~a\n"
            "    return 0\n"
        )

    def test_int64_ordering_comparison(self):
        self._check(
            "def int main():\n"
            "    int64 a = 5\n"
            "    int64 b = 3\n"
            "    bool r = a > b\n"
            "    return 0\n"
        )

    def test_int64_equality(self):
        self._check(
            "def int main():\n"
            "    int64 a = 5\n"
            "    int64 b = 3\n"
            "    bool r = a == b\n"
            "    return 0\n"
        )

    def test_widening_cast_int_to_int64(self):
        self._check(
            "def int main():\n"
            "    int a = 5\n"
            "    int64 b = int64(a)\n"
            "    return 0\n"
        )

    def test_narrowing_cast_int64_to_int(self):
        self._check(
            "def int main():\n"
            "    int64 a = 5\n"
            "    int b = int(a)\n"
            "    return 0\n"
        )

    def test_cast_int64_to_int8(self):
        self._check(
            "def int main():\n"
            "    int64 a = 5\n"
            "    int8 b = int8(a)\n"
            "    return 0\n"
        )

    def test_int64_as_function_parameter_and_return(self):
        self._check(
            "def int64 identity(int64 x):\n"
            "    return x\n"
            "\n"
            "def int main():\n"
            "    return 0\n"
        )

    def test_int64_struct_field(self):
        self._check(
            "type S struct:\n"
            "    int64 v\n"
            "\n"
            "def int main():\n"
            "    S s = S(5)\n"
            "    return 0\n"
        )

    def test_int64_array_element_type(self):
        self._check("def int main():\n    [3]int64 arr\n    return 0\n")

    def test_int64_type_alias(self):
        self._check(
            "type MyLong = int64\n"
            "\n"
            "def int main():\n"
            "    MyLong x = 5\n"
            "    return 0\n"
        )


# ---------------------------------------------------------------------------
# int64
# ---------------------------------------------------------------------------

class TestInt64Storage:
    pytestmark = GCC_SKIP

    def test_int64_vardecl_and_return(self):
        assert_program_exit_code(
            "def int64 main():\n    int64 x = 5\n    return x\n",
            5,
        )

    def test_int64_large_literal_vardecl_and_return(self):
        assert_program_exit_code(
            "def int64 main():\n    int64 x = 9000000000\n    return x\n",
            9000000000 % 256,
        )

    def test_two_int64_locals_are_independently_stored(self):
        assert_program_exit_code(
            "def int64 main():\n"
            "    int64 a = 10\n"
            "    int64 b = 20\n"
            "    return a\n",
            10,
        )

    def test_int64_assign(self):
        assert_program_exit_code(
            "def int64 main():\n"
            "    int64 x = 1\n"
            "    x = 42\n"
            "    return x\n",
            42,
        )

    def test_int64_struct_field_read(self):
        assert_program_exit_code(
            "type S struct:\n"
            "    int64 v\n"
            "\n"
            "def int64 main():\n"
            "    S s = S(5)\n"
            "    return s.v\n",
            5,
        )

    def test_int64_struct_field_assign(self):
        assert_program_exit_code(
            "type S struct:\n"
            "    int64 v\n"
            "\n"
            "def int64 main():\n"
            "    S s = S(0)\n"
            "    s.v = 77\n"
            "    return s.v\n",
            77,
        )

    def test_int64_array_element_read(self):
        assert_program_exit_code(
            "def int64 main():\n"
            "    [3]int64 arr = [1, 2, 3]\n"
            "    return arr[2]\n",
            3,
        )

    def test_int64_array_index_assign(self):
        assert_program_exit_code(
            "def int64 main():\n"
            "    [3]int64 arr = [0, 0, 0]\n"
            "    arr[1] = 99\n"
            "    return arr[1]\n",
            99,
        )

    def test_int64_addition(self):
        assert_program_exit_code(
            "def int64 main():\n"
            "    int64 a = 5\n"
            "    int64 b = 3\n"
            "    return a + b\n",
            8,
        )

    def test_int64_addition_beyond_32bit_range(self):
        assert_program_exit_code(
            "def int main():\n"
            "    int64 a = 5000000000\n"
            "    int64 b = 3000000000\n"
            "    int64 c = a + b\n"
            "    int64 threshold = 4000000000\n"
            "    if c > threshold:\n"
            "        return 1\n"
            "    return 0\n",
            1,
        )

    def test_int64_subtraction(self):
        assert_program_exit_code(
            "def int64 main():\n"
            "    int64 a = 10\n"
            "    int64 b = 3\n"
            "    return a - b\n",
            7,
        )

    def test_int64_multiplication_beyond_32bit_range(self):
        assert_program_exit_code(
            "def int main():\n"
            "    int64 a = 100000\n"
            "    int64 b = 100000\n"
            "    int64 c = a * b\n"
            "    int64 threshold = 9000000000\n"
            "    if c > threshold:\n"
            "        return 1\n"
            "    return 0\n",
            1,
        )

    def test_int64_division_beyond_32bit_range(self):
        assert_program_exit_code(
            "def int main():\n"
            "    int64 a = 10000000000\n"
            "    int64 b = 3\n"
            "    int64 c = a / b\n"
            "    int64 threshold = 3000000000\n"
            "    if c > threshold:\n"
            "        return 1\n"
            "    return 0\n",
            1,
        )

    def test_int64_modulo(self):
        assert_program_exit_code(
            "def int64 main():\n"
            "    int64 a = 100\n"
            "    int64 b = 7\n"
            "    return a % b\n",
            100 % 7,
        )

    def test_int64_bitwise_and_beyond_32bit_range(self):
        assert_program_exit_code(
            "def int main():\n"
            "    int64 a = 5000000000\n"
            "    int64 b = 6000000000\n"
            "    int64 c = a & b\n"
            "    int64 threshold = 4000000000\n"
            "    if c > threshold:\n"
            "        return 1\n"
            "    return 0\n",
            1,
        )

    def test_int64_bitwise_or(self):
        assert_program_exit_code(
            "def int64 main():\n"
            "    int64 a = 12\n"
            "    int64 b = 10\n"
            "    return a | b\n",
            12 | 10,
        )

    def test_int64_bitwise_xor(self):
        assert_program_exit_code(
            "def int64 main():\n"
            "    int64 a = 12\n"
            "    int64 b = 10\n"
            "    return a ^ b\n",
            12 ^ 10,
        )

    def test_int64_shift_left(self):
        assert_program_exit_code(
            "def int64 main():\n"
            "    int64 a = 3\n"
            "    int64 b = 4\n"
            "    return a << b\n",
            3 << 4,
        )

    def test_int64_shift_right(self):
        assert_program_exit_code(
            "def int64 main():\n"
            "    int64 a = 48\n"
            "    int64 b = 2\n"
            "    return a >> b\n",
            48 >> 2,
        )

    def test_int64_negate_beyond_32bit_range(self):
        assert_program_exit_code(
            "def int main():\n"
            "    int64 a = 5000000000\n"
            "    int64 b = -a\n"
            "    int64 threshold = -4000000000\n"
            "    if b < threshold:\n"
            "        return 1\n"
            "    return 0\n",
            1,
        )

    def test_int64_complement(self):
        assert_program_exit_code(
            "def int64 main():\n"
            "    int64 a = 5\n"
            "    return ~a\n",
            256 + (~5),
        )

    def test_int64_ordering_comparison_beyond_32bit_range(self):
        assert_program_exit_code(
            "def int main():\n"
            "    int64 a = 5000000001\n"
            "    int64 b = 5000000000\n"
            "    if a > b:\n"
            "        return 1\n"
            "    return 0\n",
            1,
        )

    def test_int64_equality(self):
        assert_program_exit_code(
            "def int main():\n"
            "    int64 a = 5\n"
            "    int64 b = 5\n"
            "    if a == b:\n"
            "        return 1\n"
            "    return 0\n",
            1,
        )

    def test_widening_cast_int_to_int64(self):
        assert_program_exit_code(
            "def int64 main():\n"
            "    int a = 5\n"
            "    int64 b = int64(a)\n"
            "    int64 c = 3\n"
            "    return b + c\n",
            8,
        )

    def test_widening_cast_int8_to_int64(self):
        assert_program_exit_code(
            "def int main():\n"
            "    int8 a = -5\n"
            "    int64 b = int64(a)\n"
            "    int64 threshold = -3\n"
            "    if b < threshold:\n"
            "        return 1\n"
            "    return 0\n",
            1,
        )

    def test_widening_cast_uint8_to_int64(self):
        assert_program_exit_code(
            "def int main():\n"
            "    uint8 a = 200\n"
            "    int64 b = int64(a)\n"
            "    int64 threshold = 100\n"
            "    if b > threshold:\n"
            "        return 1\n"
            "    return 0\n",
            1,
        )

    def test_narrowing_cast_int64_to_int(self):
        raw = 5000000123 % (2 ** 32)
        assert_program_exit_code(
            "def int main():\n"
            "    int64 a = 5000000123\n"
            "    return int(a)\n",
            raw % 256,
        )

    def test_narrowing_cast_int64_to_int8(self):
        assert_program_exit_code(
            "def int8 main():\n"
            "    int64 a = 300\n"
            "    return int8(a)\n",
            44,
        )

    def test_cast_to_int8_truncates_immediately(self):
        assert_program_exit_code(
            "def int8 main():\n"
            "    int64 a = 300\n"
            "    int8 b = 7\n"
            "    return int8(a) / b\n",
            44 // 7,
        )

    def test_int64_function_parameter_and_return(self):
        assert_program_exit_code(
            "def int64 identity(int64 x):\n"
            "    return x\n"
            "\n"
            "def int64 main():\n"
            "    return identity(100)\n",
            100,
        )

    def test_int64_function_argument_beyond_32bit_range(self):
        assert_program_exit_code(
            "def int checkParam(int64 x):\n"
            "    int64 threshold = 4000000000\n"
            "    if x > threshold:\n"
            "        return 1\n"
            "    return 0\n"
            "\n"
            "def int main():\n"
            "    return checkParam(5000000000)\n",
            1,
        )

    def test_int64_return_value_beyond_32bit_range(self):
        assert_program_exit_code(
            "def int64 identity(int64 x):\n"
            "    return x\n"
            "\n"
            "def int main():\n"
            "    int64 threshold = 4000000000\n"
            "    int64 result = identity(5000000000)\n"
            "    if result > threshold:\n"
            "        return 1\n"
            "    return 0\n",
            1,
        )

    def test_multiple_int64_parameters_beyond_32bit_range(self):
        assert_program_exit_code(
            "def int64 addThem(int64 a, int64 b):\n"
            "    return a + b\n"
            "\n"
            "def int main():\n"
            "    int64 threshold = 8000000000\n"
            "    int64 result = addThem(5000000000, 4000000000)\n"
            "    if result > threshold:\n"
            "        return 1\n"
            "    return 0\n",
            1,
        )

    def test_struct_with_int64_field_as_parameter(self):
        assert_program_exit_code(
            "type Big struct:\n"
            "    int64 v\n"
            "\n"
            "def int checkBig(Big b):\n"
            "    int64 threshold = 4000000000\n"
            "    if b.v > threshold:\n"
            "        return 1\n"
            "    return 0\n"
            "\n"
            "def int main():\n"
            "    Big b = Big(5000000000)\n"
            "    return checkBig(b)\n",
            1,
        )

    def test_method_with_int64_argument_beyond_32bit_range(self):
        assert_program_exit_code(
            "type S struct:\n"
            "    int v\n"
            "    def int checkArg(s, int64 x):\n"
            "        int64 threshold = 4000000000\n"
            "        if x > threshold:\n"
            "            return 1\n"
            "        return 0\n"
            "\n"
            "def int main():\n"
            "    S s = S(1)\n"
            "    return s.checkArg(5000000000)\n",
            1,
        )

    def test_int64_array_equality_beyond_32bit_range(self):
        assert_program_exit_code(
            "def int main():\n"
            "    [2]int64 a = [5000000000, 6000000000]\n"
            "    [2]int64 b = [5000000000, 6000000000]\n"
            "    if a == b:\n"
            "        return 1\n"
            "    return 0\n",
            1,
        )

    def test_int64_array_zero_init(self):
        assert_program_exit_code(
            "def int main():\n"
            "    [3]int64 arr\n"
            "    int64 s = arr[0] + arr[1] + arr[2]\n"
            "    int64 zero = 0\n"
            "    if s == zero:\n"
            "        return 1\n"
            "    return 0\n",
            1,
        )

    def test_int64_array_copy_beyond_32bit_range(self):
        assert_program_exit_code(
            "def int main():\n"
            "    [2]int64 a = [5000000000, 6000000000]\n"
            "    [2]int64 b = [0, 0]\n"
            "    b = a\n"
            "    int64 threshold = 4000000000\n"
            "    if b[0] > threshold:\n"
            "        return 1\n"
            "    return 0\n",
            1,
        )

    def test_int64_array_parameter_beyond_32bit_range(self):
        assert_program_exit_code(
            "def int64 sumFirst([2]int64 arr):\n"
            "    return arr[0]\n"
            "\n"
            "def int main():\n"
            "    [2]int64 a = [5000000000, 6000000000]\n"
            "    int64 threshold = 4000000000\n"
            "    int64 result = sumFirst(a)\n"
            "    if result > threshold:\n"
            "        return 1\n"
            "    return 0\n",
            1,
        )

    def test_print_int64_now_supported(self):
        assert_program_stdout(
            "def int main():\n    int64 x = 5\n    print(x)\n    return 0\n",
            "5\n",
        )


# ---------------------------------------------------------------------------
# int64
# ---------------------------------------------------------------------------

class TestInt64Print:
    pytestmark = GCC_SKIP

    def test_print_positive_int64(self):
        assert_program_stdout(
            "def int main():\n    int64 x = 5\n    print(x)\n    return 0\n",
            "5\n",
        )

    def test_print_negative_int64(self):
        assert_program_stdout(
            "def int main():\n    int64 x = -5\n    print(x)\n    return 0\n",
            "-5\n",
        )

    def test_print_int64_beyond_32bit_range(self):
        assert_program_stdout(
            "def int main():\n    int64 x = 9000000000\n    print(x)\n    return 0\n",
            "9000000000\n",
        )

    def test_print_negative_int64_beyond_32bit_range(self):
        assert_program_stdout(
            "def int main():\n    int64 x = -9000000000\n    print(x)\n    return 0\n",
            "-9000000000\n",
        )

    def test_print_int64_max_boundary(self):
        assert_program_stdout(
            "def int main():\n    int64 x = 9223372036854775807\n    print(x)\n    return 0\n",
            "9223372036854775807\n",
        )

    def test_print_int64_min_boundary(self):
        assert_program_stdout(
            "def int main():\n    int64 x = -9223372036854775808\n    print(x)\n    return 0\n",
            "-9223372036854775808\n",
        )

    def test_print_int64_zero(self):
        assert_program_stdout(
            "def int main():\n    int64 x = 0\n    print(x)\n    return 0\n",
            "0\n",
        )

    def test_print_int64_non_variable_expression(self):
        assert_program_stdout(
            "def int main():\n"
            "    int64 a = 5000000000\n"
            "    int64 b = 3000000000\n"
            "    print(a + b)\n"
            "    return 0\n",
            "8000000000\n",
        )

    def test_print_int64_array_with_negative_and_large_elements(self):
        assert_program_stdout(
            "def int main():\n"
            "    [3]int64 arr = [-5000000000, 0, 5000000000]\n"
            "    print(arr)\n"
            "    return 0\n",
            "[3]int[-5000000000, 0, 5000000000]\n",
        )

    def test_print_struct_with_int64_field(self):
        assert_program_stdout(
            "type Big struct:\n"
            "    int64 v\n"
            "\n"
            "def int main():\n"
            "    Big b = Big(-9000000000)\n"
            "    print(b)\n"
            "    return 0\n",
            "Big(v: -9000000000)\n",
        )

    def test_print_slice_of_int64(self):
        assert_program_stdout(
            "def int main():\n"
            "    []int64 s = []int64[1, -2, 5000000000]\n"
            "    print(s)\n"
            "    return 0\n",
            "[]int[1, -2, 5000000000]\n",
        )

    def test_print_array_of_structs_with_int64_field(self):
        assert_program_stdout(
            "type Big struct:\n"
            "    int64 v\n"
            "\n"
            "def int main():\n"
            "    [2]Big arr = [Big(-5000000000), Big(5000000000)]\n"
            "    print(arr)\n"
            "    return 0\n",
            "[2]Big[Big(v: -5000000000), Big(v: 5000000000)]\n",
        )


class TestInt64RegressionsFoundDuringPrintStep:
    pytestmark = GCC_SKIP

    def test_slice_of_int64_read_without_print(self):
        assert_program_exit_code(
            "def int main():\n"
            "    []int64 s = []int64[1, -2, 5000000000]\n"
            "    int64 threshold = 4000000000\n"
            "    if s[2] > threshold:\n"
            "        return 1\n"
            "    return 0\n",
            1,
        )

    def test_slice_of_int64_negative_element_without_print(self):
        assert_program_exit_code(
            "def int main():\n"
            "    []int64 s = []int64[1, -2, 5000000000]\n"
            "    int64 threshold = -1\n"
            "    if s[1] < threshold:\n"
            "        return 1\n"
            "    return 0\n",
            1,
        )

    def test_append_int64_large_value(self):
        assert_program_exit_code(
            "def int main():\n"
            "    []int64 s = []int64[1, 2]\n"
            "    s = append(s, 5000000000)\n"
            "    int64 threshold = 4000000000\n"
            "    if s[2] > threshold:\n"
            "        return 1\n"
            "    return 0\n",
            1,
        )

    def test_index_assign_int64_large_value(self):
        assert_program_exit_code(
            "def int main():\n"
            "    [3]int64 arr = [0, 0, 0]\n"
            "    arr[1] = 5000000000\n"
            "    int64 threshold = 4000000000\n"
            "    if arr[1] > threshold:\n"
            "        return 1\n"
            "    return 0\n",
            1,
        )

    def test_field_assign_int64_large_value(self):
        assert_program_exit_code(
            "type S struct:\n"
            "    int64 v\n"
            "\n"
            "def int main():\n"
            "    S s = S(0)\n"
            "    s.v = 5000000000\n"
            "    int64 threshold = 4000000000\n"
            "    if s.v > threshold:\n"
            "        return 1\n"
            "    return 0\n",
            1,
        )

    def test_struct_literal_int64_field_large_value(self):
        assert_program_exit_code(
            "type Big struct:\n"
            "    int64 v\n"
            "\n"
            "def int main():\n"
            "    [2]Big arr = [Big(1), Big(5000000000)]\n"
            "    int64 threshold = 4000000000\n"
            "    if arr[1].v > threshold:\n"
            "        return 1\n"
            "    return 0\n",
            1,
        )

    def test_negative_literal_widening_uses_negq(self):
        assert_program_exit_code(
            "def int main():\n"
            "    int64 x = -5\n"
            "    int64 threshold = -3\n"
            "    if x < threshold:\n"
            "        return 1\n"
            "    return 0\n",
            1,
        )

    def test_large_negative_literal_widening(self):
        assert_program_exit_code(
            "def int main():\n"
            "    int64 x = -9000000000\n"
            "    int64 threshold = -8000000000\n"
            "    if x < threshold:\n"
            "        return 1\n"
            "    return 0\n",
            1,
        )


class TestTypedArrayLiterals:
    pytestmark = GCC_SKIP

    def test_typed_literal_as_vardecl_initializer(self):
        assert_exit_code(
            "    [3]int arr = [3]int[1, 2, 3]\n"
            "    return arr[0] + arr[1] + arr[2]",
            6,
        )

    def test_typed_literal_matching_vardecl_type_is_redundant_but_valid(self):
        assert_stdout(
            "    [3]int arr = [3]int[1, 2, 3]\n"
            "    print(arr)\n"
            "    return 0",
            "[3]int[1, 2, 3]\n",
        )

    def test_bare_typed_literal_statement(self):
        assert_exit_code(
            "    [3]int[1, 2, 3]\n"
            "    return 42",
            42,
        )

    def test_bare_statement_with_side_effecting_element(self):
        assert_program_stdout(
            "def int se():\n"
            "    print(99)\n"
            "    return 1\n"
            "\n"
            "def int main():\n"
            "    [2]int[se(), 2]\n"
            "    return 0\n",
            "99\n",
        )

    def test_single_element_typed_literal(self):
        assert_exit_code(
            "    [1]int arr = [1]int[7]\n"
            "    return arr[0]",
            7,
        )

    def test_untyped_single_element_literal_still_works(self):
        assert_exit_code(
            "    [1]int arr = [5]\n"
            "    return arr[0]",
            5,
        )

    def test_2d_typed_literal(self):
        assert_exit_code(
            "    [2][2]int arr = [2][2]int[[1, 2], [3, 4]]\n"
            "    return arr[0][0] + arr[1][1]",
            5,
        )

    def test_bare_2d_typed_literal_recurses_for_side_effects(self):
        assert_program_stdout(
            "def int se():\n"
            "    print(7)\n"
            "    return 1\n"
            "\n"
            "def int main():\n"
            "    [2][2]int[[se(), 2], [3, 4]]\n"
            "    return 0\n",
            "7\n",
        )

    def test_size_mismatch_is_rejected(self):
        assert_semantic_error(
            "    [3]int arr = [3]int[1, 2]\n"
            "    return 0",
            match="declares type .*size 3.*but has 2 element",
        )

    def test_element_type_mismatch_is_rejected(self):
        assert_semantic_error(
            "    [3]int arr = [3]int[1, true, 3]\n"
            "    return 0",
            match="declares element type int, but element 2 is bool",
        )

    def test_literal_type_mismatched_with_vardecl_is_rejected(self):
        assert_semantic_error(
            "    [3]bool arr = [3]int[1, 2, 3]\n"
            "    return 0",
            match="Cannot initialize",
        )

    def test_typed_literal_as_call_argument(self):
        assert_program_exit_code(
            "def int sum3([3]int arr):\n"
            "    return arr[0] + arr[1] + arr[2]\n"
            "\n"
            "def int main():\n"
            "    return sum3([3]int[1, 2, 3])\n",
            6,
        )

    def test_non_literal_array_element_in_bare_statement_not_supported(self):
        source = (
            "def int main():\n"
            "    [3]int arr = [1, 2, 3]\n"
            "    [1][3]int[arr]\n"
            "    return 0\n"
        )
        ast = _parse(source)
        analyze(ast)
        with pytest.raises(IRError, match="assign the literal to a variable first"):
            generate_asm(ast, target=ASM_TARGET)


class TestBoundsChecking:
    """Every array access is bounds-checked at runtime."""
    pytestmark = GCC_SKIP

    def test_index_too_large_aborts(self):
        assert_crashes_with_sigabrt(
            "    [3]int arr = [1, 2, 3]\n"
            "    int i = 5\n"
            "    return arr[i]"
        )

    def test_negative_index_aborts(self):
        assert_crashes_with_sigabrt(
            "    [3]int arr = [1, 2, 3]\n"
            "    int i = 0 - 1\n"
            "    return arr[i]"
        )

    def test_valid_boundary_index_does_not_abort(self):
        assert_exit_code(
            "    [3]int arr = [10, 20, 30]\n"
            "    int i = 2\n"
            "    return arr[i]",
            30,
        )

    def test_panic_message_survives_piped_output(self):
        result = compile_and_run(
            "def int main():\n"
            "    [3]int arr = [1, 2, 3]\n"
            "    int i = 5\n"
            "    return arr[i]\n"
        )
        assert result.returncode == -signal.SIGABRT
        assert "array index out of bounds" in result.stdout


# ---------------------------------------------------------------------------
# Size-based stack safety
# ---------------------------------------------------------------------------

class TestHeapAllocatedArrays:
    pytestmark = GCC_SKIP

    def test_exactly_at_threshold_stays_on_stack(self):
        n = 2048  # 2048 * 8 = 16384, exactly the threshold
        source = (
            f"def int main():\n"
            f"    [{n}]int arr\n"
            f"    arr[0] = 1\n"
            f"    return arr[0]\n"
        )
        ast = _parse(source)
        analyze(ast)
        mallocs = _heap_allocations(ast)
        assert not mallocs

    def test_just_over_threshold_is_heap_allocated(self):
        n = 2049  # 2049 * 8 = 16392, one int over the threshold
        source = (
            f"def int main():\n"
            f"    [{n}]int arr\n"
            f"    arr[0] = 1\n"
            f"    return arr[0]\n"
        )
        ast = _parse(source)
        analyze(ast)
        mallocs = _heap_allocations(ast)
        assert mallocs
        assert 16392 in mallocs

    def test_heap_allocated_local_basic_read_write(self):
        assert_exit_code(
            "    [10000]int big\n"
            "    big[0] = 42\n"
            "    big[9999] = 99\n"
            "    return big[0] + big[9999]",
            141,
        )

    def test_heap_allocated_local_with_literal_initializer(self):
        n = 4200  # 4200 * 4 = 16800 bytes, over the threshold
        elems = ', '.join(str(i % 10) for i in range(n))
        assert_program_exit_code(
            f"def int main():\n"
            f"    [{n}]int arr = [{elems}]\n"
            f"    return arr[0] + arr[1] + arr[9] + arr[{n - 1}]\n",
            19,  # 0 + 1 + 9 + (4199 % 10 == 9)
        )

    def test_heap_allocated_value_semantics(self):
        assert_exit_code(
            "    [10000]int a\n"
            "    [10000]int b\n"
            "    a[0] = 1\n"
            "    b[0] = 0\n"
            "    b = a\n"
            "    b[0] = 999\n"
            "    return a[0] == 1 and b[0] == 999",
            1,
            return_type="bool",
        )

    def test_heap_allocated_2d_array_value_semantics(self):
        assert_exit_code(
            "    [100][100]int grid\n"
            "    grid[0][0] = 1\n"
            "    grid[99][99] = 2\n"
            "    [100][100]int copy = grid\n"
            "    copy[0][0] = 999\n"
            "    return grid[0][0] == 1 and copy[0][0] == 999 and copy[99][99] == 2",
            1,
            return_type="bool",
        )

    def test_heap_allocated_array_as_function_parameter(self):
        assert_program_exit_code(
            "def int sum_first_and_last([10000]int arr):\n"
            "    return arr[0] + arr[9999]\n"
            "\n"
            "def int main():\n"
            "    [10000]int big\n"
            "    big[0] = 5\n"
            "    big[9999] = 7\n"
            "    return sum_first_and_last(big)\n",
            12,
        )

    def test_heap_allocated_parameter_value_semantics(self):
        assert_program_exit_code(
            "def int mutate([10000]int arr):\n"
            "    arr[0] = 999\n"
            "    return arr[0]\n"
            "\n"
            "def bool main():\n"
            "    [10000]int big\n"
            "    big[0] = 1\n"
            "    int result = mutate(big)\n"
            "    return result == 999 and big[0] == 1\n",
            1,
        )

    def test_heap_allocated_array_as_return_type(self):
        assert_program_exit_code(
            "def [10000]int make():\n"
            "    [10000]int r\n"
            "    r[0] = 42\n"
            "    r[9999] = 84\n"
            "    return r\n"
            "\n"
            "def int main():\n"
            "    [10000]int x = make()\n"
            "    return x[0] + x[9999]\n",
            126,
        )

    def test_heap_allocated_parameter_and_return_combined(self):
        assert_program_exit_code(
            "def [10000]int double_first([10000]int arr):\n"
            "    [10000]int result\n"
            "    result[0] = arr[0] * 2\n"
            "    return result\n"
            "\n"
            "def int main():\n"
            "    [10000]int a\n"
            "    a[0] = 21\n"
            "    [10000]int b = double_first(a)\n"
            "    return b[0]\n",
            42,
        )

    def test_multiple_heap_allocated_parameters(self):
        assert_program_exit_code(
            "def int sum_firsts([10000]int a, [10000]int b, [10000]int c):\n"
            "    return a[0] + b[0] + c[0]\n"
            "\n"
            "def int main():\n"
            "    [10000]int x\n"
            "    [10000]int y\n"
            "    [10000]int z\n"
            "    x[0] = 1\n"
            "    y[0] = 2\n"
            "    z[0] = 3\n"
            "    return sum_firsts(x, y, z)\n",
            6,
        )

    def test_mixed_parameter_types_with_heap_allocated_array(self):
        assert_program_exit_code(
            "def int mix(int a, str s, [3]int small, [10000]int big):\n"
            "    int slen_check = 0\n"
            "    if s == 'hi':\n"
            "        slen_check = 1\n"
            "    return a + slen_check + small[0] + big[0]\n"
            "\n"
            "def int main():\n"
            "    [10000]int huge\n"
            "    huge[0] = 100\n"
            "    [3]int small = [2, 0, 0]\n"
            "    return mix(1, 'hi', small, huge)\n",
            104,
        )


# ---------------------------------------------------------------------------
# Slices
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Array escape analysis
# ---------------------------------------------------------------------------

class TestArrayEscapeAnalysis:
    pytestmark = GCC_SKIP

    def test_small_sliced_parameter_returned_no_longer_corrupts(self):
        assert_program_stdout(
            "def []int sliceints([5]int arr):\n"
            "    []int a = arr[:]\n"
            "    return a\n"
            "\n"
            "def int helper(int x):\n"
            "    int a = x + 1\n"
            "    int b = a + 1\n"
            "    return a + b\n"
            "\n"
            "def int main():\n"
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    []int sl = sliceints(arr)\n"
            "    int junk = helper(1)\n"
            "    junk = helper(2)\n"
            "    junk = helper(3)\n"
            "    print(sl)\n"
            "    return 0\n",
            "[]int[1, 2, 3, 4, 5]\n",
        )

    def test_small_sliced_parameter_returned_is_actually_heap_allocated(self):
        source = (
            "def []int sliceints([5]int arr):\n"
            "    []int a = arr[:]\n"
            "    return a\n"
            "\n"
            "def int main():\n"
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    []int sl = sliceints(arr)\n"
            "    return sl[0]\n"
        )
        ast = _parse(source)
        analyze(ast)
        mallocs = _heap_allocations(ast)
        assert mallocs
        assert 40 in mallocs  # 5 ints * 8 bytes

    def test_local_array_sliced_but_not_returned_stays_on_the_stack(self):
        source = (
            "def int main():\n"
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    []int s = arr[0:2]\n"
            "    return s[0] + s[1]\n"
        )
        ast = _parse(source)
        analyze(ast)
        mallocs = _heap_allocations(ast)
        assert not mallocs

    def test_array_passed_by_value_not_sliced_stays_on_the_stack(self):
        source = (
            "def int helper([5]int a):\n"
            "    return a[0]\n"
            "\n"
            "def int main():\n"
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    return helper(arr)\n"
        )
        ast = _parse(source)
        analyze(ast)
        mallocs = _heap_allocations(ast)
        assert not mallocs

    def test_transitive_reslicing_chain_escapes_correctly(self):
        assert_program_exit_code(
            "def []int make():\n"
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    []int s1 = arr[0:3]\n"
            "    []int s2 = s1[0:2]\n"
            "    return s2\n"
            "\n"
            "def int helper(int x):\n"
            "    int a = x + 1\n"
            "    return a\n"
            "\n"
            "def int main():\n"
            "    []int r = make()\n"
            "    int junk = helper(1)\n"
            "    junk = helper(2)\n"
            "    junk = helper(3)\n"
            "    return r[0] + r[1]\n",
            3,
        )

    def test_append_chain_escapes_correctly(self):
        assert_program_exit_code(
            "def []int make():\n"
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    []int s1 = arr[0:2]\n"
            "    []int s2 = append(s1, 99)\n"
            "    return s2\n"
            "\n"
            "def int helper(int x):\n"
            "    int a = x + 1\n"
            "    return a\n"
            "\n"
            "def int main():\n"
            "    []int r = make()\n"
            "    int junk = helper(1)\n"
            "    junk = helper(2)\n"
            "    return r[0] + r[1] + r[2]\n",
            102,
        )

    def test_slicing_a_row_of_a_multi_dimensional_array_escapes_the_whole_array(self):
        assert_program_exit_code(
            "def []int getRow():\n"
            "    [2][3]int matrix = [[1, 2, 3], [4, 5, 6]]\n"
            "    return matrix[1][0:2]\n"
            "\n"
            "def int helper(int x):\n"
            "    int a = x + 1\n"
            "    return a\n"
            "\n"
            "def int main():\n"
            "    []int r = getRow()\n"
            "    int junk = helper(1)\n"
            "    junk = helper(2)\n"
            "    return r[0] + r[1]\n",
            9,
        )

    def test_slice_passed_to_a_non_escaping_parameter_stays_on_the_stack(self):
        source = (
            "def int sumFirstTwo([]int s):\n"
            "    return s[0] + s[1]\n"
            "\n"
            "def int main():\n"
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    []int s = arr[0:3]\n"
            "    return sumFirstTwo(s)\n"
        )
        ast = _parse(source)
        analyze(ast)
        mallocs = _heap_allocations(ast)
        assert not mallocs

    def test_slice_passed_to_a_parameter_that_escapes_is_promoted(self):
        source = (
            "def []int keep([]int s):\n"
            "    return s\n"
            "\n"
            "def int main():\n"
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    []int s = keep(arr[0:3])\n"
            "    return s[0]\n"
        )
        ast = _parse(source)
        analyze(ast)
        mallocs = _heap_allocations(ast)
        assert mallocs

    def test_array_of_slices_element_escapes_correctly(self):
        assert_program_stdout(
            "def [1][]int makeRows():\n"
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    [1][]int rows\n"
            "    rows[0] = arr[0:2]\n"
            "    return rows\n"
            "\n"
            "def int helper(int x):\n"
            "    int a = x + 1\n"
            "    int b = a + 1\n"
            "    return a + b\n"
            "\n"
            "def int main():\n"
            "    [1][]int r = makeRows()\n"
            "    int junk = helper(1)\n"
            "    junk = helper(2)\n"
            "    junk = helper(3)\n"
            "    print(r[0])\n"
            "    return 0\n",
            "[]int[1, 2]\n",
        )

    def test_array_of_slices_element_is_actually_heap_allocated(self):
        source = (
            "def [1][]int makeRows():\n"
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    [1][]int rows\n"
            "    rows[0] = arr[0:2]\n"
            "    return rows\n"
            "\n"
            "def int main():\n"
            "    [1][]int r = makeRows()\n"
            "    return r[0][0]\n"
        )
        ast = _parse(source)
        analyze(ast)
        mallocs = _heap_allocations(ast)
        assert mallocs

    def test_slice_of_slices_element_escapes_correctly(self):
        assert_exit_code(
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    [][]int rows = [][]int[[]int[]]\n"
            "    rows[0] = arr[0:2]\n"
            "    return len(rows[0])",
            2,
        )

    def test_reading_a_container_element_directly_escapes_it(self):
        assert_program_exit_code(
            "def []int makeRow():\n"
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    [1][]int rows\n"
            "    rows[0] = arr[0:2]\n"
            "    return rows[0]\n"
            "\n"
            "def int helper(int x):\n"
            "    int a = x + 1\n"
            "    return a\n"
            "\n"
            "def int main():\n"
            "    []int r = makeRow()\n"
            "    int junk = helper(1)\n"
            "    junk = helper(2)\n"
            "    return r[0] + r[1]\n",
            3,
        )

    def test_container_element_passed_to_a_non_escaping_parameter_stays_on_the_stack(self):
        source = (
            "def int sumFirstTwo([]int s):\n"
            "    return s[0] + s[1]\n"
            "\n"
            "def int main():\n"
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    [1][]int rows\n"
            "    rows[0] = arr[0:2]\n"
            "    []int e = rows[0]\n"
            "    return sumFirstTwo(e)\n"
        )
        ast = _parse(source)
        analyze(ast)
        mallocs = _heap_allocations(ast)
        assert not mallocs

    def test_container_element_never_read_does_not_trigger_promotion(self):
        source = (
            "def int main():\n"
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    [1][]int rows\n"
            "    rows[0] = arr[0:2]\n"
            "    return rows[0][0]\n"
        )
        ast = _parse(source)
        analyze(ast)
        mallocs = _heap_allocations(ast)
        assert not mallocs

    def test_slice_variable_backed_by_local_array_assigned_into_container_element(self):
        assert_program_exit_code(
            "def [1][]int makeRows():\n"
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    []int s = arr[0:3]\n"
            "    [1][]int rows\n"
            "    rows[0] = s\n"
            "    return rows\n"
            "\n"
            "def int helper(int x):\n"
            "    int a = x + 1\n"
            "    return a\n"
            "\n"
            "def int main():\n"
            "    [1][]int r = makeRows()\n"
            "    int junk = helper(1)\n"
            "    junk = helper(2)\n"
            "    return r[0][0] + r[0][1] + r[0][2]\n",
            6,
        )

    def test_append_into_a_container_element_escapes_correctly(self):
        assert_program_exit_code(
            "def []int makeRow():\n"
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    [1][]int rows\n"
            "    rows[0] = arr[0:2]\n"
            "    []int s = append(rows[0], 99)\n"
            "    return s\n"
            "\n"
            "def int helper(int x):\n"
            "    int a = x + 1\n"
            "    return a\n"
            "\n"
            "def int main():\n"
            "    []int r = makeRow()\n"
            "    int junk = helper(1)\n"
            "    junk = helper(2)\n"
            "    return r[0] + r[1] + r[2]\n",
            102,
        )

    def test_deeply_nested_container_escapes_correctly(self):
        assert_program_stdout(
            "def [1][1][]int makeMatrix():\n"
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    [1][1][]int matrix\n"
            "    matrix[0][0] = arr[0:2]\n"
            "    return matrix\n"
            "\n"
            "def int helper(int x):\n"
            "    int a = x + 1\n"
            "    int b = a + 1\n"
            "    return a + b\n"
            "\n"
            "def int main():\n"
            "    [1][1][]int m = makeMatrix()\n"
            "    int junk = helper(1)\n"
            "    junk = helper(2)\n"
            "    junk = helper(3)\n"
            "    print(m[0][0])\n"
            "    return 0\n",
            "[]int[1, 2]\n",
        )

    def test_deeply_nested_container_is_actually_heap_allocated(self):
        source = (
            "def [1][1][]int makeMatrix():\n"
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    [1][1][]int matrix\n"
            "    matrix[0][0] = arr[0:2]\n"
            "    return matrix\n"
            "\n"
            "def int main():\n"
            "    [1][1][]int m = makeMatrix()\n"
            "    return m[0][0][0]\n"
        )
        ast = _parse(source)
        analyze(ast)
        mallocs = _heap_allocations(ast)
        assert mallocs

    def test_chained_reslicing_with_no_intermediate_variable_escapes_correctly(self):
        assert_program_stdout(
            "def []int make():\n"
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    []int s1 = arr[0:3]\n"
            "    []int s2 = s1[0:3][0:2]\n"
            "    return s2\n"
            "\n"
            "def int helper(int x):\n"
            "    int a = x + 1\n"
            "    int b = a + 1\n"
            "    return a + b\n"
            "\n"
            "def int main():\n"
            "    []int s = make()\n"
            "    int junk = helper(1)\n"
            "    junk = helper(2)\n"
            "    junk = helper(3)\n"
            "    print(s)\n"
            "    return 0\n",
            "[]int[1, 2]\n",
        )

    def test_chained_reslicing_directly_off_an_array_escapes_correctly(self):
        source = (
            "def []int make():\n"
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    []int s = arr[0:3][0:2]\n"
            "    return s\n"
            "\n"
            "def int main():\n"
            "    []int s = make()\n"
            "    return s[0]\n"
        )
        ast = _parse(source)
        analyze(ast)
        mallocs = _heap_allocations(ast)
        assert mallocs

    def test_scalar_read_through_a_slice_element_does_not_escape(self):
        source = (
            "def int main():\n"
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    [1][]int rows\n"
            "    rows[0] = arr[0:2]\n"
            "    return rows[0][0]\n"
        )
        ast = _parse(source)
        analyze(ast)
        mallocs = _heap_allocations(ast)
        assert not mallocs


class TestSlices:
    pytestmark = GCC_SKIP

    def test_basic_slice_declare_and_index_read(self):
        assert_exit_code(
            "    [5]int arr = [10, 20, 30, 40, 50]\n"
            "    []int s = arr[1:4]\n"
            "    return s[0] + s[1] + s[2]",
            90,
        )

    def test_omitted_bounds(self):
        assert_exit_code(
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    []int a = arr[:]\n"
            "    []int b = arr[2:]\n"
            "    []int c = arr[:3]\n"
            "    return a[4] + b[0] + c[2]",
            11,
        )

    def test_slicing_a_slice(self):
        assert_exit_code(
            "    [6]int arr = [1, 2, 3, 4, 5, 6]\n"
            "    []int s = arr[1:5]\n"
            "    []int s2 = s[1:3]\n"
            "    return s2[0] + s2[1]",
            7,
        )

    def test_indexing_slice_with_variable_index(self):
        assert_exit_code(
            "    [5]int arr = [10, 20, 30, 40, 50]\n"
            "    []int s = arr[1:4]\n"
            "    int i = 1\n"
            "    return s[i]",
            30,
        )

    def test_slicing_outer_dimension_of_2d_array(self):
        assert_exit_code(
            "    [3][3]int matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]\n"
            "    [][3]int rows = matrix[0:2]\n"
            "    return rows[0][0] + rows[1][2]",
            7,
        )

    def test_whole_slice_assignment(self):
        assert_exit_code(
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    []int s1 = arr[0:3]\n"
            "    []int s2 = arr[2:5]\n"
            "    s2 = s1\n"
            "    return s2[0] + s2[1] + s2[2]",
            6,
        )

    def test_slice_write_mutates_underlying_array(self):
        assert_exit_code(
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    []int s = arr[1:4]\n"
            "    s[0] = 999\n"
            "    return arr[1] == 999 and s[0] == 999",
            1,
            return_type="bool",
        )

    def test_overlapping_slices_alias_each_others_writes(self):
        assert_exit_code(
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    []int s1 = arr[0:3]\n"
            "    []int s2 = arr[1:4]\n"
            "    s1[1] = 777\n"
            "    return s2[0] == 777",
            1,
            return_type="bool",
        )

    def test_slice_parameter(self):
        assert_program_exit_code(
            "def int first([]int s):\n"
            "    return s[0]\n"
            "\n"
            "def int main():\n"
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    []int s = arr[1:4]\n"
            "    return first(s)\n",
            2,
        )

    def test_slice_return(self):
        assert_program_exit_code(
            "def []int make():\n"
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    return arr[1:4]\n"
            "\n"
            "def int main():\n"
            "    []int s = make()\n"
            "    return s[0]\n",
            2,
        )


class TestSliceBoundsChecking:
    """Slice bounds have their own check and message."""
    pytestmark = GCC_SKIP

    def test_index_into_slice_out_of_bounds_aborts(self):
        assert_crashes_with_sigabrt(
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    []int s = arr[1:4]\n"
            "    int i = 10\n"
            "    return s[i]"
        )

    def test_low_greater_than_high_aborts(self):
        assert_crashes_with_sigabrt(
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    int lo = 3\n"
            "    int hi = 1\n"
            "    []int s = arr[lo:hi]\n"
            "    return s[0]"
        )

    def test_high_greater_than_length_aborts(self):
        assert_crashes_with_sigabrt(
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    int hi = 10\n"
            "    []int s = arr[0:hi]\n"
            "    return s[0]"
        )

    def test_negative_low_aborts(self):
        assert_crashes_with_sigabrt(
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    int lo = 0 - 1\n"
            "    []int s = arr[lo:3]\n"
            "    return s[0]"
        )

    def test_low_equals_high_equals_length_is_valid(self):
        assert_exit_code(
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    []int s = arr[5:5]\n"
            "    return 42",
            42,
        )

    def test_slice_bounds_panic_message(self):
        result = compile_and_run(
            "def int main():\n"
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    int hi = 10\n"
            "    []int s = arr[0:hi]\n"
            "    return s[0]\n"
        )
        assert result.returncode == -signal.SIGABRT
        assert "slice bounds out of range" in result.stdout


# ---------------------------------------------------------------------------
# Slice parameters and return values
# ---------------------------------------------------------------------------

class TestCapAwareSlicing:
    """Re-slicing may extend up to the capacity, as in Go."""
    pytestmark = GCC_SKIP

    def test_reslice_extends_beyond_len_but_within_cap(self):
        assert_exit_code(
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    []int s = arr[0:2]\n"
            "    []int t = s[0:5]\n"
            "    return t[0] + t[1] + t[2] + t[3] + t[4]",
            15,
        )

    def test_extending_beyond_cap_still_aborts(self):
        assert_crashes_with_sigabrt(
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    []int s = arr[0:2]\n"
            "    []int t = s[0:6]\n"
            "    return 0"
        )

    def test_reslice_starting_partway_through_inherits_remaining_capacity(self):
        assert_exit_code(
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    []int s = arr[1:2]\n"
            "    []int t = s[0:4]\n"
            "    return t[0] + t[1] + t[2] + t[3]",
            2 + 3 + 4 + 5,
        )

    def test_omitted_high_still_defaults_to_len_not_cap(self):
        assert_exit_code(
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    []int s = arr[0:2]\n"
            "    []int t = s[1:]\n"
            "    return len(t)",
            1,
        )


# ---------------------------------------------------------------------------
# `append`
# ---------------------------------------------------------------------------

class TestAppend:
    pytestmark = GCC_SKIP

    def test_basic_append(self):
        assert_program_stdout(
            "def int main():\n"
            "    []int x = []int[1, 2]\n"
            "    []int y = append(x, 3)\n"
            "    print(x)\n"
            "    print(y)\n"
            "    return 0\n",
            "[]int[1, 2]\n[]int[1, 2, 3]\n",
        )

    def test_append_reuses_backing_array_when_capacity_allows(self):
        assert_exit_code(
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    []int s = arr[0:2]\n"
            "    []int t = append(s, 99)\n"
            "    return arr[2]",
            99,
        )

    def test_append_past_capacity_reallocates_without_disturbing_original(self):
        assert_exit_code(
            "    []int x = []int[1, 2]\n"
            "    []int y = append(x, 3)\n"
            "    y[0] = 999\n"
            "    return x[0]",
            1,
        )

    def test_second_append_reuses_the_first_reallocations_spare_capacity(self):
        assert_exit_code(
            "    []int x = []int[1, 2]\n"
            "    []int y = append(x, 3)\n"
            "    y = append(y, 4)\n"
            "    return y[0] + y[1] + y[2] + y[3]",
            10,
        )

    def test_append_to_none(self):
        assert_exit_code(
            "    []int x = none\n"
            "    []int y = append(x, 42)\n"
            "    return y[0]",
            42,
        )

    def test_append_with_str_element(self):
        assert_stdout(
            "    []str x = []str['a', 'b']\n"
            "    []str y = append(x, 'c')\n"
            "    print(y)\n"
            "    return 0",
            "[]str['a', 'b', 'c']\n",
        )

    def test_append_with_array_element_type(self):
        assert_exit_code(
            "    [][2]int rows = [][2]int[[1, 2]]\n"
            "    [][2]int rows2 = append(rows, [2]int[3, 4])\n"
            "    return rows2[0][0] + rows2[0][1] + rows2[1][0] + rows2[1][1]",
            1 + 2 + 3 + 4,
        )

    def test_append_nested_slice_construction(self):
        assert_exit_code(
            "    [][]int rows = [][]int[[1, 2]]\n"
            "    [][]int rows2 = append(rows, [5, 6])\n"
            "    return rows2[0][0] + rows2[0][1] + rows2[1][0] + rows2[1][1]",
            1 + 2 + 5 + 6,
        )

    def test_many_appends_in_a_loop_retain_every_value(self):
        assert_program_exit_code(
            "def int main():\n"
            "    []int x = none\n"
            "    int i = 0\n"
            "    while i < 300:\n"
            "        x = append(x, i)\n"
            "        i = i + 1\n"
            "    int total = 0\n"
            "    int j = 0\n"
            "    while j < 300:\n"
            "        total = total + x[j]\n"
            "        j = j + 1\n"
            "    return total - 30000\n",
            (44850 - 30000) % 256,
        )

    def test_growth_policy_boundary_at_256(self):
        assert_program_exit_code(
            "def int main():\n"
            "    []int x = none\n"
            "    int i = 0\n"
            "    while i < 257:\n"
            "        x = append(x, 1)\n"
            "        i = i + 1\n"
            "    return len(x)\n",
            257 % 256,
        )

    def test_wrong_argument_count_is_rejected(self):
        assert_semantic_error(
            "    []int x = []int[1, 2]\n"
            "    []int y = append(x)\n"
            "    return 0",
            match="expects exactly 2 arguments",
        )

    def test_non_slice_first_argument_is_rejected(self):
        assert_semantic_error(
            "    []int y = append(5, 3)\n"
            "    return 0",
            match="requires a slice as its first argument",
        )

    def test_element_type_mismatch_is_rejected(self):
        assert_semantic_error(
            "    []int x = []int[1, 2]\n"
            "    []int y = append(x, true)\n"
            "    return 0",
        )

    def test_cannot_be_redefined_as_a_function(self):
        source = (
            "def int append([]int a, int b):\n"
            "    return 1\n"
            "\n"
            "def int main():\n"
            "    return 0\n"
        )
        with pytest.raises(SemanticError, match="builtin"):
            analyze(_parse(source))

    def test_first_argument_can_be_an_unnamed_slice_literal(self):
        assert_exit_code(
            "    []int s = append([]int[], 1)\n"
            "    return s[0]",
            1,
        )

    def test_first_argument_can_be_an_unnamed_non_empty_slice_literal(self):
        assert_exit_code(
            "    []int s = append([]int[1, 2], 3)\n"
            "    return s[0] + s[1] + s[2]",
            6,
        )

    def test_first_argument_can_be_an_unnamed_reslice(self):
        assert_exit_code(
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    []int s = append(arr[0:2], 99)\n"
            "    return s[0] + s[1] + s[2]",
            102,
        )

    def test_unnamed_reslice_still_reuses_backing_array_when_capacity_allows(self):
        assert_exit_code(
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    []int t = append(arr[0:2], 99)\n"
            "    return arr[2]",
            99,
        )

    def test_first_argument_can_be_an_unnamed_slice_returning_call(self):
        assert_program_exit_code(
            "def []int makeSlice():\n"
            "    return []int[10, 20]\n"
            "\n"
            "def int main():\n"
            "    []int s = append(makeSlice(), 30)\n"
            "    return s[0] + s[1] + s[2]\n",
            60,
        )

    def test_nested_append_calls(self):
        assert_exit_code(
            "    []int s = append(append([]int[1], 2), 3)\n"
            "    return s[0] + s[1] + s[2]",
            6,
        )

    def test_unnamed_slice_of_slices_literal(self):
        assert_exit_code(
            "    [][]int rows = append([][]int[], [1, 2])\n"
            "    return rows[0][0] + rows[0][1]",
            3,
        )

    def test_append_a_named_struct_value(self):
        assert_program_exit_code(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def int main():\n"
            "    Point p1\n"
            "    p1.x = 1\n"
            "    p1.y = 2\n"
            "    Point p2\n"
            "    p2.x = 3\n"
            "    p2.y = 4\n"
            "    []Point s = none\n"
            "    s = append(s, p1)\n"
            "    s = append(s, p2)\n"
            "    return s[0].x + s[1].y\n",
            5,
        )

    def test_append_a_struct_literal(self):
        assert_program_exit_code(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def int main():\n"
            "    []Point s = none\n"
            "    s = append(s, Point(1, 2))\n"
            "    s = append(s, Point(3, 4))\n"
            "    return s[0].x + s[1].y\n",
            5,
        )

    def test_append_result_indexed_directly(self):
        assert_program_exit_code(
            "def int main():\n"
            "    []int base = [1, 2]\n"
            "    return append(base, 3)[2]\n",
            3,
        )

    def test_bare_append_statement_does_not_mutate_original(self):
        assert_program_exit_code(
            "def int main():\n"
            "    []int base = [1, 2]\n"
            "    append(base, 3)\n"
            "    return len(base)\n",
            2,
        )

    def test_append_result_indexed_as_function_argument(self):
        assert_program_exit_code(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def int sumPoint(Point p):\n"
            "    return p.x + p.y\n"
            "\n"
            "def int main():\n"
            "    []Point s = [Point(1, 2)]\n"
            "    return sumPoint(append(s, Point(3, 4))[0])\n",
            3,
        )

    def test_append_result_indexed_as_vardecl_initializer(self):
        assert_program_exit_code(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def int main():\n"
            "    []Point s = [Point(1, 2)]\n"
            "    Point p = append(s, Point(3, 4))[0]\n"
            "    return p.x + p.y\n",
            3,
        )

    def test_append_result_indexed_as_equality_operand(self):
        assert_program_exit_code(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def int main():\n"
            "    []Point s = [Point(1, 2)]\n"
            "    Point other = Point(1, 2)\n"
            "    if append(s, Point(3, 4))[0] == other:\n"
            "        return 1\n"
            "    return 0\n",
            1,
        )

    def test_nested_append_as_slice_argument(self):
        assert_program_exit_code(
            "def int main():\n"
            "    []int s = none\n"
            "    s = append(append(s, 1), 2)\n"
            "    return s[0] + s[1]\n",
            3,
        )


class TestBareExpressionStatements:
    """Expressions used as discarded statements."""

    pytestmark = GCC_SKIP

    def test_bare_none_statement_is_a_no_op(self):
        assert_program_exit_code(
            "def int main():\n"
            "    none\n"
            "    return 42\n",
            42,
        )

    def test_bare_composite_returning_call_statement(self):
        assert_program_exit_code(
            "def [3]int makeArr():\n"
            "    return [1, 2, 3]\n"
            "\n"
            "def int main():\n"
            "    makeArr()\n"
            "    return 42\n",
            42,
        )


class TestSliceParametersAndReturns:
    pytestmark = GCC_SKIP

    def test_multiple_slice_parameters(self):
        assert_program_exit_code(
            "def int sum_two([]int a, []int b):\n"
            "    return a[0] + b[0]\n"
            "\n"
            "def int main():\n"
            "    [3]int x = [10, 20, 30]\n"
            "    [3]int y = [1, 2, 3]\n"
            "    []int sx = x[0:3]\n"
            "    []int sy = y[0:3]\n"
            "    return sum_two(sx, sy)\n",
            11,
        )

    def test_slice_interleaved_with_scalar_parameters(self):
        assert_program_exit_code(
            "def int f(int a, []int s, int b):\n"
            "    return a + s[0] + s[1] + b\n"
            "\n"
            "def int main():\n"
            "    [3]int arr = [10, 20, 30]\n"
            "    []int s = arr[0:3]\n"
            "    return f(1, s, 2)\n",
            33,
        )

    def test_writing_through_a_slice_parameter_mutates_callers_array(self):
        assert_program_exit_code(
            "def mutate([]int s):\n"
            "    s[0] = 42\n"
            "\n"
            "def int main():\n"
            "    [3]int arr = [1, 2, 3]\n"
            "    []int s = arr[0:3]\n"
            "    mutate(s)\n"
            "    return arr[0]\n",
            42,
        )

    def test_recursive_function_with_a_slice_parameter(self):
        assert_program_exit_code(
            "def int sum_slice([]int s, int i):\n"
            "    if i >= 5:\n"
            "        return 0\n"
            "    return s[i] + sum_slice(s, i + 1)\n"
            "\n"
            "def int main():\n"
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    []int s = arr[0:5]\n"
            "    return sum_slice(s, 0)\n",
            15,
        )

    def test_forwarding_a_slice_returning_calls_result(self):
        assert_program_exit_code(
            "def []int inner():\n"
            "    [3]int arr = [7, 8, 9]\n"
            "    return arr[0:3]\n"
            "\n"
            "def []int outer():\n"
            "    return inner()\n"
            "\n"
            "def int main():\n"
            "    []int s = outer()\n"
            "    return s[0] + s[1] + s[2]\n",
            24,
        )

    def test_printing_a_slice_parameter(self):
        assert_program_stdout(
            "def show([]int s):\n"
            "    print(s)\n"
            "\n"
            "def int main():\n"
            "    [3]int arr = [4, 5, 6]\n"
            "    []int s = arr[0:3]\n"
            "    show(s)\n"
            "    return 0\n",
            "[]int[4, 5, 6]\n",
        )

    def test_reslicing_a_received_slice_parameter(self):
        assert_program_exit_code(
            "def int second_half([]int s):\n"
            "    []int half = s[2:4]\n"
            "    return half[0] + half[1]\n"
            "\n"
            "def int main():\n"
            "    [4]int arr = [10, 20, 30, 40]\n"
            "    []int s = arr[0:4]\n"
            "    return second_half(s)\n",
            70,
        )

    def test_returning_none_from_a_slice_returning_function(self):
        assert_program_exit_code(
            "def []int maybe(bool give):\n"
            "    if give:\n"
            "        [2]int arr = [1, 2]\n"
            "        return arr[0:2]\n"
            "    return none\n"
            "\n"
            "def bool main():\n"
            "    []int s = maybe(false)\n"
            "    return s == none\n",
            1,
        )

    def test_array_parameter_and_slice_parameter_together(self):
        assert_program_exit_code(
            "def int combo([3]int arr, []int s):\n"
            "    return arr[0] + s[0]\n"
            "\n"
            "def int main():\n"
            "    [3]int a = [100, 200, 300]\n"
            "    [2]int b = [5, 6]\n"
            "    []int s = b[0:2]\n"
            "    return combo(a, s)\n",
            105,
        )

    def test_slice_parameter_with_heap_allocated_array_parameter(self):
        assert_program_exit_code(
            "def int combo([5000]int big, []int s):\n"
            "    return big[0] + big[4999] + s[0]\n"
            "\n"
            "def int main():\n"
            "    [5000]int huge\n"
            "    huge[0] = 1\n"
            "    huge[4999] = 2\n"
            "    [1]int small = [100]\n"
            "    []int s = small[0:1]\n"
            "    return combo(huge, s)\n",
            103,
        )

    def test_mix_of_real_slice_and_none_arguments(self):
        assert_program_exit_code(
            "def bool f([]int a, []int b):\n"
            "    return a != none and b == none\n"
            "\n"
            "def bool main():\n"
            "    [3]int arr = [1, 2, 3]\n"
            "    []int s = arr[0:3]\n"
            "    return f(s, none)\n",
            1,
        )

    def test_exactly_six_slots_from_two_slice_parameters(self):
        assert_program_exit_code(
            "def int f([]int a, []int b):\n"
            "    return a[0] + b[0]\n"
            "\n"
            "def int main():\n"
            "    [1]int x = [1]\n"
            "    [1]int y = [2]\n"
            "    []int sx = x[0:1]\n"
            "    []int sy = y[0:1]\n"
            "    return f(sx, sy)\n",
            3,
        )

    def test_seven_slots_from_two_slices_and_a_scalar_works_via_the_stack(self):
        assert_program_exit_code(
            "def int f([]int a, []int b, int c):\n"
            "    return a[0] + b[0] + c\n"
            "\n"
            "def int main():\n"
            "    [1]int x = [1]\n"
            "    [1]int y = [2]\n"
            "    []int sx = x[0:1]\n"
            "    []int sy = y[0:1]\n"
            "    return f(sx, sy, 3)\n",
            6,
        )

    def test_a_slice_argument_straddling_the_register_stack_boundary(self):
        assert_program_exit_code(
            "def int f(int a, int b, int c, int d, int e, []int s):\n"
            "    return a + b + c + d + e + s[0] + len(s)\n"
            "\n"
            "def int main():\n"
            "    [3]int arr = [10, 20, 30]\n"
            "    []int s = arr[0:3]\n"
            "    return f(1, 2, 3, 4, 5, s)\n",
            1 + 2 + 3 + 4 + 5 + 10 + 3,
        )

    def test_one_slice_and_three_scalars_are_exactly_six_slots(self):
        assert_program_exit_code(
            "def int f(int a, []int s, int b, int c):\n"
            "    return a + s[0] + b + c\n"
            "\n"
            "def int main():\n"
            "    [1]int x = [10]\n"
            "    []int sx = x[0:1]\n"
            "    return f(1, sx, 2, 3)\n",
            16,
        )

    def test_reslice_as_call_argument(self):
        assert_program_exit_code(
            "def int sum3([]int s):\n"
            "    return s[0] + s[1] + s[2]\n"
            "\n"
            "def int main():\n"
            "    [3]int arr = [1, 2, 3]\n"
            "    return sum3(arr[0:3])\n",
            6,
        )

    def test_slice_returning_call_as_argument(self):
        assert_program_exit_code(
            "def []int makeSlice():\n"
            "    return []int[7, 8, 9]\n"
            "\n"
            "def int sum3([]int s):\n"
            "    return s[0] + s[1] + s[2]\n"
            "\n"
            "def int main():\n"
            "    return sum3(makeSlice())\n",
            24,
        )

    def test_two_unnamed_slices_alive_in_the_same_call(self):
        assert_program_exit_code(
            "def int addPairs([]int a, []int b):\n"
            "    return a[0] + a[1] + b[0] + b[1]\n"
            "\n"
            "def int main():\n"
            "    return addPairs([]int[1, 2], []int[3, 4])\n",
            10,
        )

    def test_nested_unnamed_slice_materialization(self):
        assert_program_exit_code(
            "def []int identity([]int s):\n"
            "    return s\n"
            "\n"
            "def int sum3([]int s):\n"
            "    return s[0] + s[1] + s[2]\n"
            "\n"
            "def int main():\n"
            "    return sum3(identity([]int[1, 2, 3]))\n",
            6,
        )

    def test_slice_typed_field_as_argument(self):
        assert_program_exit_code(
            "type Holder struct:\n"
            "    []int xs\n"
            "\n"
            "def int sum3([]int s):\n"
            "    return s[0] + s[1] + s[2]\n"
            "\n"
            "def int main():\n"
            "    Holder h = Holder([]int[1, 2, 3])\n"
            "    return sum3(h.xs)\n",
            6,
        )

    def test_slice_typed_index_as_argument(self):
        assert_program_exit_code(
            "def int sum2([]int s):\n"
            "    return s[0] + s[1]\n"
            "\n"
            "def int main():\n"
            "    [2][]int rows = [[]int[1, 2], []int[3, 4]]\n"
            "    return sum2(rows[0]) + sum2(rows[1])\n",
            10,
        )

    def test_append_result_as_argument(self):
        assert_program_exit_code(
            "def int sum3([]int s):\n"
            "    return s[0] + s[1] + s[2]\n"
            "\n"
            "def int main():\n"
            "    []int base = []int[1, 2]\n"
            "    return sum3(append(base, 3))\n",
            6,
        )


# ---------------------------------------------------------------------------
# Indexing unnamed slices
# ---------------------------------------------------------------------------

class TestIndexingUnnamedSlices:
    pytestmark = GCC_SKIP

    def test_indexing_a_sliced_array_directly(self):
        assert_exit_code(
            "    [3]int arr = [1, 2, 3]\n"
            "    return arr[:][0]",
            1,
        )

    def test_indexing_a_sliced_2d_array_directly(self):
        assert_exit_code(
            "    [2][2]int arr = [[1, 2], [3, 4]]\n"
            "    return arr[:][0][0]",
            1,
        )

    def test_deep_chain_reslicing_then_indexing(self):
        assert_exit_code(
            "    [5]int arr = [10, 20, 30, 40, 50]\n"
            "    return arr[:][1:3][0]",
            20,
        )

    def test_named_slice_resliced_unnamed_then_indexed(self):
        assert_exit_code(
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    []int s = arr[1:4]\n"
            "    return s[:][0:2][1]",
            3,
        )

    def test_slice_returning_call_as_unnamed_base(self):
        assert_program_exit_code(
            "def []int make():\n"
            "    [3]int arr = [7, 8, 9]\n"
            "    return arr[0:3]\n"
            "\n"
            "def int main():\n"
            "    return make()[0]\n",
            7,
        )

    def test_slice_returning_call_chained_deeper(self):
        assert_program_exit_code(
            "def []int make():\n"
            "    [3]int arr = [7, 8, 9]\n"
            "    return arr[0:3]\n"
            "\n"
            "def int main():\n"
            "    return make()[0:2][1]\n",
            8,
        )

    def test_writing_through_an_unnamed_sliced_index(self):
        assert_exit_code(
            "    [3]int arr = [1, 2, 3]\n"
            "    arr[:][0] = 99\n"
            "    return arr[0]",
            99,
        )

    def test_very_deep_nesting_of_unnamed_slices(self):
        assert_stdout(
            "    [4]int arr = [11, 22, 33, 44]\n"
            "    print(arr[:][:][:][0])\n"
            "    return 0",
            "11\n",
        )

    def test_indexing_out_of_bounds_on_an_unnamed_slice_aborts(self):
        assert_crashes_with_sigabrt(
            "    [3]int arr = [1, 2, 3]\n"
            "    return arr[:][10]"
        )

    def test_two_independent_materializations_in_one_expression(self):
        assert_stdout(
            "    [3]int arr1 = [10, 20, 30]\n"
            "    [3]int arr2 = [1, 2, 3]\n"
            "    print(arr1[:][0] + arr2[:][0])\n"
            "    return 0",
            "11\n",
        )

    def test_materialization_inside_a_slice_bound_expression(self):
        assert_stdout(
            "    [5]int arr = [100, 200, 300, 400, 500]\n"
            "    [1]int idx_holder = [2]\n"
            "    print(arr[idx_holder[:][0]:5][0])\n"
            "    return 0",
            "300\n",
        )


# ---------------------------------------------------------------------------
# Printing arrays and slices
# ---------------------------------------------------------------------------

class TestPrintArraysAndSlices:
    pytestmark = GCC_SKIP

    def test_print_array_of_int(self):
        assert_stdout(
            "    [3]int arr = [1, 2, 3]\n"
            "    print(arr)\n"
            "    return 0",
            "[3]int[1, 2, 3]\n",
        )

    def test_print_slice_of_int(self):
        assert_stdout(
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    []int s = arr[1:4]\n"
            "    print(s)\n"
            "    return 0",
            "[]int[2, 3, 4]\n",
        )

    def test_nested_2d_array_prints_its_own_name_at_every_level(self):
        assert_stdout(
            "    [2][3]int matrix = [[1, 2, 3], [4, 5, 6]]\n"
            "    print(matrix)\n"
            "    return 0",
            "[2][3]int[[3]int[1, 2, 3], [3]int[4, 5, 6]]\n",
        )

    def test_str_elements_are_quoted(self):
        assert_stdout(
            "    [3]str names = ['alice', 'bob', 'carol']\n"
            "    print(names)\n"
            "    return 0",
            "[3]str['alice', 'bob', 'carol']\n",
        )

    def test_bool_elements(self):
        assert_stdout(
            "    [3]bool flags = [true, false, true]\n"
            "    print(flags)\n"
            "    return 0",
            "[3]bool[true, false, true]\n",
        )

    def test_empty_slice_prints_with_no_trailing_comma(self):
        assert_stdout(
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    []int s = arr[5:5]\n"
            "    print(s)\n"
            "    return 0",
            "[]int[]\n",
        )

    def test_slice_of_2d_array_outer_dimension(self):
        assert_stdout(
            "    [3][3]int matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]\n"
            "    [][3]int rows = matrix[0:2]\n"
            "    print(rows)\n"
            "    return 0",
            "[][3]int[[3]int[1, 2, 3], [3]int[4, 5, 6]]\n",
        )

    def test_printing_a_slice_of_a_slice(self):
        assert_stdout(
            "    [6]int arr = [1, 2, 3, 4, 5, 6]\n"
            "    []int s = arr[1:5]\n"
            "    []int s2 = s[1:3]\n"
            "    print(s2)\n"
            "    return 0",
            "[]int[3, 4]\n",
        )

    def test_multiple_prints_each_get_exactly_one_newline(self):
        assert_stdout(
            "    [2]int a = [1, 2]\n"
            "    [2]int b = [3, 4]\n"
            "    print(a)\n"
            "    print(b)\n"
            "    return 0",
            "[2]int[1, 2]\n[2]int[3, 4]\n",

        )


    def test_array_literal_as_direct_print_argument(self):
        assert_stdout(
            "    print([1, 2, 3])\n"
            "    return 0",
            "[3]int[1, 2, 3]\n",
        )

    def test_typed_array_literal_as_direct_print_argument(self):
        assert_stdout(
            "    print([3]int[1, 2, 3])\n"
            "    return 0",
            "[3]int[1, 2, 3]\n",
        )

    def test_array_returning_call_as_direct_print_argument(self):
        assert_program_stdout(
            "def [3]int makeArr():\n"
            "    return [7, 8, 9]\n"
            "\n"
            "def int main():\n"
            "    print(makeArr())\n"
            "    return 0\n",
            "[3]int[7, 8, 9]\n",
        )

    def test_nested_array_literal_as_direct_print_argument(self):
        assert_stdout(
            "    print([[1, 2], [3, 4]])\n"
            "    return 0",
            "[2][2]int[[2]int[1, 2], [2]int[3, 4]]\n",
        )

    def test_array_of_structs_literal_as_direct_print_argument(self):
        assert_program_stdout(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def int main():\n"
            "    print([Point(1, 2), Point(3, 4)])\n"
            "    return 0\n",
            "[2]Point[Point(x: 1, y: 2), Point(x: 3, y: 4)]\n",
        )

    def test_two_unnamed_array_literals_printed_in_sequence(self):
        assert_stdout(
            "    print([1, 2])\n"
            "    print([3, 4, 5])\n"
            "    return 0",
            "[2]int[1, 2]\n[3]int[3, 4, 5]\n",
        )

    def test_slice_literal_as_direct_print_argument(self):
        assert_stdout(
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    print(arr[1:3])\n"
            "    return 0",
            "[]int[2, 3]\n",
        )

    def test_typed_slice_literal_as_direct_print_argument(self):
        assert_stdout(
            "    print([]int[1, 2, 3])\n"
            "    return 0",
            "[]int[1, 2, 3]\n",
        )

    def test_slice_returning_call_as_direct_print_argument(self):
        assert_program_stdout(
            "def []int makeSlice():\n"
            "    return []int[7, 8, 9]\n"
            "\n"
            "def int main():\n"
            "    print(makeSlice())\n"
            "    return 0\n",
            "[]int[7, 8, 9]\n",
        )

    def test_nested_unnamed_slice_as_direct_print_argument(self):
        assert_stdout(
            "    print([][]int[[]int[1, 2], []int[3, 4]])\n"
            "    return 0",
            "[][]int[[]int[1, 2], []int[3, 4]]\n",
        )

    def test_slice_of_structs_literal_as_direct_print_argument(self):
        assert_program_stdout(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def int main():\n"
            "    print([]Point[Point(1, 2), Point(3, 4)])\n"
            "    return 0\n",
            "[]Point[Point(x: 1, y: 2), Point(x: 3, y: 4)]\n",
        )

    def test_two_unnamed_slices_printed_in_sequence(self):
        assert_stdout(
            "    print([]int[1, 2])\n"
            "    print([]int[3, 4, 5])\n"
            "    return 0",
            "[]int[1, 2]\n[]int[3, 4, 5]\n",
        )

    def test_mixed_unnamed_array_and_slice_prints(self):
        assert_stdout(
            "    print([1, 2])\n"
            "    print([]int[3, 4])\n"
            "    return 0",
            "[2]int[1, 2]\n[]int[3, 4]\n",
        )

    def test_nested_unnamed_slice_materialization_in_print(self):
        assert_program_stdout(
            "def []int identity([]int s):\n"
            "    return s\n"
            "\n"
            "def int main():\n"
            "    print(identity([]int[1, 2, 3]))\n"
            "    return 0\n",
            "[]int[1, 2, 3]\n",
        )

    def test_slice_of_a_slice_typed_field_as_direct_print_argument(self):
        assert_program_stdout(
            "type Holder struct:\n"
            "    []int xs\n"
            "\n"
            "def int main():\n"
            "    Holder h = Holder([]int[10, 20, 30, 40])\n"
            "    print(h.xs[1:3])\n"
            "    return 0\n",
            "[]int[20, 30]\n",
        )


# ---------------------------------------------------------------------------
# `none`
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Slice literals
# ---------------------------------------------------------------------------

class TestSliceLiterals:
    pytestmark = GCC_SKIP

    def test_typed_slice_literal_as_vardecl_initializer(self):
        assert_exit_code(
            "    []int slice = []int[1, 2, 3]\n"
            "    return slice[0] + slice[1] + slice[2]",
            6,
        )

    def test_typed_slice_literal_in_assign(self):
        assert_exit_code(
            "    []int slice = []int[1, 2, 3]\n"
            "    slice = []int[4, 5, 6]\n"
            "    return slice[0] + slice[1] + slice[2]",
            15,
        )

    def test_empty_typed_slice_literal_in_assign(self):
        assert_program_stdout(
            "def int main():\n"
            "    []int slice = []int[1, 2, 3]\n"
            "    slice = []int[]\n"
            "    print(slice)\n"
            "    return 0\n",
            "[]int[]\n",
        )

    def test_untyped_literal_as_slice_vardecl_initializer(self):
        assert_exit_code(
            "    []int slice = [1, 2, 3]\n"
            "    return slice[0] + slice[1] + slice[2]",
            6,
        )

    def test_untyped_literal_as_slice_assign_value(self):
        assert_exit_code(
            "    []int s = []int[1, 2, 3]\n"
            "    s = [4, 5, 6]\n"
            "    return s[0] + s[1] + s[2]",
            15,
        )

    def test_named_array_is_not_auto_compatible_with_slice_target(self):
        assert_semantic_error(
            "    [3]int arr = [1, 2, 3]\n"
            "    []int s = arr\n"
            "    return 0",
            match="Cannot initialize",
        )

    def test_empty_slice_literal_is_not_nil(self):
        assert_exit_code(
            "    []int s = []int[1, 2, 3]\n"
            "    s = []int[]\n"
            "    return s == none",
            0,
            return_type="bool",
        )

    def test_mutation_through_a_slice_literal_backed_slice(self):
        assert_exit_code(
            "    []int s = []int[1, 2, 3]\n"
            "    s[0] = 99\n"
            "    return s[0]",
            99,
        )

    def test_slice_length_matches_literal_element_count(self):
        assert_exit_code(
            "    []int s = []int[7, 8, 9, 10]\n"
            "    return s[3]",
            10,
        )

    def test_multi_dimensional_slice_literal(self):
        assert_exit_code(
            "    [][2]int rows = [][2]int[[1, 2], [3, 4]]\n"
            "    return rows[0][0] + rows[1][1]",
            5,
        )

    def test_slice_literal_element_type_mismatch_is_rejected(self):
        assert_semantic_error(
            "    []bool s = []bool[1, 2, 3]\n"
            "    return 0",
            match="declares element type bool, but element 1 is int",
        )

    def test_untyped_literal_element_type_mismatch_against_slice_target_is_rejected(self):
        assert_semantic_error(
            "    []bool s = [1, 2, 3]\n"
            "    return 0",
            match="must all be bool",
        )

    def test_bare_slice_literal_statement_with_side_effecting_element(self):
        assert_program_stdout(
            "def int se():\n"
            "    print(42)\n"
            "    return 1\n"
            "\n"
            "def int main():\n"
            "    []int[se(), 2, 3]\n"
            "    return 0\n",
            "42\n",
        )

    def test_bare_slice_of_existing_array_statement(self):
        assert_exit_code(
            "    [3]int arr = [1, 2, 3]\n"
            "    arr[:]\n"
            "    return 0",
            0,
        )

    def test_bare_out_of_range_slice_statement_aborts(self):
        assert_crashes_with_sigabrt(
            "    [3]int arr = [1, 2, 3]\n"
            "    arr[0:10]\n"
            "    return 0"
        )

    def test_slice_literal_as_call_argument(self):
        assert_program_exit_code(
            "def int f([]int s):\n"
            "    return s[0]\n"
            "\n"
            "def int main():\n"
            "    return f([]int[1, 2, 3])\n",
            1,
        )


# ---------------------------------------------------------------------------
# Nested slices
# ---------------------------------------------------------------------------

class TestNestedSlices:
    pytestmark = GCC_SKIP

    def test_typed_nested_slice_literal(self):
        assert_program_exit_code(
            "def int main():\n"
            "    [][]int rows = [][]int[[1, 2], [3, 4]]\n"
            "    []int r0 = rows[0]\n"
            "    []int r1 = rows[1]\n"
            "    return r0[0] + r0[1] + r1[0] + r1[1]\n",
            10,
        )

    def test_untyped_nested_slice_literal(self):
        assert_program_exit_code(
            "def int main():\n"
            "    [][]int rows = [[1, 2], [3, 4]]\n"
            "    []int r0 = rows[0]\n"
            "    []int r1 = rows[1]\n"
            "    return r0[0] + r0[1] + r1[0] + r1[1]\n",
            10,
        )

    def test_deeply_nested_slice_of_slice_of_slice(self):
        assert_program_exit_code(
            "def int main():\n"
            "    [][][]int x = [][][]int[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]\n"
            "    [][]int mid = x[0]\n"
            "    []int inner = mid[1]\n"
            "    return inner[0] + inner[1]\n",
            7,
        )

    def test_fixed_size_array_of_slices_construction(self):
        assert_exit_code(
            "    [2][]int rows = [2][]int[[1, 2], [3, 4]]\n"
            "    return rows[0][0] + rows[0][1] + rows[1][0] + rows[1][1]",
            10,
        )

    def test_array_of_slices_literal_via_reassignment_regression(self):
        assert_exit_code(
            "    [2][]int rows = [2][]int[[5, 6], [7, 8]]\n"
            "    rows = [2][]int[[1, 2], [3, 4]]\n"
            "    return rows[0][0] + rows[0][1] + rows[1][0] + rows[1][1]",
            10,
        )

    def test_empty_nested_slice_literal_is_not_nil(self):
        assert_exit_code(
            "    [][]int x = [][]int[[]int[], []int[]]\n"
            "    return x[0] == none",
            0,
            return_type="bool",
        )

    def test_printing_array_of_slices(self):
        assert_stdout(
            "    [2][]int rows = [2][]int[[1, 2], [3, 4]]\n"
            "    print(rows)\n"
            "    return 0",
            "[2][]int[[]int[1, 2], []int[3, 4]]\n",
        )


    def test_return_bare_slice_literal_from_slice_returning_function(self):
        assert_program_exit_code(
            "def []int makeSlice():\n"
            "    return [7, 8, 9]\n"
            "\n"
            "def int main():\n"
            "    []int s = makeSlice()\n"
            "    return s[0] + s[1] + s[2]\n",
            24,
        )

    def test_return_bare_slice_literal_result_used_via_index_assign(self):
        assert_program_exit_code(
            "def []int makeSlice():\n"
            "    return [7, 8, 9]\n"
            "\n"
            "def int main():\n"
            "    [2][]int rows\n"
            "    rows[0] = [1, 2]\n"
            "    rows[1] = makeSlice()\n"
            "    return rows[0][0] + rows[0][1] + rows[1][0] + rows[1][1] + rows[1][2]\n",
            27,
        )


class TestChainedSliceIndexing:
    """`rows[0][1]` with no intermediate variable."""
    pytestmark = GCC_SKIP

    def test_chained_index_into_array_of_slices(self):
        assert_exit_code(
            "    [][]int rows = [][]int[[1, 2], [3, 4]]\n"
            "    return rows[0][0] + rows[0][1] + rows[1][0] + rows[1][1]",
            10,
        )

    def test_triple_chained_index(self):
        assert_exit_code(
            "    [][][]int x = [][][]int[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]\n"
            "    return x[0][1][0] + x[1][0][1]",
            9,
        )


class TestArrayOfSlicesCopying:
    pytestmark = GCC_SKIP

    def test_variable_to_variable_copy(self):
        assert_exit_code(
            "    [2][]int x = [2][]int[[1, 2], [3, 4]]\n"
            "    [2][]int y = x\n"
            "    return y[0][0] + y[0][1] + y[1][0] + y[1][1]",
            10,
        )

    def test_as_function_parameter(self):
        assert_program_exit_code(
            "def int sumIt([2][]int rows):\n"
            "    return rows[0][0] + rows[0][1] + rows[1][0] + rows[1][1]\n"
            "\n"
            "def int main():\n"
            "    [2][]int x = [2][]int[[1, 2], [3, 4]]\n"
            "    return sumIt(x)\n",
            10,
        )

    def test_returned_by_value(self):
        assert_program_exit_code(
            "def [2][]int makeRows():\n"
            "    [2][]int x = [2][]int[[5, 6], [7, 8]]\n"
            "    return x\n"
            "\n"
            "def int main():\n"
            "    [2][]int y = makeRows()\n"
            "    return y[0][0] + y[0][1] + y[1][0] + y[1][1]\n",
            26,
        )

    def test_shallow_copy_semantics(self):
        assert_exit_code(
            "    [][]int x = [][]int[[1, 2], [3, 4]]\n"
            "    [][]int y = x\n"
            "    y[0][0] = 99\n"
            "    return x[0][0]",
            99,
        )


class TestIndexAssignIntoArrayOfSlices:
    """`rows[i] = value` where the element is a slice."""
    pytestmark = GCC_SKIP

    def test_assign_named_slice_variable(self):
        assert_exit_code(
            "    [2][]int rows = [2][]int[[1, 2], [3, 4]]\n"
            "    []int other = []int[9, 9, 9]\n"
            "    rows[0] = other\n"
            "    return rows[0][0] + rows[0][1] + rows[0][2]",
            27,
        )

    def test_assign_typed_slice_literal(self):
        assert_exit_code(
            "    [2][]int rows = [2][]int[[1, 2], [3, 4]]\n"
            "    rows[0] = []int[9, 9, 9]\n"
            "    return rows[0][0] + rows[0][1] + rows[0][2]",
            27,
        )

    def test_assign_untyped_literal(self):
        assert_exit_code(
            "    [2][]int rows = [2][]int[[1, 2], [3, 4]]\n"
            "    rows[0] = [9, 9, 9]\n"
            "    return rows[0][0] + rows[0][1] + rows[0][2]",
            27,
        )

    def test_scalar_index_assign_still_works(self):
        assert_exit_code(
            "    [3]int arr = [1, 2, 3]\n"
            "    arr[1] = 99\n"
            "    return arr[1]",
            99,
        )

    def test_str_index_assign_still_works(self):
        assert_stdout(
            "    [2]str arr = ['a', 'b']\n"
            "    arr[0] = 'z'\n"
            "    print(arr)\n"
            "    return 0",
            "[2]str['z', 'b']\n",
        )


class TestNone:
    pytestmark = GCC_SKIP

    def test_slice_vardecl_with_none(self):
        assert_stdout(
            "    []int s = none\n"
            "    print(s)\n"
            "    return 0",
            "[]int[]\n",
        )

    def test_slice_assign_with_none(self):
        assert_stdout(
            "    [3]int arr = [1, 2, 3]\n"
            "    []int s = arr[0:3]\n"
            "    s = none\n"
            "    print(s)\n"
            "    return 0",
            "[]int[]\n",
        )

    def test_slice_index_assign_with_none(self):
        assert_program_exit_code(
            "def int main():\n"
            "    [1][]int rows\n"
            "    rows[0] = none\n"
            "    if rows[0] == none:\n"
            "        return 1\n"
            "    return 0\n",
            1,
        )

    def test_slice_field_assign_with_none(self):
        assert_program_exit_code(
            "type Box struct:\n"
            "    []int values\n"
            "\n"
            "def int main():\n"
            "    Box b\n"
            "    b.values = none\n"
            "    if b.values == none:\n"
            "        return 1\n"
            "    return 0\n",
            1,
        )

    def test_return_bare_none_from_slice_returning_function(self):
        assert_program_exit_code(
            "def []int makeNone():\n"
            "    return none\n"
            "\n"
            "def int main():\n"
            "    []int s = makeNone()\n"
            "    if s == none:\n"
            "        return 1\n"
            "    return 0\n",
            1,
        )

    def test_return_bare_none_from_pointer_returning_function(self):
        assert_program_stdout(
            "def *int maybeGet(bool flag):\n"
            "    if flag:\n"
            "        int x = 5\n"
            "        return &x\n"
            "    return none\n"
            "\n"
            "def int main():\n"
            "    *int p = maybeGet(false)\n"
            "    if p == none:\n"
            "        print('got none')\n"
            "    return 0\n",
            "got none\n",
        )

    def test_return_address_still_works_from_the_same_pointer_returning_function(self):
        assert_program_stdout(
            "def *int maybeGet(bool flag):\n"
            "    if flag:\n"
            "        int x = 5\n"
            "        return &x\n"
            "    return none\n"
            "\n"
            "def int main():\n"
            "    *int p = maybeGet(true)\n"
            "    if p == none:\n"
            "        print('got none')\n"
            "    else:\n"
            "        print(*p)\n"
            "    return 0\n",
            "5\n",
        )

    def test_none_valued_slice_equals_none(self):
        assert_exit_code(
            "    []int s = none\n"
            "    return s == none",
            1,
            return_type="bool",
        )

    def test_real_empty_slice_is_not_equal_to_none(self):
        assert_exit_code(
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    []int s = arr[5:5]\n"
            "    return s == none",
            0,
            return_type="bool",
        )

    def test_real_nonempty_slice_is_not_equal_to_none(self):
        assert_exit_code(
            "    [3]int arr = [1, 2, 3]\n"
            "    []int s = arr[0:3]\n"
            "    return s == none",
            0,
            return_type="bool",
        )

    def test_none_on_the_left_side(self):
        assert_exit_code(
            "    []int s = none\n"
            "    return none == s",
            1,
            return_type="bool",
        )

    def test_not_equal_with_none(self):
        assert_exit_code(
            "    [3]int arr = [1, 2, 3]\n"
            "    []int s = arr[0:3]\n"
            "    return s != none",
            1,
            return_type="bool",
        )

    def test_indexing_a_none_valued_slice_aborts(self):
        assert_crashes_with_sigabrt(
            "    []int s = none\n"
            "    return s[0]"
        )

    def test_printing_a_none_valued_slice(self):
        assert_stdout(
            "    []int s = none\n"
            "    print(s)\n"
            "    return 0",
            "[]int[]\n",
        )

    def test_reslicing_a_none_valued_slice_at_zero_zero(self):
        assert_exit_code(
            "    []int s = none\n"
            "    []int s2 = s[0:0]\n"
            "    return s2 == none",
            1,
            return_type="bool",
        )

    def test_int_vardecl_with_none_is_rejected(self):
        assert_semantic_error(
            "    int x = none\n"
            "    return 0",
            match="Cannot initialize",
        )

    def test_str_vardecl_with_none_is_rejected(self):
        assert_semantic_error(
            "    str x = none\n"
            "    return 0",
            match="Cannot initialize",
        )

    def test_array_vardecl_with_none_is_rejected(self):
        assert_semantic_error(
            "    [3]int arr = none\n"
            "    return 0",
            match="Cannot initialize",
        )

    def test_print_bare_none_is_rejected(self):
        assert_semantic_error(
            "    print(none)\n"
            "    return 0",
            match="cannot be called with a bare 'none'",
        )

    def test_comparing_none_to_none_is_rejected(self):
        assert_semantic_error(
            "    return none == none",
            match="does not support slice, void, sum type, dict, or none operands",
            return_type="bool",
        )

    def test_comparing_int_to_none_is_rejected(self):
        assert_semantic_error(
            "    return 5 == none",
            match="does not support slice, void, sum type, dict, or none operands",
            return_type="bool",
        )

    def test_none_as_a_slice_argument(self):
        result = compile_and_run(
            "def int first([]int s):\n"
            "    return s[0]\n"
            "\n"
            "def int main():\n"
            "    return first(none)\n"
        )
        assert result.returncode == -signal.SIGABRT

    def test_none_as_a_pointer_argument_does_not_corrupt_a_later_parameter(self):
        assert_program_stdout(
            "def int check(*int p, int y):\n"
            "    if p == none:\n"
            "        return y\n"
            "    return -1\n"
            "\n"
            "def int main():\n"
            "    print(check(none, 42))\n"
            "    return 0\n",
            "42\n",
        )

    def test_none_as_a_pointer_argument_reads_back_as_none(self):
        assert_program_stdout(
            "def bool isNull(*int p):\n"
            "    return p == none\n"
            "\n"
            "def int main():\n"
            "    if isNull(none):\n"
            "        print('yes null')\n"
            "    return 0\n",
            "yes null\n",
        )

    def test_none_as_mixed_slice_and_pointer_arguments_in_one_call(self):
        assert_program_stdout(
            "def bool takesBoth([]int s, *int p):\n"
            "    return s == none and p == none\n"
            "\n"
            "def int main():\n"
            "    if takesBoth(none, none):\n"
            "        print('both none')\n"
            "    return 0\n",
            "both none\n",
        )


# ---------------------------------------------------------------------------
# Semantic analysis
# ---------------------------------------------------------------------------

class TestSemanticErrors:


    def test_reference_to_undeclared_variable(self):
        assert_semantic_error(
            "    return a",
            match="undeclared variable",
        )

    def test_assignment_to_undeclared_variable(self):
        assert_semantic_error(
            "    a = 1\n"
            "    return 0",
            match="undeclared variable",
        )

    def test_double_declaration(self):
        assert_semantic_error(
            "    int a = 1\n"
            "    int a = 2\n"
            "    return a",
            match="already declared",
        )

    def test_declare_before_use_is_enforced_in_textual_order(self):
        assert_semantic_error(
            "    a = 1\n"
            "    int a\n"
            "    return a",
            match="undeclared variable",
        )

    def test_self_referential_initializer(self):
        assert_semantic_error(
            "    int a = a\n"
            "    return a",
            match="undeclared variable",
        )

    def test_valid_program_does_not_raise(self):
        ast = _parse("def int main():\n    int a = 1\n    return a\n")
        analyze(ast)  # should not raise


    def test_initializer_type_mismatch(self):
        assert_semantic_error(
            "    int a = true\n"
            "    return a",
            match="Cannot initialize",
        )

    def test_assignment_type_mismatch(self):
        assert_semantic_error(
            "    bool a = true\n"
            "    a = 1\n"
            "    return a",
            return_type="bool",
            match="Cannot assign",
        )

    def test_return_type_mismatch_bool_where_int_expected(self):
        assert_semantic_error(
            "    return 3 < 5",
            return_type="int",
            match="declared to return",
        )

    def test_return_type_mismatch_int_where_bool_expected(self):
        assert_semantic_error(
            "    return 1",
            return_type="bool",
            match="declared to return",
        )


    def test_not_requires_bool_not_int(self):
        assert_semantic_error(
            "    return not 0",
            return_type="bool",
            match="requires a bool operand",
        )

    def test_negate_requires_int_not_bool(self):
        assert_semantic_error(
            "    return -true",
            match="requires an int, int8, uint8, or int32 operand",
        )

    def test_complement_requires_int_not_bool(self):
        assert_semantic_error(
            "    return ~true",
            match="requires an int, int8, uint8, or int32 operand",
        )

    def test_arithmetic_requires_int_operands(self):
        assert_semantic_error(
            "    return true - false",
            match="requires two operands of the same integer type",
        )

    def test_add_requires_two_int_or_two_str_operands(self):
        assert_semantic_error(
            "    return true + false",
            match="requires two operands of the same integer type",
        )

    def test_ordering_comparison_requires_int_operands(self):
        assert_semantic_error(
            "    return true < false",
            return_type="bool",
            match="requires two operands of the same integer type",
        )

    def test_chained_ordering_comparison_is_rejected(self):
        assert_semantic_error(
            "    return 1 < 2 < 3",
            return_type="bool",
            match="requires two operands of the same integer type",
        )

    def test_logical_and_requires_bool_operands(self):
        assert_semantic_error(
            "    return 1 and 0",
            return_type="bool",
            match="requires bool operands",
        )

    def test_logical_or_requires_bool_operands(self):
        assert_semantic_error(
            "    return 1 or 0",
            return_type="bool",
            match="requires bool operands",
        )

    def test_equality_cannot_compare_int_to_bool(self):
        assert_semantic_error(
            "    return 1 == true",
            return_type="bool",
            match="Cannot compare",
        )

    def test_equality_same_type_is_valid(self):
        ast = _parse("def bool main():\n    return 1 == 1\n")
        analyze(ast)  # should not raise
        ast = _parse("def bool main():\n    return true == false\n")
        analyze(ast)  # should not raise


    def test_float_literal_is_rejected(self):
        assert_semantic_error(
            "    return 2.5",
            match="not a whole number",
        )


    def test_if_condition_must_be_bool(self):
        assert_semantic_error(
            "    if 1:\n"
            "        return 1\n"
            "    return 0",
            match="'if' condition must be bool",
        )

    def test_elif_condition_must_be_bool(self):
        assert_semantic_error(
            "    if false:\n"
            "        return 1\n"
            "    elif 1:\n"
            "        return 2\n"
            "    return 0",
            match="'if' condition must be bool",
        )

    def test_variable_declared_in_if_does_not_leak_outside(self):
        assert_semantic_error(
            "    if true:\n"
            "        int a = 1\n"
            "    return a",
            match="undeclared variable",
        )

    def test_variable_declared_in_then_not_visible_in_else(self):
        assert_semantic_error(
            "    if true:\n"
            "        int a = 1\n"
            "    else:\n"
            "        return a",
            match="undeclared variable",
        )

    def test_same_name_in_sibling_branches_is_allowed(self):
        ast = _parse(
            "def int main():\n"
            "    if true:\n"
            "        int a = 1\n"
            "        return a\n"
            "    else:\n"
            "        int a = 2\n"
            "        return a\n"
        )
        analyze(ast)  # should not raise

    def test_shadowing_outer_variable_in_if_is_allowed(self):
        ast = _parse(
            "def int main():\n"
            "    int a = 1\n"
            "    if true:\n"
            "        int a = 2\n"
            "        return a\n"
            "    return a\n"
        )
        analyze(ast)  # should not raise

    def test_double_declaration_within_same_if_branch_is_rejected(self):
        assert_semantic_error(
            "    if true:\n"
            "        int a = 1\n"
            "        int a = 2\n"
            "        return a\n"
            "    return 0",
            match="already declared",
        )


    def test_while_condition_must_be_bool(self):
        assert_semantic_error(
            "    while 1:\n"
            "        return 1\n"
            "    return 0",
            match="'while' condition must be bool",
        )

    def test_break_outside_loop_is_rejected(self):
        assert_semantic_error(
            "    break\n"
            "    return 0",
            match="'break' outside of a loop",
        )

    def test_continue_outside_loop_is_rejected(self):
        assert_semantic_error(
            "    continue\n"
            "    return 0",
            match="'continue' outside of a loop",
        )

    def test_break_inside_if_inside_while_is_allowed(self):
        ast = _parse(
            "def int main():\n"
            "    while true:\n"
            "        if true:\n"
            "            break\n"
            "    return 0\n"
        )
        analyze(ast)  # should not raise

    def test_break_inside_if_not_inside_while_is_rejected(self):
        assert_semantic_error(
            "    if true:\n"
            "        break\n"
            "    return 0",
            match="'break' outside of a loop",
        )

    def test_break_after_loop_ends_is_rejected(self):
        assert_semantic_error(
            "    while true:\n"
            "        return 1\n"
            "    break\n"
            "    return 0",
            match="'break' outside of a loop",
        )

    def test_variable_declared_in_while_does_not_leak_outside(self):
        assert_semantic_error(
            "    while true:\n"
            "        int a = 1\n"
            "    return a",
            match="undeclared variable",
        )

    def test_break_in_outer_loop_after_inner_loop_ends_is_allowed(self):
        ast = _parse(
            "def int main():\n"
            "    while true:\n"
            "        int j = 0\n"
            "        while j < 3:\n"
            "            j = j + 1\n"
            "        break\n"
            "    return 0\n"
        )
        analyze(ast)  # should not raise


    def test_add_rejects_mixed_int_and_str(self):
        assert_semantic_error(
            "    str a = 'hello'\n"
            "    int b = 5\n"
            "    return a + b",
            return_type="str",
            match="requires two operands of the same integer type",
        )

    def test_subtract_rejects_str_operands(self):
        assert_semantic_error(
            "    str a = 'hello'\n"
            "    str b = 'world'\n"
            "    return a - b",
            return_type="str",
            match="requires two operands of the same integer type",
        )

    def test_ordering_comparison_rejects_str_operands(self):
        assert_semantic_error(
            "    str a = 'hello'\n"
            "    str b = 'world'\n"
            "    return a < b",
            return_type="bool",
            match="requires two operands of the same integer type",
        )

    def test_equality_rejects_str_compared_to_int(self):
        assert_semantic_error(
            "    str a = 'hello'\n"
            "    return a == 5",
            return_type="bool",
            match="Cannot compare",
        )

    def test_str_equality_same_type_is_valid(self):
        ast = _parse(
            "def bool main():\n"
            "    str a = 'hello'\n"
            "    str b = 'hello'\n"
            "    return a == b\n"
        )
        analyze(ast)  # should not raise

    def test_initializer_type_mismatch_int_into_str(self):
        assert_semantic_error(
            "    str a = 5\n"
            "    return 0",
            match="Cannot initialize",
        )

    def test_assignment_type_mismatch_str_into_int(self):
        assert_semantic_error(
            "    int a = 5\n"
            "    a = 'hello'\n"
            "    return a",
            match="Cannot assign",
        )

    def test_return_type_mismatch_str_where_int_expected(self):
        assert_semantic_error(
            "    return 'hello'",
            return_type="int",
            match="declared to return",
        )

    def test_concatenation_type_checks_as_valid_str(self):
        ast = _parse(
            "def str main():\n"
            "    str a = 'hello'\n"
            "    str b = ' world'\n"
            "    str c = a + b\n"
            "    return c\n"
        )
        analyze(ast)  # should not raise


    def test_call_to_undeclared_function(self):
        assert_semantic_error(
            "    return foo(1)",
            match="undeclared function",
        )

    def test_duplicate_function_name(self):
        assert_program_semantic_error(
            "def int foo():\n"
            "    return 1\n"
            "\n"
            "def int foo():\n"
            "    return 2\n",
            match="already declared",
        )

    def test_call_wrong_argument_count(self):
        assert_program_semantic_error(
            "def int add(int a, int b):\n"
            "    return a + b\n"
            "\n"
            "def int main():\n"
            "    return add(1)\n",
            match="expects 2 argument",
        )

    def test_call_wrong_argument_type(self):
        assert_program_semantic_error(
            "def int add(int a, int b):\n"
            "    return a + b\n"
            "\n"
            "def int main():\n"
            "    return add(1, 'hello')\n",
            match="should be int, got str",
        )

    def test_duplicate_parameter_name(self):
        assert_program_semantic_error(
            "def int add(int a, int a):\n"
            "    return a\n",
            match="already declared",
        )

    def test_recursive_call_is_allowed(self):
        ast = _parse(
            "def int fact(int n):\n"
            "    if n == 0:\n"
            "        return 1\n"
            "    return n * fact(n - 1)\n"
        )
        analyze(ast)  # should not raise

    def test_mutual_recursion_with_forward_reference_is_allowed(self):
        ast = _parse(
            "def bool is_even(int n):\n"
            "    if n == 0:\n"
            "        return true\n"
            "    return is_odd(n - 1)\n"
            "\n"
            "def bool is_odd(int n):\n"
            "    if n == 0:\n"
            "        return false\n"
            "    return is_even(n - 1)\n"
        )
        analyze(ast)  # should not raise


    def test_print_wrong_argument_count_zero(self):
        assert_semantic_error(
            "    print()\n"
            "    return 0",
            match="expects exactly 1 argument",
        )

    def test_print_wrong_argument_count_multiple(self):
        assert_semantic_error(
            "    print(1, 2)\n"
            "    return 0",
            match="expects exactly 1 argument",
        )

    def test_print_argument_must_still_be_well_typed(self):
        assert_semantic_error(
            "    print(undeclared_variable)\n"
            "    return 0",
            match="undeclared variable",
        )

    def test_redefining_print_is_rejected(self):
        assert_program_semantic_error(
            "def int print(int x):\n"
            "    return x\n"
            "\n"
            "def int main():\n"
            "    return 0\n",
            match="builtin",
        )

    def test_print_accepts_int_bool_and_str_without_raising(self):
        for arg in ("5", "true", "'hello'"):
            ast = _parse(f"def int main():\n    print({arg})\n    return 0\n")
            analyze(ast)  # should not raise


    def test_modulo_requires_int_operands(self):
        assert_semantic_error(
            "    return true % 2",
            match="requires two operands of the same integer type",
        )

    def test_bitwise_and_requires_int_operands(self):
        assert_semantic_error(
            "    return true & false",
            match="requires two operands of the same integer type",
        )

    def test_bitwise_or_requires_int_operands(self):
        assert_semantic_error(
            "    return 'x' | 1",
            match="requires two operands of the same integer type",
        )

    def test_bitwise_xor_requires_int_operands(self):
        assert_semantic_error(
            "    return true ^ true",
            match="requires two operands of the same integer type",
        )

    def test_shift_left_requires_int_operands(self):
        assert_semantic_error(
            "    return 'x' << 1",
            match="requires two operands of the same integer type",
        )

    def test_shift_right_requires_int_operands(self):
        assert_semantic_error(
            "    return true >> 1",
            match="requires two operands of the same integer type",
        )

    def test_bitwise_and_equality_precedence_is_a_type_error(self):
        assert_semantic_error(
            "    return 1 & 2 == 2",
            match="requires two operands of the same integer type",
        )

    def test_bitwise_and_equality_with_explicit_parens_is_valid(self):
        ast = _parse("def bool main():\n    return (1 & 2) == 2\n")
        analyze(ast)  # should not raise


    def test_array_literal_size_mismatch(self):
        assert_semantic_error(
            "    [3]int arr = [1, 2]\n"
            "    return arr[0]",
            match="Cannot initialize",
        )

    def test_ragged_2d_array_literal_is_rejected(self):
        assert_semantic_error(
            "    [2][3]int matrix = [[1, 2, 3], [4, 5]]\n"
            "    return matrix[0][0]",
            match="elements must all be .*to match the declared element type",
        )

    def test_heterogeneous_array_literal_is_rejected(self):
        assert_semantic_error(
            "    [3]int arr = [1, true, 3]\n"
            "    return arr[0]",
            match="elements must all be .*to match the declared element type",
        )

    def test_non_int_array_index_is_rejected(self):
        assert_semantic_error(
            "    [3]int arr = [1, 2, 3]\n"
            "    return arr[true]",
            match="Index must be int",
        )

    def test_indexing_a_non_array_value_is_rejected(self):
        assert_semantic_error(
            "    int x = 5\n"
            "    return x[0]",
            match="only arrays, slices, str, and dict support indexing",
        )

    def test_indexing_past_available_dimensions_is_rejected(self):
        assert_semantic_error(
            "    [2][3]int matrix = [[1, 2, 3], [4, 5, 6]]\n"
            "    return matrix[0][0][0]",
            match="only arrays, slices, str, and dict support indexing",
        )

    def test_wrong_element_type_in_index_assignment_is_rejected(self):
        assert_semantic_error(
            "    [3]int arr = [1, 2, 3]\n"
            "    arr[0] = true\n"
            "    return arr[0]",
            match="Cannot assign a value of type bool to an array element of type int",
        )

    def test_mismatched_array_equality_comparison_is_rejected(self):
        assert_semantic_error(
            "    [3]int a = [1, 2, 3]\n"
            "    [4]int b = [1, 2, 3, 4]\n"
            "    return a == b",
            match="arrays must have the same length and element type",
            return_type="bool",
        )

    def test_array_of_incomparable_structs_equality_comparison_is_rejected(self):
        assert_program_semantic_error(
            "type Holder struct:\n"
            "    []int xs\n"
            "\n"
            "def bool main():\n"
            "    [2]Holder a\n"
            "    [2]Holder b\n"
            "    return a == b\n",
            match="array equality isn't defined yet when the elements "
                  "are \\(or contain\\) a slice",
        )

    def test_array_of_slices_equality_comparison_is_rejected(self):
        assert_semantic_error(
            "    [2][]int a\n"
            "    [2][]int b\n"
            "    return a == b",
            match="array equality isn't defined yet when the elements "
                  "are \\(or contain\\) a slice",
            return_type="bool",
        )

    def test_array_of_sum_types_equality_comparison_is_rejected(self):
        ast = _parse(
            "type Circle struct:\n"
            "    int radius\n"
            "\n"
            "type Square struct:\n"
            "    int64 side\n"
            "\n"
            "type Shape is Circle | Square\n"
            "\n"
            "def bool main():\n"
            "    [2]Shape a = [Circle(1), Circle(2)]\n"
            "    [2]Shape b = [Circle(1), Circle(2)]\n"
            "    return a == b\n"
        )
        with pytest.raises(
            SemanticError,
            match="array equality isn't defined yet when the elements "
                  "are \\(or contain\\) a slice, sum type, or dict",
        ):
            analyze(ast)

    def test_array_of_dicts_equality_comparison_is_rejected(self):
        assert_semantic_error(
            "    [2]dict[str]int a\n"
            "    [2]dict[str]int b\n"
            "    return a == b",
            match="array equality isn't defined yet when the elements "
                  "are \\(or contain\\) a slice, sum type, or dict",
            return_type="bool",
        )

    def test_struct_with_a_dict_typed_field_equality_comparison_is_rejected(self):
        ast = _parse(
            "type Wrapper struct:\n"
            "    dict[str]int d\n"
            "\n"
            "def bool main():\n"
            "    Wrapper a\n"
            "    Wrapper b\n"
            "    return a == b\n"
        )
        with pytest.raises(SemanticError, match="struct equality isn't defined yet when a field"):
            analyze(ast)

    def test_array_as_function_param_and_return_type_checks_correctly(self):
        ast = _parse(
            "def [3]int make_array(int a, int b, int c):\n"
            "    [3]int result = [a, b, c]\n"
            "    return result\n"
            "\n"
            "def int main():\n"
            "    [3]int r = make_array(1, 2, 3)\n"
            "    return r[0]\n"
        )
        analyze(ast)  # should not raise


    def test_type_mismatch_reports_the_expressions_own_line_and_column(self):
        ast = _parse(
            "def int main():\n"      # line 1
            "    bool b = true\n"    # line 2
            "    int x = b + 1\n"
            "    return x\n"
        )
        with pytest.raises(SemanticError, match=re.escape("at line 3, column 13")):
            analyze(ast)

    def test_undeclared_variable_reports_the_reference_site(self):
        ast = _parse(
            "def int main():\n"   # line 1
            "    return nope\n"   # line 2 -- 'nope' at column 12
        )
        with pytest.raises(SemanticError, match=re.escape("at line 2, column 12")):
            analyze(ast)

    def test_double_declaration_reports_the_second_declarations_own_line(self):
        ast = _parse(
            "def int main():\n"    # line 1
            "    int a = 1\n"      # line 2 -- the first, unproblematic declaration
            "    int a = 2\n"      # line 3 -- the redeclaration itself, 'int' at column 5
            "    return a\n"
        )
        with pytest.raises(SemanticError, match=re.escape("at line 3, column 5")):
            analyze(ast)

    def test_break_outside_loop_reports_the_break_itself(self):
        ast = _parse(
            "def int main():\n"  # line 1
            "    break\n"        # line 2 -- 'break' at column 5
            "    return 0\n"
        )
        with pytest.raises(SemanticError, match=re.escape("at line 2, column 5")):
            analyze(ast)

    def test_duplicate_struct_reports_the_second_structs_own_line(self):
        ast = _parse(
            "type A struct:\n"           # line 1 -- the first, unproblematic declaration
            "    int x\n"
            "\n"
            "type A struct:\n"           # line 4 -- the redeclaration itself, 'type' at column 1
            "    int y\n"
            "\n"
            "def int main():\n"
            "    return 0\n"
        )
        with pytest.raises(SemanticError, match=re.escape("at line 4, column 1")):
            analyze(ast)


# ---------------------------------------------------------------------------
# Single-line comments
# ---------------------------------------------------------------------------

class TestComments:
    pytestmark = GCC_SKIP

    def test_comment_only_line_inside_a_block(self):
        assert_exit_code(
            "    # just a comment, does nothing\n"
            "    return 42",
            42,
        )

    def test_trailing_comment_after_a_statement(self):
        assert_exit_code(
            "    int x = 5  # set x to five\n"
            "    return x",
            5,
        )

    def test_trailing_comment_on_the_def_line_itself(self):
        assert_program_exit_code(
            "def int add(int a, int b):  # adds two ints\n"
            "    return a + b\n"
            "\n"
            "def int main():\n"
            "    return add(3, 4)\n",
            7,
        )

    def test_leading_comment_before_a_function_definition(self):
        assert_program_exit_code(
            "# This function adds two numbers.\n"
            "def int add(int a, int b):\n"
            "    return a + b\n"
            "\n"
            "def int main():\n"
            "    return add(3, 4)\n",
            7,
        )

    def test_hash_inside_a_string_literal_is_not_a_comment(self):
        assert_stdout(
            "    str s = 'value # 42'\n"
            "    print(s)\n"
            "    return 0",
            "value # 42\n",
        )

    def test_comment_only_line_as_first_content_in_a_block(self):
        assert_exit_code(
            "    # nothing real here yet\n"
            "    if true:\n"
            "        return 1\n"
            "    return 0",
            1,
        )

    def test_comment_only_line_does_not_affect_dedent(self):
        assert_exit_code(
            "    if true:\n"
            "        int x = 1\n"
            "    # back to the outer level, just a comment\n"
            "    return 99",
            99,
        )

    def test_multiple_consecutive_comment_only_lines(self):
        assert_exit_code(
            "    # first comment\n"
            "    # second comment\n"
            "    # third comment\n"
            "    return 7",
            7,
        )

    def test_empty_comment(self):
        assert_exit_code(
            "    #\n"
            "    return 3",
            3,
        )

    def test_comment_as_the_last_line_of_the_file_no_trailing_newline(self):
        assert_program_exit_code(
            "def int main():\n"
            "    return 0\n"
            "# trailing comment, no newline after this one",
            0,
        )

    def test_comments_do_not_disturb_multi_level_nesting(self):
        assert_exit_code(
            "    # top level\n"
            "    int total = 0\n"
            "    int i = 0\n"
            "    while i < 5:\n"
            "        # inside the loop\n"
            "        if i == 2:\n"
            "            # inside the if\n"
            "            total = total + 10\n"
            "        else:\n"
            "            # inside the else\n"
            "            total = total + 1\n"
            "        # back at loop level\n"
            "        i = i + 1\n"
            "    # after the loop\n"
            "    return total",
            14,  # 1 + 1 + 10 + 1 + 1 = 14
        )

# ---------------------------------------------------------------------------
# Structs
# ---------------------------------------------------------------------------

class TestStructs:
    pytestmark = GCC_SKIP

    def test_type_struct_form_is_equivalent_to_bare_struct_form(self):
        assert_program_exit_code(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "    def int sum(self):\n"
            "        return self.x + self.y\n"
            "\n"
            "def int main():\n"
            "    Point p = Point(3, 4)\n"
            "    return p.sum()\n",
            7,
        )

    def test_basic_field_read_and_write(self):
        assert_program_exit_code(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def int main():\n"
            "    Point p\n"
            "    p.x = 3\n"
            "    p.y = 4\n"
            "    return p.x + p.y\n",
            7,
        )

    def test_field_to_field_copy(self):
        assert_program_exit_code(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def int main():\n"
            "    Point p\n"
            "    p.x = 3\n"
            "    p.y = 4\n"
            "    Point q\n"
            "    q.x = p.x\n"
            "    q.y = p.y\n"
            "    return q.x + q.y\n",
            7,
        )

    def test_whole_struct_assignment(self):
        assert_program_exit_code(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def int main():\n"
            "    Point p\n"
            "    p.x = 3\n"
            "    p.y = 4\n"
            "    Point q\n"
            "    q = p\n"
            "    return q.x + q.y\n",
            7,
        )

    def test_value_semantics_local_copy(self):
        assert_program_exit_code(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def int main():\n"
            "    Point p\n"
            "    p.x = 3\n"
            "    p.y = 4\n"
            "    Point q = p\n"
            "    q.x = 99\n"
            "    return p.x\n",
            3,
        )

    def test_value_semantics_across_a_call(self):
        assert_program_exit_code(
            "type Point struct:\n"
            "    int x\n"
            "\n"
            "def int mutate(Point p):\n"
            "    p.x = 999\n"
            "    return p.x\n"
            "\n"
            "def int main():\n"
            "    Point p\n"
            "    p.x = 5\n"
            "    int result = mutate(p)\n"
            "    return p.x\n",
            5,
        )

    def test_nested_struct_field_access(self):
        assert_program_exit_code(
            "type Inner struct:\n"
            "    int v\n"
            "type Outer struct:\n"
            "    Inner inner\n"
            "\n"
            "def int main():\n"
            "    Outer o\n"
            "    o.inner.v = 42\n"
            "    return o.inner.v\n",
            42,
        )

    def test_struct_field_containing_another_struct_assigned_wholesale(self):
        assert_program_exit_code(
            "type Inner struct:\n"
            "    int v\n"
            "type Outer struct:\n"
            "    Inner inner\n"
            "\n"
            "def int main():\n"
            "    Inner i\n"
            "    i.v = 7\n"
            "    Outer o\n"
            "    o.inner = i\n"
            "    return o.inner.v\n",
            7,
        )

    def test_array_of_structs(self):
        assert_program_exit_code(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def int main():\n"
            "    [3]Point pts\n"
            "    pts[0].x = 1\n"
            "    pts[0].y = 2\n"
            "    pts[1].x = 3\n"
            "    pts[1].y = 4\n"
            "    return pts[0].x + pts[0].y + pts[1].x + pts[1].y\n",
            10,
        )

    def test_array_typed_field_indexed(self):
        assert_program_exit_code(
            "type Row struct:\n"
            "    [3]int data\n"
            "\n"
            "def int main():\n"
            "    Row r\n"
            "    r.data[0] = 10\n"
            "    r.data[2] = 20\n"
            "    return r.data[0] + r.data[2]\n",
            30,
        )

    def test_struct_as_function_parameter(self):
        assert_program_exit_code(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def int sumPoint(Point p):\n"
            "    return p.x + p.y\n"
            "\n"
            "def int main():\n"
            "    Point p\n"
            "    p.x = 3\n"
            "    p.y = 4\n"
            "    return sumPoint(p)\n",
            7,
        )

    def test_struct_as_function_return_type(self):
        assert_program_exit_code(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def Point makePoint(int x, int y):\n"
            "    Point p\n"
            "    p.x = x\n"
            "    p.y = y\n"
            "    return p\n"
            "\n"
            "def int main():\n"
            "    Point p = makePoint(3, 4)\n"
            "    return p.x + p.y\n",
            7,
        )

    def test_forwarding_a_struct_returning_call(self):
        assert_program_exit_code(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def Point makePoint(int x, int y):\n"
            "    Point p\n"
            "    p.x = x\n"
            "    p.y = y\n"
            "    return p\n"
            "\n"
            "def Point forwardPoint(int x, int y):\n"
            "    return makePoint(x, y)\n"
            "\n"
            "def int main():\n"
            "    Point p = forwardPoint(3, 4)\n"
            "    return p.x + p.y\n",
            7,
        )

    def test_passing_a_struct_field_as_an_argument(self):
        assert_program_exit_code(
            "type Inner struct:\n"
            "    int v\n"
            "type Outer struct:\n"
            "    Inner inner\n"
            "\n"
            "def int getV(Inner i):\n"
            "    return i.v\n"
            "\n"
            "def int main():\n"
            "    Outer o\n"
            "    o.inner.v = 42\n"
            "    return getV(o.inner)\n",
            42,
        )

    def test_large_struct_is_heap_allocated(self):
        assert_program_exit_code(
            "type Big struct:\n"
            "    [5000]int data\n"
            "\n"
            "def int useBig(Big b):\n"
            "    return b.data[0] + b.data[4999]\n"
            "\n"
            "def int main():\n"
            "    Big b\n"
            "    b.data[0] = 10\n"
            "    b.data[4999] = 20\n"
            "    return useBig(b)\n",
            30,
        )

    def test_large_struct_actually_uses_malloc(self):
        source = (
            "type Big struct:\n"
            "    [5000]int data\n"
            "\n"
            "def int main():\n"
            "    Big b\n"
            "    return b.data[0]\n"
        )
        ast = _parse(source)
        analyze(ast)
        mallocs = _heap_allocations(ast)
        assert mallocs

    def test_forward_reference(self):
        assert_program_exit_code(
            "type A struct:\n"
            "    B b\n"
            "type B struct:\n"
            "    int v\n"
            "\n"
            "def int main():\n"
            "    A a\n"
            "    a.b.v = 5\n"
            "    return a.b.v\n",
            5,
        )

    def test_duplicate_struct_name_is_rejected(self):
        source = (
            "type Foo struct:\n"
            "    int a\n"
            "type Foo struct:\n"
            "    int b\n"
            "\n"
            "def int main():\n"
            "    return 0\n"
        )
        with pytest.raises(SemanticError, match="already declared"):
            analyze(_parse(source))

    def test_duplicate_field_name_is_rejected(self):
        source = (
            "type Foo struct:\n"
            "    int a\n"
            "    int a\n"
            "\n"
            "def int main():\n"
            "    return 0\n"
        )
        with pytest.raises(SemanticError, match="already declared"):
            analyze(_parse(source))

    def test_unknown_field_access_is_rejected(self):
        source = (
            "type Point struct:\n"
            "    int x\n"
            "\n"
            "def int main():\n"
            "    Point p\n"
            "    return p.y\n"
        )
        with pytest.raises(SemanticError, match="no field"):
            analyze(_parse(source))

    def test_field_access_on_non_struct_type_is_rejected(self):
        source = (
            "def int main():\n"
            "    int x = 5\n"
            "    return x.foo\n"
        )
        with pytest.raises(SemanticError, match="non-struct"):
            analyze(_parse(source))

    def test_wrong_typed_field_assignment_is_rejected(self):
        source = (
            "type Point struct:\n"
            "    int x\n"
            "\n"
            "def int main():\n"
            "    Point p\n"
            "    p.x = true\n"
            "    return 0\n"
        )
        with pytest.raises(SemanticError, match="Cannot assign"):
            analyze(_parse(source))

    def test_unknown_struct_type_name_is_rejected(self):
        source = (
            "def int main():\n"
            "    Bar b\n"
            "    return 0\n"
        )
        with pytest.raises(SemanticError, match="Unknown type"):
            analyze(_parse(source))

    def test_direct_self_containment_is_rejected(self):
        source = (
            "type Foo struct:\n"
            "    Foo f\n"
            "\n"
            "def int main():\n"
            "    return 0\n"
        )
        with pytest.raises(SemanticError, match="cannot contain itself"):
            analyze(_parse(source))

    def test_mutual_cycle_is_rejected(self):
        source = (
            "type A struct:\n"
            "    B b\n"
            "type B struct:\n"
            "    A a\n"
            "\n"
            "def int main():\n"
            "    return 0\n"
        )
        with pytest.raises(SemanticError, match="cannot contain itself"):
            analyze(_parse(source))

    def test_cycle_via_array_field_is_rejected(self):
        source = (
            "type A struct:\n"
            "    [5]B b\n"
            "type B struct:\n"
            "    A a\n"
            "\n"
            "def int main():\n"
            "    return 0\n"
        )
        with pytest.raises(SemanticError, match="cannot contain itself"):
            analyze(_parse(source))

    def test_struct_containing_array_of_different_struct_is_fine(self):
        ast = _parse(
            "type Point struct:\n"
            "    int x\n"
            "type Triangle struct:\n"
            "    [3]Point vertices\n"
            "\n"
            "def int main():\n"
            "    return 0\n"
        )
        analyze(ast)  # should not raise

    def test_nominal_typing_two_structs_with_same_fields_are_different_types(self):
        source = (
            "type A struct:\n"
            "    int v\n"
            "type B struct:\n"
            "    int v\n"
            "\n"
            "def int useA(A a):\n"
            "    return a.v\n"
            "\n"
            "def int main():\n"
            "    B b\n"
            "    b.v = 5\n"
            "    return useA(b)\n"
        )
        with pytest.raises(SemanticError, match="should be A, got B"):
            analyze(_parse(source))

    def test_slice_typed_field_basic_read_and_write(self):
        assert_program_exit_code(
            "type Row struct:\n"
            "    []int values\n"
            "\n"
            "def int main():\n"
            "    Row r\n"
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    r.values = arr[0:3]\n"
            "    return r.values[0] + r.values[1] + r.values[2]\n",
            6,
        )

    def test_slice_literal_field_in_struct_literal_regression(self):
        assert_program_exit_code(
            "type Row struct:\n"
            "    []int values\n"
            "\n"
            "def int main():\n"
            "    Row r = Row([10, 20, 30])\n"
            "    return r.values[0] + r.values[1] + r.values[2]\n",
            60,
        )

    def test_array_of_slices_field_is_supported(self):
        assert_program_exit_code(
            "type Rows struct:\n"
            "    [2][]int values\n"
            "\n"
            "def int main():\n"
            "    Rows r\n"
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    r.values[0] = arr[0:2]\n"
            "    r.values[1] = arr[2:4]\n"
            "    return r.values[0][0] + r.values[1][0]\n",
            4,
        )

    def test_slice_field_nested_through_another_struct_is_supported(self):
        assert_program_exit_code(
            "type Inner struct:\n"
            "    []int values\n"
            "type Outer struct:\n"
            "    Inner inner\n"
            "\n"
            "def int main():\n"
            "    Outer o\n"
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    o.inner.values = arr[0:2]\n"
            "    return o.inner.values[0] + o.inner.values[1]\n",
            3,
        )

    def test_ordinary_array_field_is_not_rejected(self):
        ast = _parse(
            "type Fixed struct:\n"
            "    [3]int values\n"
            "\n"
            "def int main():\n"
            "    Fixed f\n"
            "    return f.values[0]\n"
        )
        analyze(ast)  # should not raise

    def test_self_referential_struct_via_slice_field(self):
        assert_program_exit_code(
            "type Node struct:\n"
            "    int value\n"
            "    []Node children\n"
            "\n"
            "def int main():\n"
            "    Node n\n"
            "    n.value = 42\n"
            "    return n.value\n",
            42,
        )

    def test_none_flows_into_a_slice_typed_field(self):
        assert_program_exit_code(
            "type Row struct:\n"
            "    []int values\n"
            "\n"
            "def int main():\n"
            "    Row r\n"
            "    r.values = none\n"
            "    return len(r.values)\n",
            0,
        )

    def test_untyped_array_literal_flows_into_a_slice_typed_field(self):
        assert_program_exit_code(
            "type Row struct:\n"
            "    []int values\n"
            "\n"
            "def int main():\n"
            "    Row r\n"
            "    r.values = [1, 2, 3]\n"
            "    return r.values[0] + r.values[1] + r.values[2]\n",
            6,
        )

    def test_writing_a_scalar_element_of_a_slice_typed_field(self):
        assert_program_exit_code(
            "type Row struct:\n"
            "    []int values\n"
            "\n"
            "def int main():\n"
            "    Row r\n"
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    r.values = arr[0:3]\n"
            "    r.values[0] = 99\n"
            "    return r.values[0]\n",
            99,
        )

    def test_wrong_typed_value_assigned_to_a_slice_typed_field_is_rejected(self):
        source = (
            "type Row struct:\n"
            "    []int values\n"
            "\n"
            "def int main():\n"
            "    Row r\n"
            "    r.values = 5\n"
            "    return 0\n"
        )
        with pytest.raises(SemanticError, match="Cannot assign"):
            analyze(_parse(source))


    def test_struct_escaping_via_return_promotes_slice_field_backing(self):
        assert_program_exit_code(
            "type Row struct:\n"
            "    []int values\n"
            "\n"
            "def Row makeRow():\n"
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    Row r\n"
            "    r.values = arr[0:3]\n"
            "    return r\n"
            "\n"
            "def int clobber():\n"
            "    [20]int junk\n"
            "    int i = 0\n"
            "    while i < 20:\n"
            "        junk[i] = 999\n"
            "        i = i + 1\n"
            "    return junk[0]\n"
            "\n"
            "def int main():\n"
            "    Row r = makeRow()\n"
            "    int j = clobber()\n"
            "    return r.values[0] + r.values[1] + r.values[2]\n",
            6,
        )

    def test_non_escaping_slice_field_stays_stack_allocated(self):
        source = (
            "type Row struct:\n"
            "    []int values\n"
            "\n"
            "def int main():\n"
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    Row r\n"
            "    r.values = arr[0:3]\n"
            "    return r.values[0]\n"
        )
        ast = _parse(source)
        analyze(ast)
        mallocs = _heap_allocations(ast)
        assert not mallocs

    def test_reslicing_a_struct_slice_field_escapes_correctly(self):
        assert_program_exit_code(
            "type Row struct:\n"
            "    []int values\n"
            "\n"
            "def []int makeSub():\n"
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    Row r\n"
            "    r.values = arr[0:3]\n"
            "    []int s = r.values[0:2]\n"
            "    return s\n"
            "\n"
            "def int clobber():\n"
            "    [20]int junk\n"
            "    int i = 0\n"
            "    while i < 20:\n"
            "        junk[i] = 999\n"
            "        i = i + 1\n"
            "    return junk[0]\n"
            "\n"
            "def int main():\n"
            "    []int s = makeSub()\n"
            "    int j = clobber()\n"
            "    return s[0] + s[1]\n",
            3,
        )

    def test_struct_to_struct_copy_propagates_slice_field_backing(self):
        assert_program_exit_code(
            "type Row struct:\n"
            "    []int values\n"
            "\n"
            "def Row makeRow():\n"
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    Row r\n"
            "    r.values = arr[0:3]\n"
            "    Row q = r\n"
            "    return q\n"
            "\n"
            "def int clobber():\n"
            "    [20]int junk\n"
            "    int i = 0\n"
            "    while i < 20:\n"
            "        junk[i] = 999\n"
            "        i = i + 1\n"
            "    return junk[0]\n"
            "\n"
            "def int main():\n"
            "    Row r = makeRow()\n"
            "    int j = clobber()\n"
            "    return r.values[0] + r.values[1]\n",
            3,
        )

    def test_struct_slice_field_as_parameter_escapes_correctly(self):
        assert_program_exit_code(
            "type Row struct:\n"
            "    []int values\n"
            "\n"
            "def []int extract(Row r):\n"
            "    return r.values\n"
            "\n"
            "def []int makeSub():\n"
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    Row r\n"
            "    r.values = arr[0:3]\n"
            "    return extract(r)\n"
            "\n"
            "def int clobber():\n"
            "    [20]int junk\n"
            "    int i = 0\n"
            "    while i < 20:\n"
            "        junk[i] = 999\n"
            "        i = i + 1\n"
            "    return junk[0]\n"
            "\n"
            "def int main():\n"
            "    []int s = makeSub()\n"
            "    int j = clobber()\n"
            "    return s[0] + s[1]\n",
            3,
        )

    def test_append_on_a_struct_slice_field_escapes_correctly(self):
        assert_program_exit_code(
            "type Row struct:\n"
            "    []int values\n"
            "\n"
            "def []int makeAppended():\n"
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    Row r\n"
            "    r.values = arr[0:2]\n"
            "    []int s = append(r.values, 99)\n"
            "    return s\n"
            "\n"
            "def int clobber():\n"
            "    [20]int junk\n"
            "    int i = 0\n"
            "    while i < 20:\n"
            "        junk[i] = 999\n"
            "        i = i + 1\n"
            "    return junk[0]\n"
            "\n"
            "def int main():\n"
            "    []int s = makeAppended()\n"
            "    int j = clobber()\n"
            "    return s[0] + s[1] + s[2]\n",
            102,
        )

    def test_compound_assignment_to_a_field_now_works(self):
        assert_program_exit_code(
            "type Point struct:\n"
            "    int x\n"
            "\n"
            "def int main():\n"
            "    Point p = Point(5)\n"
            "    p.x += 1\n"
            "    return p.x\n",
            expected=6,
        )


class TestStructLiterals:
    """`Name(arg1, arg2, ...)`."""

    pytestmark = GCC_SKIP

    def test_basic_construction_via_var_decl(self):
        assert_program_exit_code(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def int main():\n"
            "    Point p = Point(3, 4)\n"
            "    return p.x + p.y\n",
            7,
        )

    def test_construction_via_plain_assign(self):
        assert_program_exit_code(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def int main():\n"
            "    Point p = Point(1, 1)\n"
            "    p = Point(3, 4)\n"
            "    return p.x + p.y\n",
            7,
        )

    def test_str_field(self):
        assert_program_exit_code(
            "type Person struct:\n"
            "    int age\n"
            "    str name\n"
            "\n"
            "def int main():\n"
            "    Person p = Person(5, 'hi')\n"
            "    if p.name == 'hi':\n"
            "        return p.age\n"
            "    return -1\n",
            5,
        )

    def test_array_typed_field(self):
        assert_program_exit_code(
            "type Row struct:\n"
            "    [3]int values\n"
            "\n"
            "def int main():\n"
            "    Row r = Row([1, 2, 3])\n"
            "    return r.values[0] + r.values[1] + r.values[2]\n",
            6,
        )

    def test_slice_typed_field_with_none(self):
        assert_program_exit_code(
            "type Holder struct:\n"
            "    []int xs\n"
            "\n"
            "def int main():\n"
            "    Holder h = Holder(none)\n"
            "    if h.xs == none:\n"
            "        return 42\n"
            "    return -1\n",
            42,
        )

    def test_slice_typed_field_with_a_real_slice(self):
        assert_program_exit_code(
            "type Holder struct:\n"
            "    []int xs\n"
            "\n"
            "def int main():\n"
            "    [3]int arr = [10, 20, 30]\n"
            "    Holder h = Holder(arr[:])\n"
            "    return h.xs[0] + h.xs[1] + h.xs[2]\n",
            60,
        )

    def test_struct_typed_field_via_a_named_variable(self):
        assert_program_exit_code(
            "type Inner struct:\n"
            "    int v\n"
            "\n"
            "type Outer struct:\n"
            "    Inner i\n"
            "    int b\n"
            "\n"
            "def int main():\n"
            "    Inner inner = Inner(9)\n"
            "    Outer o = Outer(inner, 2)\n"
            "    return o.i.v + o.b\n",
            11,
        )

    def test_large_struct_literal_is_heap_allocated(self):
        assert_program_exit_code(
            "type Big struct:\n"
            "    [2100]int a\n"
            "    [2100]int b\n"
            "    int tag\n"
            "\n"
            "def int main():\n"
            "    [2100]int fa\n"
            "    [2100]int fb\n"
            "    Big big = Big(fa, fb, 99)\n"
            "    return big.tag\n",
            99,
        )

    def test_large_struct_literal_actually_uses_malloc(self):
        source = (
            "type Big struct:\n"
            "    [1050]int a\n"
            "    [1050]int b\n"
            "    int tag\n"
            "\n"
            "def int main():\n"
            "    [1050]int fa\n"
            "    [1050]int fb\n"
            "    Big big = Big(fa, fb, 99)\n"
            "    return big.tag\n"
        )
        ast = _parse(source)
        analyze(ast)
        assert len(_heap_allocations(ast)) == 1

    def test_small_struct_literal_does_not_use_malloc(self):
        source = (
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def int main():\n"
            "    Point p = Point(3, 4)\n"
            "    return p.x + p.y\n"
        )
        ast = _parse(source)
        analyze(ast)
        mallocs = _heap_allocations(ast)
        assert not mallocs

    def test_value_semantics_mutating_a_copy_does_not_affect_the_original(self):
        assert_program_exit_code(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def int main():\n"
            "    Point p = Point(3, 4)\n"
            "    Point q = p\n"
            "    q.x = 100\n"
            "    return p.x + q.x\n",
            103,
        )

    def test_resulting_struct_can_be_passed_to_a_function(self):
        assert_program_exit_code(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def int sumPoint(Point p):\n"
            "    return p.x + p.y\n"
            "\n"
            "def int main():\n"
            "    Point p = Point(3, 4)\n"
            "    return sumPoint(p)\n",
            7,
        )

    def test_wrong_argument_count_is_rejected(self):
        assert_program_semantic_error(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def int main():\n"
            "    Point p = Point(3)\n"
            "    return 0\n",
            match="expects 2 argument",
        )

    def test_wrong_argument_type_is_rejected(self):
        assert_program_semantic_error(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def int main():\n"
            "    Point p = Point('nope', 4)\n"
            "    return 0\n",
            match="should be int",
        )

    def test_struct_literal_as_direct_function_argument(self):
        assert_program_exit_code(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def int sumPoint(Point p):\n"
            "    return p.x + p.y\n"
            "\n"
            "def int main():\n"
            "    return sumPoint(Point(3, 4))\n",
            7,
        )

    def test_struct_literal_as_direct_return_value(self):
        assert_program_exit_code(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def Point makePoint():\n"
            "    return Point(3, 4)\n"
            "\n"
            "def int main():\n"
            "    Point p = makePoint()\n"
            "    return p.x + p.y\n",
            7,
        )

    def test_large_struct_literal_as_direct_return_value(self):
        assert_program_exit_code(
            "type Big struct:\n"
            "    [5000]int data\n"
            "    int tag\n"
            "\n"
            "def Big makeBig():\n"
            "    [5000]int filler\n"
            "    return Big(filler, 77)\n"
            "\n"
            "def int main():\n"
            "    Big b = makeBig()\n"
            "    return b.tag\n",
            77,
        )

    def test_struct_literal_return_value_with_a_slice_typed_field(self):
        assert_program_exit_code(
            "type Holder struct:\n"
            "    []int xs\n"
            "\n"
            "def Holder makeHolder():\n"
            "    [3]int arr = [10, 20, 30]\n"
            "    return Holder(arr[:])\n"
            "\n"
            "def int main():\n"
            "    Holder h = makeHolder()\n"
            "    return h.xs[0] + h.xs[1] + h.xs[2]\n",
            60,
        )

    def test_nested_struct_literal_as_return_value(self):
        assert_program_exit_code(
            "type Inner struct:\n"
            "    int v\n"
            "type Outer struct:\n"
            "    Inner i\n"
            "    int b\n"
            "\n"
            "def Outer makeOuter():\n"
            "    return Outer(Inner(9), 2)\n"
            "\n"
            "def int main():\n"
            "    Outer o = makeOuter()\n"
            "    return o.i.v + o.b\n",
            11,
        )

    def test_struct_literal_returned_from_a_function_with_no_declared_return_type_is_rejected(self):
        assert_program_semantic_error(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def noReturnType():\n"
            "    return Point(1, 2)\n"
            "\n"
            "def int main():\n"
            "    return 0\n",
            match="no declared return type",
        )

    def test_wrong_struct_type_as_return_value_is_rejected(self):
        assert_program_semantic_error(
            "type A struct:\n"
            "    int x\n"
            "type B struct:\n"
            "    int y\n"
            "\n"
            "def A makeA():\n"
            "    return B(1)\n"
            "\n"
            "def int main():\n"
            "    return 0\n",
            match="declared to return",
        )

    def test_nested_struct_literal_via_var_decl(self):
        assert_program_exit_code(
            "type Inner struct:\n"
            "    int v\n"
            "\n"
            "type Outer struct:\n"
            "    Inner i\n"
            "    int b\n"
            "\n"
            "def int main():\n"
            "    Outer o = Outer(Inner(9), 2)\n"
            "    return o.i.v + o.b\n",
            11,
        )

    def test_nested_struct_literal_via_assign(self):
        assert_program_exit_code(
            "type Inner struct:\n"
            "    int v\n"
            "type Outer struct:\n"
            "    Inner i\n"
            "    int b\n"
            "\n"
            "def int main():\n"
            "    Outer o = Outer(Inner(1), 2)\n"
            "    o = Outer(Inner(9), 8)\n"
            "    return o.i.v + o.b\n",
            17,
        )

    def test_three_levels_of_nested_struct_literals(self):
        assert_program_exit_code(
            "type C struct:\n"
            "    int v\n"
            "type B struct:\n"
            "    C c\n"
            "    int w\n"
            "type A struct:\n"
            "    B b\n"
            "    int u\n"
            "\n"
            "def int main():\n"
            "    A a = A(B(C(1), 2), 3)\n"
            "    return a.b.c.v + a.b.w + a.u\n",
            6,
        )

    def test_nested_struct_literal_with_an_array_typed_field(self):
        assert_program_exit_code(
            "type Row struct:\n"
            "    [2]int values\n"
            "type Grid struct:\n"
            "    Row r\n"
            "    int tag\n"
            "\n"
            "def int main():\n"
            "    Grid g = Grid(Row([5, 6]), 9)\n"
            "    return g.r.values[0] + g.r.values[1] + g.tag\n",
            20,
        )

    def test_type_error_inside_a_nested_struct_literal_is_still_reported(self):
        assert_program_semantic_error(
            "type Inner struct:\n"
            "    int v\n"
            "type Outer struct:\n"
            "    Inner i\n"
            "    int b\n"
            "\n"
            "def int main():\n"
            "    Outer o = Outer(Inner('nope'), 2)\n"
            "    return 0\n",
            match="should be int",
        )

    def test_struct_literal_as_field_assign_value(self):
        assert_program_exit_code(
            "type Inner struct:\n"
            "    int v\n"
            "\n"
            "type Outer struct:\n"
            "    Inner i\n"
            "\n"
            "def int main():\n"
            "    Inner inner = Inner(1)\n"
            "    Outer o = Outer(inner)\n"
            "    o.i = Inner(2)\n"
            "    return o.i.v\n",
            2,
        )

    def test_struct_literal_as_index_assign_value(self):
        assert_program_exit_code(
            "type Point struct:\n"
            "    int x\n"
            "\n"
            "def int main():\n"
            "    [3]Point pts\n"
            "    pts[0] = Point(1)\n"
            "    pts[1] = Point(2)\n"
            "    return pts[0].x + pts[1].x\n",
            3,
        )

    def test_nested_struct_literal_as_index_assign_value(self):
        assert_program_exit_code(
            "type Inner struct:\n"
            "    int v\n"
            "type Outer struct:\n"
            "    Inner i\n"
            "    int b\n"
            "\n"
            "def int main():\n"
            "    [1]Outer os\n"
            "    os[0] = Outer(Inner(9), 2)\n"
            "    return os[0].i.v + os[0].b\n",
            11,
        )

    def test_nested_struct_literal_as_field_assign_value(self):
        assert_program_exit_code(
            "type Inner struct:\n"
            "    int v\n"
            "type Mid struct:\n"
            "    Inner i\n"
            "type Outer struct:\n"
            "    Mid m\n"
            "\n"
            "def int main():\n"
            "    Outer o\n"
            "    o.m = Mid(Inner(7))\n"
            "    return o.m.i.v\n",
            7,
        )

    def test_struct_literal_with_slice_field_as_index_assign_value(self):
        assert_program_exit_code(
            "type Holder struct:\n"
            "    []int xs\n"
            "\n"
            "def int main():\n"
            "    [1]Holder hs\n"
            "    hs[0] = Holder([]int[1, 2, 3])\n"
            "    return hs[0].xs[0] + hs[0].xs[1] + hs[0].xs[2]\n",
            6,
        )

    def test_struct_literal_with_slice_field_as_field_assign_value(self):
        assert_program_exit_code(
            "type Holder struct:\n"
            "    []int xs\n"
            "type Wrapper struct:\n"
            "    Holder h\n"
            "\n"
            "def int main():\n"
            "    Wrapper w\n"
            "    w.h = Holder([]int[10, 20])\n"
            "    return w.h.xs[0] + w.h.xs[1]\n",
            30,
        )

    def test_large_struct_literal_as_index_assign_value(self):
        assert_program_exit_code(
            "type Big struct:\n"
            "    [5000]int data\n"
            "    int tag\n"
            "\n"
            "def int main():\n"
            "    [1]Big bigs\n"
            "    [5000]int filler\n"
            "    bigs[0] = Big(filler, 88)\n"
            "    return bigs[0].tag\n",
            88,
        )

    def test_large_struct_literal_as_field_assign_value(self):
        assert_program_exit_code(
            "type Big struct:\n"
            "    [5000]int data\n"
            "    int tag\n"
            "type Wrapper struct:\n"
            "    Big b\n"
            "\n"
            "def int main():\n"
            "    Wrapper w\n"
            "    [5000]int filler\n"
            "    w.b = Big(filler, 99)\n"
            "    return w.b.tag\n",
            99,
        )

    def test_struct_literal_as_bare_statement(self):
        assert_program_stdout(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def int se():\n"
            "    print(99)\n"
            "    return 1\n"
            "\n"
            "def int main():\n"
            "    Point(se(), 2)\n"
            "    return 0\n",
            "99\n",
        )

    def test_struct_and_function_name_collision_is_rejected(self):
        assert_program_semantic_error(
            "type Point struct:\n"
            "    int x\n"
            "\n"
            "def int Point():\n"
            "    return 0\n"
            "\n"
            "def int main():\n"
            "    return 0\n",
            match="collides with a struct",
        )

    def test_struct_named_after_a_builtin_is_rejected(self):
        assert_program_semantic_error(
            "type print struct:\n"
            "    int x\n"
            "\n"
            "def int main():\n"
            "    return 0\n",
            match="builtin",
        )

    def test_print_a_struct_built_via_literal(self):
        assert_program_stdout(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def int main():\n"
            "    Point p = Point(3, 4)\n"
            "    print(p)\n"
            "    return 0\n",
            "Point(x: 3, y: 4)\n",
        )


# ---------------------------------------------------------------------------
# Argument materialization
# ---------------------------------------------------------------------------

class TestArgumentMaterialization:
    pytestmark = GCC_SKIP

    def test_array_literal_as_argument(self):
        assert_program_exit_code(
            "def int sum3([3]int arr):\n"
            "    return arr[0] + arr[1] + arr[2]\n"
            "\n"
            "def int main():\n"
            "    return sum3([1, 2, 3])\n",
            6,
        )

    def test_struct_literal_as_argument(self):
        assert_program_exit_code(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def int sumPoint(Point p):\n"
            "    return p.x + p.y\n"
            "\n"
            "def int main():\n"
            "    return sumPoint(Point(3, 4))\n",
            7,
        )

    def test_array_returning_call_as_argument(self):
        assert_program_exit_code(
            "def [3]int makeArr():\n"
            "    return [7, 8, 9]\n"
            "\n"
            "def int sum3([3]int arr):\n"
            "    return arr[0] + arr[1] + arr[2]\n"
            "\n"
            "def int main():\n"
            "    return sum3(makeArr())\n",
            24,
        )

    def test_struct_returning_call_as_argument(self):
        assert_program_exit_code(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def Point makePoint():\n"
            "    Point p\n"
            "    p.x = 3\n"
            "    p.y = 4\n"
            "    return p\n"
            "\n"
            "def int sumPoint(Point p):\n"
            "    return p.x + p.y\n"
            "\n"
            "def int main():\n"
            "    return sumPoint(makePoint())\n",
            7,
        )

    def test_two_array_literals_alive_in_the_same_call(self):
        assert_program_exit_code(
            "def int addPairs([2]int a, [2]int b):\n"
            "    return a[0] + a[1] + b[0] + b[1]\n"
            "\n"
            "def int main():\n"
            "    return addPairs([1, 2], [3, 4])\n",
            10,
        )

    def test_two_struct_literals_alive_in_the_same_call(self):
        assert_program_exit_code(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def int addPoints(Point a, Point b):\n"
            "    return a.x + a.y + b.x + b.y\n"
            "\n"
            "def int main():\n"
            "    return addPoints(Point(1, 2), Point(3, 4))\n",
            10,
        )

    def test_mixed_array_and_struct_literal_in_the_same_call(self):
        assert_program_exit_code(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def int mix([2]int a, Point p):\n"
            "    return a[0] + a[1] + p.x + p.y\n"
            "\n"
            "def int main():\n"
            "    return mix([1, 2], Point(3, 4))\n",
            10,
        )

    def test_literal_argument_nested_inside_another_call(self):
        assert_program_exit_code(
            "def int sum3([3]int arr):\n"
            "    return arr[0] + arr[1] + arr[2]\n"
            "\n"
            "def int addOne(int x):\n"
            "    return x + 1\n"
            "\n"
            "def int main():\n"
            "    return addOne(sum3([1, 2, 3]))\n",
            7,
        )

    def test_literal_argument_nested_inside_an_array_literal_element(self):
        assert_program_exit_code(
            "def int sum3([3]int arr):\n"
            "    return arr[0] + arr[1] + arr[2]\n"
            "\n"
            "def int main():\n"
            "    [2]int results = [sum3([1, 2, 3]), 4]\n"
            "    return results[0] + results[1]\n",
            10,
        )

    def test_large_array_literal_argument_is_heap_allocated(self):
        elements = ", ".join(str(i % 7) for i in range(5000))
        assert_program_exit_code(
            "def int sumFirstTwo([5000]int arr):\n"
            "    return arr[0] + arr[1]\n"
            "\n"
            "def int main():\n"
            f"    return sumFirstTwo([{elements}])\n",
            1,  # 0 % 7 + 1 % 7 = 0 + 1
        )

    def test_large_array_literal_argument_actually_uses_malloc(self):
        elements = ", ".join(str(i % 7) for i in range(5000))
        source = (
            "def int sumFirstTwo([5000]int arr):\n"
            "    return arr[0] + arr[1]\n"
            "\n"
            "def int main():\n"
            f"    return sumFirstTwo([{elements}])\n"
        )
        ast = _parse(source)
        analyze(ast)
        assert _heap_allocations(ast)
        main = next(f for f in _ir_program(ast).functions if f.name == 'main')
        main_frame_size = sum(main.slot_widths.values())
        assert main_frame_size < 1024, (
            f"main's own frame slots total {main_frame_size} bytes -- expected a "
            f"small frame, with the argument literal on the heap"
        )

    def test_large_struct_literal_argument_is_heap_allocated(self):
        assert_program_exit_code(
            "type Big struct:\n"
            "    [5000]int data\n"
            "    int tag\n"
            "\n"
            "def int useBig(Big b):\n"
            "    return b.tag\n"
            "\n"
            "def int main():\n"
            "    [5000]int filler\n"
            "    return useBig(Big(filler, 42))\n",
            42,
        )

    def test_large_struct_literal_argument_actually_uses_malloc(self):
        source = (
            "type Big struct:\n"
            "    [5000]int data\n"
            "    int tag\n"
            "\n"
            "def int useBig(Big b):\n"
            "    return b.tag\n"
            "\n"
            "def int main():\n"
            "    [5000]int filler\n"
            "    return useBig(Big(filler, 42))\n"
        )
        ast = _parse(source)
        analyze(ast)
        assert _heap_allocations(ast)

    def test_small_literal_argument_does_not_use_malloc(self):
        source = (
            "def int sum3([3]int arr):\n"
            "    return arr[0] + arr[1] + arr[2]\n"
            "\n"
            "def int main():\n"
            "    return sum3([1, 2, 3])\n"
        )
        ast = _parse(source)
        analyze(ast)
        mallocs = _heap_allocations(ast)
        assert not mallocs

    def test_nested_struct_literal_as_argument(self):
        assert_program_exit_code(
            "type Inner struct:\n"
            "    int v\n"
            "\n"
            "type Outer struct:\n"
            "    Inner i\n"
            "    int b\n"
            "\n"
            "def int useOuter(Outer o):\n"
            "    return o.i.v + o.b\n"
            "\n"
            "def int main():\n"
            "    return useOuter(Outer(Inner(9), 2))\n",
            11,
        )


# ---------------------------------------------------------------------------
# NAMED-field struct construction
# ---------------------------------------------------------------------------

class TestCompositeCallAsAddressableBase:
    """A composite-returning call used as a base: `makeArr()[i]`, `makePoint().x`."""

    pytestmark = GCC_SKIP

    def test_array_returning_call_indexed_directly(self):
        assert_program_exit_code(
            "def [3]int makeArr():\n"
            "    return [7, 8, 9]\n"
            "\n"
            "def int main():\n"
            "    return makeArr()[1]\n",
            8,
        )

    def test_struct_returning_call_field_accessed_directly(self):
        assert_program_exit_code(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def Point makePoint():\n"
            "    return Point(3, 4)\n"
            "\n"
            "def int main():\n"
            "    return makePoint().x + makePoint().y\n",
            7,
        )

    def test_slice_returning_call_indexed_directly(self):
        assert_program_exit_code(
            "def []int makeSlice([5]int arr):\n"
            "    return arr[1:4]\n"
            "\n"
            "def int main():\n"
            "    return makeSlice([10, 20, 30, 40, 50])[1]\n",
            30,
        )

    def test_slice_produced_from_array_returning_call(self):
        assert_program_exit_code(
            "def [5]int makeArr():\n"
            "    return [1, 2, 3, 4, 5]\n"
            "\n"
            "def int main():\n"
            "    []int s = makeArr()[1:4]\n"
            "    return s[0] + s[1] + s[2]\n",
            9,
        )

    def test_slice_from_call_survives_a_second_unrelated_materialization(self):
        assert_program_exit_code(
            "def [5]int makeArrayA():\n"
            "    return [1, 2, 3, 4, 5]\n"
            "\n"
            "def [5]int makeArrayB():\n"
            "    return [100, 200, 300, 400, 500]\n"
            "\n"
            "def int main():\n"
            "    []int s = makeArrayA()[1:4]\n"
            "    int unrelated = makeArrayB()[0]\n"
            "    return s[0] + s[1] + s[2] + unrelated\n",
            109,
        )

    def test_multiple_composite_calls_as_base_in_one_function(self):
        assert_program_exit_code(
            "def [3]int makeA():\n"
            "    return [1, 2, 3]\n"
            "\n"
            "def [3]int makeB():\n"
            "    return [10, 20, 30]\n"
            "\n"
            "def [3]int makeC():\n"
            "    return [100, 200, 300]\n"
            "\n"
            "def int main():\n"
            "    int a = makeA()[0]\n"
            "    int b = makeB()[1]\n"
            "    int c = makeC()[2]\n"
            "    return a + b + c\n",
            65,
        )

    def test_composite_call_as_base_inside_a_loop(self):
        assert_program_exit_code(
            "def [3]int makeArr(int seed):\n"
            "    return [seed, seed + 1, seed + 2]\n"
            "\n"
            "def int main():\n"
            "    int total = 0\n"
            "    int i = 0\n"
            "    while i < 1000:\n"
            "        total = total + makeArr(i)[0]\n"
            "        i = i + 1\n"
            "    return total % 256\n",
            44,
        )

    def test_small_array_returning_call_indexed_stays_on_stack(self):
        source = (
            "def [3]int makeSmall():\n"
            "    return [1, 2, 3]\n"
            "\n"
            "def int main():\n"
            "    return makeSmall()[0]\n"
        )
        ast = _parse(source)
        analyze(ast)
        mallocs = _heap_allocations(ast)
        assert not mallocs

    def test_large_array_returning_call_indexed_is_heap_allocated(self):
        n = 4097  # one int over the 16384-byte threshold
        source = (
            f"def [{n}]int makeBig():\n"
            f"    [{n}]int arr\n"
            f"    arr[0] = 1\n"
            f"    arr[{n - 1}] = 2\n"
            f"    return arr\n"
            f"\n"
            f"def int main():\n"
            f"    return makeBig()[0]\n"
        )
        ast = _parse(source)
        analyze(ast)
        mallocs = _heap_allocations(ast)
        assert mallocs

    def test_slice_production_from_call_always_heap_allocates_regardless_of_size(self):
        source = (
            "def [3]int makeTiny():\n"
            "    return [1, 2, 3]\n"
            "\n"
            "def int main():\n"
            "    []int s = makeTiny()[0:2]\n"
            "    return s[0]\n"
        )
        ast = _parse(source)
        analyze(ast)
        mallocs = _heap_allocations(ast)
        assert mallocs


class TestNamedStructLiterals:
    pytestmark = GCC_SKIP

    def test_named_construction_both_fields(self):
        assert_program_exit_code(
            "type A struct:\n"
            "    int x\n"
            "    str y\n"
            "\n"
            "def int main():\n"
            "    A a = A(x=1, y='hi')\n"
            "    if a.y == 'hi':\n"
            "        return a.x\n"
            "    return -1\n",
            1,
        )

    def test_named_fields_out_of_declaration_order(self):
        assert_program_exit_code(
            "type A struct:\n"
            "    int x\n"
            "    str y\n"
            "\n"
            "def int main():\n"
            "    A a = A(y='hi', x=7)\n"
            "    return a.x\n",
            7,
        )

    def test_partial_construction_first_field_only(self):
        assert_program_exit_code(
            "type A struct:\n"
            "    int x\n"
            "    str y\n"
            "\n"
            "def int main():\n"
            "    A a = A(x=42)\n"
            "    return a.x\n",
            42,
        )

    def test_partial_construction_second_field_only(self):
        assert_program_exit_code(
            "type A struct:\n"
            "    int x\n"
            "    str y\n"
            "\n"
            "def int main():\n"
            "    A a = A(y='hello')\n"
            "    if a.y == 'hello':\n"
            "        return 1\n"
            "    return 0\n",
            1,
        )

    def test_nested_named_struct_literal(self):
        assert_program_exit_code(
            "type Inner struct:\n"
            "    int v\n"
            "type Outer struct:\n"
            "    Inner i\n"
            "    int b\n"
            "\n"
            "def int main():\n"
            "    Outer o = Outer(i=Inner(v=9), b=2)\n"
            "    return o.i.v + o.b\n",
            11,
        )

    def test_named_construction_as_function_argument(self):
        assert_program_exit_code(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def int sumPoint(Point p):\n"
            "    return p.x + p.y\n"
            "\n"
            "def int main():\n"
            "    return sumPoint(Point(x=3, y=4))\n",
            7,
        )

    def test_named_construction_as_return_value(self):
        assert_program_exit_code(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def Point makePoint():\n"
            "    return Point(x=3, y=4)\n"
            "\n"
            "def int main():\n"
            "    Point p = makePoint()\n"
            "    return p.x + p.y\n",
            7,
        )

    def test_named_construction_as_index_assign_value(self):
        assert_program_exit_code(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def int main():\n"
            "    [2]Point pts\n"
            "    pts[0] = Point(x=1, y=2)\n"
            "    pts[1] = Point(y=4, x=3)\n"
            "    return pts[0].x + pts[1].y\n",
            5,
        )

    def test_named_construction_as_field_assign_value(self):
        assert_program_exit_code(
            "type Inner struct:\n"
            "    int v\n"
            "type Outer struct:\n"
            "    Inner i\n"
            "\n"
            "def int main():\n"
            "    Outer o\n"
            "    o.i = Inner(v=8)\n"
            "    return o.i.v\n",
            8,
        )

    def test_named_construction_as_array_literal_element(self):
        assert_program_exit_code(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def int main():\n"
            "    [2]Point pts = [Point(x=1, y=2), Point(x=3, y=4)]\n"
            "    return pts[0].x + pts[1].y\n",
            5,
        )

    def test_named_construction_with_array_typed_field(self):
        assert_program_exit_code(
            "type Row struct:\n"
            "    [3]int values\n"
            "\n"
            "def int main():\n"
            "    Row r = Row(values=[5, 6, 7])\n"
            "    return r.values[0] + r.values[1] + r.values[2]\n",
            18,
        )

    def test_named_construction_with_slice_typed_field(self):
        assert_program_exit_code(
            "type Holder struct:\n"
            "    []int xs\n"
            "\n"
            "def int main():\n"
            "    Holder h = Holder(xs=[]int[1, 2, 3])\n"
            "    return h.xs[0] + h.xs[1] + h.xs[2]\n",
            6,
        )

    def test_large_named_struct_literal(self):
        assert_program_exit_code(
            "type Big struct:\n"
            "    [5000]int data\n"
            "    int tag\n"
            "\n"
            "def int main():\n"
            "    [5000]int filler\n"
            "    Big b = Big(data=filler, tag=99)\n"
            "    return b.tag\n",
            99,
        )

    def test_unknown_field_name_is_rejected(self):
        assert_program_semantic_error(
            "type A struct:\n"
            "    int x\n"
            "    str y\n"
            "\n"
            "def int main():\n"
            "    A a = A(z=1)\n"
            "    return 0\n",
            match="has no field 'z'",
        )

    def test_duplicate_field_name_is_rejected(self):
        assert_program_semantic_error(
            "type A struct:\n"
            "    int x\n"
            "    str y\n"
            "\n"
            "def int main():\n"
            "    A a = A(x=1, x=2)\n"
            "    return 0\n",
            match="specified more than once",
        )

    def test_wrong_type_for_named_field_is_rejected(self):
        assert_program_semantic_error(
            "type A struct:\n"
            "    int x\n"
            "    str y\n"
            "\n"
            "def int main():\n"
            "    A a = A(x='wrong')\n"
            "    return 0\n",
            match="should be int",
        )

    def test_named_arguments_rejected_for_ordinary_function(self):
        assert_program_semantic_error(
            "def int foo(int x):\n"
            "    return x\n"
            "\n"
            "def int main():\n"
            "    return foo(x=1)\n",
            match="only supported for struct literals",
        )

    def test_named_arguments_rejected_for_print(self):
        assert_program_semantic_error(
            "type Point struct:\n"
            "    int x\n"
            "\n"
            "def int main():\n"
            "    print(x=1)\n"
            "    return 0\n",
            match="only supported for struct literals",
        )

    def test_positional_then_named_is_a_parse_error(self):
        source = (
            "type A struct:\n"
            "    int x\n"
            "    str y\n"
            "\n"
            "def int main():\n"
            "    A a = A(1, y='a')\n"
            "    return 0\n"
        )
        with pytest.raises(ParseError, match="Cannot mix positional and named"):
            _parse(source)

    def test_named_then_positional_is_a_parse_error(self):
        source = (
            "type A struct:\n"
            "    int x\n"
            "    str y\n"
            "\n"
            "def int main():\n"
            "    A a = A(x=1, 'a')\n"
            "    return 0\n"
        )
        with pytest.raises(ParseError, match="Cannot mix positional and named"):
            _parse(source)

    def test_double_equals_in_argument_is_not_mistaken_for_named(self):
        assert_program_exit_code(
            "type A struct:\n"
            "    bool x\n"
            "\n"
            "def int main():\n"
            "    int x = 1\n"
            "    A a = A(x == 1)\n"
            "    if a.x:\n"
            "        return 1\n"
            "    return 0\n",
            1,
        )


# ---------------------------------------------------------------------------
# Named construction zero-fill
# ---------------------------------------------------------------------------

class TestNamedStructLiteralZeroFill:
    pytestmark = GCC_SKIP

    def test_omitted_int_field(self):
        assert_program_exit_code(
            "type A struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def int main():\n"
            "    A a = A(x=5)\n"
            "    return a.x + a.y\n",
            5,
        )

    def test_omitted_bool_field(self):
        assert_program_exit_code(
            "type A struct:\n"
            "    int x\n"
            "    bool flag\n"
            "\n"
            "def int main():\n"
            "    A a = A(x=1)\n"
            "    if a.flag:\n"
            "        return 1\n"
            "    return 0\n",
            0,
        )

    def test_omitted_str_field(self):
        assert_program_stdout(
            "type Person struct:\n"
            "    int age\n"
            "    str name\n"
            "\n"
            "def int main():\n"
            "    Person p = Person(age=30)\n"
            "    print(p.name + 'x')\n"
            "    return 0\n",
            "x\n",
        )

    def test_omitted_slice_field(self):
        assert_program_exit_code(
            "type Holder struct:\n"
            "    int tag\n"
            "    []int xs\n"
            "\n"
            "def int main():\n"
            "    Holder h = Holder(tag=1)\n"
            "    return len(h.xs)\n",
            0,
        )

    def test_omitted_array_field(self):
        assert_program_exit_code(
            "type Row struct:\n"
            "    int tag\n"
            "    [3]int values\n"
            "\n"
            "def int main():\n"
            "    Row r = Row(tag=1)\n"
            "    return r.values[0] + r.values[1] + r.values[2]\n",
            0,
        )

    def test_omitted_struct_field(self):
        assert_program_exit_code(
            "type Inner struct:\n"
            "    int v\n"
            "type Outer struct:\n"
            "    int tag\n"
            "    Inner inner\n"
            "\n"
            "def int main():\n"
            "    Outer o = Outer(tag=1)\n"
            "    return o.inner.v\n",
            0,
        )

    def test_multiple_omitted_fields_interspersed_with_provided_ones(self):
        assert_program_exit_code(
            "type Five struct:\n"
            "    int a\n"
            "    int b\n"
            "    int c\n"
            "    int d\n"
            "    int e\n"
            "\n"
            "def int main():\n"
            "    Five f = Five(a=1, c=3, e=5)\n"
            "    return f.a + f.b + f.c + f.d + f.e\n",
            9,
        )

    def test_omitted_array_field_followed_by_a_provided_scalar_field(self):
        assert_program_exit_code(
            "type Triple struct:\n"
            "    [5000]int mid\n"
            "    int c\n"
            "\n"
            "def int main():\n"
            "    Triple t = Triple(c=7)\n"
            "    return t.mid[2500] + t.c\n",
            7,
        )

    def test_array_field_via_returning_call_followed_by_sibling_field(self):
        assert_program_exit_code(
            "def [5000]int makeArr():\n"
            "    [5000]int a\n"
            "    a[0] = 7\n"
            "    return a\n"
            "\n"
            "type Triple struct:\n"
            "    int a\n"
            "    [5000]int mid\n"
            "    int c\n"
            "\n"
            "def int main():\n"
            "    Triple t = Triple(1, makeArr(), 2)\n"
            "    return t.a + t.mid[0] + t.c\n",
            10,
        )

    def test_nested_named_literal_with_omitted_inner_field(self):
        assert_program_exit_code(
            "type Inner struct:\n"
            "    int a\n"
            "    int b\n"
            "type Outer struct:\n"
            "    Inner inner\n"
            "    int tag\n"
            "\n"
            "def int main():\n"
            "    Outer o = Outer(inner=Inner(a=1), tag=2)\n"
            "    return o.inner.a + o.inner.b + o.tag\n",
            3,
        )

    def test_all_but_one_field_omitted(self):
        assert_program_exit_code(
            "type Triple struct:\n"
            "    int a\n"
            "    int b\n"
            "    int c\n"
            "\n"
            "def int main():\n"
            "    Triple t = Triple(b=5)\n"
            "    return t.a + t.b + t.c\n",
            5,
        )

    def test_three_consecutive_omitted_composite_fields(self):
        assert_program_exit_code(
            "type Inner struct:\n"
            "    int v\n"
            "type Big struct:\n"
            "    [5000]int arr\n"
            "    Inner inner\n"
            "    []int sl\n"
            "    int tag\n"
            "\n"
            "def int main():\n"
            "    Big b = Big(tag=42)\n"
            "    return b.arr[100] + b.inner.v + len(b.sl) + b.tag\n",
            42,
        )


# ---------------------------------------------------------------------------
# Array/slice literals with struct-typed elements
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Struct literal array-field addresses
# ---------------------------------------------------------------------------

class TestStructLiteralArrayFieldAddressRegression:
    pytestmark = GCC_SKIP

    def test_array_returning_call_into_a_non_first_struct_field(self):
        assert_program_exit_code(
            "def [5000]int makeArr():\n"
            "    [5000]int a\n"
            "    a[4999] = 77\n"
            "    return a\n"
            "\n"
            "type Big struct:\n"
            "    int tag\n"
            "    [5000]int data\n"
            "\n"
            "def int main():\n"
            "    Big b = Big(1, makeArr())\n"
            "    return b.data[4999] % 256\n",
            77,
        )

    def test_preceding_field_is_not_corrupted(self):
        assert_program_exit_code(
            "def [5000]int makeArr():\n"
            "    [5000]int a\n"
            "    a[4999] = 77\n"
            "    return a\n"
            "\n"
            "type Big struct:\n"
            "    int tag\n"
            "    [5000]int data\n"
            "\n"
            "def int main():\n"
            "    Big b = Big(1, makeArr())\n"
            "    return b.tag\n",
            1,
        )

    def test_struct_returning_call_into_a_non_first_struct_field(self):
        assert_program_exit_code(
            "type Inner struct:\n"
            "    [5000]int data\n"
            "\n"
            "def Inner makeInner():\n"
            "    [5000]int a\n"
            "    a[4999] = 42\n"
            "    return Inner(a)\n"
            "\n"
            "type Outer struct:\n"
            "    int tag\n"
            "    Inner inner\n"
            "\n"
            "def int main():\n"
            "    Outer o = Outer(9, makeInner())\n"
            "    return o.tag + o.inner.data[4999]\n",
            51,
        )


# ---------------------------------------------------------------------------
# Implicit zero-value initialization
# ---------------------------------------------------------------------------

class TestImplicitZeroValue:
    pytestmark = GCC_SKIP

    def test_int_zero_value(self):
        assert_exit_code("    int a\n    return a", 0)

    def test_bool_zero_value(self):
        assert_exit_code(
            "    bool b\n"
            "    if b:\n"
            "        return 1\n"
            "    return 0",
            0,
        )

    def test_int8_zero_value(self):
        assert_exit_code("    int8 v\n    return int(v)", 0)

    def test_uint8_zero_value(self):
        assert_exit_code("    uint8 v\n    return int(v)", 0)

    def test_int64_zero_value(self):
        assert_exit_code("    int64 v\n    return int(v)", 0)

    def test_str_zero_value_prints_as_empty(self):
        assert_stdout("    str s\n    print(s)\n    return 0", "\n")

    def test_str_zero_value_is_a_real_string_not_a_null_pointer(self):
        assert_stdout(
            "    str s\n"
            "    str t = s + 'hi'\n"
            "    print(t)\n"
            "    return 0",
            "hi\n",
        )

    def test_str_zero_value_equals_empty_string_literal(self):
        assert_exit_code(
            "    str s\n"
            "    if s == '':\n"
            "        return 1\n"
            "    return 0",
            1,
        )

    def test_slice_zero_value_prints_as_empty_with_zero_length(self):
        assert_exit_code(
            "    []int s\n"
            "    print(s)\n"
            "    return len(s)",
            0,
        )
        assert_stdout("    []int s\n    print(s)\n    return 0", "[]int[]\n")

    def test_slice_zero_value_equals_none(self):
        assert_exit_code(
            "    []int s\n"
            "    if s == none:\n"
            "        return 1\n"
            "    return 0",
            1,
        )

    def test_append_to_a_zero_valued_slice(self):
        assert_exit_code(
            "    []int s\n"
            "    s = append(s, 42)\n"
            "    return s[0]",
            42,
        )

    def test_array_zero_value(self):
        assert_exit_code(
            "    [3]int arr\n"
            "    return arr[0] + arr[1] + arr[2]",
            0,
        )

    def test_large_heap_allocated_array_zero_value(self):
        assert_exit_code(
            "    [10000]int arr\n"
            "    int sum = 0\n"
            "    int i = 0\n"
            "    while i < 10000:\n"
            "        sum = sum + arr[i]\n"
            "        i = i + 1\n"
            "    return sum",
            0,
        )

    def test_array_of_str_zero_value(self):
        assert_stdout(
            "    [2]str names\n"
            "    print(names[0])\n"
            "    print(names[1])\n"
            "    return 0",
            "\n\n",
        )

    def test_array_of_slice_zero_value(self):
        assert_exit_code(
            "    [2][]int lists\n"
            "    return len(lists[0]) + len(lists[1])",
            0,
        )

    def test_struct_with_scalar_fields_zero_value(self):
        assert_program_exit_code(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def int main():\n"
            "    Point p\n"
            "    return p.x + p.y\n",
            0,
        )

    def test_struct_with_str_field_zero_value(self):
        assert_program_stdout(
            "type Person struct:\n"
            "    str name\n"
            "    int age\n"
            "\n"
            "def int main():\n"
            "    Person p\n"
            "    print(p.name)\n"
            "    return 0\n",
            "\n",
        )

    def test_struct_with_slice_field_zero_value(self):
        assert_program_stdout(
            "type Holder struct:\n"
            "    []int xs\n"
            "\n"
            "def int main():\n"
            "    Holder h\n"
            "    print(h.xs)\n"
            "    return 0\n",
            "[]int[]\n",
        )

    def test_nested_struct_field_zero_value(self):
        assert_program_exit_code(
            "type Inner struct:\n"
            "    int v\n"
            "type Outer struct:\n"
            "    Inner i\n"
            "    int b\n"
            "\n"
            "def int main():\n"
            "    Outer o\n"
            "    return o.i.v + o.b\n",
            0,
        )

    def test_heap_allocated_struct_with_non_first_array_field(self):
        assert_program_exit_code(
            "type Big struct:\n"
            "    int tag\n"
            "    [5000]int data\n"
            "\n"
            "def int main():\n"
            "    Big b\n"
            "    return b.tag + b.data[4999]\n",
            0,
        )

    def test_scalar_array_scalar_sibling_fields_survive_zero_init_stack(self):
        assert_program_exit_code(
            "type Triple struct:\n"
            "    int a\n"
            "    [3]int mid\n"
            "    int c\n"
            "\n"
            "def int main():\n"
            "    Triple t\n"
            "    t.a = 1\n"
            "    t.c = 2\n"
            "    return t.a + t.mid[1] + t.c\n",
            3,
        )

    def test_scalar_array_scalar_sibling_fields_survive_zero_init_heap(self):
        assert_program_exit_code(
            "type Triple struct:\n"
            "    int a\n"
            "    [5000]int mid\n"
            "    int c\n"
            "\n"
            "def int main():\n"
            "    Triple t\n"
            "    t.a = 1\n"
            "    t.c = 2\n"
            "    return t.a + t.mid[2500] + t.c\n",
            3,
        )

    def test_array_of_structs_zero_value(self):
        assert_program_exit_code(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def int main():\n"
            "    [3]Point pts\n"
            "    return pts[0].x + pts[1].y + pts[2].x\n",
            0,
        )

    def test_array_of_structs_with_str_field(self):
        assert_program_stdout(
            "type Person struct:\n"
            "    str name\n"
            "    int age\n"
            "\n"
            "def int main():\n"
            "    [2]Person people\n"
            "    print(people[0].name)\n"
            "    print(people[1].name)\n"
            "    return 0\n",
            "\n\n",
        )

    def test_array_of_structs_with_array_field(self):
        assert_program_exit_code(
            "type Bag struct:\n"
            "    [10]int items\n"
            "\n"
            "def int main():\n"
            "    [2]Bag bags\n"
            "    bags[0].items[5] = 99\n"
            "    return bags[0].items[5] + bags[1].items[5] + bags[1].items[0]\n",
            99,
        )

    def test_doubly_nested_zero_value(self):
        assert_program_stdout(
            "type Item struct:\n"
            "    str name\n"
            "    int qty\n"
            "type Container struct:\n"
            "    int tag\n"
            "    [2]Item items\n"
            "\n"
            "def int main():\n"
            "    Container c\n"
            "    print(c.items[0].name)\n"
            "    return 0\n",
            "\n",
        )

    def test_function_parameters_are_never_implicitly_zeroed(self):
        assert_program_exit_code(
            "def int identity(int x):\n"
            "    return x\n"
            "\n"
            "def int main():\n"
            "    return identity(7)\n",
            7,
        )

    def test_explicit_array_initializer_on_heap_allocated_var_still_works(self):
        elements = ", ".join(str(1 if i in (0, 1, 2, 4999) else 0) for i in range(5000))
        assert_exit_code(
            f"    [5000]int arr = [{elements}]\n"
            f"    return arr[0] + arr[1] + arr[2] + arr[4999]",
            4,
        )

    def test_explicit_struct_initializer_on_heap_allocated_var_still_works(self):
        assert_program_exit_code(
            "type Big struct:\n"
            "    int tag\n"
            "    [5000]int data\n"
            "\n"
            "def int main():\n"
            "    [5000]int filler\n"
            "    filler[0] = 5\n"
            "    Big b = Big(9, filler)\n"
            "    return b.tag + b.data[0]\n",
            14,
        )

    def test_partial_named_struct_literal_now_zero_fills(self):
        assert_program_exit_code(
            "type A struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def int main():\n"
            "    A a = A(x=5)\n"
            "    return a.x + a.y\n",
            5,
        )


class TestArraysOfStructs:
    pytestmark = GCC_SKIP

    def test_array_literal_of_struct_variables(self):
        assert_program_exit_code(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def int main():\n"
            "    Point p1\n"
            "    p1.x = 1\n"
            "    p1.y = 2\n"
            "    Point p2\n"
            "    p2.x = 3\n"
            "    p2.y = 4\n"
            "    [2]Point arr = [p1, p2]\n"
            "    return arr[0].x + arr[1].y\n",
            5,
        )

    def test_array_literal_of_struct_literals(self):
        assert_program_exit_code(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def int main():\n"
            "    [2]Point pts = [Point(1, 2), Point(3, 4)]\n"
            "    return pts[0].x + pts[1].y\n",
            5,
        )

    def test_fully_typed_array_literal_of_struct_literals(self):
        assert_program_exit_code(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def int main():\n"
            "    [2]Point pts = [2]Point[Point(1, 2), Point(3, 4)]\n"
            "    return pts[0].x + pts[1].y\n",
            5,
        )

    def test_slice_typed_literal_of_struct_literals(self):
        assert_program_exit_code(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def int main():\n"
            "    []Point pts = []Point[Point(1, 2), Point(3, 4)]\n"
            "    return pts[0].x + pts[1].y\n",
            5,
        )

    def test_nested_typed_array_of_struct_literals(self):
        assert_program_exit_code(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def int main():\n"
            "    [1][2]Point grid = [1][2]Point[[2]Point[Point(1,2), Point(3,4)]]\n"
            "    return grid[0][0].x + grid[0][1].y\n",
            5,
        )

    def test_mixed_element_kinds_in_one_array_literal(self):
        assert_program_exit_code(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def Point makePoint():\n"
            "    Point p\n"
            "    p.x = 10\n"
            "    p.y = 20\n"
            "    return p\n"
            "\n"
            "def int main():\n"
            "    Point p2\n"
            "    p2.x = 5\n"
            "    p2.y = 6\n"
            "    [3]Point pts = [Point(1, 2), p2, makePoint()]\n"
            "    return pts[0].x + pts[1].y + pts[2].x\n",
            17,
        )

    def test_indexed_struct_element_in_array_literal(self):
        assert_program_exit_code(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def int main():\n"
            "    [2]Point base = [Point(1, 2), Point(3, 4)]\n"
            "    [2]Point copy = [base[0], base[1]]\n"
            "    return copy[0].x + copy[1].y\n",
            5,
        )

    def test_struct_field_that_is_an_array_of_structs_built_via_nested_literal(self):
        assert_program_exit_code(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "type Pair struct:\n"
            "    [2]Point pts\n"
            "\n"
            "def int main():\n"
            "    Pair pr = Pair([Point(1, 2), Point(3, 4)])\n"
            "    return pr.pts[0].x + pr.pts[1].y\n",
            5,
        )

    def test_value_semantics_mutating_a_copy_does_not_affect_the_original(self):
        assert_program_exit_code(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def int main():\n"
            "    [2]Point a = [Point(1, 2), Point(3, 4)]\n"
            "    [2]Point b = a\n"
            "    b[0].x = 100\n"
            "    return a[0].x + b[0].x\n",
            101,
        )

    def test_large_array_of_structs_literal_is_heap_allocated(self):
        elements = ", ".join(f"Point({i}, {i})" for i in range(2500))
        assert_program_exit_code(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def int main():\n"
            f"    [2500]Point pts = [{elements}]\n"
            "    return pts[2499].x % 256\n",
            2499 % 256,
        )

    def test_large_array_of_structs_literal_actually_uses_malloc(self):
        elements = ", ".join(f"Point({i}, {i})" for i in range(2500))
        source = (
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def int main():\n"
            f"    [2500]Point pts = [{elements}]\n"
            "    return pts[2499].x % 256\n"
        )
        ast = _parse(source)
        analyze(ast)
        assert _heap_allocations(ast)

    def test_small_array_of_structs_literal_does_not_use_malloc(self):
        source = (
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def int main():\n"
            "    [2]Point pts = [Point(1, 2), Point(3, 4)]\n"
            "    return pts[0].x + pts[1].y\n"
        )
        ast = _parse(source)
        analyze(ast)
        mallocs = _heap_allocations(ast)
        assert not mallocs

    def test_mismatched_struct_types_in_array_literal_is_rejected(self):
        assert_program_semantic_error(
            "type Point struct:\n"
            "    int x\n"
            "type Other struct:\n"
            "    int y\n"
            "\n"
            "def int main():\n"
            "    Point p = Point(1)\n"
            "    Other o = Other(2)\n"
            "    [2]Point pts = [p, o]\n"
            "    return 0\n",
            match="must all be",
        )

    def test_struct_and_scalar_mixed_in_untyped_array_literal_is_rejected(self):
        assert_program_semantic_error(
            "type Point struct:\n"
            "    int x\n"
            "\n"
            "def int main():\n"
            "    [2]Point pts = [Point(1), 5]\n"
            "    return 0\n",
            match="elements must all be .*to match the declared element type",
        )

    def test_bare_typed_struct_array_literal_statement_side_effect_is_still_rejected(self):
        source = (
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def int main():\n"
            "    [2]Point[Point(1, 2), Point(3, 4)]\n"
            "    return 0\n"
        )
        ast = _parse(source)
        analyze(ast)
        with pytest.raises(IRError, match="assign the literal to a variable first"):
            generate_asm(ast, target=ASM_TARGET)

    def test_single_element_array_literal_still_parses_as_untyped(self):
        assert_program_exit_code(
            "def int main():\n"
            "    [1]int a = [5]\n"
            "    return a[0]\n",
            5,
        )


# ---------------------------------------------------------------------------
# Printing structs
# ---------------------------------------------------------------------------

class TestPrintStructs:
    pytestmark = GCC_SKIP

    def test_basic_struct(self):
        assert_program_stdout(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def int main():\n"
            "    Point p\n"
            "    p.x = 1\n"
            "    p.y = 2\n"
            "    print(p)\n"
            "    return 0\n",
            "Point(x: 1, y: 2)\n",
        )

    def test_single_field_struct(self):
        assert_program_stdout(
            "type Wrapper struct:\n"
            "    int value\n"
            "\n"
            "def int main():\n"
            "    Wrapper w\n"
            "    w.value = 42\n"
            "    print(w)\n"
            "    return 0\n",
            "Wrapper(value: 42)\n",
        )

    def test_struct_with_bool_field(self):
        assert_program_stdout(
            "type Flag struct:\n"
            "    bool on\n"
            "\n"
            "def int main():\n"
            "    Flag f\n"
            "    f.on = true\n"
            "    print(f)\n"
            "    return 0\n",
            "Flag(on: true)\n",
        )

    def test_struct_with_str_field_is_quoted(self):
        assert_program_stdout(
            "type Person struct:\n"
            "    str name\n"
            "    int age\n"
            "\n"
            "def int main():\n"
            "    Person p\n"
            "    p.name = 'alice'\n"
            "    p.age = 30\n"
            "    print(p)\n"
            "    return 0\n",
            "Person(name: 'alice', age: 30)\n",
        )

    def test_nested_struct(self):
        assert_program_stdout(
            "type Inner struct:\n"
            "    int v\n"
            "\n"
            "type Outer struct:\n"
            "    Inner inner\n"
            "\n"
            "def int main():\n"
            "    Outer o\n"
            "    o.inner.v = 99\n"
            "    print(o)\n"
            "    return 0\n",
            "Outer(inner: Inner(v: 99))\n",
        )

    def test_struct_with_array_field(self):
        assert_program_stdout(
            "type Row struct:\n"
            "    [3]int values\n"
            "\n"
            "def int main():\n"
            "    Row r\n"
            "    r.values = [1, 2, 3]\n"
            "    print(r)\n"
            "    return 0\n",
            "Row(values: [3]int[1, 2, 3])\n",
        )

    def test_struct_with_slice_field(self):
        assert_program_stdout(
            "type Row struct:\n"
            "    []int values\n"
            "\n"
            "def int main():\n"
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    Row r\n"
            "    r.values = arr[0:3]\n"
            "    print(r)\n"
            "    return 0\n",
            "Row(values: []int[1, 2, 3])\n",
        )

    def test_struct_with_empty_slice_field(self):
        assert_program_stdout(
            "type Row struct:\n"
            "    []int values\n"
            "\n"
            "def int main():\n"
            "    Row r\n"
            "    r.values = none\n"
            "    print(r)\n"
            "    return 0\n",
            "Row(values: []int[])\n",
        )

    def test_array_of_structs(self):
        assert_program_stdout(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def int main():\n"
            "    [2]Point pts\n"
            "    pts[0].x = 1\n"
            "    pts[0].y = 2\n"
            "    pts[1].x = 3\n"
            "    pts[1].y = 4\n"
            "    print(pts)\n"
            "    return 0\n",
            "[2]Point[Point(x: 1, y: 2), Point(x: 3, y: 4)]\n",
        )

    def test_slice_of_structs(self):
        assert_program_stdout(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def int main():\n"
            "    [2]Point arr\n"
            "    arr[0].x = 1\n"
            "    arr[0].y = 2\n"
            "    arr[1].x = 3\n"
            "    arr[1].y = 4\n"
            "    []Point s = arr[0:2]\n"
            "    print(s)\n"
            "    return 0\n",
            "[]Point[Point(x: 1, y: 2), Point(x: 3, y: 4)]\n",
        )

    def test_struct_field_access_as_print_argument(self):
        assert_program_stdout(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "type Container struct:\n"
            "    Point p\n"
            "\n"
            "def int main():\n"
            "    Container c\n"
            "    c.p.x = 5\n"
            "    c.p.y = 6\n"
            "    print(c.p)\n"
            "    return 0\n",
            "Point(x: 5, y: 6)\n",
        )

    def test_struct_array_index_as_print_argument(self):
        assert_program_stdout(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def int main():\n"
            "    [2]Point pts\n"
            "    pts[0].x = 7\n"
            "    pts[0].y = 8\n"
            "    print(pts[0])\n"
            "    return 0\n",
            "Point(x: 7, y: 8)\n",
        )

    def test_self_referential_struct_tree(self):
        assert_program_stdout(
            "type Node struct:\n"
            "    int value\n"
            "    []Node children\n"
            "\n"
            "def int main():\n"
            "    [2]Node kids\n"
            "    kids[0].value = 2\n"
            "    kids[0].children = none\n"
            "    kids[1].value = 3\n"
            "    kids[1].children = none\n"
            "    Node root\n"
            "    root.value = 1\n"
            "    root.children = kids[0:2]\n"
            "    print(root)\n"
            "    return 0\n",
            "Node(value: 1, children: []Node[Node(value: 2, children: []Node[]), Node(value: 3, children: []Node[])])\n",
        )

    def test_multiple_struct_prints_each_get_exactly_one_newline(self):
        assert_program_stdout(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def int main():\n"
            "    Point a\n"
            "    a.x = 1\n"
            "    a.y = 2\n"
            "    Point b\n"
            "    b.x = 3\n"
            "    b.y = 4\n"
            "    print(a)\n"
            "    print(b)\n"
            "    return 0\n",
            "Point(x: 1, y: 2)\nPoint(x: 3, y: 4)\n",
        )

    def test_struct_returning_call_as_direct_print_argument(self):
        assert_program_stdout(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def Point makePoint():\n"
            "    Point p\n"
            "    p.x = 1\n"
            "    p.y = 2\n"
            "    return p\n"
            "\n"
            "def int main():\n"
            "    print(makePoint())\n"
            "    return 0\n",
            "Point(x: 1, y: 2)\n",
        )


class TestPrintStructLiterals:
    """`print(Circle(5))` with a literal argument."""

    pytestmark = GCC_SKIP

    def test_basic_struct_literal(self):
        assert_program_stdout(
            "type Circle struct:\n"
            "    int radius\n"
            "\n"
            "def int main():\n"
            "    print(Circle(5))\n"
            "    return 0\n",
            "Circle(radius: 5)\n",
        )

    def test_multi_field_struct_literal(self):
        assert_program_stdout(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def int main():\n"
            "    print(Point(3, 4))\n"
            "    return 0\n",
            "Point(x: 3, y: 4)\n",
        )

    def test_named_argument_struct_literal(self):
        assert_program_stdout(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def int main():\n"
            "    print(Point(x=9, y=8))\n"
            "    return 0\n",
            "Point(x: 9, y: 8)\n",
        )

    def test_nested_struct_literal(self):
        assert_program_stdout(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "type Rectangle struct:\n"
            "    Point topLeft\n"
            "    int width\n"
            "    int height\n"
            "\n"
            "def int main():\n"
            "    print(Rectangle(Point(1, 2), 10, 20))\n"
            "    return 0\n",
            "Rectangle(topLeft: Point(x: 1, y: 2), width: 10, height: 20)\n",
        )

    def test_struct_literal_field_is_a_call_expression(self):
        assert_program_stdout(
            "type Circle struct:\n"
            "    int radius\n"
            "\n"
            "def int makeRadius():\n"
            "    return 42\n"
            "\n"
            "def int main():\n"
            "    print(Circle(makeRadius()))\n"
            "    return 0\n",
            "Circle(radius: 42)\n",
        )

    def test_multiple_struct_literal_print_calls_in_one_function(self):
        assert_program_stdout(
            "type Circle struct:\n"
            "    int radius\n"
            "\n"
            "def int main():\n"
            "    print(Circle(1))\n"
            "    print(Circle(2))\n"
            "    return 0\n",
            "Circle(radius: 1)\nCircle(radius: 2)\n",
        )

    def test_struct_literal_as_a_field_access_base_works(self):
        assert_program_stdout(
            "type Circle struct:\n"
            "    int radius\n"
            "\n"
            "def int main():\n"
            "    print(Circle(5).radius)\n"
            "    return 0\n",
            "5\n",
        )

    def test_struct_literal_field_assign_base_is_allowed_though_pointless(self):
        assert_program_exit_code(
            "type Circle struct:\n"
            "    int radius\n"
            "\n"
            "def int main():\n"
            "    Circle(10).radius = 999\n"
            "    return 0\n",
            0,
        )

    def test_struct_literal_as_an_index_base_is_still_rejected(self):
        assert_program_semantic_error(
            "type Circle struct:\n"
            "    int radius\n"
            "\n"
            "def int main():\n"
            "    return Circle(5)[0]\n",
            match="is a struct literal, which is only allowed",
        )

    def test_struct_literal_as_a_binary_operand_works(self):
        assert_program_exit_code(
            "type Circle struct:\n"
            "    int radius\n"
            "\n"
            "def int main():\n"
            "    Circle c = Circle(5)\n"
            "    if c == Circle(5):\n"
            "        return 1\n"
            "    return 0\n",
            1,
        )

    def test_struct_literal_as_a_binary_operand_on_the_left_works(self):
        assert_program_exit_code(
            "type Circle struct:\n"
            "    int radius\n"
            "\n"
            "def int main():\n"
            "    Circle c = Circle(5)\n"
            "    if Circle(5) == c:\n"
            "        return 1\n"
            "    return 0\n",
            1,
        )

    def test_two_struct_literals_compared_directly_works(self):
        assert_program_exit_code(
            "type Circle struct:\n"
            "    int radius\n"
            "\n"
            "def int main():\n"
            "    if Circle(5) == Circle(5):\n"
            "        return 1\n"
            "    return 0\n",
            1,
        )


# ---------------------------------------------------------------------------
# AST pretty-printing
# ---------------------------------------------------------------------------
class TestCompoundAssignmentThroughAddresses:
    """`arr[i] += 1`, `s.field += 1`, `*p += 1`."""

    def test_compound_index_assignment(self):
        assert_program_exit_code(
            "def int main():\n"
            "    [3]int arr = [1, 2, 3]\n"
            "    arr[0] += 10\n"
            "    arr[1] -= 1\n"
            "    arr[2] *= 3\n"
            "    return arr[0] + arr[1] + arr[2]\n",
            expected=11 + 1 + 9,
        )

    def test_compound_field_assignment(self):
        assert_program_exit_code(
            "type Circle struct:\n"
            "    int radius\n"
            "\n"
            "def int main():\n"
            "    Circle c = Circle(5)\n"
            "    c.radius += 10\n"
            "    return c.radius\n",
            expected=15,
        )

    def test_compound_deref_assignment(self):
        assert_program_exit_code(
            "def int main():\n"
            "    int x = 5\n"
            "    *int p = &x\n"
            "    *p += 10\n"
            "    return x\n",
            expected=15,
        )

    def test_index_expression_is_evaluated_exactly_once(self):
        assert_program_exit_code(
            "def int nextIndex(*int counter):\n"
            "    int current = *counter\n"
            "    *counter = current + 1\n"
            "    return current\n"
            "\n"
            "def int main():\n"
            "    [3]int arr = [10, 20, 30]\n"
            "    int counter = 0\n"
            "    *int p = &counter\n"
            "    arr[nextIndex(p)] += 100\n"
            "    return counter\n",
            expected=1,
        )

    def test_field_base_index_expression_is_evaluated_exactly_once(self):
        assert_program_exit_code(
            "type Circle struct:\n"
            "    int radius\n"
            "\n"
            "def int nextIndex(*int counter):\n"
            "    int current = *counter\n"
            "    *counter = current + 1\n"
            "    return current\n"
            "\n"
            "def int main():\n"
            "    [2]Circle circles = [Circle(1), Circle(2)]\n"
            "    int counter = 0\n"
            "    *int p = &counter\n"
            "    circles[nextIndex(p)].radius += 100\n"
            "    return counter\n",
            expected=1,
        )

    def test_compound_assignment_to_a_bool_element_is_rejected(self):
        assert_program_semantic_error(
            "def int main():\n"
            "    [3]bool arr = [true, false, true]\n"
            "    arr[0] += true\n"
            "    return 0\n",
            match="requires two operands of the same integer type",
        )

    def test_compound_assignment_to_a_str_element_is_rejected(self):
        assert_program_semantic_error(
            "def int main():\n"
            "    []str arr = ['a', 'b']\n"
            "    arr[0] += 'c'\n"
            "    return 0\n",
            match="Compound assignment \\('\\+='\\) to a str-typed target",
        )

    def test_compound_assignment_to_a_str_field_is_rejected(self):
        assert_program_semantic_error(
            "type Holder struct:\n"
            "    str s\n"
            "\n"
            "def int main():\n"
            "    Holder h = Holder('a')\n"
            "    h.s += 'b'\n"
            "    return 0\n",
            match="Compound assignment \\('\\+='\\) to a str-typed target",
        )

    def test_all_ten_compound_operators_parse_and_run(self):
        assert_program_exit_code(
            "def int main():\n"
            "    [10]int arr = [12, 12, 12, 12, 12, 12, 12, 12, 12, 12]\n"
            "    arr[0] += 4\n"
            "    arr[1] -= 4\n"
            "    arr[2] *= 4\n"
            "    arr[3] /= 4\n"
            "    arr[4] %= 5\n"
            "    arr[5] &= 4\n"
            "    arr[6] |= 3\n"
            "    arr[7] ^= 4\n"
            "    arr[8] <<= 2\n"
            "    arr[9] >>= 2\n"
            "    int total = 0\n"
            "    int i = 0\n"
            "    while i < 10:\n"
            "        total = total + arr[i]\n"
            "        i = i + 1\n"
            "    return total\n",
            expected=16 + 8 + 48 + 3 + 2 + 4 + 15 + 8 + 48 + 3,
        )


class TestASTPrettyPrinting:
    def test_leaf_node_renders_compactly(self):
        assert Constant(value=1).pretty() == "Constant(value=1)"

    def test_zero_field_node_renders_with_no_arguments(self):
        assert Break().pretty() == "Break()"

    def test_short_binary_expression_stays_on_one_line(self):
        ast = _parse("def int main():\n    return a + b\n")
        return_stmt = ast.functions[0].body[0]
        assert return_stmt.pretty() == (
            "Return(value=Binary(op='+', left=Variable(name='a'), right=Variable(name='b')))"
        )

    def test_operator_symbol_is_quoted_not_bare(self):
        ast = _parse("def int main():\n    return a == b\n")
        return_stmt = ast.functions[0].body[0]
        assert return_stmt.pretty() == (
            "Return(value=Binary(op='==', left=Variable(name='a'), right=Variable(name='b')))"
        )

    def test_nested_expression_expands_into_an_indented_tree(self):
        ast = _parse(
            "def int main():\n"
            "    if x < y and (y * 2 + 1) > x:\n"
            "        return 1\n"
            "    return 0\n"
        )
        if_stmt = ast.functions[0].body[0]
        assert if_stmt.pretty() == (
            "If(\n"
            "    condition=Binary(\n"
            "        op='and',\n"
            "        left=Binary(op='<', left=Variable(name='x'), right=Variable(name='y')),\n"
            "        right=Binary(\n"
            "            op='>',\n"
            "            left=Binary(\n"
            "                op='+',\n"
            "                left=Binary(op='*', left=Variable(name='y'), right=Constant(value=2)),\n"
            "                right=Constant(value=1),\n"
            "            ),\n"
            "            right=Variable(name='x'),\n"
            "        ),\n"
            "    ),\n"
            "    then_body=[Return(value=Constant(value=1))],\n"
            "    else_body=None,\n"
            "    is_match=False,\n"
            "    match_arm_count=None,\n"
            ")"
        )

    def test_empty_list_renders_as_bare_brackets(self):
        ast = _parse("def int empty():\n    return 0\n")
        fn = ast.functions[0]
        assert "params=[]" in fn.pretty()

    def test_resolved_type_is_never_shown(self):
        ast = _parse("def int main():\n    return 1 + 2\n")
        analyze(ast)
        return_stmt = ast.functions[0].body[0]
        assert return_stmt.value.resolved_type is not None  # analysis really did run
        assert "resolved_type" not in return_stmt.pretty()

    def test_self_referential_struct_field_type_renders_correctly(self):
        ast = _parse(
            "type Node struct:\n"
            "    int value\n"
            "    []Node children\n"
            "\n"
            "def int main():\n"
            "    return 0\n"
        )
        struct_def = ast.structs[0]
        assert struct_def.pretty() == (
            "StructDef(\n"
            "    name='Node',\n"
            "    fields=[\n"
            "        StructField(name='value', field_type='int'),\n"
            "        StructField(name='children', field_type=SliceTypeExpr(element_type='Node')),\n"
            "    ],\n"
            "    methods=[],\n"
            ")"
        )

    def test_new_node_subclass_needs_no_pretty_method_of_its_own(self):
        @dataclass
        class _FakeFutureNode(Node):
            label: str
            payload: Optional[Node] = None

        node = _FakeFutureNode(label='widget', payload=_FakeFutureNode(label='inner'))
        assert node.pretty() == (
            "_FakeFutureNode(label='widget', payload=_FakeFutureNode(label='inner', payload=None))"
        )


# ---------------------------------------------------------------------------
# for x in y / for x, y in z
# ---------------------------------------------------------------------------

class TestForInParsing:
    def test_single_binding_parses_into_forin(self):
        ast = _parse(
            "def int main():\n"
            "    for x in arr:\n"
            "        print(x)\n"
            "    return 0\n"
        )
        stmt = ast.functions[0].body[0]
        assert isinstance(stmt, ForIn)
        assert stmt.binding_names == ['x']

    def test_two_binding_parses_into_forin(self):
        ast = _parse(
            "def int main():\n"
            "    for i, x in arr:\n"
            "        print(x)\n"
            "    return 0\n"
        )
        stmt = ast.functions[0].body[0]
        assert isinstance(stmt, ForIn)
        assert stmt.binding_names == ['i', 'x']

    def test_three_clause_for_still_parses_into_for(self):
        ast = _parse(
            "def int main():\n"
            "    for int i = 0; i < 10; i += 1:\n"
            "        print(i)\n"
            "    return 0\n"
        )
        stmt = ast.functions[0].body[0]
        assert isinstance(stmt, For)

    def test_struct_typed_three_clause_init_still_disambiguates_correctly(self):
        ast = _parse(
            "def int main():\n"
            "    for Point p = Point(1, 2); i < 10; i += 1:\n"
            "        print(p.x)\n"
            "    return 0\n"
        )
        stmt = ast.functions[0].body[0]
        assert isinstance(stmt, For)

    def test_forin_iterable_parses_as_an_ordinary_expression(self):
        ast = _parse(
            "def int main():\n"
            "    for x in make_arr():\n"
            "        print(x)\n"
            "    return 0\n"
        )
        stmt = ast.functions[0].body[0]
        assert isinstance(stmt, ForIn)
        assert isinstance(stmt.iterable, Call)

    def test_forin_body_supports_break_and_continue(self):
        ast = _parse(
            "def int main():\n"
            "    for x in arr:\n"
            "        if x == 5:\n"
            "            break\n"
            "        continue\n"
            "    return 0\n"
        )
        stmt = ast.functions[0].body[0]
        assert isinstance(stmt, ForIn)
        assert isinstance(stmt.body[0].then_body[0], Break)
        assert isinstance(stmt.body[1], Continue)

    def test_forin_missing_in_is_rejected(self):
        with pytest.raises(ParseError):
            _parse(
                "def int main():\n"
                "    for x:\n"
                "        print(x)\n"
                "    return 0\n"
            )


# ---------------------------------------------------------------------------
# for x in y / for x, y in z
# ---------------------------------------------------------------------------

class TestForInSemantics:
    def test_array_single_binding_analyzes_correctly(self):
        ast = _parse(
            "def int main():\n"
            "    [3]int arr = [1, 2, 3]\n"
            "    for x in arr:\n"
            "        print(x)\n"
            "    return 0\n"
        )
        analyze(ast)  # should not raise

    def test_array_two_binding_analyzes_correctly(self):
        ast = _parse(
            "def int main():\n"
            "    [3]int arr = [1, 2, 3]\n"
            "    for i, x in arr:\n"
            "        print(i)\n"
            "        print(x)\n"
            "    return 0\n"
        )
        analyze(ast)  # should not raise

    def test_slice_single_binding_analyzes_correctly(self):
        ast = _parse(
            "def int main():\n"
            "    []int s = [1, 2, 3]\n"
            "    for x in s:\n"
            "        print(x)\n"
            "    return 0\n"
        )
        analyze(ast)  # should not raise

    def test_dict_single_binding_analyzes_correctly(self):
        ast = _parse(
            "def int main():\n"
            "    dict[str]int d = dict[str]int{'a': 1}\n"
            "    for k in d:\n"
            "        print(k)\n"
            "    return 0\n"
        )
        analyze(ast)  # should not raise

    def test_dict_two_binding_analyzes_correctly(self):
        ast = _parse(
            "def int main():\n"
            "    dict[str]int d = dict[str]int{'a': 1}\n"
            "    for k, v in d:\n"
            "        print(k)\n"
            "        print(v)\n"
            "    return 0\n"
        )
        analyze(ast)  # should not raise

    def test_bare_call_iterable_is_rejected(self):
        ast = _parse(
            "def []int make():\n"
            "    []int s = [1, 2, 3]\n"
            "    return s\n"
            "\n"
            "def int main():\n"
            "    for x in make():\n"
            "        print(x)\n"
            "    return 0\n"
        )
        with pytest.raises(
            SemanticError,
            match="'for ... in' does not support a function call result",
        ):
            analyze(ast)

    def test_append_call_as_iterable_is_rejected(self):
        ast = _parse(
            "def int main():\n"
            "    []int s = [1, 2, 3]\n"
            "    for x in append(s, 4):\n"
            "        print(x)\n"
            "    return 0\n"
        )
        with pytest.raises(
            SemanticError,
            match="'for ... in' does not support a function call result",
        ):
            analyze(ast)

    def test_field_rooted_in_call_iterable_is_rejected(self):
        ast = _parse(
            "type Box struct:\n"
            "    []int items\n"
            "\n"
            "def Box makeBox():\n"
            "    return Box([1, 2, 3])\n"
            "\n"
            "def int main():\n"
            "    for x in makeBox().items:\n"
            "        print(x)\n"
            "    return 0\n"
        )
        with pytest.raises(
            SemanticError,
            match="'for ... in' does not support a function call result",
        ):
            analyze(ast)

    def test_index_rooted_in_call_iterable_is_rejected(self):
        ast = _parse(
            "def [2][]int makeRows():\n"
            "    return [[1, 2], [3, 4]]\n"
            "\n"
            "def int main():\n"
            "    for x in makeRows()[0]:\n"
            "        print(x)\n"
            "    return 0\n"
        )
        with pytest.raises(
            SemanticError,
            match="'for ... in' does not support a function call result",
        ):
            analyze(ast)

    def test_slice_rooted_in_call_iterable_is_rejected(self):
        ast = _parse(
            "def []int makeSlice():\n"
            "    return [1, 2, 3, 4, 5]\n"
            "\n"
            "def int main():\n"
            "    for x in makeSlice()[1:3]:\n"
            "        print(x)\n"
            "    return 0\n"
        )
        with pytest.raises(
            SemanticError,
            match="'for ... in' does not support a function call result",
        ):
            analyze(ast)

    def test_array_literal_iterable_analyzes_correctly(self):
        ast = _parse(
            "def int main():\n"
            "    for x in [1, 2, 3]:\n"
            "        print(x)\n"
            "    return 0\n"
        )
        analyze(ast)  # should not raise

    def test_dict_literal_iterable_analyzes_correctly(self):
        ast = _parse(
            "def int main():\n"
            "    for k, v in dict[str]int{'a': 1, 'b': 2}:\n"
            "        print(k)\n"
            "    return 0\n"
        )
        analyze(ast)  # should not raise

    def test_reslicing_a_variable_iterable_analyzes_correctly(self):
        ast = _parse(
            "def int main():\n"
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    for x in arr[1:3]:\n"
            "        print(x)\n"
            "    return 0\n"
        )
        analyze(ast)  # should not raise

    def test_non_addressable_iterable_is_rejected(self):
        ast = _parse(
            "def int main():\n"
            "    int n = 3\n"
            "    for x in n + 1:\n"
            "        print(x)\n"
            "    return 0\n"
        )
        with pytest.raises(
            SemanticError,
            match="'for ... in' requires a variable, field, index, "
                  "slice, or array/dict/str literal",
        ):
            analyze(ast)

    def test_non_collection_iterable_type_is_rejected(self):
        ast = _parse(
            "def int main():\n"
            "    int y = 5\n"
            "    for x in y:\n"
            "        print(x)\n"
            "    return 0\n"
        )
        with pytest.raises(
            SemanticError,
            match="'for ... in' requires an array, slice, dict, or str as its own iterable",
        ):
            analyze(ast)

    def test_element_binding_type_is_actually_enforced(self):
        ast = _parse(
            "def int main():\n"
            "    [3]str names = ['a', 'b', 'c']\n"
            "    for x in names:\n"
            "        int y = x\n"
            "    return 0\n"
        )
        with pytest.raises(SemanticError, match="Cannot initialize 'y'"):
            analyze(ast)

    def test_index_binding_is_typed_int(self):
        ast = _parse(
            "def int main():\n"
            "    [3]str names = ['a', 'b', 'c']\n"
            "    for i, x in names:\n"
            "        str y = i\n"
            "    return 0\n"
        )
        with pytest.raises(SemanticError, match="Cannot initialize 'y'"):
            analyze(ast)

    def test_dict_key_and_value_bindings_are_distinctly_typed(self):
        ast = _parse(
            "def int main():\n"
            "    dict[str]int ages = dict[str]int{'a': 1}\n"
            "    for k, v in ages:\n"
            "        int x = v\n"
            "        str y = k\n"
            "    return 0\n"
        )
        analyze(ast)  # should not raise: v is int, k is str -- both correct

        ast_wrong = _parse(
            "def int main():\n"
            "    dict[str]int ages = dict[str]int{'a': 1}\n"
            "    for k, v in ages:\n"
            "        int x = k\n"
            "    return 0\n"
        )
        with pytest.raises(SemanticError, match="Cannot initialize 'x'"):
            analyze(ast_wrong)

    def test_binding_name_is_scoped_to_the_loop(self):
        assert_semantic_error(
            "    [3]int arr = [1, 2, 3]\n"
            "    for x in arr:\n"
            "        print(x)\n"
            "    print(x)\n"
            "    return 0\n",
            return_type="int",
            match="Reference to undeclared variable 'x'",
        )

    def test_break_and_continue_work_inside_the_body(self):
        ast = _parse(
            "def int main():\n"
            "    [3]int arr = [1, 2, 3]\n"
            "    for x in arr:\n"
            "        if x == 2:\n"
            "            break\n"
            "        continue\n"
            "    return 0\n"
        )
        analyze(ast)  # should not raise

    def test_address_of_a_for_in_binding_is_now_supported(self):
        assert_program_stdout(
            "def *int last_element_address([]int arr):\n"
            "    *int p = none\n"
            "    for x in arr:\n"
            "        p = &x\n"
            "    return p\n"
            "\n"
            "def int main():\n"
            "    []int s = [10, 20, 30]\n"
            "    *int p = last_element_address(s)\n"
            "    print(*p)\n"
            "    return 0\n",
            "30\n",
        )

    def test_escaping_for_in_binding_gets_a_fresh_allocation_per_iteration(self):
        assert_program_stdout(
            "def *int identity(*int p):\n"
            "    return p\n"
            "\n"
            "def int main():\n"
            "    []int s = [1, 2, 3]\n"
            "    *int p1 = none\n"
            "    *int p2 = none\n"
            "    for i, x in s:\n"
            "        if i == 0:\n"
            "            p1 = identity(&x)\n"
            "        if i == 2:\n"
            "            p2 = identity(&x)\n"
            "    print(*p1)\n"
            "    print(*p2)\n"
            "    return 0\n",
            "1\n3\n",
        )

    def test_for_in_binding_address_not_escaping_the_function_still_shares_one_slot(self):
        assert_program_stdout(
            "def int main():\n"
            "    []int s = [1, 2, 3]\n"
            "    *int p1 = none\n"
            "    *int p2 = none\n"
            "    for i, x in s:\n"
            "        if i == 0:\n"
            "            p1 = &x\n"
            "        if i == 2:\n"
            "            p2 = &x\n"
            "    print(*p1)\n"
            "    print(*p2)\n"
            "    return 0\n",
            "3\n3\n",
        )

    def test_address_of_a_shadowed_name_inside_the_body_is_still_allowed(self):
        ast = _parse(
            "def int main():\n"
            "    [3]int arr = [1, 2, 3]\n"
            "    for x in arr:\n"
            "        if x > 0:\n"
            "            int x = 5\n"
            "            *int p = &x\n"
            "    return 0\n"
        )
        analyze(ast)  # should not raise


# ---------------------------------------------------------------------------
# for x in y / for x, y in z
# ---------------------------------------------------------------------------

class TestForInArraySlice:
    def test_array_single_binding(self):
        assert_program_stdout(
            "def int main():\n"
            "    [5]int arr = [10, 20, 30, 40, 50]\n"
            "    for x in arr:\n"
            "        print(x)\n"
            "    return 0\n",
            "10\n20\n30\n40\n50\n",
        )

    def test_array_two_binding(self):
        assert_program_stdout(
            "def int main():\n"
            "    [3]int arr = [10, 20, 30]\n"
            "    for i, x in arr:\n"
            "        print(i)\n"
            "        print(x)\n"
            "    return 0\n",
            "0\n10\n1\n20\n2\n30\n",
        )

    def test_slice_single_binding(self):
        assert_program_stdout(
            "def int main():\n"
            "    []int s = [10, 20, 30]\n"
            "    for x in s:\n"
            "        print(x)\n"
            "    return 0\n",
            "10\n20\n30\n",
        )

    def test_slice_two_binding(self):
        assert_program_stdout(
            "def int main():\n"
            "    []int s = [10, 20, 30]\n"
            "    for i, x in s:\n"
            "        print(i)\n"
            "        print(x)\n"
            "    return 0\n",
            "0\n10\n1\n20\n2\n30\n",
        )

    def test_empty_slice_iterates_zero_times(self):
        assert_program_stdout(
            "def int main():\n"
            "    []int s = []\n"
            "    for x in s:\n"
            "        print(x)\n"
            "    print('done')\n"
            "    return 0\n",
            "done\n",
        )

    def test_str_elements(self):
        assert_program_stdout(
            "def int main():\n"
            "    [3]str names = ['alice', 'bob', 'carol']\n"
            "    for name in names:\n"
            "        print(name)\n"
            "    return 0\n",
            "alice\nbob\ncarol\n",
        )

    def test_struct_elements(self):
        assert_program_stdout(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def int main():\n"
            "    [2]Point pts = [Point(1, 2), Point(3, 4)]\n"
            "    for p in pts:\n"
            "        print(p.x)\n"
            "        print(p.y)\n"
            "    return 0\n",
            "1\n2\n3\n4\n",
        )

    def test_break_exits_the_loop_early(self):
        assert_program_stdout(
            "def int main():\n"
            "    [5]int arr = [10, 20, 30, 40, 50]\n"
            "    for x in arr:\n"
            "        if x == 30:\n"
            "            break\n"
            "        print(x)\n"
            "    return 0\n",
            "10\n20\n",
        )

    def test_continue_skips_to_the_next_element(self):
        assert_program_stdout(
            "def int main():\n"
            "    [5]int arr = [10, 20, 30, 40, 50]\n"
            "    for x in arr:\n"
            "        if x % 20 == 0:\n"
            "            continue\n"
            "        print(x)\n"
            "    return 0\n",
            "10\n30\n50\n",
        )

    def test_safe_in_place_element_mutation_does_not_panic(self):
        assert_program_stdout(
            "def int main():\n"
            "    []int s = [1, 2, 3]\n"
            "    for x in s:\n"
            "        s[0] = 999\n"
            "    print(s[0])\n"
            "    return 0\n",
            "999\n",
        )

    def test_slice_reallocation_during_iteration_panics(self):
        assert_crashes_with_sigabrt(
            "    []int s = [1]\n"
            "    int i = 0\n"
            "    while i < 100:\n"
            "        s = append(s, i)\n"
            "        i = i + 1\n"
            "    for x in s:\n"
            "        s = append(s, 999)\n"
            "    print('should not reach here')\n"
            "    return 0\n"
        )

    def test_index_based_iterable_with_mutated_index_does_not_false_panic(self):
        assert_program_stdout(
            "type Rows = [2][]int\n"
            "\n"
            "def int main():\n"
            "    Rows rows = [[1, 2], [10, 20]]\n"
            "    int i = 0\n"
            "    for x in rows[i]:\n"
            "        print(x)\n"
            "        i = 1\n"
            "    return 0\n",
            "1\n2\n",
        )

    def test_index_based_iterable_still_detects_genuine_reallocation(self):
        assert_crashes_with_sigabrt(
            "    [2][]int rows = [[1], [10, 20]]\n"
            "    int i = 0\n"
            "    int j = 0\n"
            "    while j < 100:\n"
            "        rows[0] = append(rows[0], j)\n"
            "        j = j + 1\n"
            "    for x in rows[i]:\n"
            "        rows[0] = append(rows[0], 999)\n"
            "    print('should not reach here')\n"
            "    return 0\n"
        )

    def test_array_literal_iterable_iterates_correctly(self):
        assert_program_stdout(
            "def int main():\n"
            "    for x in [10, 20, 30]:\n"
            "        print(x)\n"
            "    return 0\n",
            "10\n20\n30\n",
        )

    def test_reslicing_an_array_iterates_correctly(self):
        assert_program_stdout(
            "def int main():\n"
            "    [5]int arr = [10, 20, 30, 40, 50]\n"
            "    for x in arr[1:4]:\n"
            "        print(x)\n"
            "    return 0\n",
            "20\n30\n40\n",
        )

    def test_reslicing_a_slice_two_binding_iterates_correctly(self):
        assert_program_stdout(
            "def int main():\n"
            "    []int s = [1, 2, 3, 4, 5]\n"
            "    for i, x in s[2:5]:\n"
            "        print(i)\n"
            "        print(x)\n"
            "    return 0\n",
            "0\n3\n1\n4\n2\n5\n",
        )

    def test_reslicing_root_reallocation_panics(self):
        assert_crashes_with_sigabrt(
            "    []int s = [1, 2, 3, 4, 5]\n"
            "    int j = 0\n"
            "    while j < 100:\n"
            "        s = append(s, j)\n"
            "        j = j + 1\n"
            "    for x in s[0:len(s)]:\n"
            "        s = append(s, 999)\n"
            "    print('should not reach here')\n"
            "    return 0\n"
        )

    def test_reslicing_safe_in_place_mutation_does_not_panic(self):
        assert_program_stdout(
            "def int main():\n"
            "    []int s = [1, 2, 3, 4, 5]\n"
            "    for i, x in s[1:4]:\n"
            "        s[i + 1] = x * 100\n"
            "    for x in s:\n"
            "        print(x)\n"
            "    return 0\n",
            "1\n200\n300\n400\n5\n",
        )

    def test_reslicing_an_array_root_needs_no_mutation_check(self):
        assert_program_stdout(
            "def int main():\n"
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    for x in arr[1:4]:\n"
            "        print(x)\n"
            "    return 0\n",
            "2\n3\n4\n",
        )

    def test_array_needs_no_mutation_check_at_all(self):
        assert_program_stdout(
            "def int main():\n"
            "    [3]int arr = [1, 2, 3]\n"
            "    for x in arr:\n"
            "        arr[0] = 999\n"
            "    print(arr[0])\n"
            "    return 0\n",
            "999\n",
        )

    def test_oversized_struct_element_is_heap_allocated_correctly(self):
        assert_program_stdout(
            "type Huge struct:\n"
            "    [5000]int data\n"
            "\n"
            "def int main():\n"
            "    [2]Huge items\n"
            "    items[0].data[0] = 111\n"
            "    items[1].data[0] = 222\n"
            "    for h in items:\n"
            "        print(h.data[0])\n"
            "    return 0\n",
            "111\n222\n",
        )

    def test_slice_element_escaping_via_return_is_tracked_correctly(self):
        assert_program_stdout(
            "def []int find_matching(int target_len):\n"
            "    [][]int lists = [[1, 2], [3, 4, 5], [6]]\n"
            "    []int result = []\n"
            "    for x in lists:\n"
            "        if len(x) == target_len:\n"
            "            result = x\n"
            "    return result\n"
            "\n"
            "def int main():\n"
            "    []int found = find_matching(3)\n"
            "    print(len(found))\n"
            "    print(found[0])\n"
            "    print(found[1])\n"
            "    print(found[2])\n"
            "    return 0\n",
            "3\n3\n4\n5\n",
        )

    def test_dict_iteration_now_works(self):
        assert_program_stdout(
            "def int main():\n"
            "    dict[str]int d = dict[str]int{'a': 1}\n"
            "    for k in d:\n"
            "        print(k)\n"
            "    return 0\n",
            "a\n",
        )

# ---------------------------------------------------------------------------
# for k in d / for k, v in d
# ---------------------------------------------------------------------------

class TestForInDict:
    def test_single_key_binding(self):
        assert_program_stdout(
            "def int main():\n"
            "    dict[str]int ages = dict[str]int{'alice': 25}\n"
            "    for k in ages:\n"
            "        print(k)\n"
            "    return 0\n",
            "alice\n",
        )

    def test_key_value_binding_str_keyed(self):
        assert_program_stdout(
            "def int main():\n"
            "    dict[str]int ages = dict[str]int{'alice': 25, 'bob': 30, 'carol': 35}\n"
            "    int total = 0\n"
            "    for k, v in ages:\n"
            "        total = total + v\n"
            "    print(total)\n"
            "    return 0\n",
            "90\n",
        )

    def test_key_value_binding_int_keyed(self):
        assert_program_stdout(
            "def int main():\n"
            "    dict[int]str codes = dict[int]str{1: 'one', 2: 'two', 3: 'three'}\n"
            "    int total = 0\n"
            "    for k, v in codes:\n"
            "        total = total + k\n"
            "    print(total)\n"
            "    return 0\n",
            "6\n",
        )

    def test_tombstones_are_correctly_skipped(self):
        assert_program_stdout(
            "def int main():\n"
            "    dict[int]int nums = dict[int]int{0: 0}\n"
            "    int i = 1\n"
            "    while i < 20:\n"
            "        nums[i] = i * 10\n"
            "        i = i + 1\n"
            "    i = 0\n"
            "    while i < 20:\n"
            "        if i % 2 == 0:\n"
            "            del(nums, i)\n"
            "        i = i + 1\n"
            "    int found_odd = 0\n"
            "    int found_even = 0\n"
            "    for k, v in nums:\n"
            "        if k % 2 == 1:\n"
            "            found_odd = found_odd + 1\n"
            "        else:\n"
            "            found_even = found_even + 1\n"
            "    print(found_odd)\n"
            "    print(found_even)\n"
            "    return 0\n",
            "10\n0\n",
        )

    def test_struct_valued_dict(self):
        assert_program_stdout(
            "type Point struct:\n"
            "    int x\n"
            "    int y\n"
            "\n"
            "def int main():\n"
            "    dict[str]Point pts = dict[str]Point{'origin': Point(0, 0), 'a': Point(3, 4)}\n"
            "    int sumx = 0\n"
            "    for k, p in pts:\n"
            "        sumx = sumx + p.x\n"
            "    print(sumx)\n"
            "    return 0\n",
            "3\n",
        )

    def test_nil_dict_iterates_zero_times(self):
        assert_program_stdout(
            "def int main():\n"
            "    dict[str]int d\n"
            "    int count = 0\n"
            "    for k in d:\n"
            "        count = count + 1\n"
            "    print(count)\n"
            "    return 0\n",
            "0\n",
        )

    def test_break_exits_the_loop_early(self):
        assert_program_stdout(
            "def int main():\n"
            "    dict[int]int d = dict[int]int{1: 10, 2: 20, 3: 30}\n"
            "    int count = 0\n"
            "    for k in d:\n"
            "        count = count + 1\n"
            "        if count == 2:\n"
            "            break\n"
            "    print(count)\n"
            "    return 0\n",
            "2\n",
        )

    def test_continue_skips_to_the_next_bucket(self):
        assert_program_stdout(
            "def int main():\n"
            "    dict[str]int d = dict[str]int{'a': 1, 'b': 2, 'c': 3}\n"
            "    int seen = 0\n"
            "    for k, v in d:\n"
            "        if v == 2:\n"
            "            continue\n"
            "        seen = seen + 1\n"
            "    print(seen)\n"
            "    return 0\n",
            "2\n",
        )

    def test_safe_in_place_value_overwrite_does_not_panic(self):
        assert_program_stdout(
            "def int main():\n"
            "    dict[str]int d = dict[str]int{'a': 1, 'b': 2, 'c': 3}\n"
            "    for k in d:\n"
            "        d[k] = 999\n"
            "    print(d['a'])\n"
            "    print(d['b'])\n"
            "    print(d['c'])\n"
            "    return 0\n",
            "999\n999\n999\n",
        )

    def test_safe_delete_during_iteration_does_not_panic(self):
        assert_program_stdout(
            "def int main():\n"
            "    dict[str]int d = dict[str]int{'x': 1, 'y': 2}\n"
            "    for k in d:\n"
            "        if k == 'x':\n"
            "            del(d, 'x')\n"
            "    print('x' in d)\n"
            "    print('y' in d)\n"
            "    return 0\n",
            "false\ntrue\n",
        )

    def test_growth_triggering_insert_during_iteration_panics(self):
        assert_crashes_with_sigabrt(
            "    dict[int]int d = dict[int]int{1: 1}\n"
            "    int i = 0\n"
            "    while i < 100:\n"
            "        d[i] = i\n"
            "        i = i + 1\n"
            "    for k in d:\n"
            "        d[9999 + k] = k\n"
            "    print('should not reach here')\n"
            "    return 0\n"
        )

    def test_dict_literal_iterable_iterates_correctly(self):
        assert_program_stdout(
            "def int main():\n"
            "    for k, v in dict[str]int{'a': 1, 'b': 2}:\n"
            "        print(k)\n"
            "        print(v)\n"
            "    return 0\n",
            "a\n1\nb\n2\n",
        )


_ESCAPE_PRELUDE = (
    "type S struct:\n"
    "    []int s\n"
    "type C struct:\n"
    "    []int s\n"
    "type D struct:\n"
    "    int k\n"
    "type U is C | D\n"
    "def int noise(int a):\n"
    "    [32]int z = [a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a]\n"
    "    return z[3]\n"
)

_ESCAPE_CASES = {
    'pointer_param_field': (
        "def fill(*S out):\n"
        "    [3]int a = [1,2,3]\n"
        "    out.s = a[:]\n"
        "def int main():\n"
        "    S v = S(none)\n"
        "    fill(&v)\n"
        "    noise(9)\n"
        "    print(v.s)\n"
        "    return 0\n",
        "[]int[1, 2, 3]\n"),
    'dict_param': (
        "def put(dict[int][]int d):\n"
        "    [3]int b = [1,2,3]\n"
        "    d[0] = b[:]\n"
        "def int main():\n"
        "    dict[int][]int d = dict[int][]int{}\n"
        "    put(d)\n"
        "    noise(9)\n"
        "    print(d[0])\n"
        "    return 0\n",
        "[]int[1, 2, 3]\n"),
    'slice_param': (
        "def put([][]int rows):\n"
        "    [3]int b = [1,2,3]\n"
        "    rows[0] = b[:]\n"
        "def int main():\n"
        "    [][]int r = [][]int[none]\n"
        "    put(r)\n"
        "    noise(9)\n"
        "    print(r[0])\n"
        "    return 0\n",
        "[]int[1, 2, 3]\n"),
    'append_element': (
        "def [][]int f():\n"
        "    [3]int a = [1,2,3]\n"
        "    [][]int s = none\n"
        "    s = append(s, a[:])\n"
        "    return s\n"
        "def int main():\n"
        "    [][]int r = f()\n"
        "    noise(9)\n"
        "    print(r)\n"
        "    return 0\n",
        "[][]int[[]int[1, 2, 3]]\n"),
    'slice_literal_returned': (
        "def [][]int f():\n"
        "    [3]int a = [1,2,3]\n"
        "    return [][]int[a[:]]\n"
        "def int main():\n"
        "    [][]int r = f()\n"
        "    noise(9)\n"
        "    print(r)\n"
        "    return 0\n",
        "[][]int[[]int[1, 2, 3]]\n"),
    'slice_literal_via_var': (
        "def [][]int f():\n"
        "    [3]int a = [1,2,3]\n"
        "    [][]int x = [][]int[a[:]]\n"
        "    return x\n"
        "def int main():\n"
        "    [][]int r = f()\n"
        "    noise(9)\n"
        "    print(r)\n"
        "    return 0\n",
        "[][]int[[]int[1, 2, 3]]\n"),
    'sum_variable': (
        "def U f():\n"
        "    [3]int a = [1,2,3]\n"
        "    C c = C(none)\n"
        "    c.s = a[:]\n"
        "    U u = c\n"
        "    return u\n"
        "def int main():\n"
        "    U r = f()\n"
        "    noise(9)\n"
        "    print(r)\n"
        "    return 0\n",
        "C(s: []int[1, 2, 3])\n"),
    'narrowing_binding': (
        "def []int f():\n"
        "    [3]int a = [1,2,3]\n"
        "    C c0 = C(none)\n"
        "    c0.s = a[:]\n"
        "    [1]U us = [c0]\n"
        "    if us[0] is C as c:\n"
        "        return c.s\n"
        "    return none\n"
        "def int main():\n"
        "    []int r = f()\n"
        "    noise(9)\n"
        "    print(r)\n"
        "    return 0\n",
        "[]int[1, 2, 3]\n"),
    'narrowed_variable': (
        "def []int f():\n"
        "    [3]int a = [1,2,3]\n"
        "    C c0 = C(none)\n"
        "    c0.s = a[:]\n"
        "    U u = c0\n"
        "    if u is C:\n"
        "        return u.s\n"
        "    return none\n"
        "def int main():\n"
        "    []int r = f()\n"
        "    noise(9)\n"
        "    print(r)\n"
        "    return 0\n",
        "[]int[1, 2, 3]\n"),
    'local_pointer_store': (
        "def S f():\n"
        "    [3]int a = [1,2,3]\n"
        "    S v = S(none)\n"
        "    *S q = &v\n"
        "    q.s = a[:]\n"
        "    return v\n"
        "def int main():\n"
        "    S r = f()\n"
        "    noise(9)\n"
        "    print(r)\n"
        "    return 0\n",
        "S(s: []int[1, 2, 3])\n"),
    'dict_literal': (
        "def dict[int][]int f():\n"
        "    [3]int a = [1,2,3]\n"
        "    return dict[int][]int{0: a[:]}\n"
        "def int main():\n"
        "    dict[int][]int r = f()\n"
        "    noise(9)\n"
        "    print(r)\n"
        "    return 0\n",
        "dict[int][]int{0: []int[1, 2, 3]}\n"),
    'deref_array_param': (
        "def put(*[1][]int p):\n"
        "    [3]int b = [1,2,3]\n"
        "    (*p)[0] = b[:]\n"
        "def int main():\n"
        "    [1][]int r = [none]\n"
        "    put(&r)\n"
        "    noise(9)\n"
        "    print(r)\n"
        "    return 0\n",
        "[1][]int[[]int[1, 2, 3]]\n"),
    'for_in_binding': (
        "def [][]int f():\n"
        "    [3]int a = [1,2,3]\n"
        "    [2][]int rows = [none, none]\n"
        "    [][]int out = none\n"
        "    rows[0] = a[:]\n"
        "    for row in rows:\n"
        "        out = append(out, row)\n"
        "    return out\n"
        "def int main():\n"
        "    [][]int r = f()\n"
        "    noise(9)\n"
        "    print(r)\n"
        "    return 0\n",
        "[][]int[[]int[1, 2, 3], []int[]]\n"),
    'call_inside_cast': (
        "def int keep([]int s, *S w):\n"
        "    w.s = s\n"
        "    return 1\n"
        "def int g(*S w):\n"
        "    [3]int a = [1,2,3]\n"
        "    return int(int64(keep(a[:], w)))\n"
        "def int main():\n"
        "    S w = S(none)\n"
        "    int k = g(&w)\n"
        "    noise(9)\n"
        "    print(w.s)\n"
        "    return 0\n",
        "[]int[1, 2, 3]\n"),
    'pointer_chain': (
        "type N struct:\n"
        "    *N next\n"
        "    []int s\n"
        "def N f():\n"
        "    [3]int a = [1,2,3]\n"
        "    N inner = N(none, a[:])\n"
        "    N outer = N(&inner, none)\n"
        "    return outer\n"
        "def int main():\n"
        "    N r = f()\n"
        "    noise(9)\n"
        "    print(r.next.s)\n"
        "    return 0\n",
        "[]int[1, 2, 3]\n"),
}


class TestEscapeSoundness:

    @pytest.mark.parametrize('name', sorted(_ESCAPE_CASES))
    def test_no_dangling_slice(self, name):
        source, expected = _ESCAPE_CASES[name]
        assert_program_stdout(_ESCAPE_PRELUDE + source, expected)
