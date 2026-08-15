"""
test_compiler.py

Consolidated pytest suite for the lexer -> parser -> codegen pipeline.

Every test here compiles a small program through the real pipeline,
assembles and links it with gcc, actually *runs* the resulting binary,
and checks its exit code (or, for the short-circuit "control" tests,
that it crashes with a specific signal). This is deliberately not just
inspecting the generated assembly text -- actually executing the binary
is the strongest available check that codegen is correct, not merely
plausible-looking.

Requires `gcc` (and `as`/`ld`, which it drives) on PATH. If it's not
found, every test in this file is skipped with a clear reason rather
than failing with a wall of "file not found" errors.

Run with:
    pytest test_compiler.py -v

Organization mirrors how these were originally developed and verified,
turn by turn, as each language feature was added:
    TestUnaryOperators              ( 7 tests)
    TestBinaryArithmetic            (18 tests)
    TestComparisons                 (12 tests)
    TestPrecedenceAndLogicalOperators (8 tests)
    TestShortCircuitEvaluation       (4 tests)
    TestVariablesAndStatements      (12 tests)
                                    ----------
                                     61 tests total
"""
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from codegen import generate_asm
from lexer import lex
from parser import Parser


GCC_AVAILABLE = shutil.which("gcc") is not None
pytestmark = pytest.mark.skipif(
    not GCC_AVAILABLE,
    reason="gcc not found on PATH; these tests compile and execute real binaries",
)

# These tests actually assemble and run the generated code, so the
# assembly's platform has to match whatever `gcc` on *this* machine will
# actually produce/link -- not be hardcoded to one platform. On macOS
# that also means explicitly targeting x86_64: this compiler only ever
# generates x86-64, but Apple Silicon Macs default to arm64, and Xcode's
# toolchain needs to be told `-arch x86_64` to assemble/link x86-64 input
# instead of rejecting it outright. The resulting binary then runs under
# Rosetta 2 (installed automatically the first time it's needed, or via
# `softwareupdate --install-rosetta`).
HOST_IS_MACOS = sys.platform == "darwin"
ASM_PLATFORM = "macos" if HOST_IS_MACOS else "linux"


# ---------------------------------------------------------------------------
# Shared compile-and-run helpers
# ---------------------------------------------------------------------------

def compile_and_run(source: str) -> subprocess.CompletedProcess:
    """Runs `source` through the real lex -> parse -> codegen pipeline,
    assembles and links the result with gcc (using ASM_PLATFORM, the
    platform this test process is actually running on -- see above), and
    runs the resulting binary.

    Returns the CompletedProcess so callers can inspect `.returncode`:
    0-255 for a normal exit, or -N if the process was killed by signal N
    (that's how Python's subprocess reports signal termination when not
    going through a shell -- see assert_crashes_with_sigfpe below).

    Uses a plain tempfile.TemporaryDirectory rather than pytest's
    tmp_path fixture so the many parametrized one-liner tests below
    don't each need to declare and thread a fixture through just to
    call this helper.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        src_path = tmp / "program.lang"
        asm_path = tmp / "program.s"
        bin_path = tmp / "program"

        src_path.write_text(source)

        tokens = lex(str(src_path))
        ast = Parser(tokens).parse_program()
        asm = generate_asm(ast, platform=ASM_PLATFORM)
        asm_path.write_text(asm)

        gcc_cmd = ["gcc"]
        if HOST_IS_MACOS:
            gcc_cmd += ["-arch", "x86_64"]
        gcc_cmd += [str(asm_path), "-o", str(bin_path)]

        result = subprocess.run(gcc_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            # Don't just let CalledProcessError's bare "exit status 1"
            # through -- that hides the one thing that actually explains
            # a compile failure. Show the real diagnostic, the command,
            # and the generated assembly so a failure here is
            # self-diagnosing instead of needing a follow-up round trip.
            pytest.fail(
                "gcc failed to assemble/link the generated program.\n"
                f"command: {' '.join(gcc_cmd)}\n"
                f"--- gcc stdout ---\n{result.stdout}\n"
                f"--- gcc stderr ---\n{result.stderr}\n"
                f"--- generated assembly ---\n{asm}"
            )

        try:
            return subprocess.run([str(bin_path)])
        except OSError as e:
            if HOST_IS_MACOS:
                pytest.fail(
                    f"Failed to execute the compiled x86_64 binary ({e}). "
                    "On Apple Silicon this usually means Rosetta 2 isn't "
                    "installed -- try `softwareupdate --install-rosetta`."
                )
            raise


def assert_exit_code(body: str, expected: int) -> None:
    """Wraps `body` (the statements, one per line, each pre-indented) in
    `def int main():` and asserts the compiled-and-executed program
    exits with `expected`."""
    source = f"def int main():\n{body}\n"
    result = compile_and_run(source)
    assert result.returncode == expected, (
        f"body:\n{body}\nexpected exit {expected}, got {result.returncode}"
    )


def assert_crashes_with_sigfpe(body: str) -> None:
    """Same as assert_exit_code, but asserts the program is killed by
    SIGFPE (a real divide-by-zero trap) rather than exiting normally."""
    source = f"def int main():\n{body}\n"
    result = compile_and_run(source)
    assert result.returncode == -signal.SIGFPE, (
        f"body:\n{body}\nexpected SIGFPE crash, got exit {result.returncode}"
    )


# ---------------------------------------------------------------------------
# Unary operators: -, ~, !  (including chaining, e.g. ~-2)
# ---------------------------------------------------------------------------

class TestUnaryOperators:

    @pytest.mark.parametrize("expr,expected", [
        ("-2", 254),   # two's-complement wraparound: -2 -> 254
        ("~2", 253),   # ~2 == -3 == 253
        ("!0", 1),
        ("!5", 0),
        ("~-2", 1),    # ~(-2) == 1
        ("--2", 2),    # -(-2) == 2
        ("!-0", 1),    # -0 == 0, !0 == 1
    ])
    def test_unary_operator(self, expr, expected):
        assert_exit_code(f"    return {expr}", expected)


# ---------------------------------------------------------------------------
# Binary arithmetic: + - * / , precedence, associativity, grouping
# ---------------------------------------------------------------------------

class TestBinaryArithmetic:

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
        ("!0 + !0", 2),
        ("2 * (3 + 4) - 5", 9),
    ])
    def test_binary_arithmetic(self, expr, expected):
        assert_exit_code(f"    return {expr}", expected)


# ---------------------------------------------------------------------------
# Comparisons: == != < > <= >=
# ---------------------------------------------------------------------------

class TestComparisons:

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
        assert_exit_code(f"    return {expr}", expected)


# ---------------------------------------------------------------------------
# Precedence across every level, and 'and'/'or' chaining
# ---------------------------------------------------------------------------

class TestPrecedenceAndLogicalOperators:

    @pytest.mark.parametrize("expr,expected", [
        ("1 + 2 * 3 == 7 and 4 < 5", 1),
        ("1 + 2 * 3 == 8 and 4 < 5", 0),
        ("1 < 2 and 3 > 4 or 5 == 5", 1),
        ("1 < 2 < 3", 1),          # (1 < 2) < 3  ->  1 < 3  ->  1
        ("0 or 0 or 1", 1),
        ("0 or 0 or 0", 0),
        ("1 and 1 and 0", 0),
        ("1 and 1 and 1", 1),
    ])
    def test_precedence(self, expr, expected):
        assert_exit_code(f"    return {expr}", expected)


# ---------------------------------------------------------------------------
# Short-circuit evaluation of 'and' / 'or'
#
# A correct boolean *result* alone doesn't prove short-circuiting
# happened, since this language has no other observable side effects
# yet. The definitive check is a division-by-zero trap on the side that
# must NOT run: if it's genuinely skipped, the program returns cleanly;
# if it's evaluated anyway, the program crashes with a real SIGFPE.
# ---------------------------------------------------------------------------

class TestShortCircuitEvaluation:

    @pytest.mark.parametrize("expr,expected", [
        ("0 and (1 / 0)", 0),   # left is false -> right must be skipped
        ("1 or (1 / 0)", 1),    # left is true  -> right must be skipped
    ])
    def test_short_circuit_skips_crashing_side(self, expr, expected):
        assert_exit_code(f"    return {expr}", expected)

    @pytest.mark.parametrize("expr", [
        "1 and (1 / 0)",   # left doesn't decide -> right MUST run
        "0 or (1 / 0)",    # left doesn't decide -> right MUST run
    ])
    def test_short_circuit_control_evaluates_when_needed(self, expr):
        """The control for the pair above: proves short-circuiting is
        genuinely conditional, not just "always skip the right side" by
        coincidence. When the left side does NOT already decide the
        result, the right side must actually execute -- so this SHOULD
        crash."""
        assert_crashes_with_sigfpe(f"    return {expr}")


# ---------------------------------------------------------------------------
# Local variables: declaration, assignment, and standalone expression
# statements
# ---------------------------------------------------------------------------

class TestVariablesAndStatements:

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
        )

    def test_variable_in_short_circuit(self):
        assert_exit_code(
            "    int a = 0\n"
            "    return a and (1 / 0)",
            0,
        )

    def test_standalone_expression_statement_actually_executes(self):
        """Proof that expression statements aren't silently dropped:
        a division by zero as a bare statement, with its result
        discarded, must still crash the program."""
        assert_crashes_with_sigfpe(
            "    1 / 0\n"
            "    return 0"
        )

    def test_standalone_safe_expression_does_not_crash(self):
        assert_exit_code(
            "    1 + 1\n"
            "    return 42",
            42,
        )
