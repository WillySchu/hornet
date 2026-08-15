"""
test_compiler.py

Consolidated pytest suite for the lexer -> parser -> semantic -> codegen
pipeline.

Every execution-based test here compiles a small program through the
real pipeline, assembles and links it with gcc, actually *runs* the
resulting binary, and checks its exit code (or, for the short-circuit
"control" tests, that it crashes with a specific signal). This is
deliberately not just inspecting the generated assembly text -- actually
executing the binary is the strongest available check that codegen is
correct, not merely plausible-looking. Semantic-error tests are
different in kind: they assert that bad programs are *rejected* before
codegen ever runs, so they don't need gcc at all and aren't skipped when
it's unavailable (see GCC_SKIP below).

Requires `gcc` (and `as`/`ld`, which it drives) on PATH for the
execution-based classes. If it's not found, those are skipped with a
clear reason rather than failing with a wall of "file not found" errors.

Run with:
    pytest test_compiler.py -v

Organization:
    TestUnaryOperators                 ( 7 tests)
    TestBinaryArithmetic                (18 tests)
    TestComparisons                     (12 tests)
    TestPrecedenceAndLogicalOperators   ( 8 tests)
    TestShortCircuitEvaluation          ( 4 tests)
    TestVariablesAndStatements          (12 tests)
    TestIfStatements                    (18 tests)
    TestSemanticErrors                  (28 tests)
                                        ----------
                                        107 tests total

A NOTE ON BLOCK SCOPING AND WHY CODEGEN'S ALLOCATOR CHANGED SHAPE
-----------------------------------------------------------------
`if`/`elif`/`else` introduced real nested scopes (see semantic.py's
SCOPING section), which broke an assumption codegen's local-variable
allocator used to rely on: that a variable name always maps to exactly
one stack slot per function. Once sibling branches can each declare a
variable with the same name (`if x: int a = 1` / `else: int a = 2`,
now legitimately two different variables), stack slots have to be
keyed by which specific declaration a name resolves to at a given
point in the program, not by the name alone -- see codegen.py's LOCAL
VARIABLES section for the actual mechanism. TestIfStatements' two
same-name-in-both-branches tests exist specifically to prove that
allocator change is correct, not just that if/else branch correctly.

A NOTE ON THE TYPE SYSTEM AND WHY SEVERAL TESTS CHANGED SHAPE
-----------------------------------------------------------------
This language now has a strong static type system (see semantic.py):
`int` and `bool` are distinct types with *no* implicit conversion
between them in either direction. That's a real behavior change, not
just an addition, and it broke several tests that predate it:
  - `!0`, `!5`, `!-0` used to work (NOT applied to a raw int). Under
    strict typing, `!` requires a genuine `bool` operand -- `!0` is now
    a type error, not "not true". These are now written with real bool
    values (`!true`, `!(x == y)`, etc.) instead.
  - Every test that did `return <comparison-or-logical-expr>` from a
    `def int main()` now needs `def bool main()` instead, since
    comparisons and `and`/`or` produce `bool`, and a strict return type
    has to match exactly.
  - `0 and (1 / 0)`-style short-circuit tests used raw int operands for
    `and`/`or`, which now requires bool operands on both sides -- these
    were rewritten with `true`/`false` and a comparison
    (`(1 / 0) == 1`) standing in for "an int expression that crashes if
    evaluated, but produces a bool so `and`/`or` will accept it".
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
from semantic import SemanticError, analyze


GCC_AVAILABLE = shutil.which("gcc") is not None
GCC_SKIP = pytest.mark.skipif(
    not GCC_AVAILABLE,
    reason="gcc not found on PATH; these tests compile and execute real binaries",
)
# Applied per-class (not module-wide): TestSemanticErrors never reaches
# codegen, let alone gcc, so it shouldn't be skipped just because gcc
# happens to be missing.

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
# Shared helpers
# ---------------------------------------------------------------------------

def _parse(source: str):
    """lex + parse only, no semantic analysis -- used internally by both
    compile_and_run (which analyzes explicitly, see below) and
    assert_semantic_error (which asserts analysis itself fails)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "program.lang"
        src_path.write_text(source)
        tokens = lex(str(src_path))
        return Parser(tokens).parse_program()


def compile_and_run(source: str) -> subprocess.CompletedProcess:
    """Runs `source` through the real lex -> parse -> analyze -> codegen
    pipeline, assembles and links the result with gcc (using
    ASM_PLATFORM, the platform this test process is actually running on
    -- see above), and runs the resulting binary.

    Returns the CompletedProcess so callers can inspect `.returncode`:
    0-255 for a normal exit, or -N if the process was killed by signal N
    (that's how Python's subprocess reports signal termination when not
    going through a shell -- see assert_crashes_with_sigfpe below).

    Uses a plain tempfile.TemporaryDirectory rather than pytest's
    tmp_path fixture so the many parametrized one-liner tests below
    don't each need to declare and thread a fixture through just to
    call this helper.
    """
    ast = _parse(source)
    analyze(ast)  # every program reaching codegen in this file is expected to be well-typed

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        asm_path = tmp / "program.s"
        bin_path = tmp / "program"

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


def assert_exit_code(body: str, expected: int, return_type: str = "int") -> None:
    """Wraps `body` (the statements, one per line, each pre-indented) in
    `def {return_type} main():` and asserts the compiled-and-executed
    program exits with `expected`."""
    source = f"def {return_type} main():\n{body}\n"
    result = compile_and_run(source)
    assert result.returncode == expected, (
        f"body:\n{body}\nexpected exit {expected}, got {result.returncode}"
    )


def assert_crashes_with_sigfpe(body: str, return_type: str = "int") -> None:
    """Same as assert_exit_code, but asserts the program is killed by
    SIGFPE (a real divide-by-zero trap) rather than exiting normally."""
    source = f"def {return_type} main():\n{body}\n"
    result = compile_and_run(source)
    assert result.returncode == -signal.SIGFPE, (
        f"body:\n{body}\nexpected SIGFPE crash, got exit {result.returncode}"
    )


def assert_semantic_error(body: str, return_type: str = "int", match: str = None) -> None:
    """Asserts that `body` (wrapped in `def {return_type} main():`) is
    rejected by semantic analysis. Never reaches codegen or gcc, so
    these run regardless of GCC_AVAILABLE."""
    source = f"def {return_type} main():\n{body}\n"
    ast = _parse(source)
    with pytest.raises(SemanticError, match=match):
        analyze(ast)


# ---------------------------------------------------------------------------
# Unary operators: -, ~, !  (including chaining, e.g. ~-2)
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
        ("!true", 0),
        ("!false", 1),
        ("!!true", 1),   # chained NOT, still requires (and produces) bool at each step
    ])
    def test_bool_not(self, expr, expected):
        assert_exit_code(f"    return {expr}", expected, return_type="bool")


# ---------------------------------------------------------------------------
# Binary arithmetic: + - * / , precedence, associativity, grouping
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
# Comparisons: == != < > <= >=
#
# All of these produce `bool` now, so every one of these functions is
# declared `def bool main()`, not `def int main()` -- the exit-code
# mechanics work identically either way (the OS only ever sees the
# 0/1 that ends up in %eax), but the *type* the language sees matters
# to the analyzer.
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
#
# A correct boolean *result* alone doesn't prove short-circuiting
# happened, since this language has no other observable side effects
# yet. The definitive check is a division-by-zero trap on the side that
# must NOT run: if it's genuinely skipped, the program returns cleanly;
# if it's evaluated anyway, the program crashes with a real SIGFPE.
#
# 'and'/'or' require bool operands, so "an int expression that crashes
# if evaluated" has to be wrapped as one: `(1 / 0) == 1` is bool-typed
# (int == int), but still crashes at runtime the moment `1 / 0` actually
# executes.
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
        """The control for the pair above: proves short-circuiting is
        genuinely conditional, not just "always skip the right side" by
        coincidence. When the left side does NOT already decide the
        result, the right side must actually execute -- so this SHOULD
        crash."""
        assert_crashes_with_sigfpe(f"    return {expr}", return_type="bool")


# ---------------------------------------------------------------------------
# Local variables: declaration, assignment, and standalone expression
# statements
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


# ---------------------------------------------------------------------------
# if / elif / else: branching, elif chains, nesting, and block scoping.
#
# The scoping tests here matter more than they might look -- they're not
# just "does the value come out right", they're proof that codegen's
# stack-slot allocator correctly gives two *different* variables their
# own storage even when they share a name across sibling branches (see
# codegen.py's LOCAL VARIABLES section). Before that allocator was
# rewritten, a program declaring `a` in both an if and its else would
# have been rejected as a duplicate declaration at the codegen layer,
# even though semantic.py correctly allows it.
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
        """The key scoping case: `a` in the then-branch and `a` in the
        else-branch are independent variables that happen to share a
        name -- semantic.py allows this (separate scopes), and codegen
        must give them genuinely separate stack slots for this to come
        out right."""
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
        """Two *non-overlapping* if-blocks (not sibling branches of the
        same if) each declaring a variable called `y` -- distinct from
        the sibling-branch case above, but exercising the same
        node-identity-keyed allocation."""
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
# Semantic analysis: scope/declaration checking and the strict int/bool
# type system. These never reach codegen -- each one asserts that
# analyze() itself raises SemanticError -- so they don't need gcc and
# aren't skipped when it's missing.
# ---------------------------------------------------------------------------

class TestSemanticErrors:

    # -- scope / declaration errors --------------------------------------

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
        """A variable assigned above its own declaration is, from the
        analyzer's point of view, simply not in scope yet -- see
        semantic.py's module docstring for why that falls out of
        walking statements in program order."""
        assert_semantic_error(
            "    a = 1\n"
            "    int a\n"
            "    return a",
            match="undeclared variable",
        )

    def test_self_referential_initializer(self):
        """`int a = a` -- the right-hand `a` is checked before the new
        `a` is added to scope, so this is indistinguishable from any
        other undeclared reference."""
        assert_semantic_error(
            "    int a = a\n"
            "    return a",
            match="undeclared variable",
        )

    def test_valid_program_does_not_raise(self):
        """Sanity check on the harness itself: a well-formed program
        must NOT raise, so the tests above are actually testing
        something specific rather than everything just failing."""
        ast = _parse("def int main():\n    int a = 1\n    return a\n")
        analyze(ast)  # should not raise

    # -- initialization / assignment / return type mismatches -----------

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

    # -- operator operand-type errors ------------------------------------

    def test_not_requires_bool_not_int(self):
        assert_semantic_error(
            "    return !0",
            return_type="bool",
            match="requires a bool operand",
        )

    def test_negate_requires_int_not_bool(self):
        assert_semantic_error(
            "    return -true",
            match="requires an int operand",
        )

    def test_complement_requires_int_not_bool(self):
        assert_semantic_error(
            "    return ~true",
            match="requires an int operand",
        )

    def test_arithmetic_requires_int_operands(self):
        assert_semantic_error(
            "    return true + false",
            match="requires int operands",
        )

    def test_ordering_comparison_requires_int_operands(self):
        """< > <= >= don't accept bool -- there's no inherent ordering
        on bool in this language."""
        assert_semantic_error(
            "    return true < false",
            return_type="bool",
            match="requires int operands",
        )

    def test_chained_ordering_comparison_is_rejected(self):
        """`1 < 2 < 3` parses left-associatively as `(1 < 2) < 3` (see
        parser.py) -- and `1 < 2` produces bool, which `<` doesn't
        accept. Under implicit-conversion languages this silently means
        something most people don't intend (`(1<2)<3` happens to
        evaluate `1<3`); strict typing turns it into a hard error
        instead of a footgun."""
        assert_semantic_error(
            "    return 1 < 2 < 3",
            return_type="bool",
            match="requires int operands",
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
        """The positive control for the test above: comparing two
        values of the *same* type must NOT raise."""
        ast = _parse("def bool main():\n    return 1 == 1\n")
        analyze(ast)  # should not raise
        ast = _parse("def bool main():\n    return true == false\n")
        analyze(ast)  # should not raise

    # -- literal type errors ---------------------------------------------

    def test_float_literal_is_rejected(self):
        """There's no float type yet -- only int and bool -- even
        though the lexer's NUMBER rule matches decimals."""
        assert_semantic_error(
            "    return 2.5",
            match="not a whole number",
        )

    # -- if/elif/else: condition typing and block scoping ----------------

    def test_if_condition_must_be_bool(self):
        """No int-as-truthy shortcut -- same rule as everywhere else in
        this type system."""
        assert_semantic_error(
            "    if 1:\n"
            "        return 1\n"
            "    return 0",
            match="'if' condition must be bool",
        )

    def test_elif_condition_must_be_bool(self):
        """The condition check applies to every elif in a chain, not
        just the first `if` -- since an elif is just a nested If (see
        parser.py's If docstring), this falls out of analyze_if being
        called recursively rather than needing separate elif logic."""
        assert_semantic_error(
            "    if false:\n"
            "        return 1\n"
            "    elif 1:\n"
            "        return 2\n"
            "    return 0",
            match="'if' condition must be bool",
        )

    def test_variable_declared_in_if_does_not_leak_outside(self):
        """A variable declared inside an if-block goes out of scope
        once the block ends -- referencing it afterward is exactly as
        invalid as referencing any other undeclared name."""
        assert_semantic_error(
            "    if true:\n"
            "        int a = 1\n"
            "    return a",
            match="undeclared variable",
        )

    def test_variable_declared_in_then_not_visible_in_else(self):
        """then and else get independent scopes -- a name from one
        branch isn't visible in the other, since they're mutually
        exclusive at runtime."""
        assert_semantic_error(
            "    if true:\n"
            "        int a = 1\n"
            "    else:\n"
            "        return a",
            match="undeclared variable",
        )

    def test_same_name_in_sibling_branches_is_allowed(self):
        """The positive control for the two tests above: declaring `a`
        independently in both an if and its else must NOT raise, since
        they're separate scopes -- this is exactly the scenario that
        motivated rewriting codegen's allocator (see codegen.py's LOCAL
        VARIABLES section)."""
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
        """A block-local declaration is allowed to shadow a
        same-named variable from an enclosing scope."""
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
        """Shadowing an *enclosing* scope is fine, but the ordinary
        double-declaration rule still applies *within* a single
        branch's own scope."""
        assert_semantic_error(
            "    if true:\n"
            "        int a = 1\n"
            "        int a = 2\n"
            "        return a\n"
            "    return 0",
            match="already declared",
        )
