"""Wraps runtime/test_runtime_isolated.c -- the most thorough C-level
test coverage this package has (every type-descriptor kind, nesting,
the buffer-growth stress test, a genuinely self-referential struct,
hornet_print end-to-end) -- so it runs under `pytest tests/` like
everything else, rather than needing a separately-remembered manual
`gcc ... && ./binary` invocation.

test_runtime_isolated.c is a standalone C program, not itself a
pytest file: it #includes runtime.c directly for white-box access to
its static internals (see its own module docstring), and reports
success via exit code and an "all tests passed" stdout line, the same
contract this file checks.
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

RUNTIME_DIR = Path(__file__).resolve().parent.parent.parent / "runtime"
TEST_ISOLATED_C = RUNTIME_DIR / "test_runtime_isolated.c"

GCC_AVAILABLE = shutil.which("gcc") is not None
pytestmark = pytest.mark.skipif(not GCC_AVAILABLE, reason="gcc not available")

# Same convention as build.py/tests/test_compiler.py/etc. Not strictly
# required here the way it is for compile_and_run's own build (this
# test is self-contained C, compiled and run on the same host either
# way), but forcing x86-64 keeps this test exercising the identical
# target architecture runtime.c actually ships against in production,
# rather than silently testing arm64 on Apple Silicon instead.
HOST_IS_MACOS = sys.platform == "darwin"


def _compile_and_run(extra_flags: list[str]) -> subprocess.CompletedProcess:
    with tempfile.TemporaryDirectory() as tmpdir:
        binary = f"{tmpdir}/test_runtime_isolated"
        arch_flags = ["-arch", "x86_64"] if HOST_IS_MACOS else []
        subprocess.run(
            ["gcc", *arch_flags, "-Wall", "-Wextra", "-std=c11", *extra_flags, str(TEST_ISOLATED_C), "-o", binary],
            check=True, capture_output=True, text=True,
        )
        return subprocess.run([binary], capture_output=True, text=True)


def test_isolated_c_tests_pass():
    result = _compile_and_run([])
    assert result.returncode == 0, result.stdout + result.stderr
    assert "all tests passed" in result.stdout


def test_isolated_c_tests_pass_under_sanitizers():
    """The same suite, compiled with AddressSanitizer and
    UndefinedBehaviorSanitizer -- this is exactly the combination that
    caught a genuine misaligned-access bug in runtime.c during initial
    development (a plain, unsanitized run passed silently), so it's
    worth running as its own, separate check rather than folding into
    the plain run above and losing the distinction if one ever passes
    while the other doesn't."""
    result = _compile_and_run(["-Wpedantic", "-fsanitize=address,undefined", "-g"])
    assert result.returncode == 0, result.stdout + result.stderr
    assert "all tests passed" in result.stdout
