"""Confirms runtime.c and a separate C caller (test_linked_separately.c)
can be compiled as independent object files and linked together
correctly -- the actual cross-translation-unit path build.py's own
final link step relies on, as opposed to test_runtime_isolated.c's
own #include-based white-box testing of runtime.c's internals.

Also confirms hornet_print's own symbol visibility is exactly what's
intended: the one external entry point a future compiler call site
will call, with everything else (hornet_stringify, the buffer
helpers) kept internal.
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

RUNTIME_DIR = Path(__file__).resolve().parent.parent.parent / "runtime"
RUNTIME_C = RUNTIME_DIR / "runtime.c"
TEST_CALLER_C = RUNTIME_DIR / "test_linked_separately.c"

GCC_AVAILABLE = shutil.which("gcc") is not None
pytestmark = pytest.mark.skipif(not GCC_AVAILABLE, reason="gcc not available")

# Same convention as build.py/tests/test_compiler.py/benchmarks/
# run_benchmarks.py: without this, gcc's own default target on an
# Apple Silicon Mac is arm64, not x86-64 -- these two tests link
# object files together and run the result, so a consistent target
# matters even though nothing here mixes in Hornet-generated x86-64
# assembly directly (unlike compile_and_run's own build).
HOST_IS_MACOS = sys.platform == "darwin"


def test_runtime_and_caller_link_and_run_correctly():
    with tempfile.TemporaryDirectory() as tmpdir:
        runtime_o = f"{tmpdir}/runtime.o"
        caller_o = f"{tmpdir}/caller.o"
        binary = f"{tmpdir}/test_bin"

        arch_flags = ["-arch", "x86_64"] if HOST_IS_MACOS else []
        subprocess.run(
            ["gcc", *arch_flags, "-c", "-Wall", "-Wextra", "-std=c11", str(RUNTIME_C), "-o", runtime_o],
            check=True, capture_output=True, text=True,
        )
        subprocess.run(
            ["gcc", *arch_flags, "-c", "-Wall", "-Wextra", "-std=c11", str(TEST_CALLER_C), "-o", caller_o],
            check=True, capture_output=True, text=True,
        )
        subprocess.run(["gcc", *arch_flags, runtime_o, caller_o, "-o", binary], check=True, capture_output=True, text=True)

        result = subprocess.run([binary], capture_output=True, text=True)
        assert result.returncode == 0
        assert result.stdout == "42\n"


def test_hornet_print_is_the_only_external_symbol():
    """hornet_print is the one function meant to be called from
    outside this file; hornet_stringify and every buffer/read helper
    are internal implementation details (static linkage) that
    shouldn't leak into whatever links against this object file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        runtime_o = f"{tmpdir}/runtime.o"
        arch_flags = ["-arch", "x86_64"] if HOST_IS_MACOS else []
        subprocess.run(
            ["gcc", *arch_flags, "-c", "-Wall", "-Wextra", "-std=c11", str(RUNTIME_C), "-o", runtime_o],
            check=True, capture_output=True, text=True,
        )
        nm_result = subprocess.run(["nm", "-g", runtime_o], check=True, capture_output=True, text=True)
        external_symbols = [
            line.split()[-1] for line in nm_result.stdout.splitlines() if " T " in line
        ]
        # macOS's Mach-O convention mangles every external C symbol
        # with a leading underscore (see Emitter's own symbol()
        # method) -- nm reports that mangled name here directly, so
        # the expected name needs the identical leading underscore on
        # macOS, not just when Hornet-generated code refers to it.
        expected_name = "_hornet_print" if HOST_IS_MACOS else "hornet_print"
        assert external_symbols == [expected_name]
