"""Confirms runtime.c and a separate C caller (test_linked_separately.c)
can be compiled as independent object files and linked together
correctly -- the actual cross-translation-unit path build.py's own
final link step relies on, as opposed to test_runtime_isolated.c's
own #include-based white-box testing of runtime.c's internals.

Also confirms hornet_print/hornet_panic/hornet_slice_grow's own symbol
visibility is exactly what's intended: the three external entry points
a compiler call site actually calls, with everything else (hornet_
stringify, the buffer helpers) kept internal.
"""
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from build import c_compiler, executable_name
from tests.targets import each_runnable_target, run_binary

RUNTIME_DIR = Path(__file__).resolve().parent.parent.parent / "runtime"
RUNTIME_C = RUNTIME_DIR / "runtime.c"
TEST_CALLER_C = RUNTIME_DIR / "test_linked_separately.c"

GCC_AVAILABLE = shutil.which("gcc") is not None
pytestmark = pytest.mark.skipif(not GCC_AVAILABLE, reason="gcc not available")


@each_runnable_target
def test_runtime_and_caller_link_and_run_correctly(target):
    CC = c_compiler(target)
    with tempfile.TemporaryDirectory() as tmpdir:
        runtime_o = f"{tmpdir}/runtime.o"
        caller_o = f"{tmpdir}/caller.o"
        binary = f"{tmpdir}/{executable_name('test_bin', target)}"

        subprocess.run(
            [*CC, "-c", "-Wall", "-Wextra", "-std=c11", str(RUNTIME_C), "-o", runtime_o],
            check=True, capture_output=True, text=True,
        )
        subprocess.run(
            [*CC, "-c", "-Wall", "-Wextra", "-std=c11", str(TEST_CALLER_C), "-o", caller_o],
            check=True, capture_output=True, text=True,
        )
        subprocess.run([*CC, runtime_o, caller_o, "-o", binary], check=True, capture_output=True, text=True)

        result = run_binary(target, [binary], capture_output=True, text=True)
        assert result.returncode == 0
        assert result.stdout == "42\n"


@each_runnable_target
def test_hornet_runtime_entry_points_are_the_only_external_symbols(target):
    """The entry points listed below are the runtime's only external symbols; hornet_stringify, the
    buffer helpers, and the dict helpers are static, and don't leak into what links against it."""
    CC = c_compiler(target)
    with tempfile.TemporaryDirectory() as tmpdir:
        runtime_o = f"{tmpdir}/runtime.o"
        subprocess.run(
            [*CC, "-c", "-Wall", "-Wextra", "-std=c11", str(RUNTIME_C), "-o", runtime_o],
            check=True, capture_output=True, text=True,
        )
        nm_result = subprocess.run(["nm", "-g", runtime_o], check=True, capture_output=True, text=True)
        external_symbols = [
            line.split()[-1] for line in nm_result.stdout.splitlines() if " T " in line
        ]
        # macOS's Mach-O convention mangles every external C symbol
        # with a leading underscore (see Emitter's own symbol()
        # method) -- nm reports that mangled name here directly, so
        # the expected names need the identical leading underscore on
        # macOS, not just when Hornet-generated code refers to them.
        names = [
            "hornet_argv_get",
            "hornet_bytes",
            "hornet_dict_contains_scalar_key",
            "hornet_dict_contains_str_key",
            "hornet_dict_delete_scalar_key",
            "hornet_dict_delete_str_key",
            "hornet_dict_insert_scalar_key",
            "hornet_dict_insert_str_key",
            "hornet_dict_lookup_scalar_key",
            "hornet_dict_lookup_str_key",
            "hornet_dict_set_scalar_key",
            "hornet_dict_set_str_key",
            "hornet_close_fd",
            "hornet_error_message",
            "hornet_exit",
            "hornet_hash_bytes",
            "hornet_is_terminal",
            "hornet_open_read",
            "hornet_open_write",
            "hornet_panic",
            "hornet_panic_at",
            "hornet_print",
            "hornet_read_fd",
            "hornet_read_line",
            "hornet_slice_grow",
            "hornet_write_fd",
        ]
        expected_names = [f"_{n}" for n in names] if target.os == 'macos' else names
        assert sorted(external_symbols) == sorted(expected_names)
