"""Confirms the runtime, as build.py builds it (start.c's object and an archive of the rest), links
with a separate C caller (test_linked_separately.c) and runs -- the path build.py's own final link
step relies on.

Also confirms the runtime's external symbols are what is intended: the entry points the compiled
code and the standard library call, and otherwise only what its files share, each named `hornet__`.
And that each of its files compiles without a warning.
"""
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from build import RUNTIME_CFLAGS, c_compiler, executable_name, link_flags, runtime_objects, runtime_sources
from tests.targets import each_runnable_target, run_binary

RUNTIME_DIR = Path(__file__).resolve().parent.parent.parent / "runtime"
TEST_CALLER_C = RUNTIME_DIR / "test_linked_separately.c"

GCC_AVAILABLE = shutil.which("gcc") is not None
pytestmark = pytest.mark.skipif(not GCC_AVAILABLE, reason="gcc not available")


@each_runnable_target
def test_runtime_and_caller_link_and_run_correctly(target):
    CC = c_compiler(target)
    with tempfile.TemporaryDirectory() as tmpdir:
        caller_o = f"{tmpdir}/caller.o"
        binary = f"{tmpdir}/{executable_name('test_bin', target)}"
        subprocess.run(
            [*CC, "-c", "-Wall", "-Wextra", "-std=c11", str(TEST_CALLER_C), "-o", caller_o],
            check=True, capture_output=True, text=True,
        )
        subprocess.run([*CC, caller_o, *map(str, runtime_objects(target)), "-o", binary, *link_flags(target)],
                       check=True, capture_output=True, text=True)

        result = run_binary(target, [binary], capture_output=True, text=True)
        assert result.returncode == 0
        assert result.stdout == "42\n"


@each_runnable_target
def test_each_runtime_file_compiles_without_a_warning(target):
    for source in runtime_sources(target):
        result = subprocess.run(
            [*c_compiler(target), *RUNTIME_CFLAGS, "-c", "-Wall", "-Wextra", "-Wpedantic", str(source), "-o",
             os.devnull],
            capture_output=True, text=True)
        assert (result.returncode, result.stderr) == (0, ""), source.name


@each_runnable_target
def test_hornet_runtime_entry_points_are_the_only_external_symbols(target):
    """The entry points listed below are the runtime's only external functions, but for what its
    files share with each other (hornet__stringify, the buffer and allocation helpers), which are
    named `hornet__` so that a program's `extern` declarations don't meet them."""
    nm_result = subprocess.run(["nm", "-g", *map(str, runtime_objects(target))],
                               check=True, capture_output=True, text=True)
    external_symbols = [
        line.split()[-1] for line in nm_result.stdout.splitlines() if " T " in line
    ]
    prefix = "_" if target.os == 'macos' else ""
    shared = [name for name in external_symbols if name.startswith(prefix + "hornet__")]
    assert shared, "the runtime's files share something"
    external_symbols = [name for name in external_symbols if name not in shared]
    # macOS's Mach-O convention mangles every external C symbol
    # with a leading underscore (see Emitter's own symbol()
    # method) -- nm reports that mangled name here directly, so
    # the expected names need the identical leading underscore on
    # macOS, not just when Hornet-generated code refers to them.
    names = [
        "hornet_argv_get",
        "hornet_bytes",
        "hornet_copy_items",
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
        "hornet_alloc",
        "hornet_alloc_atomic",
        "hornet_alloc_zeroed",
        "hornet_alloc_zeroed_atomic",
        "hornet_close_fd",
        "hornet_error_message",
        "hornet_exit",
        "hornet_format_begin",
        "hornet_format_text",
        "hornet_format_value",
        "hornet_get_env",
        "hornet_hash_bytes",
        "hornet_is_terminal",
        "hornet_make_dirs",
        "hornet_make_temp_dir",
        "hornet_net_error",
        "hornet_open_read",
        "hornet_open_write",
        "hornet_panic",
        "hornet_panic_at",
        "hornet_panic_enum_value",
        "hornet_panic_index",
        "hornet_panic_missing_key",
        "hornet_panic_slice_end",
        "hornet_panic_slice_end_capacity",
        "hornet_panic_slice_order",
        "hornet_panic_slice_start",
        "hornet_panic_slice_start_capacity",
        "hornet_path_exists",
        "hornet_print",
        "hornet_process_id",
        "hornet_process_run",
        "hornet_read_fd",
        "hornet_read_line",
        "hornet_remove",
        "hornet_rename",
        "hornet_slice_grow",
        "hornet_slice_grow_atomic",
        "hornet_system_error",
        "hornet_tcp_close",
        "hornet_tcp_connect",
        "hornet_tcp_read",
        "hornet_tcp_write",
        "hornet_write_fd",
    ]
    expected_names = [f"_{n}" for n in names] if target.os == 'macos' else names
    assert sorted(external_symbols) == sorted(expected_names)
