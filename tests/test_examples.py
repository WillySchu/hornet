"""Tests for examples/wc.ht -- a real, if deliberately simple, CLI
program (no -l/-w/-c flags, no column-aligned output) exercising
get_args/read_file/read_stdin/int_to_str together, end to end.

Unlike tests/test_merge.py's own tests (which each write a fresh,
synthetic .ht file into a tmpdir), these compile the actual, checked-
in examples/wc.ht directly -- the whole point is proving the real
example program works, not a stand-in for it.
"""

import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from compile import compile_to_asm

RUNTIME_C_PATH = Path(__file__).parent.parent / "runtime" / "runtime.c"
WC_PATH = Path(__file__).parent.parent / "examples" / "wc.ht"

HOST_IS_MACOS = sys.platform == "darwin"
ASM_PLATFORM = "macos" if HOST_IS_MACOS else "linux"

GCC_AVAILABLE = shutil.which("gcc") is not None
GCC_SKIP = pytest.mark.skipif(
    not GCC_AVAILABLE,
    reason="gcc not found on PATH; these tests compile and execute real binaries",
)


def _write(tmpdir: str, name: str, content: str) -> str:
    path = Path(tmpdir) / name
    path.write_text(content)
    return str(path)


def _run_wc(tmpdir: str, args: list = None, stdin: str = None) -> subprocess.CompletedProcess:
    """Compiles the real examples/wc.ht (not a synthetic copy),
    links, and runs it with the given argv[1:]/stdin -- mirrors
    tests/test_merge.py's own _compile_and_run, but always against
    this one, fixed entry file."""
    asm = compile_to_asm(str(WC_PATH), platform=ASM_PLATFORM)

    asm_path = Path(tmpdir) / "program.s"
    asm_path.write_text(asm, encoding="latin-1")
    runtime_o = Path(tmpdir) / "runtime.o"
    runtime_cc_cmd = ["gcc"]
    if HOST_IS_MACOS:
        runtime_cc_cmd += ["-arch", "x86_64"]
    runtime_cc_cmd += ["-c", str(RUNTIME_C_PATH), "-o", str(runtime_o)]
    subprocess.run(runtime_cc_cmd, check=True, capture_output=True)
    binary = Path(tmpdir) / "program"
    gcc_cmd = ["gcc"]
    if HOST_IS_MACOS:
        gcc_cmd += ["-arch", "x86_64"]
    gcc_cmd += [str(asm_path), str(runtime_o), "-o", str(binary)]
    link = subprocess.run(gcc_cmd, capture_output=True, text=True)
    assert link.returncode == 0, f"link failed:\n{link.stderr}\n--- asm ---\n{asm}"
    return subprocess.run([str(binary), *(args or [])], input=stdin or "", capture_output=True, text=True)


@GCC_SKIP
def test_wc_single_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = _write(tmpdir, "data.txt", "line one\nline two\nline three")
        result = _run_wc(tmpdir, args=[data_path])
        # 2 lines (only 2 newline BYTES here -- the third line has no
        # trailing newline, so count_lines never counts it, matching
        # real wc's own identical newline-counting convention), 6
        # words, 28 bytes -- independently confirmed against the
        # real, system wc during this feature's own development.
        assert result.stdout == f"2 6 28 {data_path}\n"


@GCC_SKIP
def test_wc_stdin_when_no_file_argument_given():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = _run_wc(tmpdir, stdin="line one\nline two\nline three")
        assert result.stdout == "2 6 28\n"


@GCC_SKIP
def test_wc_multiple_files_prints_a_total_line():
    with tempfile.TemporaryDirectory() as tmpdir:
        first = _write(tmpdir, "first.txt", "one two three\nfour five\n")
        second = _write(tmpdir, "second.txt", "six\n")
        result = _run_wc(tmpdir, args=[first, second])
        assert result.stdout == (
            f"2 5 24 {first}\n"
            f"1 1 4 {second}\n"
            f"3 6 28 total\n"
        )


@GCC_SKIP
def test_wc_single_file_has_no_total_line():
    """A total line only ever appears with more than one file -- see
    format_counts' own len(args) > 2 check in examples/wc.ht."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = _write(tmpdir, "only.txt", "just one file\n")
        result = _run_wc(tmpdir, args=[data_path])
        assert result.stdout == f"1 3 14 {data_path}\n"
        assert "total" not in result.stdout


@GCC_SKIP
def test_wc_empty_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = _write(tmpdir, "empty.txt", "")
        result = _run_wc(tmpdir, args=[data_path])
        assert result.stdout == f"0 0 0 {data_path}\n"


@GCC_SKIP
def test_wc_panics_on_a_missing_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        missing_path = str(Path(tmpdir) / "does_not_exist.txt")
        result = _run_wc(tmpdir, args=[missing_path])
        assert result.returncode == -signal.SIGABRT
        assert "could not open file" in result.stdout
