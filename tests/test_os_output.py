"""stdlib/os.ht output and process control: write_file, write_stdout, write_stderr, exit."""

import subprocess
import tempfile
from pathlib import Path

from build import build_executable
from tests.test_compiler import GCC_SKIP, panic_message
from tests.targets import on_every_target, posix, run_binary, windows


def compile_and_run(source: str, only=None) -> subprocess.CompletedProcess:
    """Full driver (so stdlib imports resolve), then build and run for every E2E target (that `only`
    accepts; None if it accepts none)."""
    def build_and_run(target):
        with tempfile.TemporaryDirectory() as tmp:
            src, exe = Path(tmp) / 'p.ht', Path(tmp) / 'p'
            src.write_text(source)
            build_executable(str(src), str(exe), target=target)
            return run_binary(target, [exe], capture_output=True, text=True, timeout=10)
    return on_every_target(build_and_run, source, only=only)


def _run(body: str, only=None):
    return compile_and_run(
        "from 'os' import write_stdout, write_stderr, write_file, read_file, exit\n"
        "from 'errors' import Error, StrResult, IntResult, must_str, must_int\n"
        "def int main():\n" + ''.join(f"    {line}\n" for line in body.strip('\n').split('\n')), only)


@GCC_SKIP
def test_write_stdout_has_no_newline_and_stays_ordered_with_print():
    r = _run("write_stdout('a')\nprint(1)\nwrite_stdout('b')\nwrite_stdout('\\n')\nprint('c')\nreturn 0")
    assert r.stdout == "a1\nb\nc\n"


@GCC_SKIP
def test_write_stderr_goes_to_stderr():
    r = _run("write_stderr('oops\\n')\nwrite_stdout('ok')\nreturn 0")
    assert (r.stdout, r.stderr) == ('ok', 'oops\n')


@GCC_SKIP
def test_write_file_round_trips_bytes_including_nul(tmp_path):
    path = tmp_path / 'out.bin'
    r = _run(f"write_file('{path}', 'hi\\x00there\\xff')\nstr back = must_str("
             f"read_file('{path}'))\nprint(len(back))\nreturn 0")
    assert r.stdout == "9\n"
    assert path.read_bytes() == b'hi\x00there\xff'


@GCC_SKIP
def test_write_file_truncates_and_handles_large_content(tmp_path):
    path = tmp_path / 'big.txt'
    path.write_text('x' * 1_000_000)
    r = _run(
        "str s = 'abcdefghijklmnop'\n"
        "for int i = 0; i < 16; i += 1:\n"
        "    s = s + s\n"
        f"write_file('{path}', s)\n"
        f"str back = must_str(read_file('{path}'))\n"
        "print(len(back))\n"
        "print(back == s)\n"
        "return 0")
    assert r.stdout == f"{16 * 2 ** 16}\ntrue\n"
    assert path.stat().st_size == 16 * 2 ** 16


@GCC_SKIP
def test_exit_sets_code_and_keeps_earlier_output():
    r = compile_and_run(
        "from 'os' import write_stdout, exit\n"
        "def stop(int code):\n"
        "    write_stdout('bye')\n"
        "    exit(code)\n"
        "def int main():\n"
        "    print(1)\n"
        "    stop(7)\n"
        "    print(2)\n"
        "    return 0\n")
    assert (r.returncode, r.stdout) == (7, "1\nbye")


@GCC_SKIP
def test_write_file_to_missing_directory_returns_an_error(tmp_path):
    r = _run(
        f"IntResult w = write_file('{tmp_path}/no/such/dir/f', 'x')\n"
        "if w is Error:\n"
        "    print(w.message)\n"
        f"must_int(write_file('{tmp_path}/no/such/dir/f', 'x'))\n"
        "return 0")
    message = f"could not open '{tmp_path}/no/such/dir/f' for writing: No such file or directory"
    assert (r.stdout, panic_message(r.stderr)) == (message + "\n", message + "\n") and r.returncode != 0


def test_write_functions_return_bytes_written(tmp_path):
    r = _run(
        f"print(must_int(write_file('{tmp_path}/f', 'hello')))\n"
        "IntResult n = write_stdout('abc')\n"
        "print(must_int(n))\n"
        "return 0")
    assert r.stdout == "5\nabc3\n"


def test_read_file_of_a_directory_reports_an_error(tmp_path):
    body = f"StrResult c = read_file('{tmp_path}')\nif c is Error:\n    print(c.message)\nreturn 0"
    # A directory opens and then can't be read; Windows won't open one.
    on_posix, on_windows = _run(body, only=posix), _run(body, only=windows)
    assert on_posix is None or on_posix.stdout == f"could not read '{tmp_path}': read failed: Is a directory\n"
    assert on_windows is None or on_windows.stdout == f"could not open '{tmp_path}': Permission denied\n"
    assert on_posix is not None or on_windows is not None


@GCC_SKIP
def test_files_are_closed_however_reading_or_writing_them_ends(tmp_path):
    # Many more opens than the process is allowed to hold at once (where that can be limited): each
    # file must have been closed again, on the paths that fail as on those that succeed.
    path, missing = tmp_path / 'again.txt', tmp_path / 'no-such-directory' / 'x'
    source = (
        "from 'os' import write_file, read_file\n"
        "from 'errors' import Error, StrResult, IntResult\n"
        "def int main():\n"
        "    int done = 0\n"
        "    for int i = 0; i < 300; i += 1:\n"
        f"        IntResult written = write_file('{path}', 'again')\n"
        f"        StrResult back = read_file('{path}')\n"
        f"        IntResult nowhere = write_file('{missing}', 'lost')\n"
        "        if written is int and back is str and nowhere is Error:\n"
        "            done += 1\n"
        "    print(done)\n"
        "    return 0\n")

    def at_most_64_open():
        import resource
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))

    def build_and_run(target):
        with tempfile.TemporaryDirectory() as tmp:
            src, exe = Path(tmp) / 'p.ht', Path(tmp) / 'p'
            src.write_text(source)
            build_executable(str(src), str(exe), target=target)
            limit = {'preexec_fn': at_most_64_open} if posix(target) else {}  # (Wine needs more of its own)
            return run_binary(target, [exe], capture_output=True, text=True, timeout=20, **limit)
    r = on_every_target(build_and_run, source)
    assert (r.stdout, r.stderr) == ("300\n", "")
