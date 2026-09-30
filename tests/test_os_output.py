"""stdlib/os.ht output and process control: write_file, write_stdout, write_stderr, exit."""

import subprocess
import tempfile
from pathlib import Path

from build import build_executable
from tests.test_compiler import ASM_PLATFORM, GCC_SKIP


def compile_and_run(source: str) -> subprocess.CompletedProcess:
    """Full driver (so stdlib imports resolve), then run."""
    with tempfile.TemporaryDirectory() as tmp:
        src, exe = Path(tmp) / 'p.ht', Path(tmp) / 'p'
        src.write_text(source)
        build_executable(str(src), str(exe), platform=ASM_PLATFORM)
        return subprocess.run([str(exe)], capture_output=True, text=True, timeout=10)


def _run(body: str):
    return compile_and_run(
        "from 'os' import write_stdout, write_stderr, write_file, read_file, exit\n"
        "def int main():\n" + ''.join(f"    {line}\n" for line in body.strip('\n').split('\n')))


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
    r = _run(f"write_file('{path}', 'hi\\x00there\\xff')\nstr back = read_file('{path}')\nprint(len(back))\nreturn 0")
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
        f"str back = read_file('{path}')\n"
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
def test_write_file_to_missing_directory_panics(tmp_path):
    r = _run(f"write_file('{tmp_path}/no/such/dir/f', 'x')\nreturn 0")
    assert r.returncode != 0
    assert 'could not open file for writing' in r.stdout
