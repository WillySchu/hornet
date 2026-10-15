"""Programs run with their address space limited (RLIMIT_AS, as `ulimit -v`): one that allocates far
more than the limit, keeping little, finishes; one that keeps more than it ends with the out-of-memory
panic. Native Linux only: qemu reserves address space of its own, and macOS doesn't enforce the limit."""

import re
import resource
import signal
import subprocess
import sys

import pytest

from build import build_executable, run_prefix
from target import default_target
from tests.test_compiler import GCC_SKIP

LIMIT = 256 << 20

pytestmark = [
    GCC_SKIP,
    pytest.mark.skipif(sys.platform != 'linux' or run_prefix(default_target()) != [], reason='native Linux only'),
]


def _run_limited(source: str, tmp_path) -> subprocess.CompletedProcess:
    src, exe = tmp_path / 'p.ht', tmp_path / 'p'
    src.write_text(source)
    build_executable(str(src), str(exe), target=default_target())
    return subprocess.run([str(exe)], capture_output=True, text=True, timeout=120,
                          preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_AS, (LIMIT, LIMIT)))


def test_allocating_far_more_than_the_limit_keeping_little_finishes(tmp_path):
    # 3 GB in large objects (800 KB strings, and the smaller ones they grow from), and 200,000 small
    # ones, a few at a time alive.
    result = _run_limited(
        "from 'std:strings' import repeat\n"
        "def int main():\n"
        "    str chunk = repeat('abcdefgh', 1000)\n"
        "    int kept = 0\n"
        "    for int i = 0; i < 2000; i += 1:\n"
        "        str big = repeat(chunk, 100)\n"
        "        []str words\n"
        "        for int j = 0; j < 100; j += 1:\n"
        "            words = append(words, format('{}', j))\n"
        "        kept += len(big) / 100000 + len(words)\n"
        "    print(kept)\n"
        "    return 0\n", tmp_path)
    assert (result.returncode, result.stdout, result.stderr) == (0, "216000\n", "")


def test_keeping_more_than_the_limit_panics(tmp_path):
    result = _run_limited(
        "from 'std:strings' import repeat\n"
        "def int main():\n"
        "    str chunk = repeat('abcdefgh', 1000)\n"
        "    []str kept\n"
        "    for int i = 0; i < 100000; i += 1:\n"
        "        kept = append(kept, repeat(chunk, 100))\n"
        "    print(len(kept))\n"
        "    return 0\n", tmp_path)
    assert (result.returncode, result.stdout) == (-signal.SIGABRT, "")
    assert re.fullmatch(r"panic: out of memory \(allocating [0-9x ]+ bytes\)\n", result.stderr), result.stderr
