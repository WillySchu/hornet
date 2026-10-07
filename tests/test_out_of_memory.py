"""Being refused memory is a panic, `panic: out of memory (allocating N bytes)`: every allocation,
the compiled code's and the runtime's own, goes through the runtime's checked hornet_alloc and
hornet_alloc_zeroed. (Memory a system grants and can't then supply is another matter: the system
kills the program, and there is nothing to check.)"""

import re
import signal
import subprocess
import sys

import pytest

from build import build_executable, c_compiler, executable_name, link_flags, runtime_object
from compile import compile_to_asm
from target import default_target
from tests.targets import WINDOWS_ABORT_EXIT_CODE, each_runnable_target, run_binary
from tests.test_compiler import GCC_SKIP, compile_and_run

ROOT_OF_ALL = (
    "from 'os' import read_stdin\n"
    "from 'errors' import Error\n"
    "from 'strings' import Builder, repeat\n"
    "type Point struct:\n"
    "    int x\n"
    "    [3000]int big\n"
    "type Shape is Point | str | none\n"
    "def *Point make():\n"
    "    Point p\n"
    "    return &p\n"
    "def int main():\n"
    "    [4000]int local\n"
    "    []int xs = [1, 2, 3]\n"
    "    xs = append(xs, 4)\n"
    "    dict[str]int d = dict[str]int{'a': 1}\n"
    "    d['b'] = 2\n"
    "    str s = 'x' + repeat('y', 3) + format('{}', xs)\n"
    "    Shape shape = *make()\n"
    "    Builder b\n"
    "    b.write(s)\n"
    "    print(bytes(s))\n"
    "    print(shape is Point)\n"
    "    return len(d) + local[0]\n"
)


def test_no_allocation_goes_around_the_runtime(tmp_path):
    path = tmp_path / "p.ht"
    path.write_text(ROOT_OF_ALL)
    for target in ('x86_64-linux', 'aarch64-macos', 'x86_64-windows'):
        asm = compile_to_asm(str(path), target)
        calls = set(re.findall(r"\b(?:call|bl)\s+_?(\w+)", asm))
        assert {'hornet_alloc', 'hornet_alloc_zeroed'} <= calls, target
        assert not calls & {'malloc', 'calloc', 'realloc'}, target


@GCC_SKIP
def test_an_allocation_too_large_to_exist_panics():
    # 2**48 bytes: more than any of these systems has addresses for.
    result = compile_and_run(
        "def int main():\n"
        "    print('before')\n"
        "    [35184372088832]int big\n"
        "    big[0] = 1\n"
        "    return big[0]\n")
    assert (result.returncode, result.stdout, result.stderr) == (
        -signal.SIGABRT, "before\n", "panic: out of memory (allocating 281474976710656 bytes)\n")


GROWING = {
    'a slice': "    []int xs\n    while true:\n        xs = append(xs, len(xs))\n",
    'a str': "    str s = 'x'\n    while true:\n        s = s + s\n",
    'a dict': "    dict[int]int d\n    int n = 0\n    while true:\n        d[n] = n\n        n += 1\n",
    'format': "    str s = 'x'\n    while true:\n        s = format('{}{}', s, s)\n",
    'heap variables': (
        "    []*[2000]int kept\n    while true:\n        [2000]int block\n        kept = append(kept, &block)\n"),
}


@GCC_SKIP
@pytest.mark.skipif(sys.platform != 'linux', reason="uses a Linux address-space limit to have memory refused")
@pytest.mark.parametrize("body", GROWING.values(), ids=GROWING.keys())
def test_growing_without_end_under_a_memory_limit_panics(body, tmp_path):
    import resource
    (tmp_path / "p.ht").write_text("def int main():\n" + body + "    return 0\n")
    exe = tmp_path / "p"
    build_executable(str(tmp_path / "p.ht"), str(exe), target=default_target())

    def limit():
        resource.setrlimit(resource.RLIMIT_AS, (400 * 1024 * 1024, 400 * 1024 * 1024))
    result = subprocess.run([str(exe)], capture_output=True, text=True, preexec_fn=limit, timeout=120)
    assert result.returncode == -signal.SIGABRT
    assert re.fullmatch(r"panic: out of memory \(allocating \d+ bytes\)\n", result.stderr), result.stderr


C_PROGRAM = r"""
#include <stdint.h>
#include <stdio.h>
#include <string.h>
void *hornet_alloc(int64_t size);
void *hornet_alloc_zeroed(int64_t count, int64_t size);
int main(int argc, char **argv) {
    (void)argv;
    char *none = hornet_alloc(0), *zeroed = hornet_alloc_zeroed(0, 8), *some = hornet_alloc_zeroed(4, 2);
    printf("%d %d %d\n", none != NULL, zeroed != NULL, memcmp(some, "\0\0\0\0\0\0\0\0", 8) == 0);
    fflush(stdout);
    if (argc == 2) hornet_alloc_zeroed(INT64_MAX / 2, 16);   /* a product no int64 holds */
    if (argc == 3) hornet_alloc(-8);                          /* a size that went negative */
    if (argc == 4) hornet_alloc_zeroed(-1, 8);
    return 0;
}
"""


@each_runnable_target
def test_the_allocators_themselves(target, tmp_path):
    source, binary = tmp_path / "alloc.c", tmp_path / executable_name("alloc", target)
    source.write_text(C_PROGRAM)
    subprocess.run([*c_compiler(target), str(source), str(runtime_object(target)), "-o", str(binary),
                    *link_flags(target)], check=True, capture_output=True)
    aborted = (-signal.SIGABRT, WINDOWS_ABORT_EXIT_CODE)
    fine = run_binary(target, [binary], capture_output=True, text=True)
    assert (fine.returncode, fine.stdout, fine.stderr) == (0, "1 1 1\n", "")
    for arguments, message in [
        (["a"], "panic: out of memory (allocating 4611686018427387903 x 16 bytes)\n"),
        (["a", "b"], "panic: out of memory (allocating -8 bytes)\n"),
        (["a", "b", "c"], "panic: out of memory (allocating -1 x 8 bytes)\n"),
    ]:
        result = run_binary(target, [binary, *arguments], capture_output=True, text=True)
        assert result.returncode in aborted and (result.stdout, result.stderr) == ("1 1 1\n", message), arguments
