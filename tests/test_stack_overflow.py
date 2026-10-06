"""Running out of stack is a panic: the runtime catches the fault, says `panic: stack overflow`,
and ends the program as any panic does. (No source position: the fault doesn't say where the
program was.) Any other bad memory access is left to crash as before."""

import signal
import subprocess

import pytest

from build import c_compiler, executable_name, link_flags, runtime_object
from tests.targets import WINDOWS_ABORT_EXIT_CODE, each_runnable_target, run_binary
from tests.test_compiler import GCC_SKIP, compile_and_run

RUNAWAY = {
    'one function': (
        "def int depth(int n):\n"
        "    return depth(n + 1) + 1\n"
        "def int main():\n"
        "    print('before')\n"
        "    return depth(0)\n"),
    'two functions': (
        "def int ping(int n):\n"
        "    return pong(n + 1)\n"
        "def int pong(int n):\n"
        "    return ping(n) + 1\n"
        "def int main():\n"
        "    print('before')\n"
        "    return ping(0)\n"),
    # Each call takes more than a page of stack at once, so the end of the stack is stepped past.
    'a large frame': (
        "def int wide(int n):\n"
        "    [1500]int table\n"
        "    table[n % 1500] = n\n"
        "    return wide(n + 1) + table[0]\n"
        "def int main():\n"
        "    print('before')\n"
        "    return wide(0)\n"),
    'a method': (
        "type Node struct:\n"
        "    int value\n"
        "    def int sum(self, int n):\n"
        "        return self.value + self.sum(n + 1)\n"
        "def int main():\n"
        "    print('before')\n"
        "    return Node(1).sum(0)\n"),
}


@GCC_SKIP
@pytest.mark.parametrize('source', RUNAWAY.values(), ids=RUNAWAY.keys())
def test_a_runaway_recursion_panics(source):
    result = compile_and_run(source)  # every target must agree
    assert (result.returncode, result.stdout, result.stderr) == (-signal.SIGABRT, "before\n", "panic: stack overflow\n")


@GCC_SKIP
def test_a_deep_recursion_that_fits_is_untouched():
    result = compile_and_run(
        "def int depth(int n):\n"
        "    if n == 0:\n"
        "        return 0\n"
        "    return depth(n - 1) + 1\n"
        "def int main():\n"
        "    print(depth(50000))\n"
        "    return 0\n")
    assert (result.returncode, result.stdout, result.stderr) == (0, "50000\n", "")


@each_runnable_target
def test_another_bad_memory_access_still_just_crashes(target, tmp_path):
    source, binary = tmp_path / "bad.c", tmp_path / executable_name("bad", target)
    source.write_text("int main(void) { volatile int *nowhere = (int *)8; return *nowhere; }\n")
    subprocess.run([*c_compiler(target), str(source), str(runtime_object(target)), "-o", str(binary),
                    *link_flags(target)], check=True, capture_output=True)
    result = run_binary(target, [binary], capture_output=True)
    assert b"stack overflow" not in result.stderr
    assert result.returncode not in (0, -signal.SIGABRT, WINDOWS_ABORT_EXIT_CODE)
