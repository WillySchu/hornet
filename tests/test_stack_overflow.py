"""Running out of stack is a panic: the runtime catches the fault, says `panic: stack overflow in
'f'`, and ends the program as any panic does. (No source position: the fault doesn't say where the
program was. The function is the program's own that was running, found from a table of its
functions that the compiler writes.) Any other bad memory access is left to crash as before."""

import signal
import subprocess

import pytest

from build import c_compiler, executable_name, link_flags, runtime_object
from tests.targets import (
    WINDOWS_ABORT_EXIT_CODE, build_and_run, each_e2e_target, each_runnable_target, run_binary, under_rosetta)
from tests.test_compiler import GCC_SKIP, compile_and_run

# Each: a program that never stops calling, and the functions it may be said to have been in.
RUNAWAY = {
    'one function': (
        "def int depth(int n):\n"
        "    return depth(n + 1) + 1\n"
        "def int main():\n"
        "    print('before')\n"
        "    return depth(0)\n", {'depth'}),
    'two functions': (                # whichever was running
        "def int ping(int n):\n"
        "    return pong(n + 1)\n"
        "def int pong(int n):\n"
        "    return ping(n) + 1\n"
        "def int main():\n"
        "    print('before')\n"
        "    return ping(0)\n", {'ping', 'pong'}),
    # Each call takes more than a page of stack at once, so the end of the stack is stepped past.
    'a large frame': (
        "def int wide(int n):\n"
        "    [1500]int table\n"
        "    table[n % 1500] = n\n"
        "    return wide(n + 1) + table[0]\n"
        "def int main():\n"
        "    print('before')\n"
        "    return wide(0)\n", {'wide'}),
    'a method': (
        "type Node struct:\n"
        "    int value\n"
        "    def int sum(self, int n):\n"
        "        return self.value + self.sum(n + 1)\n"
        "def int main():\n"
        "    print('before')\n"
        "    return Node(1).sum(0)\n", {'Node.sum'}),
    # The stack mostly runs out inside the C these call: the function named is the program's that called it.
    'formatting on the way down': (
        "def int chatty(int n):\n"
        "    str s = format('{} {} {}', n, n, n)\n"
        "    return chatty(n + 1) + len(s)\n"
        "def int main():\n"
        "    print('before')\n"
        "    return chatty(1)\n", {'chatty'}),
    'allocating on the way down': (
        "def int grow(int n):\n"
        "    []int xs = []int[n, n]\n"
        "    return grow(n + 1) + len(xs)\n"
        "def int main():\n"
        "    print('before')\n"
        "    return grow(0)\n", {'grow'}),
    # ... and not one it had called earlier, though those have been where the stack now ends.
    'after other calls returned': (
        "def int leaf(int n):\n"
        "    return n * 2\n"
        "def int helper(int n):\n"
        "    return leaf(n) + leaf(n + 1)\n"
        "def int walk(int n):\n"
        "    int h = helper(n)\n"
        "    str s = format('{}', h)\n"
        "    return walk(n + 1) + len(s)\n"
        "def int main():\n"
        "    print('before')\n"
        "    return walk(0)\n", {'walk'}),
}


@GCC_SKIP
@each_e2e_target
@pytest.mark.parametrize('source,functions', RUNAWAY.values(), ids=RUNAWAY.keys())
def test_a_runaway_recursion_panics(source, functions, target):
    # (A target at a time: which of two functions was running is not for them to agree on.)
    result = build_and_run(source, target)
    said = {f"panic: stack overflow in '{function}'\n" for function in functions}
    assert (result.returncode, result.stdout) == (-signal.SIGABRT, "before\n")
    # Rosetta 2 can't always hand the fault to the program it is translating: when the stack runs out
    # on certain instructions it stops with an error of its own. The program still ends as aborted.
    assert result.stderr in said or (under_rosetta(target) and result.stderr.startswith("rosetta error: "))


@each_runnable_target
def test_with_no_table_of_functions_none_is_named(target, tmp_path):
    # The runtime linked with C alone: there is no program's function to name.
    source, binary = tmp_path / "deep.c", tmp_path / executable_name("deep", target)
    source.write_text(
        "volatile int sink;\n"
        "int deep(int n) { volatile char pad[256]; pad[n % 256] = 1; sink = pad[0]; return deep(n + 1) + sink; }\n"
        "int main(void) { return deep(0); }\n")
    subprocess.run([*c_compiler(target), "-O0", str(source), str(runtime_object(target)), "-o", str(binary),
                    *link_flags(target)], check=True, capture_output=True)
    result = run_binary(target, [binary], capture_output=True, text=True)
    if not under_rosetta(target):
        assert result.stderr == "panic: stack overflow\n"
    assert result.returncode in (-signal.SIGABRT, WINDOWS_ABORT_EXIT_CODE)


def test_the_table_the_runtime_reads():
    from backend.common.function_table import shown_name, table_lines
    # A function is named as it is declared: no module prefix, nor the `$` an entry file's is linked under.
    assert [shown_name(name) for name in ('main', 'depth$', 'strings$repeat', 'Node.sum', 'strings$Builder.write')] \
        == ['main', 'depth', 'repeat', 'Node.sum', 'Builder.write']
    assert table_lines(['depth$', 'main'], lambda name: '_' + name) == [
        "    .data",
        "    .globl _hornet_functions",
        "    .p2align 3",
        "_hornet_functions:",
        "    .quad _depth$, .Lhornet_function_end_0, .Lhornet_function_name_0",
        "    .quad _main, .Lhornet_function_end_1, .Lhornet_function_name_1",
        "    .quad 0, 0, 0",                               # the end
        ".Lhornet_function_name_0:",
        '    .asciz "depth"',
        ".Lhornet_function_name_1:",
        '    .asciz "main"',
    ]


def test_every_function_s_end_is_labelled_on_every_target():
    from compile import generate_asm
    from target import Target
    from tests.test_compiler import _parse, analyze
    for target in (Target('x86_64', 'linux'), Target('x86_64', 'macos'), Target('x86_64', 'windows'),
                   Target('aarch64', 'linux'), Target('aarch64', 'macos')):
        asm = generate_asm(analyze(_parse(RUNAWAY['two functions'][0])), target=target)
        prefix = '_' if target.os == 'macos' else ''
        for index, name in enumerate(('ping$', 'pong$', 'main')):
            assert asm.index(f"\n{prefix}{name}:\n") < asm.index(f"\n.Lhornet_function_end_{index}:\n")
            assert f"    .quad {prefix}{name}, .Lhornet_function_end_{index}, .Lhornet_function_name_{index}\n" in asm
        assert f"    .globl {prefix}hornet_functions\n" in asm


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
