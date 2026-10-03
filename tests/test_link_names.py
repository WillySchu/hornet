"""Linker symbols of Hornet functions (ir.typed_builder.link_name): an entry-file function never
stands in for a libc or runtime symbol of the same name."""

from compile import generate_asm
from ir.typed_builder import link_name
from tests.test_compiler import GCC_SKIP, _parse, analyze, assert_program_panics, assert_program_stdout


def test_link_names():
    assert link_name('main') == 'main'
    assert link_name('memcpy') == 'memcpy$'
    assert link_name('os$exit') == 'os$exit'
    assert link_name('Point.sum') == 'Point.sum'


def test_entry_function_and_runtime_symbol_of_one_name_stay_apart():
    asm = generate_asm(analyze(_parse(
        "def int memcpy(int a):\n"
        "    return a\n"
        "def int main():\n"
        "    str s = 'a' + 'b'\n"
        "    return memcpy(len(s))\n"
    )), 'x86_64-linux')
    assert '.globl memcpy$\n' in asm and '.globl main\n' in asm and '.globl memcpy\n' not in asm
    assert 'call    memcpy$\n' in asm  # the program's own
    assert 'call    memcpy\n' in asm  # libc's, for the concatenation


@GCC_SKIP
def test_functions_named_like_libc_and_runtime_symbols():
    names = ['memcpy', 'memcmp', 'malloc', 'calloc', 'free', 'write', 'exit', 'abort',
             'hornet_print', 'hornet_panic', 'hornet_slice_grow', 'hornet_bytes', 'hornet_dict_set_str_key']
    functions = ''.join(f"def int {name}(int x):\n    return x + {i}\n" for i, name in enumerate(names))
    total = ' + '.join(f"{name}(1)" for name in names)
    assert_program_stdout(
        functions +
        "def int main():\n"
        "    str s = 'ab' + 'cd'\n"
        "    []int xs = [1, 2]\n"
        "    xs = append(xs, 3)\n"
        "    dict[str]int d\n"
        "    d[s] = len(xs)\n"
        "    print(s + s)\n"
        "    print(s == 'abcd')\n"
        "    print(xs)\n"
        "    print(d)\n"
        "    print(bytes(s))\n"
        f"    print({total})\n"
        "    return 0\n",
        "abcdabcd\ntrue\n[]int[1, 2, 3]\ndict[str]int{'abcd': 3}\n[]uint8[97, 98, 99, 100]\n"
        f"{sum(1 + i for i in range(len(names)))}\n",
    )


@GCC_SKIP
def test_a_function_named_hornet_panic_leaves_panics_alone():
    assert_program_panics(
        "def int hornet_panic(int x):\n"
        "    return x\n"
        "def int main():\n"
        "    [3]int a\n"
        "    int i = hornet_panic(5)\n"
        "    return a[i]\n",
        "array index out of bounds",
    )
