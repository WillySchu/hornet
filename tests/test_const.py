"""Top-level `const` declarations."""

import pytest

from tests.test_compiler import GCC_SKIP, assert_program_semantic_error, assert_program_stdout


@GCC_SKIP
def test_constants_of_every_kind():
    assert_program_stdout(
        "const int LIMIT = 1 << 4\n"
        "const int32 SMALL = int32(-5)\n"
        "const bool DEBUG = LIMIT > 10 and not false\n"
        "const str GREETING = 'hi ' + 'there'\n"
        "const int8 WRAP = int8(200)\n"
        "const byte NL = \"\\n\"\n"
        "const int MIN = -9223372036854775807 - 1\n"
        "const int TWICE = LATER * 2\n"
        "const int LATER = 21\n"
        "type P struct:\n"
        "    int v\n"
        "    def int scaled(p):\n"
        "        return p.v * LIMIT\n"
        "def int main():\n"
        "    print(LIMIT)\n"
        "    print(SMALL)\n"
        "    print(DEBUG)\n"
        "    print(GREETING)\n"
        "    print(WRAP)\n"
        "    print(NL == \"\\n\")\n"
        "    print(MIN)\n"
        "    print(TWICE)\n"
        "    print(P(2).scaled())\n"
        "    []int s = []int[LIMIT, TWICE]\n"
        "    print(s)\n"
        "    if DEBUG:\n"
        "        print(len(GREETING))\n"
        "    return 0\n",
        "16\n-5\ntrue\nhi there\n-56\ntrue\n-9223372036854775808\n42\n32\n[]int[16, 42]\n8\n",
    )


@GCC_SKIP
def test_constants_across_modules(tmp_path):
    kinds = tmp_path / 'kinds.ht'
    kinds.write_text(
        "const int TK_IDENT = 1\n"
        "const int TK_NUMBER = TK_IDENT + 1\n"
        "const int _PRIVATE = 99\n"
        "const str NAME = 'kinds'\n"
        "def int private_plus(int x):\n"
        "    return x + _PRIVATE\n")
    main = tmp_path / 'main.ht'
    main.write_text(
        "from 'kinds' import TK_NUMBER, NAME as KNAME, private_plus\n"
        "import 'kinds'\n"
        "const str GREETING = 'hi ' + KNAME\n"
        "def int main():\n"
        "    print(kinds.TK_IDENT + TK_NUMBER)\n"
        "    print(GREETING)\n"
        "    print(private_plus(1))\n"
        "    return 0\n")
    from pathlib import Path
    from build import build_executable
    from tests.targets import on_every_target, run_binary
    exe = Path(tmp_path) / 'm'
    def build_and_run(target):
        build_executable(str(main), str(exe), target=target)
        return run_binary(target, [exe], capture_output=True, text=True)
    assert on_every_target(build_and_run).stdout == "3\nhi kinds\n100\n"


def test_private_constant_is_not_importable(tmp_path):
    (tmp_path / 'kinds.ht').write_text("const int _PRIVATE = 99\n")
    r = compile_and_run_expect_error(tmp_path, "from 'kinds' import _PRIVATE\ndef int main():\n    return 0\n")
    assert 'not visible outside the module' in r


def compile_and_run_expect_error(tmp_path, source: str) -> str:
    from compile import compile_to_asm
    from diagnostics import CompileError
    main = tmp_path / 'main.ht'
    main.write_text(source)
    with pytest.raises(CompileError) as e:
        compile_to_asm(str(main), 'x86_64-linux')
    return str(e.value)


@pytest.mark.parametrize('source,match', [
    ("const int A = 1\ndef int main():\n    A = 2\n    return 0\n", "Cannot assign to constant 'A'"),
    ("const int A = 1\ndef int main():\n    *int p = &A\n    return 0\n", "Cannot take the address of constant 'A'"),
    ("const int A = B\nconst int B = A\ndef int main():\n    return 0\n", "defined in terms of itself"),
    ("def int f():\n    return 1\nconst int A = f()\ndef int main():\n    return 0\n", "must be built from literals"),
    ("const int A = 'x'\ndef int main():\n    return 0\n", "declared int but its value has type str"),
    ("const int A = 1 / 0\ndef int main():\n    return 0\n", "Division by zero"),
    ("const int A = (-9223372036854775807 - 1) / -1\ndef int main():\n    return 0\n", "overflow in division"),
    ("const int f = 1\ndef int f():\n    return 0\ndef int main():\n    return 0\n", "collides with a function"),
    ("const [2]int A = [1, 2]\ndef int main():\n    return 0\n", "must be an integer type, bool, str, or an enum"),
    ("const int8 A = 300\ndef int main():\n    return 0\n", "out of range for int8"),
    ("const int A = 1\nconst int A = 2\ndef int main():\n    return 0\n", "already declared"),
])
def test_bad_constants_are_rejected(source, match):
    assert_program_semantic_error(source, match=match)


@pytest.mark.parametrize('binding', [
    "    int A = 2\n",
    "    for A in []int[1]:\n        print(A)\n",
])
def test_local_names_cannot_reuse_a_constant(tmp_path, binding):
    message = compile_and_run_expect_error(tmp_path, "const int A = 1\ndef int main():\n" + binding + "    return 0\n")
    assert "is a constant here and can't also be a variable name" in message


def test_parameter_cannot_reuse_a_constant(tmp_path):
    message = compile_and_run_expect_error(tmp_path, "const int A = 1\ndef int main(int A):\n    return 0\n")
    assert "can't also be a variable name" in message


@GCC_SKIP
def test_constant_array_sizes():
    assert_program_stdout(
        "const int N = 2\n"
        "const int8 SMALL = int8(3)\n"
        "type Pair = [N]int\n"
        "type P struct:\n"
        "    [N]int xs\n"
        "    [N * 2][SMALL]int grid\n"
        "type Q struct:\n"
        "    int y\n"
        "type PQ is P | Q\n"
        "def [N]int doubled([N]int a):\n"
        "    return [N]int[a[0] * 2, a[1] * 2]\n"
        "def int main():\n"
        "    Pair pair = [5, 6]\n"
        "    P p\n"
        "    p.xs = doubled(pair)\n"
        "    p.grid[3][2] = 7\n"
        "    PQ v = p\n"
        "    dict[str][N]int d\n"
        "    d['k'] = p.xs\n"
        "    print(d['k'])\n"
        "    print(len(p.grid))\n"
        "    print(len(p.grid[0]))\n"
        "    if v is P:\n"
        "        print(v.grid[3][2])\n"
        "    return 0\n",
        "[2]int[10, 12]\n4\n3\n7\n",
    )


@pytest.mark.parametrize('source,match', [
    ("def int main():\n    int n = 3\n    [n]int a\n    return 0\n", "Array size must be a constant expression, but 'n' isn't a constant"),
    ("def int f():\n    return 2\ndef int main():\n    [f()]int a\n    return 0\n", "Array size must be a constant expression, not a call"),
    ("const int N = 2 - 5\ndef int main():\n    [N]int a\n    return 0\n", "Array size must be positive, got -3"),
    ("const str N = 'x'\ndef int main():\n    [N]int a\n    return 0\n", "Array size must be an integer, got str"),
    ("const int N = 3\ndef int main():\n    [N]int a = [1, 2]\n    return 0\n", r"declared \[3\]int"),
])
def test_bad_constant_array_sizes_are_rejected(source, match):
    assert_program_semantic_error(source, match=match)


@GCC_SKIP
def test_constant_array_sizes_across_modules(tmp_path):
    (tmp_path / 'lib.ht').write_text(
        "const int N = 3\n"
        "const int _M = 2\n"
        "type Buf struct:\n"
        "    [N]int xs\n"
        "    [_M]int ys\n"
        "def [N]int make():\n"
        "    return [N]int[1, 2, 3]\n")
    main = tmp_path / 'main.ht'
    main.write_text(
        "from 'lib' import N, Buf, make\n"
        "import 'lib'\n"
        "def int main():\n"
        "    [N]int a = make()\n"
        "    [lib.N * 2]int b\n"
        "    Buf buf = Buf(a, [7, 8])\n"
        "    print(len(b))\n"
        "    print(buf.xs[2] + buf.ys[1])\n"
        "    return 0\n")
    from build import build_executable
    from tests.targets import on_every_target, run_binary
    exe = tmp_path / 'm'
    def build_and_run(target):
        build_executable(str(main), str(exe), target=target)
        return run_binary(target, [exe], capture_output=True, text=True)
    assert on_every_target(build_and_run).stdout == "6\n11\n"
