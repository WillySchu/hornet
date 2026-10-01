"""Programs built for aarch64-linux and run under qemu-user, checked against expected output
and against x86-64. The whole end-to-end suite also runs on aarch64-linux where it can."""

import pytest

from build import can_run
from target import Target
from tests.targets import build_and_run
from tests.test_random_programs import Gen

A64 = Target('aarch64', 'linux')
X86 = Target('x86_64', 'linux')
pytestmark = pytest.mark.skipif(not can_run(A64), reason='no aarch64-linux toolchain and qemu-user')


def _same_as_x86(source: str) -> str:
    a = build_and_run(source, A64)
    if can_run(X86):
        x = build_and_run(source, X86)
        assert (a.returncode, a.stdout) == (x.returncode, x.stdout), f"aarch64 {a}\nx86-64 {x}"
    return a


PROGRAMS = {
    'fib': ("def int fib(int n):\n    if n < 2:\n        return n\n    return fib(n - 1) + fib(n - 2)\n"
            "def int main():\n    print(fib(20))\n    return 7\n", 7, "6765\n"),
    'print_scalars': ("def int main():\n    print(-42)\n    print('hello\\n')\n    print(true)\n    print(int8(-5))\n"
                      "    print(uint8(250))\n    print(int32(-2147483648))\n    print(9223372036854775807)\n    return 0\n",
                      0, "-42\nhello\n\ntrue\n-5\n250\n-2147483648\n9223372036854775807\n"),
    'arithmetic': ("def int main():\n    int a = 1000003\n    int b = -7\n"
                   "    print(a + b)\n    print(a - b)\n    print(a * b)\n    print(a / b)\n    print(a % b)\n"
                   "    print(a & 255)\n    print(a | 1)\n    print(a ^ b)\n    print(a << 40)\n    print(b >> 1)\n"
                   "    print(-a)\n    print(~a)\n    print(a * 123456789012)\n    return 0\n", 0, None),
    'narrow_wraparound': ("def int8 add8(int8 a, int8 b):\n    return a + b\n"
                          "def uint8 mul8(uint8 a, uint8 b):\n    return a * b\n"
                          "def int main():\n    int8 x = add8(int8(100), int8(100))\n    print(x)\n    print(x < int8(0))\n"
                          "    uint8 y = mul8(uint8(20), uint8(20))\n    print(int(y))\n    print(int32(int8(200)))\n    return 0\n",
                          0, "-56\ntrue\n144\n-56\n"),
    'loops_and_branches': ("def int main():\n    int total = 0\n    for int i = 0; i < 100; i += 1:\n"
                           "        if i % 3 == 0 or i % 5 == 0:\n            total += i\n        elif i > 90:\n            break\n"
                           "    int j = 0\n    while true:\n        j += 1\n        if j >= 10:\n            break\n"
                           "    print(total)\n    print(j)\n    return total % 256\n", 1935 % 256, "1935\n10\n"),
    'many_arguments': ("def int f(int a, int b, int c, int d, int e, int g, int h, int i, int j, int k):\n"
                       "    return a + 2 * b + 3 * c + 4 * d + 5 * e + 6 * g + 7 * h + 8 * i + 9 * j + 10 * k\n"
                       "def int main():\n    print(f(1, 2, 3, 4, 5, 6, 7, 8, 9, 10))\n    return 0\n", 0, "385\n"),
    'values_across_calls': ("def int id(int x):\n    return x\n"
                            "def int main():\n    int a = 1\n    int b = 2\n    int c = 3\n    int d = 4\n    int e = 5\n"
                            "    int f = 6\n    int g = 7\n    int h = 8\n    int i = 9\n    int j = 10\n    int k = 11\n"
                            "    int s = id(a) + id(b) + id(c) + id(d) + id(e) + id(f) + id(g) + id(h) + id(i) + id(j) + id(k)\n"
                            "    print(s + a + b + c + d + e + f + g + h + i + j + k)\n    return 0\n", 0, "132\n"),
    'large_frame': ("def int main():\n    [1500]int big\n    big[0] = 5\n    big[1499] = 7\n    int x = 9\n"
                    "    print(big[0] + big[1499] + x)\n    return 0\n", 0, "21\n"),
    'copies': ("type Odd struct:\n    int8 a\n    int b\n    uint8 c\n    [5]int8 d\n"
               "def [3]int8 bump3([3]int8 x):\n    x[2] = x[2] + int8(1)\n    return x\n"
               "def [300]int fill(int n):\n    [300]int r\n    for int i = 0; i < 300; i += 1:\n        r[i] = i * n\n    return r\n"
               "def int main():\n    [3]int8 small = [int8(1), int8(2), int8(3)]\n    [3]int8 other = bump3(small)\n"
               "    print(small)\n    print(other)\n    [300]int big = fill(3)\n    [300]int copy = big\n    copy[299] = 0\n"
               "    print(big[299] + big[150] + copy[299])\n    Odd o = Odd(int8(-1), 77, uint8(200), [int8(1), int8(2), int8(3), int8(4), int8(5)])\n"
               "    Odd p = o\n    p.d[4] = int8(9)\n    print(o)\n    print(p)\n    [17]uint8 seventeen\n    seventeen[16] = uint8(42)\n"
               "    [17]uint8 s2 = seventeen\n    print(int(s2[16]))\n    return 0\n", 0,
               "[3]int8[1, 2, 3]\n[3]int8[1, 2, 4]\n1347\nOdd(a: -1, b: 77, c: 200, d: [5]int8[1, 2, 3, 4, 5])\n"
               "Odd(a: -1, b: 77, c: 200, d: [5]int8[1, 2, 3, 4, 9])\n42\n"),
    'bounds_check': ("def int main():\n    [3]int a = [1, 2, 3]\n    int i = 5\n    print(a[i])\n    return 0\n", None, None),
}


@pytest.mark.parametrize('name', sorted(PROGRAMS))
def test_program(name):
    source, code, out = PROGRAMS[name]
    r = _same_as_x86(source)
    if code is not None:
        assert r.returncode == code
    if out is not None:
        assert r.stdout == out


def test_bounds_check_panics():
    r = build_and_run(PROGRAMS['bounds_check'][0], A64)
    assert r.returncode != 0 and r.stderr == 'array index out of bounds\n'


@pytest.mark.parametrize('seed', range(0, 40, 4))
def test_random_program(seed):
    source, expected = Gen(seed).program()
    assert build_and_run(source, A64).stdout == expected


def _truncated(a: int, b: int) -> tuple:
    q = abs(a) // abs(b)
    q = q if (a < 0) == (b < 0) else -q
    return q, a - b * q


def test_division_by_constants_matches_truncating_division():
    divisors = [2, 3, 7, 10, 16, -3, -16, 1000000007, 4611686018427387904]
    numbers = [0, 1, -1, 7, -7, 2 ** 63 - 1, -2 ** 63, 123456789]
    lines = [f"    print(n / {d})\n    print(n % {d})\n" for d in divisors]
    source = ("def int main():\n    [8]int ns = [" + ", ".join(str(n) if n != -2 ** 63 else "-9223372036854775807 - 1"
                                                    for n in numbers) + "]\n"
              "    for int i = 0; i < 8; i += 1:\n        int n = ns[i]\n"
              + "".join("    " + line.replace("\n    ", "\n        ") for line in lines) + "    return 0\n")
    expected = "".join(f"{q}\n{r}\n" for n in numbers for d in divisors for q, r in [_truncated(n, d)])
    assert _same_as_x86(source).stdout == expected
