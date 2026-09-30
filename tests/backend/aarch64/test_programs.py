"""Programs built for aarch64-linux and run under qemu-user, checked against expected output
and against x86-64. (AArch64 joins the full end-to-end suite once its backend is complete.)"""

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
    assert r.returncode != 0 and 'array index out of bounds' in r.stdout


@pytest.mark.parametrize('seed', range(0, 40, 4))
def test_random_program(seed):
    source, expected = Gen(seed).program()
    assert build_and_run(source, A64).stdout == expected
