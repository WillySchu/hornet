"""int8/uint8 arithmetic wraps to 8 bits wherever the value lives (register, stack, across calls)."""

import pytest

from tests.test_compiler import GCC_SKIP, assert_program_stdout


def _wrap(v: int, signed: bool) -> int:
    v &= 255
    return v - 256 if signed and v >= 128 else v


def _div(a, b):
    q = abs(a) // abs(b)
    return q if (a < 0) == (b < 0) else -q


OPS = {
    '+': lambda a, b: a + b, '-': lambda a, b: a - b, '*': lambda a, b: a * b,
    '&': lambda a, b: a & b, '|': lambda a, b: a | b, '^': lambda a, b: a ^ b,
    '/': _div, '%': lambda a, b: a - _div(a, b) * b,
}
PAIRS = [(100, 100), (127, 1), (-128, -1), (-128, 1), (-1, -1), (-7, 2), (7, -2), (0, 5), (15, 3)]
UPAIRS = [(200, 200), (255, 1), (0, 1), (128, 2), (7, 3), (255, 255), (1, 254)]


@GCC_SKIP
@pytest.mark.parametrize('ty', ['int8', 'uint8'])
def test_binary_ops_wrap_in_registers_and_across_calls(ty):
    signed = ty == 'int8'
    pairs = PAIRS if signed else UPAIRS
    lines, expected = [], []
    for name, op in [('add', '+'), ('sub', '-'), ('mul', '*'), ('band', '&'), ('bor', '|'), ('bxor', '^'), ('div', '/'), ('mod', '%')]:
        lines.append(f"def {ty} {name}({ty} a, {ty} b):\n    return a {op} b\n")
    body = ["def int main():"]
    for k, (a, b) in enumerate(pairs):
        for name, op in [('add', '+'), ('sub', '-'), ('mul', '*'), ('band', '&'), ('bor', '|'), ('bxor', '^'), ('div', '/'), ('mod', '%')]:
            if op in '/%' and b == 0:
                continue
            body.append(f"    {ty} r_{name}{k} = {name}({ty}({a}), {ty}({b}))")
            body.append(f"    print(int(r_{name}{k}))")
            body.append(f"    print(r_{name}{k} < {ty}(0) or r_{name}{k} > {ty}(100))")
            v = _wrap(OPS[op](a, b), signed)
            expected.append(str(v))
            expected.append('true' if v < 0 or v > 100 else 'false')
    body.append("    return 0")
    assert_program_stdout('\n'.join(lines) + '\n' + '\n'.join(body) + '\n', '\n'.join(expected) + '\n')


@GCC_SKIP
def test_unary_shift_and_loop_carried_narrow_values():
    assert_program_stdout(
        "def int8 neg(int8 x):\n"
        "    return -x\n"
        "def uint8 inv(uint8 x):\n"
        "    return ~x\n"
        "def int8 shl(int8 x, int8 n):\n"
        "    return x << n\n"
        "def int main():\n"
        "    print(int(neg(int8(-128))))\n"
        "    print(neg(int8(-128)) < int8(0))\n"
        "    print(int(inv(uint8(0))))\n"
        "    print(int(shl(int8(1), int8(7))))\n"
        "    uint8 s = uint8(1)\n"
        "    int8 t = int8(1)\n"
        "    for int i = 0; i < 9; i += 1:\n"
        "        s = s * uint8(3)\n"
        "        t = t * int8(5)\n"
        "    print(int(s))\n"
        "    print(int(t))\n"
        "    print(int(s) + int(t))\n"
        "    return 0\n",
        "-128\ntrue\n255\n-128\n"
        f"{3 ** 9 % 256}\n{_wrap(5 ** 9, True)}\n{3 ** 9 % 256 + _wrap(5 ** 9, True)}\n",
    )
