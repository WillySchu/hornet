"""Tests for strength reduction."""

import random

import pytest

from folding import fold_binary_op
from ir.ir import IRBinOp, IRConst, IRFunction, IRReturn, Temp
from ops import BinaryOp
from optimize.strength_reduction import reduce_strength
from tests.test_compiler import GCC_SKIP, assert_program_stdout
from typesys import Type

BITS = {Type.INT8: 8, Type.UINT8: 8, Type.INT32: 32, Type.INT: 64, Type.INT64: 64}


def t(n: int, ty: Type = Type.INT) -> Temp:
    return Temp(id=n, type=ty)


def c(v: int, ty: Type = Type.INT) -> IRConst:
    return IRConst(v, ty)


def op(operator, left, right, ty: Type = Type.INT) -> IRBinOp:
    return IRBinOp(dst=t(1, ty), op=operator, left=left, right=right)


def reduced(instr: IRBinOp) -> IRBinOp:
    f = IRFunction(name='f', body=[instr, IRReturn(value=instr.dst)])
    reduce_strength(f)
    return f.body[0]


def test_a_power_of_two_on_either_side_becomes_a_shift():
    assert reduced(op(BinaryOp.MULTIPLY, t(0), c(8))) == op(BinaryOp.SHIFT_LEFT, t(0), c(3))
    assert reduced(op(BinaryOp.MULTIPLY, c(2), t(0))) == op(BinaryOp.SHIFT_LEFT, t(0), c(1))
    assert reduced(op(BinaryOp.MULTIPLY, t(0), c(2 ** 62))) == op(BinaryOp.SHIFT_LEFT, t(0), c(62))


def test_the_count_has_the_type_the_factor_had():
    x = t(0, Type.UINT8)
    assert reduced(op(BinaryOp.MULTIPLY, x, c(128, Type.UINT8), Type.UINT8)) == op(
        BinaryOp.SHIFT_LEFT, x, c(7, Type.UINT8), Type.UINT8)


@pytest.mark.parametrize('instr', [
    op(BinaryOp.MULTIPLY, t(0), c(12)),
    op(BinaryOp.MULTIPLY, t(0), c(1)),  # identity reduction's
    op(BinaryOp.MULTIPLY, t(0), c(0)),
    op(BinaryOp.MULTIPLY, t(0), c(-8)),
    op(BinaryOp.MULTIPLY, c(4), c(8)),  # constant folding's
    op(BinaryOp.MULTIPLY, t(0), t(2)),
    op(BinaryOp.ADD, t(0), c(8)),
    op(BinaryOp.DIVIDE, t(0), c(8)),  # (rounds toward zero, as no shift of a negative number does)
])
def test_anything_else_is_left(instr):
    assert reduced(instr) is instr


@pytest.mark.parametrize('ty', BITS)
def test_the_shift_wraps_as_the_multiplication_did(ty):
    bits = BITS[ty]
    lo, hi = (0, 255) if ty == Type.UINT8 else (-2 ** (bits - 1), 2 ** (bits - 1) - 1)
    r = random.Random(bits)
    values = range(lo, hi + 1) if bits == 8 else [lo, hi, -1, 0, 1] + [r.randint(lo, hi) for _ in range(200)]
    k = 1
    while 2 ** k <= hi:  # every factor the type can hold
        for x in values:
            assert fold_binary_op(BinaryOp.SHIFT_LEFT, x, k, ty) == fold_binary_op(BinaryOp.MULTIPLY, x, 2 ** k, ty)
        k += 1
    assert k == {8: 7 if ty == Type.INT8 else 8, 32: 31, 64: 63}[bits]


# A program multiplies what it reads at run time, at each width, by each factor that width can hold,
# and by one that is no power of two; each result is printed beside what the same shift gives.
PROGRAM = """\
def int8 small(int8 x, int k):
    if k == 1:
        return x * 2
    if k == 6:
        return x * 64
    return x * 3

def uint8 unsigned(uint8 x, int k):
    if k == 1:
        return 2 * x
    if k == 7:
        return x * 128
    return x * 3

def int32 medium(int32 x, int k):
    if k == 4:
        return x * 16
    if k == 30:
        return 1073741824 * x
    return x * 3

def int wide(int x, int k):
    if k == 3:
        return x * 8
    if k == 62:
        return x * 4611686018427387904
    return x * 3

def int main():
    [4]int8 a = [1, -1, 127, -128]
    [4]uint8 b = [1, 3, 129, 255]
    [4]int32 c = [1, -3, 2147483647, -2147483648]
    [4]int d = [1, -3, 9223372036854775807, -9223372036854775807]
    for x in a:
        print(format('{} {} {} {} {}', small(x, 1), x << 1, small(x, 6), x << 6, small(x, 0)))
    for x in b:
        print(format('{} {} {} {} {}', unsigned(x, 1), x << 1, unsigned(x, 7), x << 7, unsigned(x, 0)))
    for x in c:
        print(format('{} {} {} {} {}', medium(x, 4), x << 4, medium(x, 30), x << 30, medium(x, 0)))
    for x in d:
        print(format('{} {} {} {} {}', wide(x, 3), x << 3, wide(x, 62), x << 62, wide(x, 0)))
    return 0
"""


def _wrapped(v: int, bits: int, signed: bool = True) -> int:
    v &= (1 << bits) - 1
    return v - (1 << bits) if signed and v >= 1 << (bits - 1) else v


def _expected() -> str:
    rows = []
    for values, bits, signed, (k1, k2) in (
            ([1, -1, 127, -128], 8, True, (1, 6)), ([1, 3, 129, 255], 8, False, (1, 7)),
            ([1, -3, 2 ** 31 - 1, -2 ** 31], 32, True, (4, 30)),
            ([1, -3, 2 ** 63 - 1, -2 ** 63 + 1], 64, True, (3, 62))):
        for x in values:
            w = lambda v: _wrapped(v, bits, signed)
            rows.append(f"{w(x * 2 ** k1)} {w(x << k1)} {w(x * 2 ** k2)} {w(x << k2)} {w(x * 3)}")
    return "\n".join(rows) + "\n"


@GCC_SKIP
def test_a_program_s_products_are_what_they_were():
    assert_program_stdout(PROGRAM, _expected())
