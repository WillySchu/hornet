"""Tests for optimize/constant_folding.py -- fold_binary_op/fold_
unary_op/fold_cast's own arithmetic (including the width/truncation/
zero-divisor rules their own module docstring explains), and fold_
constants' own in-place IR rewriting."""

from parser import BinaryOp, UnaryOp
from semantic import Type
from ir.ir import IRBinOp, IRCast, IRConst, IRFunction, IRMove, IRReturn, IRUnOp, Temp
from optimize.constant_folding import fold_binary_op, fold_cast, fold_constants, fold_unary_op


# -- fold_binary_op: ordinary arithmetic --------------------------------------

def test_fold_binary_op_add():
    assert 8 == fold_binary_op(BinaryOp.ADD, 5, 3, Type.INT)


def test_fold_binary_op_subtract():
    assert 2 == fold_binary_op(BinaryOp.SUBTRACT, 5, 3, Type.INT)


def test_fold_binary_op_multiply():
    assert 15 == fold_binary_op(BinaryOp.MULTIPLY, 5, 3, Type.INT)


def test_fold_binary_op_bitwise_and():
    assert 0b0100 == fold_binary_op(BinaryOp.BITWISE_AND, 0b0110, 0b0101, Type.INT)


def test_fold_binary_op_bitwise_or():
    assert 0b0111 == fold_binary_op(BinaryOp.BITWISE_OR, 0b0110, 0b0101, Type.INT)


def test_fold_binary_op_bitwise_xor():
    assert 0b0011 == fold_binary_op(BinaryOp.BITWISE_XOR, 0b0110, 0b0101, Type.INT)


# -- fold_binary_op: comparisons, always bool, never wrapped ------------------

def test_fold_binary_op_less_than():
    assert 1 == fold_binary_op(BinaryOp.LESS_THAN, 2, 5, Type.INT)
    assert 0 == fold_binary_op(BinaryOp.LESS_THAN, 5, 2, Type.INT)


def test_fold_binary_op_greater_than():
    assert 1 == fold_binary_op(BinaryOp.GREATER_THAN, 5, 2, Type.INT)


def test_fold_binary_op_less_than_or_equal():
    assert 1 == fold_binary_op(BinaryOp.LESS_THAN_OR_EQUAL, 5, 5, Type.INT)


def test_fold_binary_op_greater_than_or_equal():
    assert 1 == fold_binary_op(BinaryOp.GREATER_THAN_OR_EQUAL, 5, 5, Type.INT)


def test_fold_binary_op_equal():
    assert 1 == fold_binary_op(BinaryOp.EQUAL, 5, 5, Type.INT)
    assert 0 == fold_binary_op(BinaryOp.EQUAL, 5, 6, Type.INT)


def test_fold_binary_op_not_equal():
    assert 1 == fold_binary_op(BinaryOp.NOT_EQUAL, 5, 6, Type.INT)


# -- fold_binary_op: division/modulo, truncating toward zero, never by a
# constant zero -- the two rules this module's own docstring calls out ------

def test_fold_binary_op_divide_positive():
    assert 3 == fold_binary_op(BinaryOp.DIVIDE, 7, 2, Type.INT)


def test_fold_binary_op_divide_truncates_toward_zero_not_floor():
    """-7 / 2 is -3 on this compiler's own runtime (idiv truncates
    toward zero) -- NOT -4, which is what Python's own // (floor
    division) would give."""
    assert -3 == fold_binary_op(BinaryOp.DIVIDE, -7, 2, Type.INT)
    assert -3 == fold_binary_op(BinaryOp.DIVIDE, 7, -2, Type.INT)
    assert 3 == fold_binary_op(BinaryOp.DIVIDE, -7, -2, Type.INT)


def test_fold_binary_op_modulo_truncating():
    """-7 % 2 is -1 on this compiler's own runtime -- matching -7 ==
    (-3 * 2) + -1, NOT Python's own %, which would give 1."""
    assert -1 == fold_binary_op(BinaryOp.MODULO, -7, 2, Type.INT)
    assert 1 == fold_binary_op(BinaryOp.MODULO, 7, -2, Type.INT)


def test_fold_binary_op_divide_by_zero_is_not_folded():
    """Division by a constant zero is a genuine, intended runtime trap
    (a hardware SIGFPE) -- folding it would silently replace a crash
    with an arbitrary value instead."""
    assert fold_binary_op(BinaryOp.DIVIDE, 5, 0, Type.INT) is None


def test_fold_binary_op_modulo_by_zero_is_not_folded():
    assert fold_binary_op(BinaryOp.MODULO, 5, 0, Type.INT) is None


# -- fold_binary_op: shifts, count masked to the operand width ---------------

def test_fold_binary_op_shift_left():
    assert 8 == fold_binary_op(BinaryOp.SHIFT_LEFT, 1, 3, Type.INT)


def test_fold_binary_op_shift_right():
    assert 1 == fold_binary_op(BinaryOp.SHIFT_RIGHT, 8, 3, Type.INT)


def test_fold_binary_op_shift_right_sign_extends():
    """An arithmetic shift right of a negative value keeps its own
    sign -- -8 >> 1 is -4, not some large positive value."""
    assert -4 == fold_binary_op(BinaryOp.SHIFT_RIGHT, -8, 1, Type.INT)


def test_fold_binary_op_shift_count_masked_to_32_bits():
    """x86's own sal/sar mask the shift count to 5 bits for anything
    but int64 -- a shift by 35 on a 32-bit value is a shift by 35 & 31
    == 3, not 35 itself."""
    assert fold_binary_op(BinaryOp.SHIFT_LEFT, 1, 3, Type.INT) == fold_binary_op(BinaryOp.SHIFT_LEFT, 1, 35, Type.INT)


def test_fold_binary_op_shift_count_masked_to_64_bits_for_int64():
    """int64 uses the 64-bit shift instruction, masking its own count
    to 6 bits instead of 5 -- a shift by 64 is a shift by 0, not 3."""
    assert 1 == fold_binary_op(BinaryOp.SHIFT_LEFT, 1, 64, Type.INT64)
    assert 8 == fold_binary_op(BinaryOp.SHIFT_LEFT, 1, 67, Type.INT64)  # 67 & 63 == 3


# -- fold_binary_op: width-aware wraparound -----------------------------------

def test_fold_binary_op_int8_wraps_on_overflow():
    """int8(100) + int8(100) wraps to -56 -- the identical two's-
    complement result the real add instruction produces on a 1-byte
    value -- not the mathematically \"correct\" 200, which doesn't fit
    in int8's own range at all."""
    assert -56 == fold_binary_op(BinaryOp.ADD, 100, 100, Type.INT8)


def test_fold_binary_op_uint8_wraps_on_overflow():
    """uint8(200) + uint8(100) wraps to 44 (300 mod 256), never
    negative -- uint8 has no sign bit to overflow into."""
    assert 44 == fold_binary_op(BinaryOp.ADD, 200, 100, Type.UINT8)


def test_fold_binary_op_int64_does_not_spuriously_truncate_to_32_bits():
    """A value that fits in int64 but not in a 32-bit int must survive
    folding unchanged -- confirms the width used for wrapping is
    genuinely read from result_type, not hardcoded to 32 bits."""
    big = 5_000_000_000
    assert big + 1 == fold_binary_op(BinaryOp.ADD, big, 1, Type.INT64)


def test_fold_binary_op_int_wraps_at_32_bits():
    assert -(2 ** 31) == fold_binary_op(BinaryOp.ADD, 2 ** 31 - 1, 1, Type.INT)


# -- fold_unary_op -------------------------------------------------------------

def test_fold_unary_op_negate():
    assert -5 == fold_unary_op(UnaryOp.NEGATE, 5, Type.INT)


def test_fold_unary_op_negate_int8_min_wraps_to_itself():
    """Negating int8's own most negative value (-128) overflows back
    to -128 in two's complement -- there's no +128 to represent it."""
    assert -128 == fold_unary_op(UnaryOp.NEGATE, -128, Type.INT8)


def test_fold_unary_op_complement():
    assert -6 == fold_unary_op(UnaryOp.COMPLEMENT, 5, Type.INT)


def test_fold_unary_op_not_zero_is_true():
    assert 1 == fold_unary_op(UnaryOp.NOT, 0, Type.BOOL)


def test_fold_unary_op_not_nonzero_is_false():
    assert 0 == fold_unary_op(UnaryOp.NOT, 1, Type.BOOL)


# -- fold_cast -- deliberately mirrors gen_cast_narrowing_into exactly -------

def test_fold_cast_int_to_int8_positive_low_byte():
    """int(300) as int8: 300's low byte (0x2C) has its high bit
    clear, so sign-extension leaves it positive -- 44, matching gen_
    cast_narrowing_into's own docstring example exactly."""
    assert 44 == fold_cast(Type.INT8, 300)


def test_fold_cast_int_to_int8_negative_low_byte():
    """int(200) as int8: 200's low byte (0xC8) has its high bit set,
    so sign-extension produces the negative two's-complement
    reinterpretation -- -56, the docstring's own other example."""
    assert -56 == fold_cast(Type.INT8, 200)


def test_fold_cast_int_to_uint8():
    assert 44 == fold_cast(Type.UINT8, 300)


def test_fold_cast_narrow_to_int64_widens():
    assert -5 == fold_cast(Type.INT64, -5)


def test_fold_cast_int64_to_int_truncates_to_32_bits_first():
    """Narrowing an int64 source down to int needs the identical
    32-bit truncation int64_to_int8/uint8 also goes through."""
    big = 5_000_000_000  # doesn't fit in a 32-bit int at all
    assert fold_cast(Type.INT, big) == fold_cast(Type.INT, big % (2 ** 32))


def test_fold_cast_int64_to_int64_still_truncates_through_32_bits():
    """gen_cast_narrowing_into's own MovSXD always re-derives its
    result from dst's 32-bit view, even for an already-int64 source --
    fold_cast has to match that exactly, not "fix" it into a true no-
    op, or a folded int64-to-int64 cast could silently disagree with
    an unfolded one."""
    big = 5_000_000_000
    assert big != fold_cast(Type.INT64, big)


def test_fold_cast_same_width_reinterpretation():
    """int8-to-uint8 (or back) is just a same-width reinterpretation
    -- take the low byte, reinterpret its sign convention."""
    assert 200 == fold_cast(Type.UINT8, -56)


# -- fold_constants: in-place IR rewriting ------------------------------------

def _fn_with(body: list) -> IRFunction:
    return IRFunction(name='f', body=body, return_type=Type.INT)


def test_fold_constants_replaces_foldable_binop_with_move():
    dst = Temp(id=0, type=Type.INT)
    fn = _fn_with([IRBinOp(dst=dst, op=BinaryOp.ADD, left=IRConst(5, Type.INT), right=IRConst(3, Type.INT))])

    fold_constants(fn)

    assert [IRMove(dst=dst, src=IRConst(8, Type.INT))] == fn.body


def test_fold_constants_preserves_dst_identity():
    """The rewritten IRMove's own dst is the SAME Temp (matching id),
    not a fresh one -- whatever already reads this Temp keeps working
    unchanged."""
    dst = Temp(id=7, type=Type.INT)
    fn = _fn_with([IRBinOp(dst=dst, op=BinaryOp.ADD, left=IRConst(1, Type.INT), right=IRConst(1, Type.INT))])

    fold_constants(fn)

    assert fn.body[0].dst == dst


def test_fold_constants_leaves_non_constant_operands_untouched():
    left = Temp(id=0, type=Type.INT)
    dst = Temp(id=1, type=Type.INT)
    original = IRBinOp(dst=dst, op=BinaryOp.ADD, left=left, right=IRConst(3, Type.INT))
    fn = _fn_with([original])

    fold_constants(fn)

    assert [original] == fn.body


def test_fold_constants_leaves_divide_by_zero_untouched():
    dst = Temp(id=0, type=Type.INT)
    original = IRBinOp(dst=dst, op=BinaryOp.DIVIDE, left=IRConst(5, Type.INT), right=IRConst(0, Type.INT))
    fn = _fn_with([original])

    fold_constants(fn)

    assert [original] == fn.body


def test_fold_constants_replaces_foldable_unop_with_move():
    dst = Temp(id=0, type=Type.INT)
    fn = _fn_with([IRUnOp(dst=dst, op=UnaryOp.NEGATE, operand=IRConst(5, Type.INT))])

    fold_constants(fn)

    assert [IRMove(dst=dst, src=IRConst(-5, Type.INT))] == fn.body


def test_fold_constants_replaces_foldable_cast_with_move():
    dst = Temp(id=0, type=Type.INT8)
    fn = _fn_with([IRCast(dst=dst, src=IRConst(300, Type.INT))])

    fold_constants(fn)

    assert [IRMove(dst=dst, src=IRConst(44, Type.INT8))] == fn.body


def test_fold_constants_only_rewrites_the_foldable_op_leaving_others_alone():
    dst_a = Temp(id=0, type=Type.INT)
    dst_b = Temp(id=1, type=Type.INT)
    foldable = IRBinOp(dst=dst_a, op=BinaryOp.ADD, left=IRConst(1, Type.INT), right=IRConst(1, Type.INT))
    not_foldable = IRBinOp(dst=dst_b, op=BinaryOp.ADD, left=dst_a, right=IRConst(1, Type.INT))
    ret = IRReturn(value=dst_b)
    fn = _fn_with([foldable, not_foldable, ret])

    fold_constants(fn)

    assert fn.body == [IRMove(dst=dst_a, src=IRConst(2, Type.INT)), not_foldable, ret]
