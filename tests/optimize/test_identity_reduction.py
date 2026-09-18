"""Tests for optimize/identity_reduction.py -- reduce_binary_op's own
identity/absorbing table (including the deliberate exclusions its own
module docstring explains), and reduce_identities' own in-place IR
rewriting."""

from parser import BinaryOp
from semantic import Type
from ir.ir import IRBinOp, IRConst, IRFunction, IRMove, IRReturn, Temp
from optimize.identity_reduction import reduce_binary_op, reduce_identities


def _temp(id_=0, t=Type.INT):
    return Temp(id=id_, type=t)


def _const(value, t=Type.INT):
    return IRConst(value, t)


# -- reduce_binary_op: ADD -----------------------------------------------------

def test_reduce_add_right_zero_returns_left():
    x = _temp()
    assert x == reduce_binary_op(BinaryOp.ADD, x, _const(0), Type.INT)


def test_reduce_add_left_zero_returns_right():
    x = _temp()
    assert x == reduce_binary_op(BinaryOp.ADD, _const(0), x, Type.INT)


def test_reduce_add_no_identity_operand_is_not_reduced():
    x = _temp()
    assert reduce_binary_op(BinaryOp.ADD, x, _const(5), Type.INT) is None


# -- reduce_binary_op: SUBTRACT ------------------------------------------------

def test_reduce_subtract_right_zero_returns_left():
    x = _temp()
    assert x == reduce_binary_op(BinaryOp.SUBTRACT, x, _const(0), Type.INT)


def test_reduce_subtract_left_zero_is_not_reduced():
    """0 - x is -x, a transformed value, not x itself -- a strength
    reduction this pass deliberately leaves alone (see its own module
    docstring)."""
    x = _temp()
    assert reduce_binary_op(BinaryOp.SUBTRACT, _const(0), x, Type.INT) is None


# -- reduce_binary_op: MULTIPLY -------------------------------------------------

def test_reduce_multiply_right_zero_is_absorbing():
    x = _temp()
    assert IRConst(0, Type.INT) == reduce_binary_op(BinaryOp.MULTIPLY, x, _const(0), Type.INT)


def test_reduce_multiply_left_zero_is_absorbing():
    x = _temp()
    assert IRConst(0, Type.INT) == reduce_binary_op(BinaryOp.MULTIPLY, _const(0), x, Type.INT)


def test_reduce_multiply_right_one_returns_left():
    x = _temp()
    assert x == reduce_binary_op(BinaryOp.MULTIPLY, x, _const(1), Type.INT)


def test_reduce_multiply_left_one_returns_right():
    x = _temp()
    assert x == reduce_binary_op(BinaryOp.MULTIPLY, _const(1), x, Type.INT)


def test_reduce_multiply_by_negative_one_is_not_reduced():
    """x * -1 is -x, a strength reduction, not an identity -- the
    result isn't x itself unchanged."""
    x = _temp()
    assert reduce_binary_op(BinaryOp.MULTIPLY, x, _const(-1), Type.INT) is None


def test_reduce_multiply_absorbing_wins_over_one_when_both_operands_are_zero_and_one_shaped():
    """0 * 1 hits the absorbing check first -- confirms check ordering
    doesn't accidentally prefer the wrong rule when a constant could
    seem to match more than one shape."""
    assert IRConst(0, Type.INT) == reduce_binary_op(BinaryOp.MULTIPLY, _const(0), _const(1), Type.INT)


# -- reduce_binary_op: DIVIDE/MODULO -- divisor-position only, never dividend -

def test_reduce_divide_right_one_returns_left():
    x = _temp()
    assert x == reduce_binary_op(BinaryOp.DIVIDE, x, _const(1), Type.INT)


def test_reduce_divide_left_zero_is_not_reduced():
    """0 / x is 0 for every x except x == 0, where the unreduced code
    traps (a hardware SIGFPE) -- reducing this away would silently
    replace that trap with an ordinary 0."""
    x = _temp()
    assert reduce_binary_op(BinaryOp.DIVIDE, _const(0), x, Type.INT) is None


def test_reduce_modulo_right_one_is_absorbing_zero():
    x = _temp()
    assert IRConst(0, Type.INT) == reduce_binary_op(BinaryOp.MODULO, x, _const(1), Type.INT)


def test_reduce_modulo_left_zero_is_not_reduced():
    """Same divide-by-zero hazard as DIVIDE's own left-zero case."""
    x = _temp()
    assert reduce_binary_op(BinaryOp.MODULO, _const(0), x, Type.INT) is None


# -- reduce_binary_op: SHIFT_LEFT/SHIFT_RIGHT ----------------------------------

def test_reduce_shift_left_by_zero_returns_left():
    x = _temp()
    assert x == reduce_binary_op(BinaryOp.SHIFT_LEFT, x, _const(0), Type.INT)


def test_reduce_shift_right_by_zero_returns_left():
    x = _temp()
    assert x == reduce_binary_op(BinaryOp.SHIFT_RIGHT, x, _const(0), Type.INT)


def test_reduce_shift_left_of_zero_is_absorbing():
    x = _temp()
    assert IRConst(0, Type.INT) == reduce_binary_op(BinaryOp.SHIFT_LEFT, _const(0), x, Type.INT)


def test_reduce_shift_right_of_zero_is_absorbing():
    x = _temp()
    assert IRConst(0, Type.INT) == reduce_binary_op(BinaryOp.SHIFT_RIGHT, _const(0), x, Type.INT)


# -- reduce_binary_op: bitwise --------------------------------------------------

def test_reduce_bitwise_and_right_zero_is_absorbing():
    x = _temp()
    assert IRConst(0, Type.INT) == reduce_binary_op(BinaryOp.BITWISE_AND, x, _const(0), Type.INT)


def test_reduce_bitwise_and_left_zero_is_absorbing():
    x = _temp()
    assert IRConst(0, Type.INT) == reduce_binary_op(BinaryOp.BITWISE_AND, _const(0), x, Type.INT)


def test_reduce_bitwise_and_all_ones_is_not_reduced():
    """x & -1 is x, but recognizing -1 as "all bits set" needs the
    operand's own runtime width -- deliberately deferred (see this
    module's own docstring)."""
    x = _temp()
    assert reduce_binary_op(BinaryOp.BITWISE_AND, x, _const(-1), Type.INT) is None


def test_reduce_bitwise_or_right_zero_returns_left():
    x = _temp()
    assert x == reduce_binary_op(BinaryOp.BITWISE_OR, x, _const(0), Type.INT)


def test_reduce_bitwise_or_left_zero_returns_right():
    x = _temp()
    assert x == reduce_binary_op(BinaryOp.BITWISE_OR, _const(0), x, Type.INT)


def test_reduce_bitwise_or_all_ones_is_not_reduced():
    x = _temp()
    assert reduce_binary_op(BinaryOp.BITWISE_OR, x, _const(-1), Type.INT) is None


def test_reduce_bitwise_xor_right_zero_returns_left():
    x = _temp()
    assert x == reduce_binary_op(BinaryOp.BITWISE_XOR, x, _const(0), Type.INT)


def test_reduce_bitwise_xor_left_zero_returns_right():
    x = _temp()
    assert x == reduce_binary_op(BinaryOp.BITWISE_XOR, _const(0), x, Type.INT)


# -- reduce_binary_op: comparisons have no identity operand at all -----------

def test_reduce_comparison_ops_are_never_reduced():
    x = _temp()
    for op in (BinaryOp.LESS_THAN, BinaryOp.GREATER_THAN, BinaryOp.EQUAL, BinaryOp.NOT_EQUAL):
        assert reduce_binary_op(op, x, _const(0), Type.INT) is None


# -- reduce_identities: in-place IR rewriting ----------------------------------

def _fn_with(body: list) -> IRFunction:
    return IRFunction(name='f', body=body, return_type=Type.INT)


def test_reduce_identities_replaces_reducible_binop_with_move():
    x = _temp(0)
    dst = _temp(1)
    fn = _fn_with([IRBinOp(dst=dst, op=BinaryOp.ADD, left=x, right=IRConst(0, Type.INT))])

    reduce_identities(fn)

    assert [IRMove(dst=dst, src=x)] == fn.body


def test_reduce_identities_preserves_dst_identity():
    x = _temp(0)
    dst = _temp(7)
    fn = _fn_with([IRBinOp(dst=dst, op=BinaryOp.ADD, left=x, right=IRConst(0, Type.INT))])

    reduce_identities(fn)

    assert fn.body[0].dst == dst


def test_reduce_identities_leaves_non_reducible_binop_untouched():
    x = _temp(0)
    dst = _temp(1)
    original = IRBinOp(dst=dst, op=BinaryOp.ADD, left=x, right=IRConst(5, Type.INT))
    fn = _fn_with([original])

    reduce_identities(fn)

    assert [original] == fn.body


def test_reduce_identities_ignores_non_binop_instructions():
    ret = IRReturn(value=IRConst(0, Type.INT))
    fn = _fn_with([ret])

    reduce_identities(fn)

    assert [ret] == fn.body


def test_reduce_identities_only_rewrites_the_reducible_op_leaving_others_alone():
    x = _temp(0)
    dst_a = _temp(1)
    dst_b = _temp(2)
    reducible = IRBinOp(dst=dst_a, op=BinaryOp.ADD, left=x, right=IRConst(0, Type.INT))
    not_reducible = IRBinOp(dst=dst_b, op=BinaryOp.ADD, left=dst_a, right=IRConst(5, Type.INT))
    ret = IRReturn(value=dst_b)
    fn = _fn_with([reducible, not_reducible, ret])

    reduce_identities(fn)

    assert fn.body == [IRMove(dst=dst_a, src=x), not_reducible, ret]


def test_reduce_identities_reduces_a_fully_constant_op_too():
    """5 + 0 is both fully constant AND has an identity operand --
    reduce_identities alone (with no constant folding involved at all)
    still produces the correct, fully-reduced IRConst(5)."""
    dst = _temp(0)
    fn = _fn_with([IRBinOp(dst=dst, op=BinaryOp.ADD, left=IRConst(5, Type.INT), right=IRConst(0, Type.INT))])

    reduce_identities(fn)

    assert [IRMove(dst=dst, src=IRConst(5, Type.INT))] == fn.body
