"""`not` binds looser than comparisons, `in`, and the arithmetic and bitwise operators, and tighter
than `and` and `or`: `not a < b` is `not (a < b)`."""

import pytest

from lexer import Lexer
from parser import Binary, BinaryOp, Parser, Unary, UnaryOp, Variable
from tests.test_compiler import GCC_SKIP, assert_semantic_error, assert_stdout

A, B, C = Variable(name='a'), Variable(name='b'), Variable(name='c')


def _not(operand):
    return Unary(op=UnaryOp.NOT, operand=operand)


def _binary(op, left, right):
    return Binary(op=op, left=left, right=right)


def _expression(text: str):
    program = Parser(Lexer(f"def bool f():\n    return {text}\n").tokenize()).parse_program()
    return program.functions[0].body[0].value


@pytest.mark.parametrize("text,expected", [
    ("not a < b", _not(_binary(BinaryOp.LESS_THAN, A, B))),
    ("not a == b", _not(_binary(BinaryOp.EQUAL, A, B))),
    ("not a in b", _not(_binary(BinaryOp.IN, A, B))),
    ("a not in b", _not(_binary(BinaryOp.IN, A, B))),
    ("not a + b < c", _not(_binary(BinaryOp.LESS_THAN, _binary(BinaryOp.ADD, A, B), C))),
    ("not a & b", _not(_binary(BinaryOp.BITWISE_AND, A, B))),
    ("not not a", _not(_not(A))),
    ("not a and b", _binary(BinaryOp.AND, _not(A), B)),
    ("not a or b", _binary(BinaryOp.OR, _not(A), B)),
    ("a and not b < c", _binary(BinaryOp.AND, A, _not(_binary(BinaryOp.LESS_THAN, B, C)))),
    ("not a < b or not b < c", _binary(BinaryOp.OR, _not(_binary(BinaryOp.LESS_THAN, A, B)),
                                       _not(_binary(BinaryOp.LESS_THAN, B, C)))),
    ("a == not b", _binary(BinaryOp.EQUAL, A, _not(B))),
    # The other prefix operators still bind tighter than any binary operator.
    ("-a < b", _binary(BinaryOp.LESS_THAN, Unary(op=UnaryOp.NEGATE, operand=A), B)),
])
def test_not_takes_a_whole_comparison(text, expected):
    assert _expression(text) == expected


@GCC_SKIP
def test_not_of_comparisons():
    assert_stdout(
        "    int x = 3\n"
        "    print(not x < 5)\n"
        "    print(not x > 5)\n"
        "    print(not x == 3)\n"
        "    print(not x in [1, 2])\n"
        "    print(not x < 5 or x == 3)\n"
        "    print(not x > 5 and x == 4)\n"
        "    return 0",
        "false\ntrue\nfalse\ntrue\ntrue\nfalse\n",
    )


def test_not_of_an_int_is_still_an_error():
    assert_semantic_error("    int x = 3\n    if not x:\n        return 1\n    return 0",
                          match="'not' requires a bool operand, got int")
