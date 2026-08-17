"""Tests for the parser"""

import parser


def test_unary_op_symbol():
    tcs = [
        {
            'op': parser.UnaryOp.NEGATE,
            'str': '-',
        },
        {
            'op': parser.UnaryOp.COMPLEMENT,
            'str': '~',
        },
        {
            'op': parser.UnaryOp.NOT,
            'str': 'not',
        },
    ]

    for tc in tcs:
        assert tc['str'] == tc['op'].symbol()


def test_binary_op_symbol():
    tcs = [
        {
            'op': parser.BinaryOp.ADD,
            'str': '+',
        },
        {
            'op': parser.BinaryOp.SUBTRACT,
            'str': '-',
        },
        {
            'op': parser.BinaryOp.MULTIPLY,
            'str': '*',
        },
        {
            'op': parser.BinaryOp.DIVIDE,
            'str': '/',
        },
        {
            'op': parser.BinaryOp.MODULO,
            'str': '%',
        },
        {
            'op': parser.BinaryOp.SHIFT_LEFT,
            'str': '<<',
        },
        {
            'op': parser.BinaryOp.SHIFT_RIGHT,
            'str': '>>',
        },
        {
            'op': parser.BinaryOp.EQUAL,
            'str': '==',
        },
        {
            'op': parser.BinaryOp.NOT_EQUAL,
            'str': '!=',
        },
        {
            'op': parser.BinaryOp.BITWISE_AND,
            'str': '&',
        },
        {
            'op': parser.BinaryOp.BITWISE_OR,
            'str': '|',
        },
        {
            'op': parser.BinaryOp.BITWISE_XOR,
            'str': '^',
        },
        {
            'op': parser.BinaryOp.AND,
            'str': 'and',
        },
        {
            'op': parser.BinaryOp.OR,
            'str': 'or',
        },
    ]

    for tc in tcs:
        assert tc['str'] == tc['op'].symbol()


def test_unescape_string_literal():
    tcs = [
        # TODO(will): Finish this test.
        {
            'input': "'asdf'",
            'expected': 'asdf',
        },
    ]

    for tc in tcs:
        assert tc['expected'] == parser._unescape_string_literal(tc['input'])
