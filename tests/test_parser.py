"""Tests for the parser"""

import re

import pytest

import lexer
import parser


TEST_TOKENS = [
    lexer.Token(lexer.TokenType.DEF, 'def', 1, 1),
    lexer.Token(lexer.TokenType.INT, 'int', 1, 5),
    lexer.Token(lexer.TokenType.IDENTIFIER, 'main', 1, 9),
    lexer.Token(lexer.TokenType.OPEN_PAREN, '(', 1, 13),
    lexer.Token(lexer.TokenType.CLOSE_PAREN, ')', 1, 14),
    lexer.Token(lexer.TokenType.COLON, ':', 1, 15),
    lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 16),

    lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
    lexer.Token(lexer.TokenType.INT, 'int', 2, 5),
    lexer.Token(lexer.TokenType.IDENTIFIER, 'i', 2, 9),
    lexer.Token(lexer.TokenType.ASSIGN, '=', 2, 11),
    lexer.Token(lexer.TokenType.NUMBER, '0', 2, 13),
    lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 14),

    lexer.Token(lexer.TokenType.BOOL, 'bool', 3, 5),
    lexer.Token(lexer.TokenType.IDENTIFIER, 'a', 3, 10),
    lexer.Token(lexer.TokenType.NEWLINE, '\n', 3, 11),

    lexer.Token(lexer.TokenType.BOOL, 'bool', 4, 5),
    lexer.Token(lexer.TokenType.IDENTIFIER, 'b', 4, 10),
    lexer.Token(lexer.TokenType.NEWLINE, '\n', 4, 11),

    lexer.Token(lexer.TokenType.NEWLINE, '\n', 5, 1),

    lexer.Token(lexer.TokenType.WHILE, 'while', 6, 5),
    lexer.Token(lexer.TokenType.IDENTIFIER, 'i', 6, 11),
    lexer.Token(lexer.TokenType.LESS_THAN, '<', 6, 13),
    lexer.Token(lexer.TokenType.NUMBER, '10', 6, 15),
    lexer.Token(lexer.TokenType.COLON, ':', 6, 17),
    lexer.Token(lexer.TokenType.NEWLINE, '\n', 6, 18),

    lexer.Token(lexer.TokenType.INDENT, '', 7, 1),
    lexer.Token(lexer.TokenType.IDENTIFIER, 'i', 7, 9),
    lexer.Token(lexer.TokenType.PLUS_ASSIGN, '+=', 7, 11),
    lexer.Token(lexer.TokenType.NUMBER, '1', 7, 14),
    lexer.Token(lexer.TokenType.NEWLINE, '\n', 7, 15),

    lexer.Token(lexer.TokenType.IF, 'if', 8, 9),
    lexer.Token(lexer.TokenType.IDENTIFIER, 'a', 8, 12),
    lexer.Token(lexer.TokenType.COLON, ':', 8, 13),
    lexer.Token(lexer.TokenType.NEWLINE, '\n', 8, 14),

    lexer.Token(lexer.TokenType.INDENT, '', 9, 1),
    lexer.Token(lexer.TokenType.CONTINUE, 'continue', 9, 13),
    lexer.Token(lexer.TokenType.NEWLINE, '\n', 9, 21),

    lexer.Token(lexer.TokenType.DEDENT, '', 10, 1),
    lexer.Token(lexer.TokenType.IF, 'if', 10, 9),
    lexer.Token(lexer.TokenType.IDENTIFIER, 'b', 10, 12),
    lexer.Token(lexer.TokenType.COLON, ':', 10, 13),
    lexer.Token(lexer.TokenType.NEWLINE, '\n', 10, 14),

    lexer.Token(lexer.TokenType.INDENT, '', 11, 1),
    lexer.Token(lexer.TokenType.BREAK, 'break', 11, 13),
    lexer.Token(lexer.TokenType.NEWLINE, '\n', 11, 18),

    lexer.Token(lexer.TokenType.DEDENT, '', 12, 1),
    lexer.Token(lexer.TokenType.DEDENT, '', 12, 1),
    lexer.Token(lexer.TokenType.RETURN, 'return', 12, 5),
    lexer.Token(lexer.TokenType.NUMBER, '0', 12, 12),
    lexer.Token(lexer.TokenType.NEWLINE, '\n', 12, 13),

    lexer.Token(lexer.TokenType.DEDENT, '', 13, 1),
    lexer.Token(lexer.TokenType.EOF, '', 13, 1),
]


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
        {
            'input': "'asdf'",
            'expected': 'asdf',
        },
        # TODO(will): Should we check to make sure the outside characters are actually quotes?
        {
            'input': "asdf",
            'expected': 'sd',
        },
        {
            'input': "'Hello World!\n'",
            'expected': 'Hello World!\n',
        },
        {
            'input': "'\"Hello World!\n'",
            'expected': '\"Hello World!\n',
        },
        {
            'input': "'\'Hello World!\n'",
            'expected': '\'Hello World!\n',
        },
        {
            'input': "'\tHello World!'",
            'expected': '\tHello World!',
        },
        {
            'input': "'Hello World!\0'",
            'expected': 'Hello World!\0',
        },
        {
            'input': "'Hello World!\r'",
            'expected': 'Hello World!\r',
        },
    ]

    for tc in tcs:
        assert tc['expected'] == parser._unescape_string_literal(tc['input'])


def test_parser_init_empty():
    tokens = []

    with pytest.raises(ValueError, match='tokens must have non zero length'):
        p = parser.Parser(tokens)


def test_parser_init_no_eof():
    tokens = [
        lexer.Token(lexer.TokenType.DEF, 'def', 1, 1),
    ]

    with pytest.raises(ValueError, match='tokens must be terminated by an EOF'):
        p = parser.Parser(tokens)


def test_parser_peek_init():
    p = parser.Parser(TEST_TOKENS)
    assert TEST_TOKENS[0] == p.peek()


def test_parser_peek_set_pos():
    p = parser.Parser(TEST_TOKENS)
    p.pos = 17
    assert TEST_TOKENS[17] == p.peek()


def test_parser_peek_set_pos_off_end():
    p = parser.Parser(TEST_TOKENS)
    p.pos = len(TEST_TOKENS)
    assert TEST_TOKENS[-1] == p.peek()


def test_parser_peek_offset():
    p = parser.Parser(TEST_TOKENS)
    assert TEST_TOKENS[52] == p.peek(52)


def test_parser_peek_offset_off_end():
    p = parser.Parser(TEST_TOKENS)
    assert TEST_TOKENS[-1] == p.peek(len(TEST_TOKENS))


def test_parser_current_init():
    p = parser.Parser(TEST_TOKENS)
    assert TEST_TOKENS[0] == p.current()


def test_parser_current_set_pos():
    p = parser.Parser(TEST_TOKENS)
    p.pos = 27
    assert TEST_TOKENS[27] == p.current()


def test_parser_current_set_pos_off_end():
    p = parser.Parser(TEST_TOKENS)
    p.pos = len(TEST_TOKENS)
    assert TEST_TOKENS[-1] == p.current()


def test_parser_at_end():
    p = parser.Parser(TEST_TOKENS)
    for i in range(len(TEST_TOKENS)):
        p.pos = i
        if i == len(TEST_TOKENS) - 1:
            assert p.at_end()
        else:
            assert not p.at_end()


def test_parser_at_end_mutliple_eof():
    p = parser.Parser(TEST_TOKENS + TEST_TOKENS)
    for i in range(2*len(TEST_TOKENS)):
        p.pos = i
        if i == len(TEST_TOKENS) - 1 or i == 2 * len(TEST_TOKENS) - 1:
            assert p.at_end()
        else:
            assert not p.at_end()


def test_parser_check_hit():
    p = parser.Parser(TEST_TOKENS)
    assert p.check(lexer.TokenType.DEF)


def test_parser_check_miss():
    p = parser.Parser(TEST_TOKENS)
    assert not p.check(lexer.TokenType.INT)


def test_parser_check_hit_multiple():
    p = parser.Parser(TEST_TOKENS)
    assert p.check(lexer.TokenType.INT, lexer.TokenType.DEF)


def test_parser_check_miss_multiple():
    p = parser.Parser(TEST_TOKENS)
    assert not p.check(lexer.TokenType.INT, lexer.TokenType.BOOL)


def test_parser_check_cannot_see_eof():
    p = parser.Parser(TEST_TOKENS)
    p.pos = len(TEST_TOKENS) - 1
    assert not p.check(lexer.TokenType.EOF)


def test_parser_advance():
    p = parser.Parser(TEST_TOKENS)
    for i in range(len(TEST_TOKENS)):
        assert TEST_TOKENS[i] == p.advance()


def test_parser_match_hit():
    p = parser.Parser(TEST_TOKENS)
    assert p.match(lexer.TokenType.DEF)
    assert p.pos == 1


def test_parser_match_miss():
    p = parser.Parser(TEST_TOKENS)
    assert not p.match(lexer.TokenType.INT)
    assert p.pos == 0


def test_expect_hit():
    p = parser.Parser(TEST_TOKENS)
    assert TEST_TOKENS[0] == p.expect(lexer.TokenType.DEF)
    assert 1 == p.pos


def test_expect_miss():
    p = parser.Parser(TEST_TOKENS)
    with pytest.raises(parser.ParseError):
        p.expect(lexer.TokenType.INT)


def test_skip_newlines():
    p = parser.Parser(TEST_TOKENS)
    p.skip_newlines()
    assert 0 == p.pos
    p.pos = 6
    p.skip_newlines()
    assert 7 == p.pos
    p.pos = 18
    p.skip_newlines()
    assert 20 == p.pos


def test_skip_newlines_all_newlines():
    p = parser.Parser([
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 1),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 1),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 3, 1),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 4, 1),
        lexer.Token(lexer.TokenType.EOF, '', 4, 2),
    ])
    p.skip_newlines()
    assert 4 == p.pos


def test_parse_program():
    p = parser.Parser(TEST_TOKENS)

    expected = parser.Program(
        functions=[
            parser.Function(name='main', return_type='int', params=[], body=[
                parser.VarDecl(name='i', var_type='int', init=parser.Constant(value=0)),
                parser.VarDecl(name='a', var_type='bool'),
                parser.VarDecl(name='b', var_type='bool'),
                parser.While(
                    condition=parser.Binary(
                        op=parser.BinaryOp.LESS_THAN,
                        left=parser.Variable(name='i'),
                        right=parser.Constant(value=10),
                    ),
                    body=[
                        parser.Assign(
                            name='i',
                            value=parser.Binary(
                                op=parser.BinaryOp.ADD,
                                left=parser.Variable(name='i'),
                                right=parser.Constant(value=1),
                            ),
                        ),
                        parser.If(
                            condition=parser.Variable(name='a'),
                            then_body=[parser.Continue()],
                        ),
                        parser.If(
                            condition=parser.Variable(name='b'),
                            then_body=[parser.Break()],
                        ),
                    ],
                ),
                parser.Return(
                    value=parser.Constant(value=0),
                ),
            ]),
        ],
    )
    program = p.parse_program()
    assert expected == program


def test_parse_function_only_eof():
    tokens = [
        lexer.Token(lexer.TokenType.EOF, '', 1, 1),
    ]
    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape("Expected 'def' to start a function definition at line 1, column 1")):
        p.parse_function()


def test_parse_function_no_type():
    tokens = [
        lexer.Token(lexer.TokenType.DEF, 'def', 1, 1),
        lexer.Token(lexer.TokenType.EOF, '', 1, 1),
    ]
    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape("Expected a type ('int', 'bool', 'str', or '[size]type'), got TokenType.EOF ('') at line 1, column 1")):
        p.parse_function()


def test_parse_function_no_open_paren():
    tokens = [
        lexer.Token(lexer.TokenType.DEF, 'def', 1, 1),
        lexer.Token(lexer.TokenType.INT, 'int', 1, 5),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'main', 1, 9),
        lexer.Token(lexer.TokenType.EOF, '', 1, 9),
    ]
    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape("Expected '(' after function name at line 1, column 9")):
        p.parse_function()


def test_parse_function_no_close_paren():
    tokens = [
        lexer.Token(lexer.TokenType.DEF, 'def', 1, 1),
        lexer.Token(lexer.TokenType.INT, 'int', 1, 5),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'main', 1, 9),
        lexer.Token(lexer.TokenType.OPEN_PAREN, '(', 1, 13),
        lexer.Token(lexer.TokenType.EOF, '', 1, 1),
    ]
    p = parser.Parser(tokens)

    # The parser checks to see if there's a closing paren, if not it assumes there are parameters.
    #  Thus it calls parse_param() which in turn calls parse_type() leading to the following error message.
    #  TODO(will): Low priority, but this should probably be a more helpful message.
    with pytest.raises(parser.ParseError, match=re.escape("Expected a type ('int', 'bool', 'str', or '[size]type'), got TokenType.EOF ('') at line 1, column 1")):
        p.parse_function()


def test_parse_function_no_colon():
    tokens = [
        lexer.Token(lexer.TokenType.DEF, 'def', 1, 1),
        lexer.Token(lexer.TokenType.INT, 'int', 1, 5),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'main', 1, 9),
        lexer.Token(lexer.TokenType.OPEN_PAREN, '(', 1, 13),
        lexer.Token(lexer.TokenType.CLOSE_PAREN, ')', 1, 14),
        lexer.Token(lexer.TokenType.EOF, '', 1, 15),
    ]
    p = parser.Parser(tokens)

    expected = parser.Function(name='main', return_type='int', body=[])

    with pytest.raises(parser.ParseError, match=re.escape("Expected ':' to start the function body at line 1, column 15")):
        p.parse_function()


def test_parse_function_no_newline():
    tokens = [
        lexer.Token(lexer.TokenType.DEF, 'def', 1, 1),
        lexer.Token(lexer.TokenType.INT, 'int', 1, 5),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'main', 1, 9),
        lexer.Token(lexer.TokenType.OPEN_PAREN, '(', 1, 13),
        lexer.Token(lexer.TokenType.CLOSE_PAREN, ')', 1, 14),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 15),
        lexer.Token(lexer.TokenType.EOF, '', 1, 16),
    ]
    p = parser.Parser(tokens)

    expected = parser.Function(name='main', return_type='int', body=[])

    with pytest.raises(parser.ParseError, match=re.escape("Expected a newline after ':' at line 1, column 16")):
        p.parse_function()


def test_parse_function_no_indent():
    tokens = [
        lexer.Token(lexer.TokenType.DEF, 'def', 1, 1),
        lexer.Token(lexer.TokenType.INT, 'int', 1, 5),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'main', 1, 9),
        lexer.Token(lexer.TokenType.OPEN_PAREN, '(', 1, 13),
        lexer.Token(lexer.TokenType.CLOSE_PAREN, ')', 1, 14),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 15),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 16),
        lexer.Token(lexer.TokenType.EOF, '', 2, 1),
    ]
    p = parser.Parser(tokens)

    expected = parser.Function(name='main', return_type='int', body=[])

    with pytest.raises(parser.ParseError, match=re.escape("Expected an indented block at line 2, column 1")):
        p.parse_function()


def test_parse_function_no_dedent():
    tokens = [
        lexer.Token(lexer.TokenType.DEF, 'def', 1, 1),
        lexer.Token(lexer.TokenType.INT, 'int', 1, 5),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'main', 1, 9),
        lexer.Token(lexer.TokenType.OPEN_PAREN, '(', 1, 13),
        lexer.Token(lexer.TokenType.CLOSE_PAREN, ')', 1, 14),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 15),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 16),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.EOF, '', 2, 5),
    ]
    p = parser.Parser(tokens)

    expected = parser.Function(name='main', return_type='int', body=[])

    with pytest.raises(parser.ParseError, match=re.escape("Expected the end of an indented block at line 2, column 5")):
        p.parse_function()


def test_parse_function_no_body():
    tokens = [
        lexer.Token(lexer.TokenType.DEF, 'def', 1, 1),
        lexer.Token(lexer.TokenType.INT, 'int', 1, 5),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'main', 1, 9),
        lexer.Token(lexer.TokenType.OPEN_PAREN, '(', 1, 13),
        lexer.Token(lexer.TokenType.CLOSE_PAREN, ')', 1, 14),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 15),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 16),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.DEDENT, '', 2, 5),
        lexer.Token(lexer.TokenType.EOF, '', 2, 6),
    ]
    p = parser.Parser(tokens)

    expected = parser.Function(name='main', return_type='int', body=[])

    with pytest.raises(parser.ParseError, match=re.escape("Expected at least one statement in this block")):
        p.parse_function()


# This works at this stage, even though nothing is returned. That will be an error in the semantic analysis step.
def test_parse_function_basic():
    tokens = [
        lexer.Token(lexer.TokenType.DEF, 'def', 1, 1),
        lexer.Token(lexer.TokenType.INT, 'int', 1, 5),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'main', 1, 9),
        lexer.Token(lexer.TokenType.OPEN_PAREN, '(', 1, 13),
        lexer.Token(lexer.TokenType.CLOSE_PAREN, ')', 1, 14),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 15),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 16),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.NUMBER, '1', 2, 6),
        lexer.Token(lexer.TokenType.DEDENT, '', 2, 5),
        lexer.Token(lexer.TokenType.EOF, '', 2, 7),
    ]
    p = parser.Parser(tokens)

    expected = parser.Function(name='main', return_type='int', body=[
        parser.ExprStmt(
            expr=parser.Constant(value=1),
        ),
    ])

    function = p.parse_function()
    assert expected == function


def test_parse_params_empty():
    tokens = [
        lexer.Token(lexer.TokenType.CLOSE_PAREN, ')', 1, 14),
        lexer.Token(lexer.TokenType.EOF, '', 2, 7),
    ]
    p = parser.Parser(tokens)

    assert [] == p.parse_params()


def test_parse_params_no_type():
    tokens = [
        lexer.Token(lexer.TokenType.IDENTIFIER, 'a', 1, 1),
        lexer.Token(lexer.TokenType.CLOSE_PAREN, ')', 1, 2),
        lexer.Token(lexer.TokenType.EOF, '', 2, 7),
    ]
    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape("Expected a type ('int', 'bool', 'str', or '[size]type'), got TokenType.IDENTIFIER ('a') at line 1, column 1")):
        p.parse_params()


def test_parse_params_no_name():
    tokens = [
        lexer.Token(lexer.TokenType.INT, 'int', 1, 1),
        lexer.Token(lexer.TokenType.CLOSE_PAREN, ')', 1, 2),
        lexer.Token(lexer.TokenType.EOF, '', 2, 7),
    ]
    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape("Expected a parameter name at line 1, column 2")):
        p.parse_params()


def test_parse_params_single_param():
    tokens = [
        lexer.Token(lexer.TokenType.INT, 'int', 1, 1),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'a', 1, 5),
        lexer.Token(lexer.TokenType.CLOSE_PAREN, ')', 1, 6),
        lexer.Token(lexer.TokenType.EOF, '', 1, 7),
    ]
    p = parser.Parser(tokens)

    expected = [
        parser.Param(name='a', type='int'),
    ]

    assert expected == p.parse_params()


def test_parse_params_two_params_no_second():
    tokens = [
        lexer.Token(lexer.TokenType.INT, 'int', 1, 1),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'a', 1, 5),
        lexer.Token(lexer.TokenType.COMMA, ',', 1, 6),
        lexer.Token(lexer.TokenType.CLOSE_PAREN, ')', 1, 7),
        lexer.Token(lexer.TokenType.EOF, '', 1, 8),
    ]
    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape("Expected a type ('int', 'bool', 'str', or '[size]type'), got TokenType.CLOSE_PAREN (')') at line 1, column 7")):
        p.parse_params()


def test_parse_params_two_params_no_name():
    tokens = [
        lexer.Token(lexer.TokenType.INT, 'int', 1, 1),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'a', 1, 5),
        lexer.Token(lexer.TokenType.COMMA, ',', 1, 6),
        lexer.Token(lexer.TokenType.BOOL, 'bool', 1, 7),
        lexer.Token(lexer.TokenType.CLOSE_PAREN, ')', 1, 11),
        lexer.Token(lexer.TokenType.EOF, '', 1, 12),
    ]
    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape("Expected a parameter name at line 1, column 11")):
        p.parse_params()


def test_parse_params_two_params():
    tokens = [
        lexer.Token(lexer.TokenType.INT, 'int', 1, 1),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'a', 1, 5),
        lexer.Token(lexer.TokenType.COMMA, ',', 1, 6),
        lexer.Token(lexer.TokenType.BOOL, 'bool', 1, 7),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'b', 1, 11),
        lexer.Token(lexer.TokenType.CLOSE_PAREN, ')', 1, 12),
        lexer.Token(lexer.TokenType.EOF, '', 1, 13),
    ]
    p = parser.Parser(tokens)

    expected = [
        parser.Param(
            name='a',
            type='int',
        ),
        parser.Param(
            name='b',
            type='bool',
        ),
    ]

    params = p.parse_params()
    assert expected == params


def test_parse_param_empty():
    tokens = [
        lexer.Token(lexer.TokenType.EOF, '', 1, 1),
    ]
    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape("Expected a type ('int', 'bool', 'str', or '[size]type'), got TokenType.EOF ('') at line 1, column 1")):
        p.parse_param()


def test_parse_param_no_name():
    tokens = [
        lexer.Token(lexer.TokenType.INT, 'int', 1, 1),
        lexer.Token(lexer.TokenType.EOF, '', 1, 4),
    ]
    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape("Expected a parameter name at line 1, column 4")):
        p.parse_param()


def test_parse_param():
    tokens = [
        lexer.Token(lexer.TokenType.INT, 'int', 1, 1),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'i', 1, 5),
        lexer.Token(lexer.TokenType.EOF, '', 1, 6),
    ]
    p = parser.Parser(tokens)

    assert parser.Param(name='i', type='int') == p.parse_param()


def test_parse_type_empty():
    tokens = [
        lexer.Token(lexer.TokenType.EOF, '', 1, 1),
    ]
    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape("Expected a type ('int', 'bool', 'str', or '[size]type'), got TokenType.EOF ('') at line 1, column 1")):
        p.parse_type()


def test_parse_type_int():
    tokens = [
        lexer.Token(lexer.TokenType.INT, 'int', 1, 1),
        lexer.Token(lexer.TokenType.EOF, '', 1, 4),
    ]
    p = parser.Parser(tokens)

    assert 'int' == p.parse_type()


def test_parse_type_str():
    tokens = [
        lexer.Token(lexer.TokenType.STR, 'str', 1, 1),
        lexer.Token(lexer.TokenType.EOF, '', 1, 4),
    ]
    p = parser.Parser(tokens)

    assert 'str' == p.parse_type()


def test_parse_type_bool():
    tokens = [
        lexer.Token(lexer.TokenType.STR, 'bool', 1, 1),
        lexer.Token(lexer.TokenType.EOF, '', 1, 5),
    ]
    p = parser.Parser(tokens)

    assert 'bool' == p.parse_type()


def test_parse_type_array():
    tokens = [
        lexer.Token(lexer.TokenType.OPEN_BRACKET, '[', 1, 1),
        lexer.Token(lexer.TokenType.NUMBER, '3', 1, 2),
        lexer.Token(lexer.TokenType.CLOSE_BRACKET, ']', 1, 3),
        lexer.Token(lexer.TokenType.INT, 'int', 1, 4),
        lexer.Token(lexer.TokenType.EOF, '', 1, 7),
    ]
    p = parser.Parser(tokens)

    assert parser.ArrayTypeExpr(size=3, element_type='int') == p.parse_type()


def test_parse_type_array_missing_size():
    tokens = [
        lexer.Token(lexer.TokenType.OPEN_BRACKET, '[', 1, 1),
        lexer.Token(lexer.TokenType.EOF, '', 1, 2),
    ]
    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape('Expected an array size (a positive integer literal) at line 1, column 2')):
        p.parse_type()


def test_parse_type_array_missing_size2():
    tokens = [
        lexer.Token(lexer.TokenType.OPEN_BRACKET, '[', 1, 1),
        lexer.Token(lexer.TokenType.CLOSE_BRACKET, ']', 1, 2),
        lexer.Token(lexer.TokenType.INT, 'int', 1, 3),
        lexer.Token(lexer.TokenType.EOF, '', 1, 6),
    ]
    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape('Expected an array size (a positive integer literal) at line 1, column 2')):
        p.parse_type()


def test_parse_type_array_missing_closing_brakcet():
    tokens = [
        lexer.Token(lexer.TokenType.OPEN_BRACKET, '[', 1, 1),
        lexer.Token(lexer.TokenType.NUMBER, '3', 1, 2),
        lexer.Token(lexer.TokenType.EOF, '', 1, 3),
    ]
    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape('Expected \']\' after array size at line 1, column 3')):
        p.parse_type()


def test_parse_type_array_missing_type():
    tokens = [
        lexer.Token(lexer.TokenType.OPEN_BRACKET, '[', 1, 1),
        lexer.Token(lexer.TokenType.NUMBER, '3', 1, 2),
        lexer.Token(lexer.TokenType.CLOSE_BRACKET, ']', 1, 3),
        lexer.Token(lexer.TokenType.EOF, '', 1, 4),
    ]
    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape("Expected a type ('int', 'bool', 'str', or '[size]type'), got TokenType.EOF ('') at line 1, column 4")):
        p.parse_type()


def test_parse_type_array_float_size():
    tokens = [
        lexer.Token(lexer.TokenType.OPEN_BRACKET, '[', 1, 1),
        lexer.Token(lexer.TokenType.NUMBER, '3.3', 1, 2),
        lexer.Token(lexer.TokenType.CLOSE_BRACKET, ']', 1, 3),
        lexer.Token(lexer.TokenType.INT, 'int', 1, 4),
        lexer.Token(lexer.TokenType.EOF, '', 1, 7),
    ]
    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape("Array size must be a whole number, got '3.3' at line 1, column 2")):
        p.parse_type()


def test_parse_type_array_negative_size():
    tokens = [
        lexer.Token(lexer.TokenType.OPEN_BRACKET, '[', 1, 1),
        lexer.Token(lexer.TokenType.NUMBER, '-3', 1, 2),
        lexer.Token(lexer.TokenType.CLOSE_BRACKET, ']', 1, 4),
        lexer.Token(lexer.TokenType.INT, 'int', 1, 5),
        lexer.Token(lexer.TokenType.EOF, '', 1, 8),
    ]
    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape("Array size must be positive, got -3 at line 1, column 2")):
        p.parse_type()


def test_parse_type_array_nested():
    tokens = [
        lexer.Token(lexer.TokenType.OPEN_BRACKET, '[', 1, 1),
        lexer.Token(lexer.TokenType.NUMBER, '3', 1, 2),
        lexer.Token(lexer.TokenType.CLOSE_BRACKET, ']', 1, 3),
        lexer.Token(lexer.TokenType.OPEN_BRACKET, '[', 1, 4),
        lexer.Token(lexer.TokenType.NUMBER, '2', 1, 5),
        lexer.Token(lexer.TokenType.CLOSE_BRACKET, ']', 1, 6),
        lexer.Token(lexer.TokenType.INT, 'int', 1, 7),
        lexer.Token(lexer.TokenType.EOF, '', 1, 10),
    ]
    p = parser.Parser(tokens)

    assert parser.ArrayTypeExpr(size=3, element_type=parser.ArrayTypeExpr(size=2, element_type='int')) == p.parse_type()


def test_parse_block_empty():
    tokens = [
        lexer.Token(lexer.TokenType.EOF, '', 1, 1),
    ]
    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape("Expected an indented block at line 1, column 1")):
        p.parse_block()


def test_parse_block_no_dedent():
    tokens = [
        lexer.Token(lexer.TokenType.INDENT, '', 1, 1),
        lexer.Token(lexer.TokenType.EOF, '', 1, 2),
    ]
    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape("Expected the end of an indented block at line 1, column 2")):
        p.parse_block()


def test_parse_block_no_statement():
    tokens = [
        lexer.Token(lexer.TokenType.INDENT, '', 1, 1),
        lexer.Token(lexer.TokenType.DEDENT, '', 1, 2),
        lexer.Token(lexer.TokenType.EOF, '', 1, 3),
    ]
    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape("Expected at least one statement in this block")):
        p.parse_block()


def test_parse_block_single_statement():
    tokens = [
        lexer.Token(lexer.TokenType.INDENT, '', 1, 1),
        lexer.Token(lexer.TokenType.NUMBER, '1', 1, 2),
        lexer.Token(lexer.TokenType.DEDENT, '', 1, 3),
        lexer.Token(lexer.TokenType.EOF, '', 1, 4),
    ]
    p = parser.Parser(tokens)

    assert [parser.ExprStmt(expr=parser.Constant(value=1))] == p.parse_block()


def test_parse_block_single_statement_ignore_newlines():
    tokens = [
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 1),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 1),
        lexer.Token(lexer.TokenType.INDENT, '', 3, 1),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 3, 2),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 4, 1),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 5, 1),
        lexer.Token(lexer.TokenType.NUMBER, '1', 6, 1),
        lexer.Token(lexer.TokenType.DEDENT, '', 6, 2),
        lexer.Token(lexer.TokenType.EOF, '', 1, 4),
    ]
    p = parser.Parser(tokens)

    assert [parser.ExprStmt(expr=parser.Constant(value=1))] == p.parse_block()


def test_parse_block_many_statements():
    tokens = [
        lexer.Token(lexer.TokenType.INDENT, '', 1, 1),
        lexer.Token(lexer.TokenType.NUMBER, '1', 1, 3),
        lexer.Token(lexer.TokenType.NUMBER, '2', 1, 5),
        lexer.Token(lexer.TokenType.NUMBER, '3', 1, 7),
        lexer.Token(lexer.TokenType.NUMBER, '4', 1, 9),
        lexer.Token(lexer.TokenType.DEDENT, '', 1, 10),
        lexer.Token(lexer.TokenType.EOF, '', 1, 11),
    ]
    p = parser.Parser(tokens)

    assert [
        parser.ExprStmt(expr=parser.Constant(value=1)),
        parser.ExprStmt(expr=parser.Constant(value=2)),
        parser.ExprStmt(expr=parser.Constant(value=3)),
        parser.ExprStmt(expr=parser.Constant(value=4)),
    ] == p.parse_block()


# Next: test_parse_statement
