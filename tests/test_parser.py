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


def test_parse_function_no_name():
    tokens = [
        lexer.Token(lexer.TokenType.DEF, 'def', 1, 1),
        lexer.Token(lexer.TokenType.EOF, '', 1, 1),
    ]
    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape("Expected a function name at line 1, column 1")):
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
    with pytest.raises(
        parser.ParseError,
        match=re.escape(
            "Expected a type ('int', 'int8', 'uint8', 'int64', 'bool', 'str', a struct name, '[size]type', or '[]type'), got TokenType.EOF ('') at line 1, column 1"
        )):
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

    with pytest.raises(
        parser.ParseError,
        match=re.escape(
            "Expected a type ('int', 'int8', 'uint8', 'int64', 'bool', 'str', a struct name, '[size]type', or '[]type'), got TokenType.CLOSE_PAREN (')') at line 1, column 7"
        )):
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

    with pytest.raises(
        parser.ParseError,
        match=re.escape(
            "Expected a type ('int', 'int8', 'uint8', 'int64', 'bool', 'str', a struct name, '[size]type', or '[]type'), got TokenType.EOF ('') at line 1, column 1"
        )):
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

    with pytest.raises(
        parser.ParseError,
        match=re.escape(
            "Expected a type ('int', 'int8', 'uint8', 'int64', 'bool', 'str', a struct name, '[size]type', or '[]type'), got TokenType.EOF ('') at line 1, column 1"
        )):
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

    with pytest.raises(parser.ParseError, match=re.escape('Expected an array size (a positive integer literal), or \']\' for a slice type at line 1, column 2')):
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

    with pytest.raises(
        parser.ParseError,
        match=re.escape(
            "Expected a type ('int', 'int8', 'uint8', 'int64', 'bool', 'str', a struct name, '[size]type', or '[]type'), got TokenType.EOF ('') at line 1, column 4"
        )):
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


def test_parse_statement_empty():
    tokens = [
        lexer.Token(lexer.TokenType.EOF, '', 1, 11),
    ]
    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape('Expected an expression, got TokenType.EOF (\'\') at line 1, column 11')):
        p.parse_statement()


def test_parse_statement_int_no_name():
    tokens = [
        lexer.Token(lexer.TokenType.INT, '', 1, 1),
        lexer.Token(lexer.TokenType.EOF, '', 1, 2),
    ]
    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape('Expected a variable name at line 1, column 2')):
        p.parse_statement()


def test_parse_statement_int():
    tokens = [
        lexer.Token(lexer.TokenType.INT, '', 1, 1),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'a', 1, 2),
        lexer.Token(lexer.TokenType.EOF, '', 1, 3),
    ]
    p = parser.Parser(tokens)

    assert parser.VarDecl(name='a', var_type='') == p.parse_statement()


def test_parse_statement_str_no_name():
    tokens = [
        lexer.Token(lexer.TokenType.STR, '', 1, 1),
        lexer.Token(lexer.TokenType.EOF, '', 1, 2),
    ]
    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape('Expected a variable name at line 1, column 2')):
        p.parse_statement()


def test_parse_statement_str():
    tokens = [
        lexer.Token(lexer.TokenType.STR, '', 1, 1),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'a', 1, 2),
        lexer.Token(lexer.TokenType.EOF, '', 1, 3),
    ]
    p = parser.Parser(tokens)

    assert parser.VarDecl(name='a', var_type='') == p.parse_statement()


def test_parse_statement_bool_no_name():
    tokens = [
        lexer.Token(lexer.TokenType.BOOL, '', 1, 1),
        lexer.Token(lexer.TokenType.EOF, '', 1, 2),
    ]
    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape('Expected a variable name at line 1, column 2')):
        p.parse_statement()


def test_parse_statement_bool():
    tokens = [
        lexer.Token(lexer.TokenType.BOOL, '', 1, 1),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'a', 1, 2),
        lexer.Token(lexer.TokenType.EOF, '', 1, 3),
    ]
    p = parser.Parser(tokens)

    assert parser.VarDecl(name='a', var_type='') == p.parse_statement()


def test_parse_statement_array_no_size():
    tokens = [
        lexer.Token(lexer.TokenType.OPEN_BRACKET, '[', 1, 1),
        lexer.Token(lexer.TokenType.EOF, '', 1, 2),
    ]
    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape('Expected an array size (a positive integer literal), or \']\' for a slice type at line 1, column 2')):
        p.parse_statement()


def test_parse_statement_array_no_close_bracket():
    tokens = [
        lexer.Token(lexer.TokenType.OPEN_BRACKET, '[', 1, 1),
        lexer.Token(lexer.TokenType.NUMBER, '3', 1, 2),
        lexer.Token(lexer.TokenType.EOF, '', 1, 3),
    ]
    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape('Expected \']\' after array size at line 1, column 3')):
        p.parse_statement()


def test_parse_statement_array_no_type():
    tokens = [
        lexer.Token(lexer.TokenType.OPEN_BRACKET, '[', 1, 1),
        lexer.Token(lexer.TokenType.NUMBER, '3', 1, 2),
        lexer.Token(lexer.TokenType.CLOSE_BRACKET, ']', 1, 3),
        lexer.Token(lexer.TokenType.EOF, '', 1, 4),
    ]
    p = parser.Parser(tokens)

    with pytest.raises(
        parser.ParseError,
        match=re.escape(
            "Expected a type ('int', 'int8', 'uint8', 'int64', 'bool', 'str', a struct name, '[size]type', or '[]type'), got TokenType.EOF ('') at line 1, column 4"
        )):
        p.parse_statement()


def test_parse_statement_array_no_name():
    tokens = [
        lexer.Token(lexer.TokenType.OPEN_BRACKET, '[', 1, 1),
        lexer.Token(lexer.TokenType.NUMBER, '3', 1, 2),
        lexer.Token(lexer.TokenType.CLOSE_BRACKET, ']', 1, 3),
        lexer.Token(lexer.TokenType.INT, 'int', 1, 4),
        lexer.Token(lexer.TokenType.EOF, '', 1, 7),
    ]
    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape('Expected a variable name at line 1, column 7')):
        p.parse_statement()


def test_parse_statement_array():
    tokens = [
        lexer.Token(lexer.TokenType.OPEN_BRACKET, '[', 1, 1),
        lexer.Token(lexer.TokenType.NUMBER, '3', 1, 2),
        lexer.Token(lexer.TokenType.CLOSE_BRACKET, ']', 1, 3),
        lexer.Token(lexer.TokenType.INT, 'int', 1, 4),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'arr', 1, 7),
        lexer.Token(lexer.TokenType.EOF, '', 1, 10),
    ]
    p = parser.Parser(tokens)

    assert parser.VarDecl(name='arr', var_type=parser.ArrayTypeExpr(size=3, element_type='int')) == p.parse_statement()


def test_parse_statement_return_no_value():
    tokens = [
        lexer.Token(lexer.TokenType.RETURN, 'return', 1, 1),
        lexer.Token(lexer.TokenType.EOF, '', 1, 7),
    ]
    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape('Expected an expression, got TokenType.EOF (\'\') at line 1, column 7')):
        p.parse_statement()


def test_parse_statement_return():
    tokens = [
        lexer.Token(lexer.TokenType.RETURN, 'return', 1, 1),
        lexer.Token(lexer.TokenType.NUMBER, '5', 1, 8),
        lexer.Token(lexer.TokenType.EOF, '', 1, 9),
    ]
    p = parser.Parser(tokens)

    assert parser.Return(value=parser.Constant(value=5)) == p.parse_statement()


def test_parse_statement_if_no_expression():
    tokens = [
        lexer.Token(lexer.TokenType.IF, 'if', 1, 1),
        lexer.Token(lexer.TokenType.EOF, '', 1, 3),
    ]
    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape('Expected an expression, got TokenType.EOF (\'\') at line 1, column 3')):
        p.parse_statement()


def test_parse_statement_if_no_colon():
    tokens = [
        lexer.Token(lexer.TokenType.IF, 'if', 1, 1),
        lexer.Token(lexer.TokenType.TRUE, 'true', 1, 3),
        lexer.Token(lexer.TokenType.EOF, '', 1, 7),
    ]
    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape('Expected \':\' to start the if body at line 1, column 7')):
        p.parse_statement()


def test_parse_statement_if_no_newline():
    tokens = [
        lexer.Token(lexer.TokenType.IF, 'if', 1, 1),
        lexer.Token(lexer.TokenType.TRUE, 'true', 1, 3),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 7),
        lexer.Token(lexer.TokenType.EOF, '', 1, 8),
    ]
    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape('Expected a newline after \':\' at line 1, column 8')):
        p.parse_statement()


def test_parse_statement_if_no_indent():
    tokens = [
        lexer.Token(lexer.TokenType.IF, 'if', 1, 1),
        lexer.Token(lexer.TokenType.TRUE, 'true', 1, 3),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 7),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 8),
        lexer.Token(lexer.TokenType.EOF, '', 1, 9),
    ]
    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape('Expected an indented block at line 1, column 9')):
        p.parse_statement()


def test_parse_statement_if_no_dedent():
    tokens = [
        lexer.Token(lexer.TokenType.IF, 'if', 1, 1),
        lexer.Token(lexer.TokenType.TRUE, 'true', 1, 3),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 7),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 8),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.EOF, '', 2, 5),
    ]
    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape('Expected the end of an indented block at line 2, column 5')):
        p.parse_statement()


def test_parse_statement_if_no_body():
    tokens = [
        lexer.Token(lexer.TokenType.IF, 'if', 1, 1),
        lexer.Token(lexer.TokenType.TRUE, 'true', 1, 3),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 7),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 8),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.DEDENT, '', 2, 5),
        lexer.Token(lexer.TokenType.EOF, '', 2, 6),
    ]
    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape('Expected at least one statement in this block')):
        p.parse_statement()


def test_parse_statement_if():
    tokens = [
        lexer.Token(lexer.TokenType.IF, 'if', 1, 1),
        lexer.Token(lexer.TokenType.TRUE, 'true', 1, 3),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 7),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 8),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.NUMBER, '1', 2, 5),
        lexer.Token(lexer.TokenType.DEDENT, '', 2, 6),
        lexer.Token(lexer.TokenType.EOF, '', 2, 7),
    ]
    p = parser.Parser(tokens)

    expected = parser.If(condition=parser.BoolLiteral(value=True), then_body=[parser.ExprStmt(expr=parser.Constant(value=1))])

    assert expected == p.parse_statement()


def test_parse_statement_while_no_condition():
    tokens = [
        lexer.Token(lexer.TokenType.WHILE, 'while', 1, 1),
        lexer.Token(lexer.TokenType.EOF, '', 1, 6),
    ]
    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape('Expected an expression, got TokenType.EOF (\'\') at line 1, column 6')):
        p.parse_statement()


def test_parse_statement_while_no_colon():
    tokens = [
        lexer.Token(lexer.TokenType.WHILE, 'while', 1, 1),
        lexer.Token(lexer.TokenType.TRUE, 'true', 1, 6),
        lexer.Token(lexer.TokenType.EOF, '', 1, 10),
    ]
    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape('Expected \':\' to start the while body at line 1, column 10')):
        p.parse_statement()


def test_parse_statement_while_no_newline():
    tokens = [
        lexer.Token(lexer.TokenType.WHILE, 'while', 1, 1),
        lexer.Token(lexer.TokenType.TRUE, 'true', 1, 6),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 10),
        lexer.Token(lexer.TokenType.EOF, '', 1, 11),
    ]
    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape('Expected a newline after \':\' at line 1, column 11')):
        p.parse_statement()


def test_parse_statement_while_no_indent():
    tokens = [
        lexer.Token(lexer.TokenType.WHILE, 'while', 1, 1),
        lexer.Token(lexer.TokenType.TRUE, 'true', 1, 6),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 10),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 11),
        lexer.Token(lexer.TokenType.EOF, '', 2, 1),
    ]
    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape('Expected an indented block at line 2, column 1')):
        p.parse_statement()


def test_parse_statement_while_no_dedent():
    tokens = [
        lexer.Token(lexer.TokenType.WHILE, 'while', 1, 1),
        lexer.Token(lexer.TokenType.TRUE, 'true', 1, 6),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 10),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 11),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.EOF, '', 2, 5),
    ]
    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape('Expected the end of an indented block at line 2, column 5')):
        p.parse_statement()


def test_parse_statement_while_no_body():
    tokens = [
        lexer.Token(lexer.TokenType.WHILE, 'while', 1, 1),
        lexer.Token(lexer.TokenType.TRUE, 'true', 1, 6),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 10),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 11),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.DEDENT, '', 2, 5),
        lexer.Token(lexer.TokenType.EOF, '', 2, 6),
    ]
    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape('Expected at least one statement in this block')):
        p.parse_statement()


def test_parse_statement_while():
    tokens = [
        lexer.Token(lexer.TokenType.WHILE, 'while', 1, 1),
        lexer.Token(lexer.TokenType.TRUE, 'true', 1, 6),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 10),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 11),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.NUMBER, '5', 2, 5),
        lexer.Token(lexer.TokenType.DEDENT, '', 2, 6),
        lexer.Token(lexer.TokenType.EOF, '', 2, 7),
    ]
    p = parser.Parser(tokens)

    expected = parser.While(condition=parser.BoolLiteral(value=True), body=[parser.ExprStmt(expr=parser.Constant(value=5))])

    assert expected == p.parse_statement()


def test_parse_statement_break():
    tokens = [
        lexer.Token(lexer.TokenType.BREAK, 'break', 1, 1),
        lexer.Token(lexer.TokenType.EOF, '', 1, 6),
    ]
    p = parser.Parser(tokens)

    expected = parser.Break()

    assert expected == p.parse_statement()


def test_parse_statement_break():
    tokens = [
        lexer.Token(lexer.TokenType.CONTINUE, 'continue', 1, 1),
        lexer.Token(lexer.TokenType.EOF, '', 1, 9),
    ]
    p = parser.Parser(tokens)

    expected = parser.Continue()

    assert expected == p.parse_statement()


def test_parse_statement_assign_no_value():
    tokens = [
        lexer.Token(lexer.TokenType.IDENTIFIER, 'a', 1, 1),
        lexer.Token(lexer.TokenType.ASSIGN, '=', 1, 2),
        lexer.Token(lexer.TokenType.EOF, '', 1, 3),
    ]
    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape('Expected an expression, got TokenType.EOF (\'\') at line 1, column 3')):
        p.parse_statement()


def test_parse_statement_assign():
    tokens = [
        lexer.Token(lexer.TokenType.IDENTIFIER, 'a', 1, 1),
        lexer.Token(lexer.TokenType.ASSIGN, '=', 1, 2),
        lexer.Token(lexer.TokenType.NUMBER, '3', 1, 3),
        lexer.Token(lexer.TokenType.EOF, '', 1, 4),
    ]
    p = parser.Parser(tokens)

    expected = parser.Assign(name='a', value=parser.Constant(value=3))

    assert expected == p.parse_statement()


# TODO(will): Test parse_expr_stmt_or_index_assign() path.


def test_parse_while_empty():
    tokens = [
        lexer.Token(lexer.TokenType.EOF, '', 1, 1),
    ]
    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape('Expected \'while\' at line 1, column 1')):
        p.parse_while()


def test_parse_while_no_condition():
    tokens = [
        lexer.Token(lexer.TokenType.WHILE, 'while', 1, 1),
        lexer.Token(lexer.TokenType.EOF, '', 1, 6),
    ]
    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape('Expected an expression, got TokenType.EOF (\'\') at line 1, column 6')):
        p.parse_while()


def test_parse_while_no_colon():
    tokens = [
        lexer.Token(lexer.TokenType.WHILE, 'while', 1, 1),
        lexer.Token(lexer.TokenType.TRUE, 'true', 1, 6),
        lexer.Token(lexer.TokenType.EOF, '', 1, 10),
    ]
    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape('Expected \':\' to start the while body at line 1, column 10')):
        p.parse_while()


def test_parse_while_no_newline():
    tokens = [
        lexer.Token(lexer.TokenType.WHILE, 'while', 1, 1),
        lexer.Token(lexer.TokenType.TRUE, 'true', 1, 6),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 10),
        lexer.Token(lexer.TokenType.EOF, '', 1, 11),
    ]
    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape('Expected a newline after \':\' at line 1, column 11')):
        p.parse_while()


def test_parse_while_no_indent():
    tokens = [
        lexer.Token(lexer.TokenType.WHILE, 'while', 1, 1),
        lexer.Token(lexer.TokenType.TRUE, 'true', 1, 6),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 10),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 11),
        lexer.Token(lexer.TokenType.EOF, '', 2, 1),
    ]
    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape('Expected an indented block at line 2, column 1')):
        p.parse_while()


def test_parse_while_no_dedent():
    tokens = [
        lexer.Token(lexer.TokenType.WHILE, 'while', 1, 1),
        lexer.Token(lexer.TokenType.TRUE, 'true', 1, 6),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 10),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 11),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.EOF, '', 2, 5),
    ]
    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape('Expected the end of an indented block at line 2, column 5')):
        p.parse_while()


def test_parse_while_no_body():
    tokens = [
        lexer.Token(lexer.TokenType.WHILE, 'while', 1, 1),
        lexer.Token(lexer.TokenType.TRUE, 'true', 1, 6),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 10),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 11),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.DEDENT, '', 2, 5),
        lexer.Token(lexer.TokenType.EOF, '', 2, 6),
    ]
    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape('Expected at least one statement in this block')):
        p.parse_while()


def test_parse_while():
    tokens = [
        lexer.Token(lexer.TokenType.WHILE, 'while', 1, 1),
        lexer.Token(lexer.TokenType.TRUE, 'true', 1, 6),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 10),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 11),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.BREAK, 'break', 2, 5),
        lexer.Token(lexer.TokenType.DEDENT, '', 2, 6),
        lexer.Token(lexer.TokenType.EOF, '', 2, 7),
    ]
    p = parser.Parser(tokens)

    expected = parser.While(condition=parser.BoolLiteral(value=True), body=[parser.Break()])

    assert expected == p.parse_while()


def test_parse_break_emtpy():
    tokens = [
        lexer.Token(lexer.TokenType.EOF, '', 1, 1),
    ]
    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape('Expected \'break\' at line 1, column 1')):
        p.parse_break()


def test_parse_break():
    tokens = [
        lexer.Token(lexer.TokenType.BREAK, 'break', 1, 1),
        lexer.Token(lexer.TokenType.EOF, '', 1, 6),
    ]
    p = parser.Parser(tokens)

    expected = parser.Break()

    assert expected == p.parse_break()


def test_parse_continue_emtpy():
    tokens = [
        lexer.Token(lexer.TokenType.EOF, '', 1, 1),
    ]
    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape('Expected \'continue\' at line 1, column 1')):
        p.parse_continue()


def test_parse_break():
    tokens = [
        lexer.Token(lexer.TokenType.CONTINUE, 'continue', 1, 1),
        lexer.Token(lexer.TokenType.EOF, '', 1, 9),
    ]
    p = parser.Parser(tokens)

    expected = parser.Continue()

    assert expected == p.parse_continue()


def test_parse_if_empty():
    tokens = [
        lexer.Token(lexer.TokenType.IF, 'if', 1, 1),
        lexer.Token(lexer.TokenType.EOF, '', 1, 3),
    ]
    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape('Expected an expression, got TokenType.EOF (\'\') at line 1, column 3')):
        p.parse_if()


def test_parse_if_no_colon():
    tokens = [
        lexer.Token(lexer.TokenType.IF, 'if', 1, 1),
        lexer.Token(lexer.TokenType.TRUE, 'true', 1, 4),
        lexer.Token(lexer.TokenType.EOF, '', 1, 8),
    ]
    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape('Expected \':\' to start the if body at line 1, column 8')):
        p.parse_if()


def test_parse_if_no_newline():
    tokens = [
        lexer.Token(lexer.TokenType.IF, 'if', 1, 1),
        lexer.Token(lexer.TokenType.TRUE, 'true', 1, 4),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 8),
        lexer.Token(lexer.TokenType.EOF, '', 1, 9),
    ]
    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape('Expected a newline after \':\' at line 1, column 9')):
        p.parse_if()


def test_parse_if_no_indent():
    tokens = [
        lexer.Token(lexer.TokenType.IF, 'if', 1, 1),
        lexer.Token(lexer.TokenType.TRUE, 'true', 1, 4),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 8),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 9),
        lexer.Token(lexer.TokenType.EOF, '', 2, 1),
    ]
    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape('Expected an indented block at line 2, column 1')):
        p.parse_if()


def test_parse_if_no_dedent():
    tokens = [
        lexer.Token(lexer.TokenType.IF, 'if', 1, 1),
        lexer.Token(lexer.TokenType.TRUE, 'true', 1, 4),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 8),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 9),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.EOF, '', 2, 5),
    ]
    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape('Expected the end of an indented block at line 2, column 5')):
        p.parse_if()


def test_parse_if_no_body():
    tokens = [
        lexer.Token(lexer.TokenType.IF, 'if', 1, 1),
        lexer.Token(lexer.TokenType.TRUE, 'true', 1, 4),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 8),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 9),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.DEDENT, '', 2, 5),
        lexer.Token(lexer.TokenType.EOF, '', 2, 6),
    ]
    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape('Expected at least one statement in this block')):
        p.parse_if()


def test_parse_if():
    tokens = [
        lexer.Token(lexer.TokenType.IF, 'if', 1, 1),
        lexer.Token(lexer.TokenType.TRUE, 'true', 1, 4),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 8),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 9),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.STRING, '\'hi\'', 2, 5),
        lexer.Token(lexer.TokenType.DEDENT, '', 2, 7),
        lexer.Token(lexer.TokenType.EOF, '', 2, 8),
    ]
    p = parser.Parser(tokens)

    expected = parser.If(condition=parser.BoolLiteral(value=True), then_body=[parser.ExprStmt(expr=parser.StringLiteral(value='hi'))])

    assert expected == p.parse_if()


def test_parse_if():
    tokens = [
        lexer.Token(lexer.TokenType.IF, 'if', 1, 1),
        lexer.Token(lexer.TokenType.TRUE, 'true', 1, 4),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 8),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 9),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.STRING, '\'hi\'', 2, 5),
        lexer.Token(lexer.TokenType.DEDENT, '', 2, 7),
        lexer.Token(lexer.TokenType.EOF, '', 2, 8),
    ]
    p = parser.Parser(tokens)

    expected = parser.If(condition=parser.BoolLiteral(value=True), then_body=[parser.ExprStmt(expr=parser.StringLiteral(value='hi'))])

    assert expected == p.parse_if()


def test_parse_if_else_no_colon():
    tokens = [
        lexer.Token(lexer.TokenType.IF, 'if', 1, 1),
        lexer.Token(lexer.TokenType.TRUE, 'true', 1, 4),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 8),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 9),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.STRING, '\'hi\'', 2, 5),
        lexer.Token(lexer.TokenType.DEDENT, '', 2, 7),
        lexer.Token(lexer.TokenType.ELSE, 'else', 2, 8),
        lexer.Token(lexer.TokenType.EOF, '', 2, 12),
    ]
    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape('Expected \':\' to start the else body at line 2, column 12')):
        p.parse_if()


def test_parse_if_else_no_newline():
    tokens = [
        lexer.Token(lexer.TokenType.IF, 'if', 1, 1),
        lexer.Token(lexer.TokenType.TRUE, 'true', 1, 4),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 8),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 9),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.STRING, '\'hi\'', 2, 5),
        lexer.Token(lexer.TokenType.DEDENT, '', 2, 7),
        lexer.Token(lexer.TokenType.ELSE, 'else', 2, 8),
        lexer.Token(lexer.TokenType.COLON, ':', 2, 12),
        lexer.Token(lexer.TokenType.EOF, '', 2, 13),
    ]
    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape('Expected a newline after \':\' at line 2, column 13')):
        p.parse_if()


def test_parse_if_else_no_indent():
    tokens = [
        lexer.Token(lexer.TokenType.IF, 'if', 1, 1),
        lexer.Token(lexer.TokenType.TRUE, 'true', 1, 4),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 8),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 9),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.STRING, '\'hi\'', 2, 5),
        lexer.Token(lexer.TokenType.DEDENT, '', 2, 7),
        lexer.Token(lexer.TokenType.ELSE, 'else', 2, 8),
        lexer.Token(lexer.TokenType.COLON, ':', 2, 12),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 13),
        lexer.Token(lexer.TokenType.EOF, '', 3, 1),
    ]
    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape('Expected an indented block at line 3, column 1')):
        p.parse_if()


def test_parse_if_else_no_dedent():
    tokens = [
        lexer.Token(lexer.TokenType.IF, 'if', 1, 1),
        lexer.Token(lexer.TokenType.TRUE, 'true', 1, 4),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 8),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 9),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.STRING, '\'hi\'', 2, 5),
        lexer.Token(lexer.TokenType.DEDENT, '', 2, 7),
        lexer.Token(lexer.TokenType.ELSE, 'else', 2, 8),
        lexer.Token(lexer.TokenType.COLON, ':', 2, 12),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 13),
        lexer.Token(lexer.TokenType.INDENT, '', 3, 1),
        lexer.Token(lexer.TokenType.EOF, '', 3, 5),
    ]
    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape('Expected the end of an indented block at line 3, column 5')):
        p.parse_if()


def test_parse_if_else_no_body():
    tokens = [
        lexer.Token(lexer.TokenType.IF, 'if', 1, 1),
        lexer.Token(lexer.TokenType.TRUE, 'true', 1, 4),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 8),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 9),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.STRING, '\'hi\'', 2, 5),
        lexer.Token(lexer.TokenType.DEDENT, '', 2, 7),
        lexer.Token(lexer.TokenType.ELSE, 'else', 2, 8),
        lexer.Token(lexer.TokenType.COLON, ':', 2, 12),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 13),
        lexer.Token(lexer.TokenType.INDENT, '', 3, 1),
        lexer.Token(lexer.TokenType.DEDENT, '', 3, 5),
        lexer.Token(lexer.TokenType.EOF, '', 3, 6),
    ]
    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape('Expected at least one statement in this block')):
        p.parse_if()


def test_parse_if_else():
    tokens = [
        lexer.Token(lexer.TokenType.IF, 'if', 1, 1),
        lexer.Token(lexer.TokenType.TRUE, 'true', 1, 4),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 8),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 9),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.STRING, '\'hi\'', 2, 5),
        lexer.Token(lexer.TokenType.DEDENT, '', 2, 7),
        lexer.Token(lexer.TokenType.ELSE, 'else', 2, 8),
        lexer.Token(lexer.TokenType.COLON, ':', 2, 12),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 13),
        lexer.Token(lexer.TokenType.INDENT, '', 3, 1),
        lexer.Token(lexer.TokenType.STRING, '\'bye\'', 3, 5),
        lexer.Token(lexer.TokenType.DEDENT, '', 3, 8),
        lexer.Token(lexer.TokenType.EOF, '', 3, 9),
    ]
    p = parser.Parser(tokens)

    expected = parser.If(
        condition=parser.BoolLiteral(value=True),
        then_body=[parser.ExprStmt(expr=parser.StringLiteral(value='hi'))],
        else_body=[parser.ExprStmt(expr=parser.StringLiteral(value='bye'))],
    )

    assert expected == p.parse_if()


def test_parse_if_elif_no_condition():
    tokens = [
        lexer.Token(lexer.TokenType.IF, 'if', 1, 1),
        lexer.Token(lexer.TokenType.TRUE, 'true', 1, 4),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 8),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 9),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.STRING, '\'hi\'', 2, 5),
        lexer.Token(lexer.TokenType.DEDENT, '', 2, 7),
        lexer.Token(lexer.TokenType.ELIF, 'elif', 2, 8),
        lexer.Token(lexer.TokenType.EOF, '', 2, 9),
    ]
    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape('Expected an expression, got TokenType.EOF (\'\') at line 2, column 9')):
        p.parse_if()


def test_parse_if_elif_no_colon():
    tokens = [
        lexer.Token(lexer.TokenType.IF, 'if', 1, 1),
        lexer.Token(lexer.TokenType.TRUE, 'true', 1, 4),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 8),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 9),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.STRING, '\'hi\'', 2, 5),
        lexer.Token(lexer.TokenType.DEDENT, '', 2, 7),
        lexer.Token(lexer.TokenType.ELIF, 'elif', 2, 8),
        lexer.Token(lexer.TokenType.FALSE, 'false', 2, 13),
        lexer.Token(lexer.TokenType.EOF, '', 2, 18),
    ]
    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape('Expected \':\' to start the if body at line 2, column 18')):
        p.parse_if()


def test_parse_if_elif_no_newline():
    tokens = [
        lexer.Token(lexer.TokenType.IF, 'if', 1, 1),
        lexer.Token(lexer.TokenType.TRUE, 'true', 1, 4),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 8),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 9),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.STRING, '\'hi\'', 2, 5),
        lexer.Token(lexer.TokenType.DEDENT, '', 2, 7),
        lexer.Token(lexer.TokenType.ELIF, 'elif', 2, 8),
        lexer.Token(lexer.TokenType.FALSE, 'false', 2, 13),
        lexer.Token(lexer.TokenType.COLON, ':', 2, 18),
        lexer.Token(lexer.TokenType.EOF, '', 2, 19),
    ]
    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape('Expected a newline after \':\' at line 2, column 19')):
        p.parse_if()


def test_parse_if_elif_no_indent():
    tokens = [
        lexer.Token(lexer.TokenType.IF, 'if', 1, 1),
        lexer.Token(lexer.TokenType.TRUE, 'true', 1, 4),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 8),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 9),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.STRING, '\'hi\'', 2, 5),
        lexer.Token(lexer.TokenType.DEDENT, '', 2, 7),
        lexer.Token(lexer.TokenType.ELIF, 'elif', 2, 8),
        lexer.Token(lexer.TokenType.FALSE, 'false', 2, 13),
        lexer.Token(lexer.TokenType.COLON, ':', 2, 18),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 19),
        lexer.Token(lexer.TokenType.EOF, '', 3, 1),
    ]
    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape('Expected an indented block at line 3, column 1')):
        p.parse_if()


def test_parse_if_elif_no_dedent():
    tokens = [
        lexer.Token(lexer.TokenType.IF, 'if', 1, 1),
        lexer.Token(lexer.TokenType.TRUE, 'true', 1, 4),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 8),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 9),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.STRING, '\'hi\'', 2, 5),
        lexer.Token(lexer.TokenType.DEDENT, '', 2, 7),
        lexer.Token(lexer.TokenType.ELIF, 'elif', 2, 8),
        lexer.Token(lexer.TokenType.FALSE, 'false', 2, 13),
        lexer.Token(lexer.TokenType.COLON, ':', 2, 18),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 19),
        lexer.Token(lexer.TokenType.INDENT, '', 3, 1),
        lexer.Token(lexer.TokenType.EOF, '', 3, 5),
    ]
    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape('Expected the end of an indented block at line 3, column 5')):
        p.parse_if()


def test_parse_if_elif_no_body():
    tokens = [
        lexer.Token(lexer.TokenType.IF, 'if', 1, 1),
        lexer.Token(lexer.TokenType.TRUE, 'true', 1, 4),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 8),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 9),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.STRING, '\'hi\'', 2, 5),
        lexer.Token(lexer.TokenType.DEDENT, '', 2, 7),
        lexer.Token(lexer.TokenType.ELIF, 'elif', 2, 8),
        lexer.Token(lexer.TokenType.FALSE, 'false', 2, 13),
        lexer.Token(lexer.TokenType.COLON, ':', 2, 18),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 19),
        lexer.Token(lexer.TokenType.INDENT, '', 3, 1),
        lexer.Token(lexer.TokenType.DEDENT, '', 3, 5),
        lexer.Token(lexer.TokenType.EOF, '', 3, 6),
    ]
    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape('Expected at least one statement in this block')):
        p.parse_if()


def test_parse_if_elif():
    tokens = [
        lexer.Token(lexer.TokenType.IF, 'if', 1, 1),
        lexer.Token(lexer.TokenType.TRUE, 'true', 1, 4),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 8),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 9),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.STRING, '\'hi\'', 2, 5),
        lexer.Token(lexer.TokenType.DEDENT, '', 2, 7),
        lexer.Token(lexer.TokenType.ELIF, 'elif', 2, 8),
        lexer.Token(lexer.TokenType.FALSE, 'false', 2, 13),
        lexer.Token(lexer.TokenType.COLON, ':', 2, 18),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 19),
        lexer.Token(lexer.TokenType.INDENT, '', 3, 1),
        lexer.Token(lexer.TokenType.STRING, '\'bye\'', 3, 5),
        lexer.Token(lexer.TokenType.DEDENT, '', 3, 9),
        lexer.Token(lexer.TokenType.EOF, '', 3, 10),
    ]
    p = parser.Parser(tokens)

    expected = parser.If(
        condition=parser.BoolLiteral(value=True),
        then_body=[parser.ExprStmt(expr=parser.StringLiteral(value='hi'))],
        else_body=[
            parser.If(
                condition=parser.BoolLiteral(value=False),
                then_body=[parser.ExprStmt(expr=parser.StringLiteral(value='bye'))],
            ),
        ],
    )

    assert expected == p.parse_if()


def test_parse_if_elif_else():
    tokens = [
        lexer.Token(lexer.TokenType.IF, 'if', 1, 1),
        lexer.Token(lexer.TokenType.TRUE, 'true', 1, 4),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 8),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 9),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.STRING, '\'hi\'', 2, 5),
        lexer.Token(lexer.TokenType.DEDENT, '', 2, 7),
        lexer.Token(lexer.TokenType.ELIF, 'elif', 2, 8),
        lexer.Token(lexer.TokenType.FALSE, 'false', 2, 13),
        lexer.Token(lexer.TokenType.COLON, ':', 2, 18),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 19),
        lexer.Token(lexer.TokenType.INDENT, '', 3, 1),
        lexer.Token(lexer.TokenType.STRING, '\'maybe\'', 3, 5),
        lexer.Token(lexer.TokenType.DEDENT, '', 3, 9),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 3, 10),
        lexer.Token(lexer.TokenType.ELSE, 'else', 4, 1),
        lexer.Token(lexer.TokenType.COLON, ':', 4, 5),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 4, 6),
        lexer.Token(lexer.TokenType.INDENT, '', 5, 1),
        lexer.Token(lexer.TokenType.STRING, '\'maybe\'', 5, 5),
        lexer.Token(lexer.TokenType.DEDENT, '', 5, 10),
        lexer.Token(lexer.TokenType.EOF, '', 5, 11),
    ]
    p = parser.Parser(tokens)

    expected = parser.If(
        condition=parser.BoolLiteral(value=True),
        then_body=[parser.ExprStmt(expr=parser.StringLiteral(value='hi'))],
        else_body=[
            parser.If(
                condition=parser.BoolLiteral(value=False),
                then_body=[parser.ExprStmt(expr=parser.StringLiteral(value='maybe'))],
            ),
            parser.ExprStmt(expr=parser.StringLiteral(value='bye')),
        ],
    )


def test_parse_if_elif_else_ignore_newlines():
    tokens = [
        lexer.Token(lexer.TokenType.IF, 'if', 1, 1),
        lexer.Token(lexer.TokenType.TRUE, 'true', 1, 4),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 8),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 9),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.STRING, '\'hi\'', 2, 5),
        lexer.Token(lexer.TokenType.DEDENT, '', 2, 7),
        lexer.Token(lexer.TokenType.ELIF, 'elif', 2, 8),
        lexer.Token(lexer.TokenType.FALSE, 'false', 2, 13),
        lexer.Token(lexer.TokenType.COLON, ':', 2, 18),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 19),
        lexer.Token(lexer.TokenType.INDENT, '', 3, 1),
        lexer.Token(lexer.TokenType.STRING, '\'maybe\'', 3, 5),
        lexer.Token(lexer.TokenType.DEDENT, '', 3, 9),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 3, 10),
        lexer.Token(lexer.TokenType.ELSE, 'else', 4, 1),
        lexer.Token(lexer.TokenType.COLON, ':', 4, 5),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 4, 6),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 5, 1),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 6, 1),
        lexer.Token(lexer.TokenType.INDENT, '', 7, 1),
        lexer.Token(lexer.TokenType.STRING, '\'maybe\'', 7, 5),
        lexer.Token(lexer.TokenType.DEDENT, '', 7, 10),
        lexer.Token(lexer.TokenType.EOF, '', 7, 11),
    ]
    p = parser.Parser(tokens)

    expected = parser.If(
        condition=parser.BoolLiteral(value=True),
        then_body=[parser.ExprStmt(expr=parser.StringLiteral(value='hi'))],
        else_body=[
            parser.If(
                condition=parser.BoolLiteral(value=False),
                then_body=[parser.ExprStmt(expr=parser.StringLiteral(value='maybe'))],
            ),
            parser.ExprStmt(expr=parser.StringLiteral(value='bye')),
        ],
    )


def test_parse_var_decl_none_empty():
    tokens = [
        lexer.Token(lexer.TokenType.EOF, '', 1, 1),
    ]

    p = parser.Parser(tokens)

    with pytest.raises(
        parser.ParseError,
        match=re.escape(
            "Expected a type ('int', 'int8', 'uint8', 'int64', 'bool', 'str', a struct name, '[size]type', or '[]type'), got TokenType.EOF ('') at line 1, column 1"
        )):
        p.parse_var_decl()


def test_parse_var_decl_none_no_name():
    tokens = [
        lexer.Token(lexer.TokenType.INT, 'int', 1, 1),
        lexer.Token(lexer.TokenType.EOF, '', 1, 4),
    ]

    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape("Expected a variable name at line 1, column 4")):
        p.parse_var_decl()


def test_parse_var_decl_none_no_assign():
    tokens = [
        lexer.Token(lexer.TokenType.INT, 'int', 1, 1),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'x', 1, 4),
        lexer.Token(lexer.TokenType.EOF, '', 1, 5),
    ]

    p = parser.Parser(tokens)

    expected = parser.VarDecl(name='x', var_type='int')

    assert expected == p.parse_var_decl()


def test_parse_var_decl_none_no_value():
    tokens = [
        lexer.Token(lexer.TokenType.INT, 'int', 1, 1),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'x', 1, 4),
        lexer.Token(lexer.TokenType.ASSIGN, '=', 1, 6),
        lexer.Token(lexer.TokenType.EOF, '', 1, 7),
    ]

    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape("Expected an expression, got TokenType.EOF (\'\') at line 1, column 7")):
        p.parse_var_decl()


def test_parse_var_decl_none():
    tokens = [
        lexer.Token(lexer.TokenType.INT, 'int', 1, 1),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'x', 1, 4),
        lexer.Token(lexer.TokenType.ASSIGN, '=', 1, 6),
        lexer.Token(lexer.TokenType.NUMBER, '1', 1, 8),
        lexer.Token(lexer.TokenType.EOF, '', 1, 9),
    ]

    p = parser.Parser(tokens)

    expected = parser.VarDecl('x', var_type='int', init=parser.Constant(value=1))

    assert expected == p.parse_var_decl()


def test_parse_var_decl_pass_type_with_type():
    tokens = [
        lexer.Token(lexer.TokenType.INT, 'int', 1, 1),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'x', 1, 4),
        lexer.Token(lexer.TokenType.ASSIGN, '=', 1, 6),
        lexer.Token(lexer.TokenType.NUMBER, '1', 1, 8),
        lexer.Token(lexer.TokenType.EOF, '', 1, 9),
    ]

    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape("Expected a variable name at line 1, column 1")):
        p.parse_var_decl('str')


def test_parse_var_decl_pass_type():
    tokens = [
        lexer.Token(lexer.TokenType.IDENTIFIER, 'x', 1, 1),
        lexer.Token(lexer.TokenType.ASSIGN, '=', 1, 3),
        lexer.Token(lexer.TokenType.NUMBER, '1', 1, 5),
        lexer.Token(lexer.TokenType.EOF, '', 1, 6),
    ]

    p = parser.Parser(tokens)

    expected = parser.VarDecl('x', var_type='int', init=parser.Constant(value=1))

    assert expected == p.parse_var_decl('int')


def test_parse_var_decl_pass_wrong_type():
    tokens = [
        lexer.Token(lexer.TokenType.IDENTIFIER, 'x', 1, 1),
        lexer.Token(lexer.TokenType.ASSIGN, '=', 1, 3),
        lexer.Token(lexer.TokenType.NUMBER, '1', 1, 5),
        lexer.Token(lexer.TokenType.EOF, '', 1, 6),
    ]

    p = parser.Parser(tokens)

    expected = parser.VarDecl('x', var_type='str', init=parser.Constant(value=1))

    assert expected == p.parse_var_decl('str')


def test_parse_assign_empty():
    tokens = [
        lexer.Token(lexer.TokenType.EOF, '', 1, 1),
    ]

    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape("Expected TokenType.IDENTIFIER, got TokenType.EOF (\'\') at line 1, column 1")):
        p.parse_assign()


def test_parse_assign_no_assign():
    tokens = [
        lexer.Token(lexer.TokenType.IDENTIFIER, 'a', 1, 1),
        lexer.Token(lexer.TokenType.EOF, '', 1, 2),
    ]

    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape("Expected an expression, got TokenType.EOF (\'\') at line 1, column 2")):
        p.parse_assign()


def test_parse_assign_no_value():
    tokens = [
        lexer.Token(lexer.TokenType.IDENTIFIER, 'a', 1, 1),
        lexer.Token(lexer.TokenType.ASSIGN, '=', 1, 2),
        lexer.Token(lexer.TokenType.EOF, '', 1, 3),
    ]

    p = parser.Parser(tokens)

    with pytest.raises(parser.ParseError, match=re.escape("Expected an expression, got TokenType.EOF (\'\') at line 1, column 3")):
        p.parse_assign()


def test_parse_assign():
    tokens = [
        lexer.Token(lexer.TokenType.IDENTIFIER, 'a', 1, 1),
        lexer.Token(lexer.TokenType.ASSIGN, '=', 1, 2),
        lexer.Token(lexer.TokenType.NUMBER, '1', 1, 3),
        lexer.Token(lexer.TokenType.EOF, '', 1, 4),
    ]

    p = parser.Parser(tokens)

    expected = parser.Assign(name='a', value=parser.Constant(value=1))

    assert expected == p.parse_assign()


def test_parse_assign_compound_addition():
    tokens = [
        lexer.Token(lexer.TokenType.IDENTIFIER, 'a', 1, 1),
        lexer.Token(lexer.TokenType.PLUS_ASSIGN, '+=', 1, 2),
        lexer.Token(lexer.TokenType.NUMBER, '1', 1, 4),
        lexer.Token(lexer.TokenType.EOF, '', 1, 5),
    ]

    p = parser.Parser(tokens)

    expected = parser.Assign(name='a', value=parser.Binary(op=parser.BinaryOp.ADD, left=parser.Variable(name='a'), right=parser.Constant(value=1)))

    assert expected == p.parse_assign()


def test_parse_assign_compound_subtraction():
    tokens = [
        lexer.Token(lexer.TokenType.IDENTIFIER, 'a', 1, 1),
        lexer.Token(lexer.TokenType.MINUS_ASSIGN, '-=', 1, 2),
        lexer.Token(lexer.TokenType.NUMBER, '1', 1, 4),
        lexer.Token(lexer.TokenType.EOF, '', 1, 5),
    ]

    p = parser.Parser(tokens)

    expected = parser.Assign(name='a', value=parser.Binary(op=parser.BinaryOp.SUBTRACT, left=parser.Variable(name='a'), right=parser.Constant(value=1)))

    assert expected == p.parse_assign()


def test_parse_assign_compound_multiplication():
    tokens = [
        lexer.Token(lexer.TokenType.IDENTIFIER, 'a', 1, 1),
        lexer.Token(lexer.TokenType.STAR_ASSIGN, '*=', 1, 2),
        lexer.Token(lexer.TokenType.NUMBER, '1', 1, 4),
        lexer.Token(lexer.TokenType.EOF, '', 1, 5),
    ]

    p = parser.Parser(tokens)

    expected = parser.Assign(name='a', value=parser.Binary(op=parser.BinaryOp.MULTIPLY, left=parser.Variable(name='a'), right=parser.Constant(value=1)))

    assert expected == p.parse_assign()


def test_parse_assign_compound_division():
    tokens = [
        lexer.Token(lexer.TokenType.IDENTIFIER, 'a', 1, 1),
        lexer.Token(lexer.TokenType.SLASH_ASSIGN, '/=', 1, 2),
        lexer.Token(lexer.TokenType.NUMBER, '1', 1, 4),
        lexer.Token(lexer.TokenType.EOF, '', 1, 5),
    ]

    p = parser.Parser(tokens)

    expected = parser.Assign(name='a', value=parser.Binary(op=parser.BinaryOp.DIVIDE, left=parser.Variable(name='a'), right=parser.Constant(value=1)))

    assert expected == p.parse_assign()


def test_parse_assign_compound_modulo():
    tokens = [
        lexer.Token(lexer.TokenType.IDENTIFIER, 'a', 1, 1),
        lexer.Token(lexer.TokenType.PERCENT_ASSIGN, '%=', 1, 2),
        lexer.Token(lexer.TokenType.NUMBER, '1', 1, 4),
        lexer.Token(lexer.TokenType.EOF, '', 1, 5),
    ]

    p = parser.Parser(tokens)

    expected = parser.Assign(name='a', value=parser.Binary(op=parser.BinaryOp.MODULO, left=parser.Variable(name='a'), right=parser.Constant(value=1)))

    assert expected == p.parse_assign()


def test_parse_assign_compound_bitwise_and():
    tokens = [
        lexer.Token(lexer.TokenType.IDENTIFIER, 'a', 1, 1),
        lexer.Token(lexer.TokenType.AMPERSAND_ASSIGN, '&=', 1, 2),
        lexer.Token(lexer.TokenType.NUMBER, '1', 1, 4),
        lexer.Token(lexer.TokenType.EOF, '', 1, 5),
    ]

    p = parser.Parser(tokens)

    expected = parser.Assign(name='a', value=parser.Binary(op=parser.BinaryOp.BITWISE_AND, left=parser.Variable(name='a'), right=parser.Constant(value=1)))

    assert expected == p.parse_assign()


def test_parse_assign_compound_bitwise_or():
    tokens = [
        lexer.Token(lexer.TokenType.IDENTIFIER, 'a', 1, 1),
        lexer.Token(lexer.TokenType.PIPE_ASSIGN, '|=', 1, 2),
        lexer.Token(lexer.TokenType.NUMBER, '1', 1, 4),
        lexer.Token(lexer.TokenType.EOF, '', 1, 5),
    ]

    p = parser.Parser(tokens)

    expected = parser.Assign(name='a', value=parser.Binary(op=parser.BinaryOp.BITWISE_OR, left=parser.Variable(name='a'), right=parser.Constant(value=1)))

    assert expected == p.parse_assign()


def test_parse_assign_compound_bitwise_xor():
    tokens = [
        lexer.Token(lexer.TokenType.IDENTIFIER, 'a', 1, 1),
        lexer.Token(lexer.TokenType.CARET_ASSIGN, '^=', 1, 2),
        lexer.Token(lexer.TokenType.NUMBER, '1', 1, 4),
        lexer.Token(lexer.TokenType.EOF, '', 1, 5),
    ]

    p = parser.Parser(tokens)

    expected = parser.Assign(name='a', value=parser.Binary(op=parser.BinaryOp.BITWISE_XOR, left=parser.Variable(name='a'), right=parser.Constant(value=1)))

    assert expected == p.parse_assign()


def test_parse_assign_compound_shift_left():
    tokens = [
        lexer.Token(lexer.TokenType.IDENTIFIER, 'a', 1, 1),
        lexer.Token(lexer.TokenType.SHIFT_LEFT_ASSIGN, '<<=', 1, 2),
        lexer.Token(lexer.TokenType.NUMBER, '1', 1, 5),
        lexer.Token(lexer.TokenType.EOF, '', 1, 6),
    ]

    p = parser.Parser(tokens)

    expected = parser.Assign(name='a', value=parser.Binary(op=parser.BinaryOp.SHIFT_LEFT, left=parser.Variable(name='a'), right=parser.Constant(value=1)))

    assert expected == p.parse_assign()


def test_parse_assign_compound_shift_right():
    tokens = [
        lexer.Token(lexer.TokenType.IDENTIFIER, 'a', 1, 1),
        lexer.Token(lexer.TokenType.SHIFT_RIGHT_ASSIGN, '>>=', 1, 2),
        lexer.Token(lexer.TokenType.NUMBER, '1', 1, 5),
        lexer.Token(lexer.TokenType.EOF, '', 1, 6),
    ]

    p = parser.Parser(tokens)

    expected = parser.Assign(name='a', value=parser.Binary(op=parser.BinaryOp.SHIFT_RIGHT, left=parser.Variable(name='a'), right=parser.Constant(value=1)))

    assert expected == p.parse_assign()


# TODO(will): Test parse_expr_stmt_or_index_assign

# TODO(will): Test parse_expression

# TODO(will): Test parse_binary

# TODO(will): Test parse_unary

# TODO(will): Test parse_postfix

# TODO(will): Test parse_index_of_slice

# TODO(will): Test parse_primary

# TODO(will): Test parse_array_literal

# TODO(will): Test parse_call

# ---------------------------------------------------------------------------
# Source positions (Node.line/col) -- one test per position-derivation
# strategy parser.py uses, not per node type (many node types share the
# same strategy): a "start token" the method itself captured, or a
# position propagated from an already-parsed child. Real line/col
# values (not all 1s) are used throughout so a test can't pass by
# accident from every position defaulting to the same number.
# ---------------------------------------------------------------------------

def test_position_constant_is_its_own_token():
    tokens = [
        lexer.Token(lexer.TokenType.NUMBER, '7', 3, 10),
        lexer.Token(lexer.TokenType.EOF, '', 3, 11),
    ]
    result = parser.Parser(tokens).parse_primary()
    assert (result.line, result.col) == (3, 10)


def test_position_variable_is_its_own_token():
    tokens = [
        lexer.Token(lexer.TokenType.IDENTIFIER, 'x', 5, 2),
        lexer.Token(lexer.TokenType.EOF, '', 5, 3),
    ]
    result = parser.Parser(tokens).parse_primary()
    assert (result.line, result.col) == (5, 2)


def test_position_unary_is_its_operator_token():
    tokens = [
        lexer.Token(lexer.TokenType.MINUS, '-', 2, 8),
        lexer.Token(lexer.TokenType.NUMBER, '1', 2, 9),
        lexer.Token(lexer.TokenType.EOF, '', 2, 10),
    ]
    result = parser.Parser(tokens).parse_unary()
    assert (result.line, result.col) == (2, 8)


def test_position_binary_is_left_operands_not_operators():
    """`a + b`, with the left operand deliberately placed on an
    earlier line than '+' -- proves the Binary node takes its
    position from the left operand, not from wherever the operator
    or the whole expression happens to sit."""
    tokens = [
        lexer.Token(lexer.TokenType.IDENTIFIER, 'a', 1, 1),
        lexer.Token(lexer.TokenType.PLUS, '+', 2, 1),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'b', 2, 3),
        lexer.Token(lexer.TokenType.EOF, '', 2, 4),
    ]
    result = parser.Parser(tokens).parse_binary()
    assert (result.line, result.col) == (1, 1)


def test_position_binary_chain_keeps_the_original_leftmost_operand():
    """`a - b - c` -- left-associative, so this is (a - b) - c; both
    the inner and outer Binary should still report a's position, not
    the intermediate (a - b) result's own (nonexistent) token."""
    tokens = [
        lexer.Token(lexer.TokenType.IDENTIFIER, 'a', 4, 4),
        lexer.Token(lexer.TokenType.MINUS, '-', 4, 6),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'b', 4, 8),
        lexer.Token(lexer.TokenType.MINUS, '-', 4, 10),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'c', 4, 12),
        lexer.Token(lexer.TokenType.EOF, '', 4, 13),
    ]
    result = parser.Parser(tokens).parse_binary()
    assert (result.line, result.col) == (4, 4)


def test_position_postfix_chain_is_the_base_not_the_dot_or_bracket():
    """`a.b[0]` -- both the Field and the outer Index should point at
    `a`, not at '.' or '['."""
    tokens = [
        lexer.Token(lexer.TokenType.IDENTIFIER, 'a', 6, 1),
        lexer.Token(lexer.TokenType.DOT, '.', 6, 2),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'b', 6, 3),
        lexer.Token(lexer.TokenType.OPEN_BRACKET, '[', 6, 4),
        lexer.Token(lexer.TokenType.NUMBER, '0', 6, 5),
        lexer.Token(lexer.TokenType.CLOSE_BRACKET, ']', 6, 6),
        lexer.Token(lexer.TokenType.EOF, '', 6, 7),
    ]
    result = parser.Parser(tokens).parse_postfix()
    assert isinstance(result, parser.Index)
    assert (result.line, result.col) == (6, 1)
    assert (result.array.line, result.array.col) == (6, 1)  # the Field


def test_position_array_literal_is_its_open_bracket():
    tokens = [
        lexer.Token(lexer.TokenType.OPEN_BRACKET, '[', 7, 9),
        lexer.Token(lexer.TokenType.NUMBER, '1', 7, 10),
        lexer.Token(lexer.TokenType.CLOSE_BRACKET, ']', 7, 11),
        lexer.Token(lexer.TokenType.EOF, '', 7, 12),
    ]
    result = parser.Parser(tokens).parse_array_literal()
    assert (result.line, result.col) == (7, 9)


def test_position_call_is_its_name_token():
    tokens = [
        lexer.Token(lexer.TokenType.IDENTIFIER, 'foo', 8, 3),
        lexer.Token(lexer.TokenType.OPEN_PAREN, '(', 8, 6),
        lexer.Token(lexer.TokenType.CLOSE_PAREN, ')', 8, 7),
        lexer.Token(lexer.TokenType.EOF, '', 8, 8),
    ]
    result = parser.Parser(tokens).parse_call()
    assert (result.line, result.col) == (8, 3)


def test_position_var_decl_is_its_type_token():
    tokens = [
        lexer.Token(lexer.TokenType.INT, 'int', 9, 1),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'x', 9, 5),
        lexer.Token(lexer.TokenType.EOF, '', 9, 6),
    ]
    result = parser.Parser(tokens).parse_var_decl()
    assert (result.line, result.col) == (9, 1)


def test_position_var_decl_via_parse_statement_struct_typed_branch():
    """The OTHER path into parse_var_decl -- parse_statement's own
    two-identifier lookahead already parsed the type before calling
    parse_var_decl, so this exercises the start_tok handoff between
    them, not parse_var_decl's own self.current() fallback."""
    tokens = [
        lexer.Token(lexer.TokenType.IDENTIFIER, 'Point', 10, 1),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'p', 10, 7),
        lexer.Token(lexer.TokenType.EOF, '', 10, 8),
    ]
    result = parser.Parser(tokens).parse_statement()
    assert isinstance(result, parser.VarDecl)
    assert (result.line, result.col) == (10, 1)


def test_position_assign_is_its_name_token():
    tokens = [
        lexer.Token(lexer.TokenType.IDENTIFIER, 'x', 11, 5),
        lexer.Token(lexer.TokenType.ASSIGN, '=', 11, 7),
        lexer.Token(lexer.TokenType.NUMBER, '1', 11, 9),
        lexer.Token(lexer.TokenType.EOF, '', 11, 10),
    ]
    result = parser.Parser(tokens).parse_assign()
    assert (result.line, result.col) == (11, 5)


def test_position_compound_assign_desugared_binary_matches_name_token():
    """`x += 1` desugars to Assign(Binary(...)) -- both the outer
    Assign and the synthesized Binary/Variable should take x's own
    position, not the operator's."""
    tokens = [
        lexer.Token(lexer.TokenType.IDENTIFIER, 'x', 12, 5),
        lexer.Token(lexer.TokenType.PLUS_ASSIGN, '+=', 12, 7),
        lexer.Token(lexer.TokenType.NUMBER, '1', 12, 10),
        lexer.Token(lexer.TokenType.EOF, '', 12, 11),
    ]
    result = parser.Parser(tokens).parse_assign()
    assert (result.line, result.col) == (12, 5)
    assert (result.value.line, result.value.col) == (12, 5)  # the desugared Binary


def test_position_return_is_its_keyword_token():
    tokens = [
        lexer.Token(lexer.TokenType.RETURN, 'return', 13, 5),
        lexer.Token(lexer.TokenType.NUMBER, '1', 13, 12),
        lexer.Token(lexer.TokenType.EOF, '', 13, 13),
    ]
    result = parser.Parser(tokens).parse_return()
    assert (result.line, result.col) == (13, 5)


def test_position_if_is_its_keyword_token_not_the_condition():
    tokens = [
        lexer.Token(lexer.TokenType.IF, 'if', 14, 1),
        lexer.Token(lexer.TokenType.TRUE, 'true', 14, 4),
        lexer.Token(lexer.TokenType.COLON, ':', 14, 8),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 14, 9),
        lexer.Token(lexer.TokenType.INDENT, '', 15, 1),
        lexer.Token(lexer.TokenType.RETURN, 'return', 15, 5),
        lexer.Token(lexer.TokenType.NUMBER, '1', 15, 12),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 15, 13),
        lexer.Token(lexer.TokenType.DEDENT, '', 16, 1),
        lexer.Token(lexer.TokenType.EOF, '', 16, 1),
    ]
    result = parser.Parser(tokens).parse_if()
    assert (result.line, result.col) == (14, 1)


def test_position_while_is_its_keyword_token():
    tokens = [
        lexer.Token(lexer.TokenType.WHILE, 'while', 17, 1),
        lexer.Token(lexer.TokenType.TRUE, 'true', 17, 7),
        lexer.Token(lexer.TokenType.COLON, ':', 17, 11),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 17, 12),
        lexer.Token(lexer.TokenType.INDENT, '', 18, 1),
        lexer.Token(lexer.TokenType.BREAK, 'break', 18, 5),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 18, 10),
        lexer.Token(lexer.TokenType.DEDENT, '', 19, 1),
        lexer.Token(lexer.TokenType.EOF, '', 19, 1),
    ]
    result = parser.Parser(tokens).parse_while()
    assert (result.line, result.col) == (17, 1)


def test_position_break_and_continue_are_their_own_token():
    break_tokens = [
        lexer.Token(lexer.TokenType.BREAK, 'break', 20, 5),
        lexer.Token(lexer.TokenType.EOF, '', 20, 10),
    ]
    assert (parser.Parser(break_tokens).parse_break().line,
            parser.Parser(break_tokens).parse_break().col) == (20, 5)

    continue_tokens = [
        lexer.Token(lexer.TokenType.CONTINUE, 'continue', 21, 5),
        lexer.Token(lexer.TokenType.EOF, '', 21, 13),
    ]
    assert (parser.Parser(continue_tokens).parse_continue().line,
            parser.Parser(continue_tokens).parse_continue().col) == (21, 5)


def test_position_function_is_its_def_token():
    tokens = TEST_TOKENS  # `def int main(): ...` -- def is at (1, 1)
    result = parser.Parser(tokens).parse_function()
    assert (result.line, result.col) == (1, 1)


def test_position_param_is_its_type_token():
    tokens = [
        lexer.Token(lexer.TokenType.INT, 'int', 22, 10),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'a', 22, 14),
        lexer.Token(lexer.TokenType.EOF, '', 22, 15),
    ]
    result = parser.Parser(tokens).parse_param()
    assert (result.line, result.col) == (22, 10)


def test_position_new_node_defaults_to_zero_when_built_by_hand():
    """A node built directly (as every OTHER test in this file does,
    and as a hand-written AST fixture elsewhere would) has no real
    token behind it -- line/col default to 0 rather than some
    misleading guess."""
    node = parser.Constant(value=1)
    assert (node.line, node.col) == (0, 0)


# ---------------------------------------------------------------------------
# type Name struct: ... -- the only spelling for a struct declaration
# now that the hard cutover has happened (see parser.py's own
# parse_program: a bare `struct Name:` at the top level is rejected
# outright, with a specific error pointing at the new spelling).
# ---------------------------------------------------------------------------

def test_parse_type_declaration_struct_form_produces_a_struct_def():
    tokens = [
        lexer.Token(lexer.TokenType.TYPE, 'type', 1, 1),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'Point', 1, 6),
        lexer.Token(lexer.TokenType.STRUCT, 'struct', 1, 12),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 18),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 19),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.INT, 'int', 2, 5),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'x', 2, 9),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 10),
        lexer.Token(lexer.TokenType.DEDENT, '', 3, 1),
        lexer.Token(lexer.TokenType.EOF, '', 3, 1),
    ]
    result = parser.Parser(tokens).parse_type_declaration()
    assert result == parser.StructDef(name='Point', fields=[parser.StructField(name='x', field_type='int')])


def test_bare_struct_keyword_at_top_level_is_rejected():
    """The hard cutover itself: `struct Name: ...` (no leading `type`)
    used to parse (see git history around parse_struct_def, now
    removed) -- now parse_program rejects it outright with a specific
    error pointing at the replacement spelling, rather than falling
    through to some more confusing "expected def" message from
    parse_function."""
    tokens = [
        lexer.Token(lexer.TokenType.STRUCT, 'struct', 1, 1),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'Point', 1, 8),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 13),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 14),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.INT, 'int', 2, 5),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'x', 2, 9),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 10),
        lexer.Token(lexer.TokenType.DEDENT, '', 3, 1),
        lexer.Token(lexer.TokenType.EOF, '', 3, 1),
    ]
    with pytest.raises(
        parser.ParseError,
        match=re.escape(
            "Bare 'struct Name:' is no longer supported -- write "
            "'type Name struct:' instead at line 1, column 1"
        )
    ):
        parser.Parser(tokens).parse_program()


def test_parse_type_declaration_alias_form_still_works():
    """parse_type_declaration replaces parse_type_alias outright (not
    just adding to it) -- the plain alias form must still work
    unchanged."""
    tokens = [
        lexer.Token(lexer.TokenType.TYPE, 'type', 1, 1),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'MyInt', 1, 6),
        lexer.Token(lexer.TokenType.ASSIGN, '=', 1, 12),
        lexer.Token(lexer.TokenType.INT, 'int', 1, 14),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 17),
        lexer.Token(lexer.TokenType.EOF, '', 2, 1),
    ]
    result = parser.Parser(tokens).parse_type_declaration()
    assert result == parser.TypeAlias(name='MyInt', target_type='int')


def test_parse_type_declaration_none_of_assign_struct_is_raises():
    tokens = [
        lexer.Token(lexer.TokenType.TYPE, 'type', 1, 1),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'Foo', 1, 6),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'bar', 1, 10),
        lexer.Token(lexer.TokenType.EOF, '', 1, 13),
    ]
    with pytest.raises(
        parser.ParseError,
        match=re.escape(
            "Expected '=' (for a type alias), 'struct' (for a struct declaration), "
            "or 'is' (for a sum type) at line 1, column 10"
        )
    ):
        parser.Parser(tokens).parse_type_declaration()


def test_parse_program_sorts_type_struct_form_into_structs_not_aliases():
    """The dispatch in parse_program: a `type Name struct: ...`
    declaration must land in Program.structs -- never in Program.
    type_aliases, even though it's parsed by the same method (parse_
    type_declaration) an actual alias is."""
    tokens = [
        lexer.Token(lexer.TokenType.TYPE, 'type', 1, 1),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'Point', 1, 6),
        lexer.Token(lexer.TokenType.STRUCT, 'struct', 1, 12),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 18),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 19),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.INT, 'int', 2, 5),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'x', 2, 9),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 10),
        lexer.Token(lexer.TokenType.DEDENT, '', 3, 1),
        lexer.Token(lexer.TokenType.EOF, '', 3, 1),
    ]
    program = parser.Parser(tokens).parse_program()
    assert len(program.structs) == 1
    assert program.structs[0].name == 'Point'
    assert program.type_aliases == []


# ---------------------------------------------------------------------------
# type Name is Variant | Variant (| Variant)* -- a sum type declaration
# (SumTypeDef, via _parse_sum_type_body). Semantic questions (does each
# variant name actually resolve to a declared struct, are there
# duplicates) aren't this file's concern -- these only check the
# GRAMMAR: at least two variants required, arbitrary variant counts
# beyond that, and parse_program sorting the result into Program.
# sum_types specifically.
# ---------------------------------------------------------------------------

def test_parse_type_declaration_sum_type_two_variants():
    tokens = [
        lexer.Token(lexer.TokenType.TYPE, 'type', 1, 1),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'Shape', 1, 6),
        lexer.Token(lexer.TokenType.IS, 'is', 1, 12),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'Circle', 1, 15),
        lexer.Token(lexer.TokenType.PIPE, '|', 1, 22),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'Square', 1, 24),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 30),
        lexer.Token(lexer.TokenType.EOF, '', 2, 1),
    ]
    result = parser.Parser(tokens).parse_type_declaration()
    assert result == parser.SumTypeDef(name='Shape', variants=['Circle', 'Square'])
    assert isinstance(result, parser.SumTypeDef)


def test_parse_type_declaration_sum_type_more_than_two_variants():
    """No special-casing beyond two -- an arbitrary number of '|'-
    separated variants all accumulate the same way."""
    tokens = [
        lexer.Token(lexer.TokenType.TYPE, 'type', 1, 1),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'Shape', 1, 6),
        lexer.Token(lexer.TokenType.IS, 'is', 1, 12),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'Circle', 1, 15),
        lexer.Token(lexer.TokenType.PIPE, '|', 1, 22),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'Square', 1, 24),
        lexer.Token(lexer.TokenType.PIPE, '|', 1, 31),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'Triangle', 1, 33),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 41),
        lexer.Token(lexer.TokenType.EOF, '', 2, 1),
    ]
    result = parser.Parser(tokens).parse_type_declaration()
    assert result.variants == ['Circle', 'Square', 'Triangle']


def test_parse_type_declaration_sum_type_single_variant_raises():
    """The parser's own "at least two variants" check -- mirroring
    _parse_struct_body's "at least one field" -- fires on a single
    bare name with no '|' at all, before semantic.py ever sees it."""
    tokens = [
        lexer.Token(lexer.TokenType.TYPE, 'type', 1, 1),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'Shape', 1, 6),
        lexer.Token(lexer.TokenType.IS, 'is', 1, 12),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'Circle', 1, 15),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 21),
        lexer.Token(lexer.TokenType.EOF, '', 2, 1),
    ]
    with pytest.raises(
        parser.ParseError,
        match=re.escape(
            "Expected at least one '|' and a second variant in sum type "
            "'Shape' -- a sum type needs at least two variants "
            "at line 1, column 15"
        )
    ):
        parser.Parser(tokens).parse_type_declaration()


def test_parse_program_sorts_sum_type_into_sum_types():
    """The dispatch in parse_program: a `type Name is ...` declaration
    lands in Program.sum_types specifically -- never structs or
    type_aliases, even though all three share one entry point
    (parse_type_declaration)."""
    tokens = [
        lexer.Token(lexer.TokenType.TYPE, 'type', 1, 1),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'Shape', 1, 6),
        lexer.Token(lexer.TokenType.IS, 'is', 1, 12),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'Circle', 1, 15),
        lexer.Token(lexer.TokenType.PIPE, '|', 1, 22),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'Square', 1, 24),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 30),
        lexer.Token(lexer.TokenType.EOF, '', 2, 1),
    ]
    program = parser.Parser(tokens).parse_program()
    assert len(program.sum_types) == 1
    assert program.sum_types[0].name == 'Shape'
    assert program.structs == []
    assert program.type_aliases == []


# ---------------------------------------------------------------------------
# `if NAME is TypeName:` (IsCheck) -- sum type narrowing's own grammar,
# stage 1: recognized as an if/elif condition's own special shape,
# never a production inside the general expression grammar. These
# only check the GRAMMAR (the shape parses into an IsCheck with the
# right variable_name/type_name, ordinary conditions are unaffected,
# and while doesn't recognize this shape at all) -- whether either
# name refers to anything real, and whether NAME's own type actually
# narrows anywhere, are semantic.py's job, not tested here.
# ---------------------------------------------------------------------------

def test_if_condition_recognizes_is_check():
    tokens = [
        lexer.Token(lexer.TokenType.IF, 'if', 1, 1),
        lexer.Token(lexer.TokenType.IDENTIFIER, 's', 1, 4),
        lexer.Token(lexer.TokenType.IS, 'is', 1, 6),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'Circle', 1, 9),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 15),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 16),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.RETURN, 'return', 2, 5),
        lexer.Token(lexer.TokenType.NUMBER, '0', 2, 12),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 13),
        lexer.Token(lexer.TokenType.DEDENT, '', 3, 1),
        lexer.Token(lexer.TokenType.EOF, '', 3, 1),
    ]
    result = parser.Parser(tokens).parse_if()
    assert isinstance(result.condition, parser.IsCheck)
    assert result.condition.variable_name == 's'
    assert result.condition.type_name == 'Circle'


def test_elif_condition_also_recognizes_is_check():
    """_parse_if_body is shared by parse_if/parse_elif_as_if -- an
    elif's own condition gets the identical is-check recognition an
    if's does, following naturally from elif being nothing more than
    a nested If (see If's own docstring)."""
    tokens = [
        lexer.Token(lexer.TokenType.IF, 'if', 1, 1),
        lexer.Token(lexer.TokenType.TRUE, 'true', 1, 4),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 8),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 9),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.RETURN, 'return', 2, 5),
        lexer.Token(lexer.TokenType.NUMBER, '1', 2, 12),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 13),
        lexer.Token(lexer.TokenType.DEDENT, '', 3, 1),
        lexer.Token(lexer.TokenType.ELIF, 'elif', 3, 1),
        lexer.Token(lexer.TokenType.IDENTIFIER, 's', 3, 6),
        lexer.Token(lexer.TokenType.IS, 'is', 3, 8),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'Square', 3, 11),
        lexer.Token(lexer.TokenType.COLON, ':', 3, 17),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 3, 18),
        lexer.Token(lexer.TokenType.INDENT, '', 4, 1),
        lexer.Token(lexer.TokenType.RETURN, 'return', 4, 5),
        lexer.Token(lexer.TokenType.NUMBER, '2', 4, 12),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 4, 13),
        lexer.Token(lexer.TokenType.DEDENT, '', 5, 1),
        lexer.Token(lexer.TokenType.EOF, '', 5, 1),
    ]
    result = parser.Parser(tokens).parse_if()
    elif_node = result.else_body[0]
    assert isinstance(elif_node, parser.If)
    assert isinstance(elif_node.condition, parser.IsCheck)
    assert elif_node.condition.variable_name == 's'
    assert elif_node.condition.type_name == 'Square'


def test_ordinary_if_condition_is_unaffected():
    """A bare identifier NOT followed by 'is' -- the two-token
    lookahead must not misfire and must fall through to an ordinary
    parse_expression() unchanged."""
    tokens = [
        lexer.Token(lexer.TokenType.IF, 'if', 1, 1),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'flag', 1, 4),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 8),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 9),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.RETURN, 'return', 2, 5),
        lexer.Token(lexer.TokenType.NUMBER, '0', 2, 12),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 13),
        lexer.Token(lexer.TokenType.DEDENT, '', 3, 1),
        lexer.Token(lexer.TokenType.EOF, '', 3, 1),
    ]
    result = parser.Parser(tokens).parse_if()
    assert isinstance(result.condition, parser.Variable)
    assert result.condition.name == 'flag'


def test_is_check_requires_a_type_name_after_is():
    tokens = [
        lexer.Token(lexer.TokenType.IF, 'if', 1, 1),
        lexer.Token(lexer.TokenType.IDENTIFIER, 's', 1, 4),
        lexer.Token(lexer.TokenType.IS, 'is', 1, 6),
        lexer.Token(lexer.TokenType.NUMBER, '5', 1, 9),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 10),
        lexer.Token(lexer.TokenType.EOF, '', 1, 11),
    ]
    with pytest.raises(parser.ParseError, match="Expected a type name after 'is'"):
        parser.Parser(tokens).parse_if()


def test_is_check_with_index_subject_requires_as_binding():
    """`shapes[0] is Circle as c` -- subject is the Index node itself,
    variable_name is the binding, not shapes/0 in any form."""
    tokens = [
        lexer.Token(lexer.TokenType.IF, 'if', 1, 1),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'shapes', 1, 4),
        lexer.Token(lexer.TokenType.OPEN_BRACKET, '[', 1, 10),
        lexer.Token(lexer.TokenType.NUMBER, '0', 1, 11),
        lexer.Token(lexer.TokenType.CLOSE_BRACKET, ']', 1, 12),
        lexer.Token(lexer.TokenType.IS, 'is', 1, 14),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'Circle', 1, 17),
        lexer.Token(lexer.TokenType.AS, 'as', 1, 24),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'c', 1, 27),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 28),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 29),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.RETURN, 'return', 2, 5),
        lexer.Token(lexer.TokenType.NUMBER, '0', 2, 12),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 13),
        lexer.Token(lexer.TokenType.DEDENT, '', 3, 1),
        lexer.Token(lexer.TokenType.EOF, '', 3, 1),
    ]
    result = parser.Parser(tokens).parse_if()
    condition = result.condition
    assert isinstance(condition, parser.IsCheck)
    assert condition.variable_name == 'c'
    assert condition.type_name == 'Circle'
    assert isinstance(condition.subject, parser.Index)
    assert condition.subject.array.name == 'shapes'
    assert condition.subject.index.value == 0


def test_is_check_with_non_bare_subject_requires_as():
    """`shapes[0] is Circle` with no trailing `as` -- rejected with a
    message naming the actual gap, not a generic parse failure."""
    tokens = [
        lexer.Token(lexer.TokenType.IF, 'if', 1, 1),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'shapes', 1, 4),
        lexer.Token(lexer.TokenType.OPEN_BRACKET, '[', 1, 10),
        lexer.Token(lexer.TokenType.NUMBER, '0', 1, 11),
        lexer.Token(lexer.TokenType.CLOSE_BRACKET, ']', 1, 12),
        lexer.Token(lexer.TokenType.IS, 'is', 1, 14),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'Circle', 1, 17),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 23),
        lexer.Token(lexer.TokenType.EOF, '', 1, 24),
    ]
    with pytest.raises(parser.ParseError, match="Expected 'as NAME' after the type name"):
        parser.Parser(tokens).parse_if()


def test_is_check_bare_variable_can_still_take_an_optional_as_rename():
    """`x is Circle as y` -- a bare-variable subject doesn't NEED `as`
    (see test_if_condition_recognizes_is_check, subject=None there),
    but can still take one, opting into a renamed, freshly-bound copy
    rather than the zero-cost same-name narrowing `is` without `as`
    gets. Caught, by hand, as a real gap before this branch existed:
    without it, this shape fell through the bare-variable fast path
    having consumed only `x is Circle`, leaving `as y` to break the
    caller's own subsequent `:` expectation with a confusing error."""
    tokens = [
        lexer.Token(lexer.TokenType.IF, 'if', 1, 1),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'x', 1, 4),
        lexer.Token(lexer.TokenType.IS, 'is', 1, 6),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'Circle', 1, 9),
        lexer.Token(lexer.TokenType.AS, 'as', 1, 16),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'y', 1, 19),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 20),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 21),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.RETURN, 'return', 2, 5),
        lexer.Token(lexer.TokenType.NUMBER, '0', 2, 12),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 13),
        lexer.Token(lexer.TokenType.DEDENT, '', 3, 1),
        lexer.Token(lexer.TokenType.EOF, '', 3, 1),
    ]
    result = parser.Parser(tokens).parse_if()
    condition = result.condition
    assert condition.variable_name == 'y'
    assert condition.type_name == 'Circle'
    assert isinstance(condition.subject, parser.Variable)
    assert condition.subject.name == 'x'


def test_match_with_call_subject_binds_only_the_first_arm():
    """`match makeShape() as s:` -- subject is set on the OUTERMOST
    If's own condition only (the first arm in source order -- see
    parse_match's own docstring for why setting it on every arm would
    re-evaluate the subject once per arm tried). The second arm,
    nested in the first's own else_body, gets subject=None -- it
    relies on 's' already being in scope by the time it runs, not on
    building its own, second binding."""
    tokens = [
        lexer.Token(lexer.TokenType.MATCH, 'match', 1, 1),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'makeShape', 1, 7),
        lexer.Token(lexer.TokenType.OPEN_PAREN, '(', 1, 16),
        lexer.Token(lexer.TokenType.CLOSE_PAREN, ')', 1, 17),
        lexer.Token(lexer.TokenType.AS, 'as', 1, 19),
        lexer.Token(lexer.TokenType.IDENTIFIER, 's', 1, 22),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 23),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 24),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.IS, 'is', 2, 5),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'Circle', 2, 8),
        lexer.Token(lexer.TokenType.COLON, ':', 2, 14),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 15),
        lexer.Token(lexer.TokenType.INDENT, '', 3, 1),
        lexer.Token(lexer.TokenType.RETURN, 'return', 3, 9),
        lexer.Token(lexer.TokenType.NUMBER, '1', 3, 16),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 3, 17),
        lexer.Token(lexer.TokenType.DEDENT, '', 4, 5),
        lexer.Token(lexer.TokenType.IS, 'is', 4, 5),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'Square', 4, 8),
        lexer.Token(lexer.TokenType.COLON, ':', 4, 14),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 4, 15),
        lexer.Token(lexer.TokenType.INDENT, '', 5, 9),
        lexer.Token(lexer.TokenType.RETURN, 'return', 5, 9),
        lexer.Token(lexer.TokenType.NUMBER, '2', 5, 16),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 5, 17),
        lexer.Token(lexer.TokenType.DEDENT, '', 6, 1),
        lexer.Token(lexer.TokenType.DEDENT, '', 6, 1),
        lexer.Token(lexer.TokenType.EOF, '', 6, 1),
    ]
    outermost = parser.Parser(tokens).parse_match()
    assert outermost.condition.variable_name == 's'
    assert outermost.condition.type_name == 'Circle'
    assert isinstance(outermost.condition.subject, parser.Call)
    assert outermost.condition.subject.name == 'makeShape'

    second_arm = outermost.else_body[0]
    assert second_arm.condition.variable_name == 's'
    assert second_arm.condition.type_name == 'Square'
    assert second_arm.condition.subject is None


def test_while_condition_does_not_recognize_is_check():
    """Deliberately restricted to if/elif for this first cut (see
    IsCheck's own docstring) -- a while condition falls through to
    parse_expression, which stops at the bare name and leaves 'is'
    unconsumed, so this fails at the next expect(COLON) instead of
    ever producing an IsCheck."""
    tokens = [
        lexer.Token(lexer.TokenType.WHILE, 'while', 1, 1),
        lexer.Token(lexer.TokenType.IDENTIFIER, 's', 1, 7),
        lexer.Token(lexer.TokenType.IS, 'is', 1, 9),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'Circle', 1, 12),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 18),
        lexer.Token(lexer.TokenType.EOF, '', 1, 19),
    ]
    with pytest.raises(parser.ParseError, match="Expected ':' to start the while body"):
        parser.Parser(tokens).parse_while()


# ---------------------------------------------------------------------------
# `match NAME: (is TypeName: <block>)+ [else: <block>]?` -- exhaustive
# matching's own grammar. Introduces no new AST node: parse_match
# desugars entirely into an ordinary nested-If chain, one IsCheck-
# conditioned If per arm, chained through else_body exactly like an
# elif chain already is (see If's own docstring) -- these only check
# that desugaring produces the right SHAPE (is_match set on the
# outermost If alone, each arm's own condition, correct else_body
# chaining) and the grammar's own error cases; whether the chain is
# actually exhaustive is semantic.py's job, not tested here.
# ---------------------------------------------------------------------------

def test_match_desugars_into_a_nested_if_chain():
    tokens = [
        lexer.Token(lexer.TokenType.MATCH, 'match', 1, 1),
        lexer.Token(lexer.TokenType.IDENTIFIER, 's', 1, 7),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 8),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 9),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.IS, 'is', 2, 5),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'Circle', 2, 8),
        lexer.Token(lexer.TokenType.COLON, ':', 2, 14),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 15),
        lexer.Token(lexer.TokenType.INDENT, '', 3, 1),
        lexer.Token(lexer.TokenType.RETURN, 'return', 3, 9),
        lexer.Token(lexer.TokenType.NUMBER, '1', 3, 16),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 3, 17),
        lexer.Token(lexer.TokenType.DEDENT, '', 4, 5),
        lexer.Token(lexer.TokenType.IS, 'is', 4, 5),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'Square', 4, 8),
        lexer.Token(lexer.TokenType.COLON, ':', 4, 14),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 4, 15),
        lexer.Token(lexer.TokenType.INDENT, '', 5, 1),
        lexer.Token(lexer.TokenType.RETURN, 'return', 5, 9),
        lexer.Token(lexer.TokenType.NUMBER, '2', 5, 16),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 5, 17),
        lexer.Token(lexer.TokenType.DEDENT, '', 6, 1),
        lexer.Token(lexer.TokenType.DEDENT, '', 6, 1),
        lexer.Token(lexer.TokenType.EOF, '', 6, 1),
    ]
    outer = parser.Parser(tokens).parse_match()
    assert outer.is_match is True
    assert outer.match_arm_count == 2
    assert isinstance(outer.condition, parser.IsCheck)
    assert outer.condition.variable_name == 's'
    assert outer.condition.type_name == 'Circle'

    assert len(outer.else_body) == 1
    inner = outer.else_body[0]
    assert isinstance(inner, parser.If)
    assert inner.is_match is False  # only the OUTERMOST node is marked
    assert inner.match_arm_count is None
    assert inner.condition.variable_name == 's'
    assert inner.condition.type_name == 'Square'
    assert inner.else_body is None  # no trailing else -- relies on exhaustiveness


def test_match_with_explicit_else():
    """The else_body on the LAST arm is the explicit block itself,
    not one more IsCheck-conditioned If -- else is not an arm."""
    tokens = [
        lexer.Token(lexer.TokenType.MATCH, 'match', 1, 1),
        lexer.Token(lexer.TokenType.IDENTIFIER, 's', 1, 7),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 8),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 9),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.IS, 'is', 2, 5),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'Circle', 2, 8),
        lexer.Token(lexer.TokenType.COLON, ':', 2, 14),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 15),
        lexer.Token(lexer.TokenType.INDENT, '', 3, 1),
        lexer.Token(lexer.TokenType.RETURN, 'return', 3, 9),
        lexer.Token(lexer.TokenType.NUMBER, '1', 3, 16),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 3, 17),
        lexer.Token(lexer.TokenType.DEDENT, '', 4, 5),
        lexer.Token(lexer.TokenType.ELSE, 'else', 4, 5),
        lexer.Token(lexer.TokenType.COLON, ':', 4, 9),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 4, 10),
        lexer.Token(lexer.TokenType.INDENT, '', 5, 1),
        lexer.Token(lexer.TokenType.RETURN, 'return', 5, 9),
        lexer.Token(lexer.TokenType.NUMBER, '0', 5, 16),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 5, 17),
        lexer.Token(lexer.TokenType.DEDENT, '', 6, 1),
        lexer.Token(lexer.TokenType.DEDENT, '', 6, 1),
        lexer.Token(lexer.TokenType.EOF, '', 6, 1),
    ]
    outer = parser.Parser(tokens).parse_match()
    assert outer.is_match is True
    assert len(outer.else_body) == 1
    return_stmt = outer.else_body[0]
    assert isinstance(return_stmt, parser.Return)


def test_match_with_no_arms_raises():
    tokens = [
        lexer.Token(lexer.TokenType.MATCH, 'match', 1, 1),
        lexer.Token(lexer.TokenType.IDENTIFIER, 's', 1, 7),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 8),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 9),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.ELSE, 'else', 2, 5),
        lexer.Token(lexer.TokenType.COLON, ':', 2, 9),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 10),
        lexer.Token(lexer.TokenType.INDENT, '', 3, 1),
        lexer.Token(lexer.TokenType.RETURN, 'return', 3, 9),
        lexer.Token(lexer.TokenType.NUMBER, '0', 3, 16),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 3, 17),
        lexer.Token(lexer.TokenType.DEDENT, '', 4, 1),
        lexer.Token(lexer.TokenType.DEDENT, '', 4, 1),
        lexer.Token(lexer.TokenType.EOF, '', 4, 1),
    ]
    with pytest.raises(parser.ParseError, match="Expected at least one 'is' arm"):
        parser.Parser(tokens).parse_match()


def test_non_identifier_match_subject_requires_an_as_binding():
    """The restriction has loosened since sum type matching gained
    non-bare-variable subject support: a match subject that's an
    arbitrary expression is no longer rejected outright -- it just
    needs an explicit `as NAME` (see parse_match's own docstring,
    subject/binding_name), exactly like a non-bare-variable `is`
    check already does. Still rejected at the grammar level, just
    with a different, more specific message now that names the actual
    gap (a missing `as`) rather than pretending non-identifiers are
    categorically disallowed."""
    tokens = [
        lexer.Token(lexer.TokenType.MATCH, 'match', 1, 1),
        lexer.Token(lexer.TokenType.NUMBER, '5', 1, 7),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 8),
        lexer.Token(lexer.TokenType.EOF, '', 1, 9),
    ]
    with pytest.raises(parser.ParseError, match="Expected 'as NAME' after the match subject"):
        parser.Parser(tokens).parse_match()


def test_match_requires_is_or_else_between_arms():
    tokens = [
        lexer.Token(lexer.TokenType.MATCH, 'match', 1, 1),
        lexer.Token(lexer.TokenType.IDENTIFIER, 's', 1, 7),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 8),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 9),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'oops', 2, 5),
        lexer.Token(lexer.TokenType.EOF, '', 2, 9),
    ]
    with pytest.raises(parser.ParseError, match="Expected 'is' \\(a match arm\\) or 'else'"):
        parser.Parser(tokens).parse_match()


def test_parse_statement_dispatches_to_match():
    """parse_statement's own dispatch (`if self.check(TokenType.MATCH):
    return self.parse_match()`) -- every other test in this section
    calls parse_match directly, bypassing that dispatch line entirely,
    so this is the one test that actually goes through parse_program's
    ordinary statement-parsing path."""
    tokens = [
        lexer.Token(lexer.TokenType.MATCH, 'match', 1, 1),
        lexer.Token(lexer.TokenType.IDENTIFIER, 's', 1, 7),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 8),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 9),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.IS, 'is', 2, 5),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'Circle', 2, 8),
        lexer.Token(lexer.TokenType.COLON, ':', 2, 14),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 15),
        lexer.Token(lexer.TokenType.INDENT, '', 3, 1),
        lexer.Token(lexer.TokenType.RETURN, 'return', 3, 9),
        lexer.Token(lexer.TokenType.NUMBER, '1', 3, 16),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 3, 17),
        lexer.Token(lexer.TokenType.DEDENT, '', 4, 1),
        lexer.Token(lexer.TokenType.DEDENT, '', 4, 1),
        lexer.Token(lexer.TokenType.EOF, '', 4, 1),
    ]
    result = parser.Parser(tokens).parse_statement()
    assert isinstance(result, parser.If)
    assert result.is_match is True


# ---------------------------------------------------------------------------
# Pointers, stage 1: grammar only. `*T` in type position (PointerTypeExpr),
# `&`/`*` as new unary operators (UnaryOp.ADDRESS_OF/DEREFERENCE, reusing
# the existing AMPERSAND/STAR tokens -- no new lexer tokens needed at
# all), and `*pointer = value` as a new assignment target (DerefAssign).
#
# The one genuinely hard part: '*' can start EITHER a pointer TYPE
# (`*Circle p`, a VarDecl) OR a dereference EXPRESSION (`*p = 5`, an
# assignment) -- unlike every other type-starting token, which this
# parser (deliberately, with no symbol table) can always tell apart
# from an expression by shape alone. parse_statement resolves this by
# speculatively parsing a type and backtracking (restoring self.pos)
# if it turns out not to be followed by a variable name -- these tests
# check that disambiguation directly, via the real lexer, since hand-
# building tokens for a speculative-parse-and-backtrack path is more
# error-prone than just running the whole thing end to end.
# ---------------------------------------------------------------------------

def _parse_program(source: str) -> parser.Program:
    tokens = lexer.Lexer(source).tokenize()
    return parser.Parser(tokens).parse_program()


def test_pointer_type_in_a_var_decl():
    prog = _parse_program(
        "def int main():\n"
        "    *int p\n"
        "    return 0\n"
    )
    var_type = prog.functions[0].body[0].var_type
    assert isinstance(var_type, parser.PointerTypeExpr)
    assert var_type.pointee_type == 'int'


def test_pointer_to_struct_type_in_a_var_decl():
    """The genuinely ambiguous shape (`*Circle p`), resolved correctly:
    a type name followed by a SECOND identifier (the variable's own
    name) commits to a VarDecl."""
    prog = _parse_program(
        "type Circle struct:\n"
        "    int radius\n"
        "\n"
        "def int main():\n"
        "    *Circle p\n"
        "    return 0\n"
    )
    var_type = prog.functions[0].body[0].var_type
    assert isinstance(var_type, parser.PointerTypeExpr)
    assert var_type.pointee_type == 'Circle'


def test_pointer_to_pointer_type_parses_without_restriction():
    """parse_type/PointerTypeExpr don't reject `**int` themselves --
    see PointerTypeExpr's own docstring for why that's semantic.py's
    job instead."""
    prog = _parse_program(
        "def int main():\n"
        "    **int p\n"
        "    return 0\n"
    )
    var_type = prog.functions[0].body[0].var_type
    assert isinstance(var_type, parser.PointerTypeExpr)
    assert isinstance(var_type.pointee_type, parser.PointerTypeExpr)
    assert var_type.pointee_type.pointee_type == 'int'


def test_address_of_expression():
    prog = _parse_program(
        "def int main():\n"
        "    int x = 5\n"
        "    *int p = &x\n"
        "    return 0\n"
    )
    addr_expr = prog.functions[0].body[1].init
    assert isinstance(addr_expr, parser.Unary)
    assert addr_expr.op == parser.UnaryOp.ADDRESS_OF
    assert isinstance(addr_expr.operand, parser.Variable)
    assert addr_expr.operand.name == 'x'


def test_dereference_read_expression():
    prog = _parse_program(
        "def int main():\n"
        "    int x = 5\n"
        "    *int p = &x\n"
        "    int y = *p\n"
        "    return y\n"
    )
    deref_expr = prog.functions[0].body[2].init
    assert isinstance(deref_expr, parser.Unary)
    assert deref_expr.op == parser.UnaryOp.DEREFERENCE
    assert isinstance(deref_expr.operand, parser.Variable)
    assert deref_expr.operand.name == 'p'


def test_dereference_assignment_is_not_mistaken_for_a_pointer_var_decl():
    """The critical disambiguation case: `*p = 10` is a DerefAssign,
    NOT a (nonsensical) attempt at a pointer-typed VarDecl whose
    pointee type happens to be spelled 'p'."""
    prog = _parse_program(
        "def int main():\n"
        "    int x = 5\n"
        "    *int p = &x\n"
        "    *p = 10\n"
        "    return x\n"
    )
    deref_assign = prog.functions[0].body[2]
    assert isinstance(deref_assign, parser.DerefAssign)
    assert isinstance(deref_assign.pointer, parser.Variable)
    assert deref_assign.pointer.name == 'p'
    assert deref_assign.value == parser.Constant(value=10)


def test_ordinary_multiplication_is_unaffected():
    prog = _parse_program(
        "def int main():\n"
        "    int x = 3 * 4\n"
        "    return x\n"
    )
    mult_expr = prog.functions[0].body[0].init
    assert isinstance(mult_expr, parser.Binary)
    assert mult_expr.op == parser.BinaryOp.MULTIPLY


def test_dereference_of_a_non_type_falls_through_when_speculative_parse_fails():
    """`*5` as its own bare statement -- the speculative parse_type()
    call inside parse_statement itself raises (5 is never a valid
    type), not just "no identifier follows a successfully-parsed
    type" -- a DIFFERENT branch of the same backtracking logic than
    the *p-vs-*Circle-p cases above, worth its own direct test. Must
    be `*5` as parse_statement's OWN first token, not merely appearing
    on the right of an assignment (`x = *5`), which never reaches
    parse_statement's own STAR-handling branch at all -- parse_
    expression already handles a dereference wherever it's just one
    operand among others."""
    prog = _parse_program(
        "def int main():\n"
        "    *5\n"
        "    return 0\n"
    )
    stmt = prog.functions[0].body[0]
    assert isinstance(stmt, parser.ExprStmt)
    deref_expr = stmt.expr
    assert isinstance(deref_expr, parser.Unary)
    assert deref_expr.op == parser.UnaryOp.DEREFERENCE
    assert deref_expr.operand == parser.Constant(value=5)


def test_compound_assignment_through_a_dereference_now_parses():
    """`*p += 1` -- was rejected at parse time before this stage; now
    parses into a DerefAssign carrying compound_op, mirroring
    IndexAssign/FieldAssign exactly."""
    program = _parse_program(
        "def int main():\n"
        "    int x = 5\n"
        "    *int p = &x\n"
        "    *p += 1\n"
        "    return x\n"
    )
    stmt = program.functions[0].body[2]
    assert isinstance(stmt, parser.DerefAssign)
    assert stmt.compound_op == parser.BinaryOp.ADD


# ---------------------------------------------------------------------------
# `extern [type] NAME(params)` -- an external function declaration, no
# body, no ':'. Introduces one new AST node (ExternFunctionDecl, distinct
# from Function -- see its own docstring in parser.py for why) and one
# new Program list (extern_functions), sorted at parse time exactly like
# StructDef/SumTypeDef already are. Grammar mirrors parse_function's own
# return-type/name/params handling exactly, just without the trailing
# ':'/NEWLINE/body a real function always has.
# ---------------------------------------------------------------------------

def test_extern_function_with_scalar_return_and_params():
    program = _parse_program(
        "extern int abs(int n)\n"
        "def int main():\n"
        "    return abs(-5)\n"
    )
    assert len(program.extern_functions) == 1
    ext = program.extern_functions[0]
    assert isinstance(ext, parser.ExternFunctionDecl)
    assert ext.name == 'abs'
    assert ext.return_type == 'int'
    assert len(ext.params) == 1
    assert ext.params[0].name == 'n'
    assert ext.params[0].type == 'int'
    assert len(program.functions) == 1  # main only -- abs is NOT a Function


def test_extern_function_with_no_return_type_is_void():
    """No return type at all (not a 'void' keyword -- Hornet has none,
    see Function's own docstring) means void, identical to an ordinary
    Function's own convention."""
    program = _parse_program(
        "extern free(*int8 p)\n"
        "def int main():\n"
        "    return 0\n"
    )
    ext = program.extern_functions[0]
    assert ext.name == 'free'
    assert ext.return_type is None


def test_extern_function_with_pointer_return_type():
    program = _parse_program(
        "extern *int8 malloc(int64 size)\n"
        "def int main():\n"
        "    return 0\n"
    )
    ext = program.extern_functions[0]
    assert ext.name == 'malloc'
    assert isinstance(ext.return_type, parser.PointerTypeExpr)
    assert ext.return_type.pointee_type == 'int8'


def test_extern_function_with_no_params():
    program = _parse_program(
        "extern int getpid()\n"
        "def int main():\n"
        "    return getpid()\n"
    )
    ext = program.extern_functions[0]
    assert ext.params == []


def test_extern_function_with_multiple_params():
    program = _parse_program(
        "extern *int8 memcpy(*int8 dst, *int8 src, int64 n)\n"
        "def int main():\n"
        "    return 0\n"
    )
    ext = program.extern_functions[0]
    assert [p.name for p in ext.params] == ['dst', 'src', 'n']


def test_extern_function_order_independent_from_ordinary_functions():
    """Mirrors how ordinary functions/structs are already collected
    into their own Program list regardless of source order -- an
    extern declaration appearing AFTER the function that calls it
    parses identically to one appearing before."""
    program = _parse_program(
        "def int main():\n"
        "    return abs(-5)\n"
        "\n"
        "extern int abs(int n)\n"
    )
    assert len(program.extern_functions) == 1
    assert program.extern_functions[0].name == 'abs'
    assert len(program.functions) == 1


def test_extern_function_requires_a_name():
    with pytest.raises(parser.ParseError, match="Expected a function name"):
        _parse_program("extern int (int n)\n")


def test_extern_function_requires_parens():
    with pytest.raises(parser.ParseError, match="Expected '\\(' after function name"):
        _parse_program("extern int abs\n")
