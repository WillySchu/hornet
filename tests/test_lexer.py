"""Tests for the lexer."""

import lexer


def test_token_eq():
    tcs = [
        {
            'first': lexer.Token(lexer.TokenType.DEF, 'def', 0, 0),
            'second': lexer.Token(lexer.TokenType.DEF, 'def', 0, 0),
            'eq': True,
        },
        {
            'first': lexer.Token(lexer.TokenType.DEF, 'def', 0, 0),
            'second': lexer.Token(lexer.TokenType.INT, 'int', 0, 0),
            'eq': False,
        },
        {
            'first': lexer.Token(lexer.TokenType.INT, 'int', 0, 1),
            'second': lexer.Token(lexer.TokenType.INT, 'int', 0, 0),
            'eq': False,
        },
        {
            'first': lexer.Token(lexer.TokenType.INT, 'int', 0, 0),
            'second': lexer.Token(lexer.TokenType.INT, 'int', 1, 0),
            'eq': False,
        },
    ]
    for tc in tcs:
        if tc['eq']:
            assert tc['first'] == tc['second']
        else:
            assert not tc['first'] == tc['second']

# Initial grammar.

def test_lex_return_2():
    res = lexer.lex('tests/return_2.ht')
    #assert res[11].col == 0
    expected = [
        lexer.Token(lexer.TokenType.DEF, 'def', 1, 1),
        lexer.Token(lexer.TokenType.INT, 'int', 1, 5),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'main', 1, 9),
        lexer.Token(lexer.TokenType.OPEN_PAREN, '(', 1, 13),
        lexer.Token(lexer.TokenType.CLOSE_PAREN, ')', 1, 14),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 15),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 16),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.RETURN, 'return', 2, 5),
        lexer.Token(lexer.TokenType.NUMBER, '2', 2, 12),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 13),
        lexer.Token(lexer.TokenType.DEDENT, '', 3, 1),
        lexer.Token(lexer.TokenType.EOF, '', 3, 1),
    ]
    assert expected == res


# Test Punctuation.

def test_lex_brackets():
    res = lexer.lex('tests/brackets.ht')
    expected = [
        lexer.Token(lexer.TokenType.DEF, 'def', 1, 1),
        lexer.Token(lexer.TokenType.INT, 'int', 1, 5),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'main', 1, 9),
        lexer.Token(lexer.TokenType.OPEN_PAREN, '(', 1, 13),
        lexer.Token(lexer.TokenType.CLOSE_PAREN, ')', 1, 14),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 15),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 16),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.OPEN_BRACKET, '[', 2, 5),
        lexer.Token(lexer.TokenType.NUMBER, '2', 2, 6),
        lexer.Token(lexer.TokenType.CLOSE_BRACKET, ']', 2, 7),
        lexer.Token(lexer.TokenType.INT, 'int', 2, 8),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'arr', 2, 12),
        lexer.Token(lexer.TokenType.ASSIGN, '=', 2, 16),
        lexer.Token(lexer.TokenType.OPEN_BRACKET, '[', 2, 18),
        lexer.Token(lexer.TokenType.NUMBER, '1', 2, 19),
        lexer.Token(lexer.TokenType.COMMA, ',', 2, 20),
        lexer.Token(lexer.TokenType.NUMBER, '2', 2, 22),
        lexer.Token(lexer.TokenType.CLOSE_BRACKET, ']', 2, 23),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 24),
        lexer.Token(lexer.TokenType.RETURN, 'return', 3, 5),
        lexer.Token(lexer.TokenType.NUMBER, '0', 3, 12),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 3, 13),
        lexer.Token(lexer.TokenType.DEDENT, '', 4, 1),
        lexer.Token(lexer.TokenType.EOF, '', 4, 1),
    ]
    assert expected == res


def test_lex_comma():
    res = lexer.lex('tests/comma.ht')
    expected = [
        lexer.Token(lexer.TokenType.DEF, 'def', 1, 1),
        lexer.Token(lexer.TokenType.INT, 'int', 1, 5),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'main', 1, 9),
        lexer.Token(lexer.TokenType.OPEN_PAREN, '(', 1, 13),
        lexer.Token(lexer.TokenType.INT, 'int', 1, 14),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'a', 1, 18),
        lexer.Token(lexer.TokenType.CLOSE_PAREN, ')', 1, 19),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 20),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 21),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.RETURN, 'return', 2, 5),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'a', 2, 12),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 13),
        lexer.Token(lexer.TokenType.DEDENT, '', 3, 1),
        lexer.Token(lexer.TokenType.EOF, '', 3, 1),
    ]
    assert expected == res


# Unary Ops.

def test_lex_negation():
    res = lexer.lex('tests/negation.ht')
    expected = [
        lexer.Token(lexer.TokenType.DEF, 'def', 1, 1),
        lexer.Token(lexer.TokenType.INT, 'int', 1, 5),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'main', 1, 9),
        lexer.Token(lexer.TokenType.OPEN_PAREN, '(', 1, 13),
        lexer.Token(lexer.TokenType.CLOSE_PAREN, ')', 1, 14),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 15),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 16),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.RETURN, 'return', 2, 5),
        lexer.Token(lexer.TokenType.MINUS, '-', 2, 12),
        lexer.Token(lexer.TokenType.NUMBER, '2', 2, 13),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 14),
        lexer.Token(lexer.TokenType.DEDENT, '', 3, 1),
        lexer.Token(lexer.TokenType.EOF, '', 3, 1),
    ]
    assert expected == res


def test_lex_bitwise_complement():
    res = lexer.lex('tests/bitwise_complement.ht')
    expected = [
        lexer.Token(lexer.TokenType.DEF, 'def', 1, 1),
        lexer.Token(lexer.TokenType.INT, 'int', 1, 5),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'main', 1, 9),
        lexer.Token(lexer.TokenType.OPEN_PAREN, '(', 1, 13),
        lexer.Token(lexer.TokenType.CLOSE_PAREN, ')', 1, 14),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 15),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 16),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.RETURN, 'return', 2, 5),
        lexer.Token(lexer.TokenType.TILDE, '~', 2, 12),
        lexer.Token(lexer.TokenType.NUMBER, '2', 2, 13),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 14),
        lexer.Token(lexer.TokenType.DEDENT, '', 3, 1),
        lexer.Token(lexer.TokenType.EOF, '', 3, 1),
    ]
    assert expected == res

# Binary Ops.

def test_assignment():
    res = lexer.lex('tests/assigment.ht')
    expected = [
        lexer.Token(lexer.TokenType.DEF, 'def', 1, 1),
        lexer.Token(lexer.TokenType.INT, 'int', 1, 5),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'main', 1, 9),
        lexer.Token(lexer.TokenType.OPEN_PAREN, '(', 1, 13),
        lexer.Token(lexer.TokenType.CLOSE_PAREN, ')', 1, 14),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 15),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 16),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.INT, 'int', 2, 5),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'a', 2, 9),
        lexer.Token(lexer.TokenType.ASSIGN, '=', 2, 11),
        lexer.Token(lexer.TokenType.NUMBER, '1', 2, 13),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 14),
        lexer.Token(lexer.TokenType.RETURN, 'return', 3, 5),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'a', 3, 12),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 3, 13),
        lexer.Token(lexer.TokenType.DEDENT, '', 4, 1),
        lexer.Token(lexer.TokenType.EOF, '', 4, 1),
    ]
    assert expected == res

def test_lex_addition():
    res = lexer.lex('tests/addition.ht')
    expected = [
        lexer.Token(lexer.TokenType.DEF, 'def', 1, 1),
        lexer.Token(lexer.TokenType.INT, 'int', 1, 5),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'main', 1, 9),
        lexer.Token(lexer.TokenType.OPEN_PAREN, '(', 1, 13),
        lexer.Token(lexer.TokenType.CLOSE_PAREN, ')', 1, 14),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 15),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 16),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.RETURN, 'return', 2, 5),
        lexer.Token(lexer.TokenType.NUMBER, '2', 2, 12),
        lexer.Token(lexer.TokenType.PLUS, '+', 2, 14),
        lexer.Token(lexer.TokenType.NUMBER, '3', 2, 16),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 17),
        lexer.Token(lexer.TokenType.DEDENT, '', 3, 1),
        lexer.Token(lexer.TokenType.EOF, '', 3, 1),
    ]
    assert expected == res


def test_lex_subtraction():
    res = lexer.lex('tests/subtraction.ht')
    expected = [
        lexer.Token(lexer.TokenType.DEF, 'def', 1, 1),
        lexer.Token(lexer.TokenType.INT, 'int', 1, 5),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'main', 1, 9),
        lexer.Token(lexer.TokenType.OPEN_PAREN, '(', 1, 13),
        lexer.Token(lexer.TokenType.CLOSE_PAREN, ')', 1, 14),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 15),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 16),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.RETURN, 'return', 2, 5),
        lexer.Token(lexer.TokenType.NUMBER, '2', 2, 12),
        lexer.Token(lexer.TokenType.MINUS, '-', 2, 14),
        lexer.Token(lexer.TokenType.NUMBER, '3', 2, 16),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 17),
        lexer.Token(lexer.TokenType.DEDENT, '', 3, 1),
        lexer.Token(lexer.TokenType.EOF, '', 3, 1),
    ]
    assert expected == res


def test_lex_multiplication():
    res = lexer.lex('tests/multiplication.ht')
    expected = [
        lexer.Token(lexer.TokenType.DEF, 'def', 1, 1),
        lexer.Token(lexer.TokenType.INT, 'int', 1, 5),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'main', 1, 9),
        lexer.Token(lexer.TokenType.OPEN_PAREN, '(', 1, 13),
        lexer.Token(lexer.TokenType.CLOSE_PAREN, ')', 1, 14),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 15),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 16),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.RETURN, 'return', 2, 5),
        lexer.Token(lexer.TokenType.NUMBER, '2', 2, 12),
        lexer.Token(lexer.TokenType.STAR, '*', 2, 14),
        lexer.Token(lexer.TokenType.NUMBER, '3', 2, 16),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 17),
        lexer.Token(lexer.TokenType.DEDENT, '', 3, 1),
        lexer.Token(lexer.TokenType.EOF, '', 3, 1),
    ]
    assert expected == res


def test_lex_division():
    res = lexer.lex('tests/division.ht')
    expected = [
        lexer.Token(lexer.TokenType.DEF, 'def', 1, 1),
        lexer.Token(lexer.TokenType.INT, 'int', 1, 5),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'main', 1, 9),
        lexer.Token(lexer.TokenType.OPEN_PAREN, '(', 1, 13),
        lexer.Token(lexer.TokenType.CLOSE_PAREN, ')', 1, 14),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 15),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 16),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.RETURN, 'return', 2, 5),
        lexer.Token(lexer.TokenType.NUMBER, '2', 2, 12),
        lexer.Token(lexer.TokenType.SLASH, '/', 2, 14),
        lexer.Token(lexer.TokenType.NUMBER, '3', 2, 16),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 17),
        lexer.Token(lexer.TokenType.DEDENT, '', 3, 1),
        lexer.Token(lexer.TokenType.EOF, '', 3, 1),
    ]
    assert expected == res


def test_lex_modulo():
    res = lexer.lex('tests/modulo.ht')
    expected = [
        lexer.Token(lexer.TokenType.DEF, 'def', 1, 1),
        lexer.Token(lexer.TokenType.INT, 'int', 1, 5),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'main', 1, 9),
        lexer.Token(lexer.TokenType.OPEN_PAREN, '(', 1, 13),
        lexer.Token(lexer.TokenType.CLOSE_PAREN, ')', 1, 14),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 15),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 16),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.RETURN, 'return', 2, 5),
        lexer.Token(lexer.TokenType.NUMBER, '2', 2, 12),
        lexer.Token(lexer.TokenType.PERCENT, '%', 2, 14),
        lexer.Token(lexer.TokenType.NUMBER, '3', 2, 16),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 17),
        lexer.Token(lexer.TokenType.DEDENT, '', 3, 1),
        lexer.Token(lexer.TokenType.EOF, '', 3, 1),
    ]
    assert expected == res


def test_lex_bitwise_and():
    res = lexer.lex('tests/bitwise_and.ht')
    expected = [
        lexer.Token(lexer.TokenType.DEF, 'def', 1, 1),
        lexer.Token(lexer.TokenType.INT, 'int', 1, 5),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'main', 1, 9),
        lexer.Token(lexer.TokenType.OPEN_PAREN, '(', 1, 13),
        lexer.Token(lexer.TokenType.CLOSE_PAREN, ')', 1, 14),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 15),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 16),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.RETURN, 'return', 2, 5),
        lexer.Token(lexer.TokenType.NUMBER, '2', 2, 12),
        lexer.Token(lexer.TokenType.AMPERSAND, '&', 2, 14),
        lexer.Token(lexer.TokenType.NUMBER, '3', 2, 16),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 17),
        lexer.Token(lexer.TokenType.DEDENT, '', 3, 1),
        lexer.Token(lexer.TokenType.EOF, '', 3, 1),
    ]
    assert expected == res


def test_lex_bitwise_or():
    res = lexer.lex('tests/bitwise_or.ht')
    expected = [
        lexer.Token(lexer.TokenType.DEF, 'def', 1, 1),
        lexer.Token(lexer.TokenType.INT, 'int', 1, 5),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'main', 1, 9),
        lexer.Token(lexer.TokenType.OPEN_PAREN, '(', 1, 13),
        lexer.Token(lexer.TokenType.CLOSE_PAREN, ')', 1, 14),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 15),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 16),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.RETURN, 'return', 2, 5),
        lexer.Token(lexer.TokenType.NUMBER, '2', 2, 12),
        lexer.Token(lexer.TokenType.PIPE, '|', 2, 14),
        lexer.Token(lexer.TokenType.NUMBER, '3', 2, 16),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 17),
        lexer.Token(lexer.TokenType.DEDENT, '', 3, 1),
        lexer.Token(lexer.TokenType.EOF, '', 3, 1),
    ]
    assert expected == res


def test_lex_bitwise_xor():
    res = lexer.lex('tests/bitwise_xor.ht')
    expected = [
        lexer.Token(lexer.TokenType.DEF, 'def', 1, 1),
        lexer.Token(lexer.TokenType.INT, 'int', 1, 5),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'main', 1, 9),
        lexer.Token(lexer.TokenType.OPEN_PAREN, '(', 1, 13),
        lexer.Token(lexer.TokenType.CLOSE_PAREN, ')', 1, 14),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 15),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 16),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.RETURN, 'return', 2, 5),
        lexer.Token(lexer.TokenType.NUMBER, '2', 2, 12),
        lexer.Token(lexer.TokenType.CARET, '^', 2, 14),
        lexer.Token(lexer.TokenType.NUMBER, '3', 2, 16),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 17),
        lexer.Token(lexer.TokenType.DEDENT, '', 3, 1),
        lexer.Token(lexer.TokenType.EOF, '', 3, 1),
    ]
    assert expected == res


def test_lex_shift_left():
    res = lexer.lex('tests/bitwise_shift_left.ht')
    expected = [
        lexer.Token(lexer.TokenType.DEF, 'def', 1, 1),
        lexer.Token(lexer.TokenType.INT, 'int', 1, 5),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'main', 1, 9),
        lexer.Token(lexer.TokenType.OPEN_PAREN, '(', 1, 13),
        lexer.Token(lexer.TokenType.CLOSE_PAREN, ')', 1, 14),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 15),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 16),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.RETURN, 'return', 2, 5),
        lexer.Token(lexer.TokenType.NUMBER, '2', 2, 12),
        lexer.Token(lexer.TokenType.SHIFT_LEFT, '<<', 2, 14),
        lexer.Token(lexer.TokenType.NUMBER, '3', 2, 17),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 18),
        lexer.Token(lexer.TokenType.DEDENT, '', 3, 1),
        lexer.Token(lexer.TokenType.EOF, '', 3, 1),
    ]
    assert expected == res


def test_lex_shift_right():
    res = lexer.lex('tests/bitwise_shift_right.ht')
    expected = [
        lexer.Token(lexer.TokenType.DEF, 'def', 1, 1),
        lexer.Token(lexer.TokenType.INT, 'int', 1, 5),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'main', 1, 9),
        lexer.Token(lexer.TokenType.OPEN_PAREN, '(', 1, 13),
        lexer.Token(lexer.TokenType.CLOSE_PAREN, ')', 1, 14),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 15),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 16),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.RETURN, 'return', 2, 5),
        lexer.Token(lexer.TokenType.NUMBER, '2', 2, 12),
        lexer.Token(lexer.TokenType.SHIFT_RIGHT, '>>', 2, 14),
        lexer.Token(lexer.TokenType.NUMBER, '3', 2, 17),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 18),
        lexer.Token(lexer.TokenType.DEDENT, '', 3, 1),
        lexer.Token(lexer.TokenType.EOF, '', 3, 1),
    ]
    assert expected == res


def test_lex_equal():
    res = lexer.lex('tests/equal.ht')
    expected = [
        lexer.Token(lexer.TokenType.DEF, 'def', 1, 1),
        lexer.Token(lexer.TokenType.INT, 'int', 1, 5),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'main', 1, 9),
        lexer.Token(lexer.TokenType.OPEN_PAREN, '(', 1, 13),
        lexer.Token(lexer.TokenType.CLOSE_PAREN, ')', 1, 14),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 15),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 16),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.RETURN, 'return', 2, 5),
        lexer.Token(lexer.TokenType.NUMBER, '2', 2, 12),
        lexer.Token(lexer.TokenType.EQUAL, '==', 2, 14),
        lexer.Token(lexer.TokenType.NUMBER, '3', 2, 17),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 18),
        lexer.Token(lexer.TokenType.DEDENT, '', 3, 1),
        lexer.Token(lexer.TokenType.EOF, '', 3, 1),
    ]
    assert expected == res


def test_lex_not_equal():
    res = lexer.lex('tests/not_equal.ht')
    expected = [
        lexer.Token(lexer.TokenType.DEF, 'def', 1, 1),
        lexer.Token(lexer.TokenType.INT, 'int', 1, 5),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'main', 1, 9),
        lexer.Token(lexer.TokenType.OPEN_PAREN, '(', 1, 13),
        lexer.Token(lexer.TokenType.CLOSE_PAREN, ')', 1, 14),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 15),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 16),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.RETURN, 'return', 2, 5),
        lexer.Token(lexer.TokenType.NUMBER, '2', 2, 12),
        lexer.Token(lexer.TokenType.NOT_EQUAL, '!=', 2, 14),
        lexer.Token(lexer.TokenType.NUMBER, '3', 2, 17),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 18),
        lexer.Token(lexer.TokenType.DEDENT, '', 3, 1),
        lexer.Token(lexer.TokenType.EOF, '', 3, 1),
    ]
    assert expected == res


def test_lex_less_than():
    res = lexer.lex('tests/less_than.ht')
    expected = [
        lexer.Token(lexer.TokenType.DEF, 'def', 1, 1),
        lexer.Token(lexer.TokenType.INT, 'int', 1, 5),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'main', 1, 9),
        lexer.Token(lexer.TokenType.OPEN_PAREN, '(', 1, 13),
        lexer.Token(lexer.TokenType.CLOSE_PAREN, ')', 1, 14),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 15),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 16),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.RETURN, 'return', 2, 5),
        lexer.Token(lexer.TokenType.NUMBER, '2', 2, 12),
        lexer.Token(lexer.TokenType.LESS_THAN, '<', 2, 14),
        lexer.Token(lexer.TokenType.NUMBER, '3', 2, 16),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 17),
        lexer.Token(lexer.TokenType.DEDENT, '', 3, 1),
        lexer.Token(lexer.TokenType.EOF, '', 3, 1),
    ]
    assert expected == res


def test_lex_greater_than():
    res = lexer.lex('tests/greater_than.ht')
    expected = [
        lexer.Token(lexer.TokenType.DEF, 'def', 1, 1),
        lexer.Token(lexer.TokenType.INT, 'int', 1, 5),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'main', 1, 9),
        lexer.Token(lexer.TokenType.OPEN_PAREN, '(', 1, 13),
        lexer.Token(lexer.TokenType.CLOSE_PAREN, ')', 1, 14),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 15),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 16),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.RETURN, 'return', 2, 5),
        lexer.Token(lexer.TokenType.NUMBER, '2', 2, 12),
        lexer.Token(lexer.TokenType.GREATER_THAN, '>', 2, 14),
        lexer.Token(lexer.TokenType.NUMBER, '3', 2, 16),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 17),
        lexer.Token(lexer.TokenType.DEDENT, '', 3, 1),
        lexer.Token(lexer.TokenType.EOF, '', 3, 1),
    ]
    assert expected == res


def test_lex_less_than_or_equal():
    res = lexer.lex('tests/less_than_or_equal.ht')
    expected = [
        lexer.Token(lexer.TokenType.DEF, 'def', 1, 1),
        lexer.Token(lexer.TokenType.INT, 'int', 1, 5),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'main', 1, 9),
        lexer.Token(lexer.TokenType.OPEN_PAREN, '(', 1, 13),
        lexer.Token(lexer.TokenType.CLOSE_PAREN, ')', 1, 14),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 15),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 16),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.RETURN, 'return', 2, 5),
        lexer.Token(lexer.TokenType.NUMBER, '2', 2, 12),
        lexer.Token(lexer.TokenType.LESS_THAN_OR_EQUAL, '<=', 2, 14),
        lexer.Token(lexer.TokenType.NUMBER, '3', 2, 17),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 18),
        lexer.Token(lexer.TokenType.DEDENT, '', 3, 1),
        lexer.Token(lexer.TokenType.EOF, '', 3, 1),
    ]
    assert expected == res


def test_lex_greater_than_or_equal():
    res = lexer.lex('tests/greater_than_or_equal.ht')
    expected = [
        lexer.Token(lexer.TokenType.DEF, 'def', 1, 1),
        lexer.Token(lexer.TokenType.INT, 'int', 1, 5),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'main', 1, 9),
        lexer.Token(lexer.TokenType.OPEN_PAREN, '(', 1, 13),
        lexer.Token(lexer.TokenType.CLOSE_PAREN, ')', 1, 14),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 15),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 16),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.RETURN, 'return', 2, 5),
        lexer.Token(lexer.TokenType.NUMBER, '2', 2, 12),
        lexer.Token(lexer.TokenType.GREATER_THAN_OR_EQUAL, '>=', 2, 14),
        lexer.Token(lexer.TokenType.NUMBER, '3', 2, 17),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 18),
        lexer.Token(lexer.TokenType.DEDENT, '', 3, 1),
        lexer.Token(lexer.TokenType.EOF, '', 3, 1),
    ]
    assert expected == res


# Test compound assigment.

def test_plus_assign():
    res = lexer.lex('tests/plus_assign.ht')
    expected = [
        lexer.Token(lexer.TokenType.DEF, 'def', 1, 1),
        lexer.Token(lexer.TokenType.INT, 'int', 1, 5),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'main', 1, 9),
        lexer.Token(lexer.TokenType.OPEN_PAREN, '(', 1, 13),
        lexer.Token(lexer.TokenType.CLOSE_PAREN, ')', 1, 14),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 15),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 16),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.RETURN, 'return', 2, 5),
        lexer.Token(lexer.TokenType.NUMBER, '2', 2, 12),
        lexer.Token(lexer.TokenType.PLUS_ASSIGN, '+=', 2, 14),
        lexer.Token(lexer.TokenType.NUMBER, '3', 2, 17),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 18),
        lexer.Token(lexer.TokenType.DEDENT, '', 3, 1),
        lexer.Token(lexer.TokenType.EOF, '', 3, 1),
    ]
    assert expected == res


def test_minus_assign():
    res = lexer.lex('tests/minus_assign.ht')
    expected = [
        lexer.Token(lexer.TokenType.DEF, 'def', 1, 1),
        lexer.Token(lexer.TokenType.INT, 'int', 1, 5),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'main', 1, 9),
        lexer.Token(lexer.TokenType.OPEN_PAREN, '(', 1, 13),
        lexer.Token(lexer.TokenType.CLOSE_PAREN, ')', 1, 14),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 15),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 16),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.RETURN, 'return', 2, 5),
        lexer.Token(lexer.TokenType.NUMBER, '2', 2, 12),
        lexer.Token(lexer.TokenType.MINUS_ASSIGN, '-=', 2, 14),
        lexer.Token(lexer.TokenType.NUMBER, '3', 2, 17),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 18),
        lexer.Token(lexer.TokenType.DEDENT, '', 3, 1),
        lexer.Token(lexer.TokenType.EOF, '', 3, 1),
    ]
    assert expected == res


def test_star_assign():
    res = lexer.lex('tests/star_assign.ht')
    expected = [
        lexer.Token(lexer.TokenType.DEF, 'def', 1, 1),
        lexer.Token(lexer.TokenType.INT, 'int', 1, 5),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'main', 1, 9),
        lexer.Token(lexer.TokenType.OPEN_PAREN, '(', 1, 13),
        lexer.Token(lexer.TokenType.CLOSE_PAREN, ')', 1, 14),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 15),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 16),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.RETURN, 'return', 2, 5),
        lexer.Token(lexer.TokenType.NUMBER, '2', 2, 12),
        lexer.Token(lexer.TokenType.STAR_ASSIGN, '*=', 2, 14),
        lexer.Token(lexer.TokenType.NUMBER, '3', 2, 17),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 18),
        lexer.Token(lexer.TokenType.DEDENT, '', 3, 1),
        lexer.Token(lexer.TokenType.EOF, '', 3, 1),
    ]
    assert expected == res


def test_slash_assign():
    res = lexer.lex('tests/slash_assign.ht')
    expected = [
        lexer.Token(lexer.TokenType.DEF, 'def', 1, 1),
        lexer.Token(lexer.TokenType.INT, 'int', 1, 5),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'main', 1, 9),
        lexer.Token(lexer.TokenType.OPEN_PAREN, '(', 1, 13),
        lexer.Token(lexer.TokenType.CLOSE_PAREN, ')', 1, 14),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 15),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 16),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.RETURN, 'return', 2, 5),
        lexer.Token(lexer.TokenType.NUMBER, '2', 2, 12),
        lexer.Token(lexer.TokenType.SLASH_ASSIGN, '/=', 2, 14),
        lexer.Token(lexer.TokenType.NUMBER, '3', 2, 17),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 18),
        lexer.Token(lexer.TokenType.DEDENT, '', 3, 1),
        lexer.Token(lexer.TokenType.EOF, '', 3, 1),
    ]
    assert expected == res


def test_percent_assign():
    res = lexer.lex('tests/percent_assign.ht')
    expected = [
        lexer.Token(lexer.TokenType.DEF, 'def', 1, 1),
        lexer.Token(lexer.TokenType.INT, 'int', 1, 5),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'main', 1, 9),
        lexer.Token(lexer.TokenType.OPEN_PAREN, '(', 1, 13),
        lexer.Token(lexer.TokenType.CLOSE_PAREN, ')', 1, 14),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 15),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 16),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.RETURN, 'return', 2, 5),
        lexer.Token(lexer.TokenType.NUMBER, '2', 2, 12),
        lexer.Token(lexer.TokenType.PERCENT_ASSIGN, '%=', 2, 14),
        lexer.Token(lexer.TokenType.NUMBER, '3', 2, 17),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 18),
        lexer.Token(lexer.TokenType.DEDENT, '', 3, 1),
        lexer.Token(lexer.TokenType.EOF, '', 3, 1),
    ]
    assert expected == res


def test_ampersand_assign():
    res = lexer.lex('tests/ampersand_assign.ht')
    expected = [
        lexer.Token(lexer.TokenType.DEF, 'def', 1, 1),
        lexer.Token(lexer.TokenType.INT, 'int', 1, 5),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'main', 1, 9),
        lexer.Token(lexer.TokenType.OPEN_PAREN, '(', 1, 13),
        lexer.Token(lexer.TokenType.CLOSE_PAREN, ')', 1, 14),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 15),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 16),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.RETURN, 'return', 2, 5),
        lexer.Token(lexer.TokenType.NUMBER, '2', 2, 12),
        lexer.Token(lexer.TokenType.AMPERSAND_ASSIGN, '&=', 2, 14),
        lexer.Token(lexer.TokenType.NUMBER, '3', 2, 17),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 18),
        lexer.Token(lexer.TokenType.DEDENT, '', 3, 1),
        lexer.Token(lexer.TokenType.EOF, '', 3, 1),
    ]
    assert expected == res


def test_pipe_assign():
    res = lexer.lex('tests/pipe_assign.ht')
    expected = [
        lexer.Token(lexer.TokenType.DEF, 'def', 1, 1),
        lexer.Token(lexer.TokenType.INT, 'int', 1, 5),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'main', 1, 9),
        lexer.Token(lexer.TokenType.OPEN_PAREN, '(', 1, 13),
        lexer.Token(lexer.TokenType.CLOSE_PAREN, ')', 1, 14),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 15),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 16),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.RETURN, 'return', 2, 5),
        lexer.Token(lexer.TokenType.NUMBER, '2', 2, 12),
        lexer.Token(lexer.TokenType.PIPE_ASSIGN, '|=', 2, 14),
        lexer.Token(lexer.TokenType.NUMBER, '3', 2, 17),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 18),
        lexer.Token(lexer.TokenType.DEDENT, '', 3, 1),
        lexer.Token(lexer.TokenType.EOF, '', 3, 1),
    ]
    assert expected == res


def test_caret_assign():
    res = lexer.lex('tests/caret_assign.ht')
    expected = [
        lexer.Token(lexer.TokenType.DEF, 'def', 1, 1),
        lexer.Token(lexer.TokenType.INT, 'int', 1, 5),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'main', 1, 9),
        lexer.Token(lexer.TokenType.OPEN_PAREN, '(', 1, 13),
        lexer.Token(lexer.TokenType.CLOSE_PAREN, ')', 1, 14),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 15),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 16),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.RETURN, 'return', 2, 5),
        lexer.Token(lexer.TokenType.NUMBER, '2', 2, 12),
        lexer.Token(lexer.TokenType.CARET_ASSIGN, '^=', 2, 14),
        lexer.Token(lexer.TokenType.NUMBER, '3', 2, 17),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 18),
        lexer.Token(lexer.TokenType.DEDENT, '', 3, 1),
        lexer.Token(lexer.TokenType.EOF, '', 3, 1),
    ]
    assert expected == res


def test_shift_left_assign():
    res = lexer.lex('tests/shift_left_assign.ht')
    expected = [
        lexer.Token(lexer.TokenType.DEF, 'def', 1, 1),
        lexer.Token(lexer.TokenType.INT, 'int', 1, 5),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'main', 1, 9),
        lexer.Token(lexer.TokenType.OPEN_PAREN, '(', 1, 13),
        lexer.Token(lexer.TokenType.CLOSE_PAREN, ')', 1, 14),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 15),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 16),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.RETURN, 'return', 2, 5),
        lexer.Token(lexer.TokenType.NUMBER, '2', 2, 12),
        lexer.Token(lexer.TokenType.SHIFT_LEFT_ASSIGN, '<<=', 2, 14),
        lexer.Token(lexer.TokenType.NUMBER, '3', 2, 18),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 19),
        lexer.Token(lexer.TokenType.DEDENT, '', 3, 1),
        lexer.Token(lexer.TokenType.EOF, '', 3, 1),
    ]
    assert expected == res


def test_shift_right_assign():
    res = lexer.lex('tests/shift_right_assign.ht')
    expected = [
        lexer.Token(lexer.TokenType.DEF, 'def', 1, 1),
        lexer.Token(lexer.TokenType.INT, 'int', 1, 5),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'main', 1, 9),
        lexer.Token(lexer.TokenType.OPEN_PAREN, '(', 1, 13),
        lexer.Token(lexer.TokenType.CLOSE_PAREN, ')', 1, 14),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 15),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 16),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.RETURN, 'return', 2, 5),
        lexer.Token(lexer.TokenType.NUMBER, '2', 2, 12),
        lexer.Token(lexer.TokenType.SHIFT_RIGHT_ASSIGN, '>>=', 2, 14),
        lexer.Token(lexer.TokenType.NUMBER, '3', 2, 18),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 19),
        lexer.Token(lexer.TokenType.DEDENT, '', 3, 1),
        lexer.Token(lexer.TokenType.EOF, '', 3, 1),
    ]
    assert expected == res


# Test keywords.

def test_lex_not():
    res = lexer.lex('tests/not.ht')
    expected = [
        lexer.Token(lexer.TokenType.DEF, 'def', 1, 1),
        lexer.Token(lexer.TokenType.INT, 'int', 1, 5),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'main', 1, 9),
        lexer.Token(lexer.TokenType.OPEN_PAREN, '(', 1, 13),
        lexer.Token(lexer.TokenType.CLOSE_PAREN, ')', 1, 14),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 15),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 16),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.RETURN, 'return', 2, 5),
        lexer.Token(lexer.TokenType.NOT, 'not', 2, 12),
        lexer.Token(lexer.TokenType.NUMBER, '2', 2, 16),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 17),
        lexer.Token(lexer.TokenType.DEDENT, '', 3, 1),
        lexer.Token(lexer.TokenType.EOF, '', 3, 1),
    ]
    assert expected == res


def test_lex_and():
    res = lexer.lex('tests/and.ht')
    expected = [
        lexer.Token(lexer.TokenType.DEF, 'def', 1, 1),
        lexer.Token(lexer.TokenType.INT, 'int', 1, 5),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'main', 1, 9),
        lexer.Token(lexer.TokenType.OPEN_PAREN, '(', 1, 13),
        lexer.Token(lexer.TokenType.CLOSE_PAREN, ')', 1, 14),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 15),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 16),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.RETURN, 'return', 2, 5),
        lexer.Token(lexer.TokenType.NUMBER, '0', 2, 12),
        lexer.Token(lexer.TokenType.AND, 'and', 2, 14),
        lexer.Token(lexer.TokenType.NUMBER, '1', 2, 18),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 19),
        lexer.Token(lexer.TokenType.DEDENT, '', 3, 1),
        lexer.Token(lexer.TokenType.EOF, '', 3, 1),
    ]
    assert expected == res


def test_lex_or():
    res = lexer.lex('tests/or.ht')
    expected = [
        lexer.Token(lexer.TokenType.DEF, 'def', 1, 1),
        lexer.Token(lexer.TokenType.INT, 'int', 1, 5),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'main', 1, 9),
        lexer.Token(lexer.TokenType.OPEN_PAREN, '(', 1, 13),
        lexer.Token(lexer.TokenType.CLOSE_PAREN, ')', 1, 14),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 15),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 16),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.RETURN, 'return', 2, 5),
        lexer.Token(lexer.TokenType.NUMBER, '0', 2, 12),
        lexer.Token(lexer.TokenType.OR, 'or', 2, 14),
        lexer.Token(lexer.TokenType.NUMBER, '1', 2, 17),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 18),
        lexer.Token(lexer.TokenType.DEDENT, '', 3, 1),
        lexer.Token(lexer.TokenType.EOF, '', 3, 1),
    ]
    assert expected == res


def test_lex_str():
    res = lexer.lex('tests/str.ht')
    expected = [
        lexer.Token(lexer.TokenType.DEF, 'def', 1, 1),
        lexer.Token(lexer.TokenType.STR, 'str', 1, 5),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'main', 1, 9),
        lexer.Token(lexer.TokenType.OPEN_PAREN, '(', 1, 13),
        lexer.Token(lexer.TokenType.CLOSE_PAREN, ')', 1, 14),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 15),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 16),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.STR, 'str', 2, 5),
        lexer.Token(lexer.TokenType.IDENTIFIER, 's', 2, 9),
        lexer.Token(lexer.TokenType.ASSIGN, '=', 2, 11),
        lexer.Token(lexer.TokenType.STRING, "'asdf'", 2, 13),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 19),
        lexer.Token(lexer.TokenType.RETURN, 'return', 3, 5),
        lexer.Token(lexer.TokenType.IDENTIFIER, 's', 3, 12),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 3, 13),
        lexer.Token(lexer.TokenType.DEDENT, '', 4, 1),
        lexer.Token(lexer.TokenType.EOF, '', 4, 1),
    ]
    assert expected == res


def test_lex_bool():
    res = lexer.lex('tests/bool.ht')
    expected = [
        lexer.Token(lexer.TokenType.DEF, 'def', 1, 1),
        lexer.Token(lexer.TokenType.BOOL, 'bool', 1, 5),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'main', 1, 10),
        lexer.Token(lexer.TokenType.OPEN_PAREN, '(', 1, 14),
        lexer.Token(lexer.TokenType.CLOSE_PAREN, ')', 1, 15),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 16),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 17),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.BOOL, 'bool', 2, 5),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'b', 2, 10),
        lexer.Token(lexer.TokenType.ASSIGN, '=', 2, 12),
        lexer.Token(lexer.TokenType.TRUE, 'true', 2, 14),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 18),
        lexer.Token(lexer.TokenType.RETURN, 'return', 3, 5),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'b', 3, 12),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 3, 13),
        lexer.Token(lexer.TokenType.DEDENT, '', 4, 1),
        lexer.Token(lexer.TokenType.EOF, '', 4, 1),
    ]
    assert expected == res


def test_lex_true():
    res = lexer.lex('tests/true.ht')
    expected = [
        lexer.Token(lexer.TokenType.DEF, 'def', 1, 1),
        lexer.Token(lexer.TokenType.BOOL, 'bool', 1, 5),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'main', 1, 10),
        lexer.Token(lexer.TokenType.OPEN_PAREN, '(', 1, 14),
        lexer.Token(lexer.TokenType.CLOSE_PAREN, ')', 1, 15),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 16),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 17),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.RETURN, 'return', 2, 5),
        lexer.Token(lexer.TokenType.TRUE, 'true', 2, 12),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 16),
        lexer.Token(lexer.TokenType.DEDENT, '', 3, 1),
        lexer.Token(lexer.TokenType.EOF, '', 3, 1),
    ]
    assert expected == res


def test_lex_false():
    res = lexer.lex('tests/false.ht')
    expected = [
        lexer.Token(lexer.TokenType.DEF, 'def', 1, 1),
        lexer.Token(lexer.TokenType.BOOL, 'bool', 1, 5),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'main', 1, 10),
        lexer.Token(lexer.TokenType.OPEN_PAREN, '(', 1, 14),
        lexer.Token(lexer.TokenType.CLOSE_PAREN, ')', 1, 15),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 16),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 17),
        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.RETURN, 'return', 2, 5),
        lexer.Token(lexer.TokenType.FALSE, 'false', 2, 12),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 17),
        lexer.Token(lexer.TokenType.DEDENT, '', 3, 1),
        lexer.Token(lexer.TokenType.EOF, '', 3, 1),
    ]
    assert expected == res


def test_if_elif_else():
    res = lexer.lex('tests/if_elif_else.ht')
    expected = [
        lexer.Token(lexer.TokenType.DEF, 'def', 1, 1),
        lexer.Token(lexer.TokenType.INT, 'int', 1, 5),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'main', 1, 9),
        lexer.Token(lexer.TokenType.OPEN_PAREN, '(', 1, 13),
        lexer.Token(lexer.TokenType.CLOSE_PAREN, ')', 1, 14),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 15),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 16),

        lexer.Token(lexer.TokenType.INDENT, '', 2, 1),
        lexer.Token(lexer.TokenType.BOOL, 'bool', 2, 5),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'a', 2, 10),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 11),

        lexer.Token(lexer.TokenType.BOOL, 'bool', 3, 5),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'b', 3, 10),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 3, 11),

        lexer.Token(lexer.TokenType.IF, 'if', 4, 5),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'a', 4, 8),
        lexer.Token(lexer.TokenType.COLON, ':', 4, 9),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 4, 10),

        lexer.Token(lexer.TokenType.INDENT, '', 5, 1),
        lexer.Token(lexer.TokenType.RETURN, 'return', 5, 9),
        lexer.Token(lexer.TokenType.NUMBER, '0', 5, 16),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 5, 17),

        lexer.Token(lexer.TokenType.DEDENT, '', 6, 1),
        lexer.Token(lexer.TokenType.ELIF, 'elif', 6, 5),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'b', 6, 10),
        lexer.Token(lexer.TokenType.COLON, ':', 6, 11),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 6, 12),

        lexer.Token(lexer.TokenType.INDENT, '', 7, 1),
        lexer.Token(lexer.TokenType.RETURN, 'return', 7, 9),
        lexer.Token(lexer.TokenType.NUMBER, '1', 7, 16),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 7, 17),

        lexer.Token(lexer.TokenType.DEDENT, '', 8, 1),
        lexer.Token(lexer.TokenType.ELSE, 'else', 8, 5),
        lexer.Token(lexer.TokenType.COLON, ':', 8, 9),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 8, 10),

        lexer.Token(lexer.TokenType.INDENT, '', 9, 1),
        lexer.Token(lexer.TokenType.RETURN, 'return', 9, 9),
        lexer.Token(lexer.TokenType.NUMBER, '2', 9, 16),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 9, 17),
        lexer.Token(lexer.TokenType.DEDENT, '', 10, 1),
        lexer.Token(lexer.TokenType.DEDENT, '', 10, 1),
        lexer.Token(lexer.TokenType.EOF, '', 10, 1),
    ]
    assert expected == res


def test_while_break_continue():
    res = lexer.lex('tests/while_break_continue.ht')
    expected = [
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
    assert expected == res


def test_none():
    res = lexer.lex('tests/none.ht')
    expected = [
        lexer.Token(lexer.TokenType.NONE, 'none', 1, 1),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 5),
        lexer.Token(lexer.TokenType.EOF, '', 2, 1),
    ]

    assert expected == res
