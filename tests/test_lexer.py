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
    #assert res[8].col == 0
    expected = [
        lexer.Token(lexer.TokenType.DEF, 'def', 1, 1),
        lexer.Token(lexer.TokenType.INT, 'int', 1, 5),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'main', 1, 9),
        lexer.Token(lexer.TokenType.OPEN_PAREN, '(', 1, 13),
        lexer.Token(lexer.TokenType.CLOSE_PAREN, ')', 1, 14),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 15),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 16),
        lexer.Token(lexer.TokenType.RETURN, 'return', 2, 5),
        lexer.Token(lexer.TokenType.NUMBER, '2', 2, 12),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 13),
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
        lexer.Token(lexer.TokenType.RETURN, 'return', 2, 5),
        lexer.Token(lexer.TokenType.MINUS, '-', 2, 12),
        lexer.Token(lexer.TokenType.NUMBER, '2', 2, 13),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 14),
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
        lexer.Token(lexer.TokenType.RETURN, 'return', 2, 5),
        lexer.Token(lexer.TokenType.TILDE, '~', 2, 12),
        lexer.Token(lexer.TokenType.NUMBER, '2', 2, 13),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 14),
        lexer.Token(lexer.TokenType.EOF, '', 3, 1),
    ]
    assert expected == res


def test_lex_logical_negation():
    res = lexer.lex('tests/logical_negation.ht')
    expected = [
        lexer.Token(lexer.TokenType.DEF, 'def', 1, 1),
        lexer.Token(lexer.TokenType.INT, 'int', 1, 5),
        lexer.Token(lexer.TokenType.IDENTIFIER, 'main', 1, 9),
        lexer.Token(lexer.TokenType.OPEN_PAREN, '(', 1, 13),
        lexer.Token(lexer.TokenType.CLOSE_PAREN, ')', 1, 14),
        lexer.Token(lexer.TokenType.COLON, ':', 1, 15),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 1, 16),
        lexer.Token(lexer.TokenType.RETURN, 'return', 2, 5),
        lexer.Token(lexer.TokenType.BANG, '!', 2, 12),
        lexer.Token(lexer.TokenType.NUMBER, '2', 2, 13),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 14),
        lexer.Token(lexer.TokenType.EOF, '', 3, 1),
    ]
    assert expected == res

# Binary Ops.

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
        lexer.Token(lexer.TokenType.RETURN, 'return', 2, 5),
        lexer.Token(lexer.TokenType.NUMBER, '2', 2, 12),
        lexer.Token(lexer.TokenType.PLUS, '+', 2, 14),
        lexer.Token(lexer.TokenType.NUMBER, '3', 2, 16),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 17),
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
        lexer.Token(lexer.TokenType.RETURN, 'return', 2, 5),
        lexer.Token(lexer.TokenType.NUMBER, '2', 2, 12),
        lexer.Token(lexer.TokenType.MINUS, '-', 2, 14),
        lexer.Token(lexer.TokenType.NUMBER, '3', 2, 16),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 17),
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
        lexer.Token(lexer.TokenType.RETURN, 'return', 2, 5),
        lexer.Token(lexer.TokenType.NUMBER, '2', 2, 12),
        lexer.Token(lexer.TokenType.STAR, '*', 2, 14),
        lexer.Token(lexer.TokenType.NUMBER, '3', 2, 16),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 17),
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
        lexer.Token(lexer.TokenType.RETURN, 'return', 2, 5),
        lexer.Token(lexer.TokenType.NUMBER, '2', 2, 12),
        lexer.Token(lexer.TokenType.SLASH, '/', 2, 14),
        lexer.Token(lexer.TokenType.NUMBER, '3', 2, 16),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 17),
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
        lexer.Token(lexer.TokenType.RETURN, 'return', 2, 5),
        lexer.Token(lexer.TokenType.NUMBER, '2', 2, 12),
        lexer.Token(lexer.TokenType.EQUAL, '==', 2, 14),
        lexer.Token(lexer.TokenType.NUMBER, '3', 2, 17),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 18),
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
        lexer.Token(lexer.TokenType.RETURN, 'return', 2, 5),
        lexer.Token(lexer.TokenType.NUMBER, '2', 2, 12),
        lexer.Token(lexer.TokenType.NOT_EQUAL, '!=', 2, 14),
        lexer.Token(lexer.TokenType.NUMBER, '3', 2, 17),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 18),
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
        lexer.Token(lexer.TokenType.RETURN, 'return', 2, 5),
        lexer.Token(lexer.TokenType.NUMBER, '2', 2, 12),
        lexer.Token(lexer.TokenType.LESS_THAN, '<', 2, 14),
        lexer.Token(lexer.TokenType.NUMBER, '3', 2, 16),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 17),
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
        lexer.Token(lexer.TokenType.RETURN, 'return', 2, 5),
        lexer.Token(lexer.TokenType.NUMBER, '2', 2, 12),
        lexer.Token(lexer.TokenType.GREATER_THAN, '>', 2, 14),
        lexer.Token(lexer.TokenType.NUMBER, '3', 2, 16),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 17),
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
        lexer.Token(lexer.TokenType.RETURN, 'return', 2, 5),
        lexer.Token(lexer.TokenType.NUMBER, '2', 2, 12),
        lexer.Token(lexer.TokenType.LESS_THAN_OR_EQUAL, '<=', 2, 14),
        lexer.Token(lexer.TokenType.NUMBER, '3', 2, 17),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 18),
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
        lexer.Token(lexer.TokenType.RETURN, 'return', 2, 5),
        lexer.Token(lexer.TokenType.NUMBER, '2', 2, 12),
        lexer.Token(lexer.TokenType.GREATER_THAN_OR_EQUAL, '>=', 2, 14),
        lexer.Token(lexer.TokenType.NUMBER, '3', 2, 17),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 18),
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
        lexer.Token(lexer.TokenType.RETURN, 'return', 2, 5),
        lexer.Token(lexer.TokenType.NOT, 'not', 2, 12),
        lexer.Token(lexer.TokenType.NUMBER, '2', 2, 16),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 17),
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
        lexer.Token(lexer.TokenType.RETURN, 'return', 2, 5),
        lexer.Token(lexer.TokenType.NUMBER, '0', 2, 12),
        lexer.Token(lexer.TokenType.AND, 'and', 2, 14),
        lexer.Token(lexer.TokenType.NUMBER, '1', 2, 18),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 19),
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
        lexer.Token(lexer.TokenType.RETURN, 'return', 2, 5),
        lexer.Token(lexer.TokenType.NUMBER, '0', 2, 12),
        lexer.Token(lexer.TokenType.OR, 'or', 2, 14),
        lexer.Token(lexer.TokenType.NUMBER, '1', 2, 17),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 18),
        lexer.Token(lexer.TokenType.EOF, '', 3, 1),
    ]
    assert expected == res
