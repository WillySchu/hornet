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
        lexer.Token(lexer.TokenType.MINUS, '-', 2, 12),
        lexer.Token(lexer.TokenType.NUMBER, '2', 2, 13),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 14),
        lexer.Token(lexer.TokenType.EOF, '', 3, 1),
    ]
    assert expected == res


def test_lex_bitwise_complement():
    res = lexer.lex('tests/bitwise_complement.ht')
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
        lexer.Token(lexer.TokenType.TILDE, '~', 2, 12),
        lexer.Token(lexer.TokenType.NUMBER, '2', 2, 13),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 14),
        lexer.Token(lexer.TokenType.EOF, '', 3, 1),
    ]
    assert expected == res


def test_lex_bitwise_complement():
    res = lexer.lex('tests/logical_negation.ht')
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
        lexer.Token(lexer.TokenType.BANG, '!', 2, 12),
        lexer.Token(lexer.TokenType.NUMBER, '2', 2, 13),
        lexer.Token(lexer.TokenType.NEWLINE, '\n', 2, 14),
        lexer.Token(lexer.TokenType.EOF, '', 3, 1),
    ]
    assert expected == res
