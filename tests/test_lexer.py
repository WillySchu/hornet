"""Tests for the lexer."""

import lexer


def test_lex():
    res = lexer.lex('tests/return_2.gl')
    assert len(res) == 1
