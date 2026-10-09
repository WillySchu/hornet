"""The parser's grammar as modules of functions over a TokenStream. Types and expressions need each
other (an array's size is an expression; a typed literal starts with a type), so each imports the
other whole: that must work whichever is loaded first."""

import subprocess
import sys
from pathlib import Path

import pytest

from lexer import Lexer, TokenType
from parser import ArrayLiteral, ArrayTypeExpr, Binary, BinaryOp, Constant, IsCheck, TokenStream, Variable
from parser import expressions, type_exprs

ROOT = Path(__file__).resolve().parent.parent


def _stream(source: str) -> TokenStream:
    return TokenStream(Lexer(source, 'f.ht').tokenize())


@pytest.mark.parametrize("first", ["parser.type_exprs", "parser.expressions", "parser", "parser.parser"])
def test_the_modules_load_whichever_comes_first(first):
    code = f"import {first}; import parser.expressions, parser.type_exprs; print('loaded')"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
    assert (result.returncode, result.stdout, result.stderr) == (0, "loaded\n", "")


def test_a_type_s_array_size_is_an_expression():
    parsed = type_exprs.parse_type(_stream("[N + 1]int"))
    assert isinstance(parsed, ArrayTypeExpr) and parsed.element_type == 'int'
    assert isinstance(parsed.size, Binary) and parsed.size.op == BinaryOp.ADD        # read by expressions.py
    assert type_exprs.parse_type(_stream("[3][]*P")) == ArrayTypeExpr(
        size=3, element_type=type_exprs.parse_type(_stream("[]*P")))


def test_an_expression_may_start_with_or_name_a_type():
    literal = expressions.parse_expression(_stream("[2]int[1, n]"))
    assert isinstance(literal, ArrayLiteral) and literal.type_expr == ArrayTypeExpr(size=2, element_type='int')
    assert [type(e) for e in literal.elements] == [Constant, Variable]              # its type read by type_exprs.py
    check = expressions.parse_expression(_stream("shape is []int"))
    assert isinstance(check, IsCheck) and check.type_name == type_exprs.parse_type(_stream("[]int"))


def test_a_function_leaves_the_stream_after_what_it_read():
    stream = _stream("a + b * 2, rest")
    expressions.parse_expression(stream)
    assert stream.current().val == ',' and stream.pos == 5
    stream = _stream("*[2]int x")
    type_exprs.parse_type(stream)
    assert stream.current().val == 'x'


def test_a_statement_is_told_from_its_first_tokens():
    from parser import Assign, ExprStmt, If, Unary, VarDecl, statements
    assert isinstance(statements.parse_statement(_stream("int n = 1\n")), VarDecl)
    assert isinstance(statements.parse_statement(_stream("n = 1\n")), Assign)
    assert isinstance(statements.parse_statement(_stream("f(n)\n")), ExprStmt)
    assert isinstance(statements.parse_statement(_stream("if n > 0:\n    n = 1\n")), If)
    # A leading `*` is tried as a pointer type first, and taken back if no name follows it.
    declaration = statements.parse_statement(_stream("*P q = none\n"))
    assert isinstance(declaration, VarDecl) and declaration.name == 'q'
    stream = _stream("*q = 5\n")
    assignment = statements.parse_statement(stream)
    assert isinstance(assignment, Assign) and isinstance(assignment.target, Unary)
    assert stream.current().type == TokenType.NEWLINE      # read from the start again, and to the line's end
