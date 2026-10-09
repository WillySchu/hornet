"""The parser's grammar as modules of functions over a TokenStream. Types and expressions need each
other (an array's size is an expression; a typed literal starts with a type), so each imports the
other whole: that must work whichever is loaded first."""

import subprocess
import sys
from pathlib import Path

import pytest

from lexer import Lexer
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
