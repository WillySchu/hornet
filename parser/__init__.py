"""The parser: a source file's tokens to its tree.

    nodes.py    the tree's node classes
    stream.py   the token stream: where the parser is in a file's tokens
    escapes.py  what a string or byte literal's text stands for
    errors.py   what it reports
    type_exprs.py    types as they are written
    expressions.py   expressions: operators, literals, calls
    parser.py   the rest of the grammar (statements, declarations), on its way to modules of their own

The entry points are here (parse, parse_tokens), with the names other code takes from the package.
"""

import argparse
from typing import List

from lexer import Token, lex
from ops import BinaryOp, UnaryOp
from parser import nodes
from parser.nodes import *  # noqa: F401,F403 (what nodes.__all__ lists: the node classes)
from parser.nodes import Program
from parser.errors import ParseError
from parser.expressions import READ_AS_A_TYPED_LITERAL
from parser.parser import Parser
from parser.stream import TokenStream

# What the package offers: the nodes, the parser and its error, the entry points below, and the two
# operator enums, which live in ops.py but which some code takes from here.
__all__ = [
    *nodes.__all__,
    'BinaryOp', 'UnaryOp', 'ParseError', 'Parser', 'TokenStream', 'READ_AS_A_TYPED_LITERAL', 'parse', 'parse_tokens',
    'main',
]


def parse_tokens(tokens: List[Token]) -> Program:
    return Parser(tokens).parse_program()


def parse(filename: str) -> Program:
    tokens = lex(filename)
    return parse_tokens(tokens)


def main():
    """The command line: `python -m parser FILE` (see __main__.py)."""
    arg_parser = argparse.ArgumentParser(description='Parser')
    arg_parser.add_argument('file', type=str, help='File to parse.')
    args = arg_parser.parse_args()
    ast = parse(args.file)
    print(ast.pretty())
