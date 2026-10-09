"""The parser as one object: `Parser(tokens).parse_program()`.

The grammar itself is functions over a TokenStream, in the modules beside this one; this is the way
in for code that has a file's tokens and wants its tree."""

from typing import List

from lexer import Token
from parser import declarations
from parser.nodes import Program
from parser.stream import TokenStream


class Parser:
    def __init__(self, tokens: List[Token]):
        self.stream = TokenStream(tokens)

    def parse_program(self) -> Program:
        return declarations.parse_program(self.stream)
