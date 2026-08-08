"""Parser

Recursive-descent parser that turns the token stream produced by Lexer
into an Abstract Syntax Tree (AST).

Grammar supported (matches what the current lexer can produce):

    program     := function* EOF
    function    := 'def' type IDENTIFIER '(' ')' ':' NEWLINE statement+
    type        := 'int'
    statement   := return_stmt
    return_stmt := 'return' expression NEWLINE?
    expression   := unary_expr
    unary_expr   := ('-' | '~' | '!') unary_expr
                   | primary_expr
    primary_expr := NUMBER

unary_expr is recursive (rather than a single optional prefix) so that
chained unary operators parse correctly -- e.g. `~-2`, `!!flag`, `--x`.
Each recurses on itself, which makes the operators right-associative:
`- - 2` parses as Unary(-, Unary(-, Constant(2))), i.e. "negate (negate
2)", which is the reading you'd expect.

NOTE ON INDENTATION
--------------------
The attached lexer's SKIP rule consumes all spaces/tabs -- including
leading indentation -- so no INDENT/DEDENT tokens ever reach the parser.
That means this parser currently has no reliable signal for "where a
block ends" other than "the next 'def' or end of file". That's enough to
correctly parse flat, single-block functions (like the example below),
but it will NOT correctly handle nested blocks (if/while/etc.) once those
are added to the language, since two nested blocks at different
indentation levels are indistinguishable from one flat block once
whitespace is stripped. When you're ready to add nested blocks, the
lexer will need to track indentation depth per line and emit INDENT/
DEDENT tokens (the classic Python-style approach), and parse_block()
below should be rewritten to consume those instead of scanning for 'def'.
"""

import argparse
from dataclasses import dataclass, field
from enum import auto, Enum
from typing import List, Union

from lexer import Token, TokenType, lex


# ---------------------------------------------------------------------------
# AST Nodes
# ---------------------------------------------------------------------------

class UnaryOp(Enum):
    NEGATE = auto()      # '-'  arithmetic negation
    COMPLEMENT = auto()  # '~'  bitwise complement
    NOT = auto()         # '!'  logical not

    def symbol(self) -> str:
        return {
            UnaryOp.NEGATE: '-',
            UnaryOp.COMPLEMENT: '~',
            UnaryOp.NOT: '!',
        }[self]


class Node:
    """Base class for all AST nodes."""

    def pretty(self) -> str:
        raise NotImplementedError


@dataclass
class Constant(Node):
    value: Union[int, float]

    def pretty(self) -> str:
        return f"Constant(value: {self.value})"


@dataclass
class Unary(Node):
    op: UnaryOp
    operand: Node

    def pretty(self) -> str:
        return f"Unary(op: {self.op.symbol()}) -> {self.operand.pretty()}"


@dataclass
class Return(Node):
    value: Node

    def pretty(self) -> str:
        return f"Return -> {self.value.pretty()}"


@dataclass
class Function(Node):
    name: str
    return_type: str
    body: List[Node] = field(default_factory=list)

    def pretty(self) -> str:
        body_str = ' -> '.join(stmt.pretty() for stmt in self.body)
        return f"Function(name: {self.name}) -> {body_str}"


@dataclass
class Program(Node):
    functions: List[Function] = field(default_factory=list)

    def pretty(self) -> str:
        return '\n'.join(f"Program -> {fn.pretty()}" for fn in self.functions)

    def __repr__(self) -> str:
        return self.pretty()


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class ParseError(Exception):
    """Raised when the parser encounters unexpected or malformed input."""


# Maps a prefix-operator token straight to the UnaryOp it represents.
_UNARY_OPS = {
    TokenType.MINUS: UnaryOp.NEGATE,
    TokenType.TILDE: UnaryOp.COMPLEMENT,
    TokenType.BANG: UnaryOp.NOT,
}


class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    # -- token helpers --------------------------------------------------

    def peek(self, offset: int = 0) -> Token:
        idx = min(self.pos + offset, len(self.tokens) - 1)
        return self.tokens[idx]

    def current(self) -> Token:
        return self.peek()

    def at_end(self) -> bool:
        return self.current().type == TokenType.EOF

    def check(self, *types: TokenType) -> bool:
        return not self.at_end() and self.current().type in types

    def advance(self) -> Token:
        tok = self.current()
        if not self.at_end():
            self.pos += 1
        return tok

    def match(self, *types: TokenType) -> bool:
        if self.check(*types):
            self.advance()
            return True
        return False

    def expect(self, type_: TokenType, message: str = None) -> Token:
        if self.check(type_):
            return self.advance()
        tok = self.current()
        msg = message or f"Expected {type_}, got {tok.type} ('{tok.val}')"
        raise ParseError(f"{msg} at line {tok.line}, column {tok.col}")

    def skip_newlines(self):
        while self.match(TokenType.NEWLINE):
            pass

    # -- grammar rules ----------------------------------------------------

    def parse_program(self) -> Program:
        functions = []
        self.skip_newlines()
        while not self.at_end():
            functions.append(self.parse_function())
            self.skip_newlines()
        return Program(functions=functions)

    def parse_function(self) -> Function:
        self.expect(TokenType.DEF, "Expected 'def' to start a function definition")
        return_type = self.parse_type()
        name_tok = self.expect(TokenType.IDENTIFIER, "Expected a function name")
        self.expect(TokenType.OPEN_PAREN, "Expected '(' after function name")
        # No parameter support yet -- the lexer has no comma token, so
        # anything beyond an empty parameter list can't be produced.
        self.expect(TokenType.CLOSE_PAREN, "Expected ')' after parameter list")
        self.expect(TokenType.COLON, "Expected ':' to start the function body")
        self.expect(TokenType.NEWLINE, "Expected a newline after ':'")

        body = self.parse_block()
        return Function(name=name_tok.val, return_type=return_type, body=body)

    def parse_type(self) -> str:
        # Only 'int' is a recognized type for now.
        tok = self.expect(TokenType.INT, "Expected a type (e.g. 'int')")
        return tok.val

    def parse_block(self) -> List[Node]:
        """Parse statements until the next 'def' (a new function) or EOF.

        See the module docstring for why this -- rather than proper
        INDENT/DEDENT tracking -- is how block boundaries are detected.
        """
        statements = []
        self.skip_newlines()
        while not self.at_end() and not self.check(TokenType.DEF):
            statements.append(self.parse_statement())
            self.skip_newlines()
        if not statements:
            tok = self.current()
            raise ParseError(
                f"Expected at least one statement in function body "
                f"at line {tok.line}, column {tok.col}"
            )
        return statements

    def parse_statement(self) -> Node:
        if self.check(TokenType.RETURN):
            return self.parse_return()
        tok = self.current()
        raise ParseError(
            f"Unexpected token {tok.type} ('{tok.val}') at line {tok.line}, "
            f"column {tok.col}"
        )

    def parse_return(self) -> Return:
        self.expect(TokenType.RETURN)
        value = self.parse_expression()
        return Return(value=value)

    def parse_expression(self) -> Node:
        return self.parse_unary()

    def parse_unary(self) -> Node:
        if self.check(*_UNARY_OPS):
            op_tok = self.advance()
            # Recurse on parse_unary (not parse_primary) so operators
            # chain: `~-2` is COMPLEMENT applied to (NEGATE applied to 2).
            operand = self.parse_unary()
            return Unary(op=_UNARY_OPS[op_tok.type], operand=operand)
        return self.parse_primary()

    def parse_primary(self) -> Node:
        # Only numeric constants are supported for now -- the lexer
        # doesn't yet emit binary operators or parens-as-grouping, so
        # there's nothing else a primary expression could be.
        if self.check(TokenType.NUMBER):
            tok = self.advance()
            value = float(tok.val) if '.' in tok.val else int(tok.val)
            return Constant(value=value)
        tok = self.current()
        raise ParseError(
            f"Expected an expression, got {tok.type} ('{tok.val}') "
            f"at line {tok.line}, column {tok.col}"
        )


# ---------------------------------------------------------------------------
# Convenience entry points
# ---------------------------------------------------------------------------

def parse_tokens(tokens: List[Token]) -> Program:
    return Parser(tokens).parse_program()


def parse(filename: str) -> Program:
    tokens = lex(filename)
    return parse_tokens(tokens)


def main():
    arg_parser = argparse.ArgumentParser(description='Parser')
    arg_parser.add_argument('file', type=str, help='File to parse.')
    args = arg_parser.parse_args()
    ast = parse(args.file)
    print(ast.pretty())


if __name__ == '__main__':
    main()
