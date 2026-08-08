"""Parser

Recursive-descent parser that turns the token stream produced by Lexer
into an Abstract Syntax Tree (AST).

Grammar supported (matches what the current lexer can produce):

    program     := function* EOF
    function    := 'def' type IDENTIFIER '(' ')' ':' NEWLINE statement+
    type        := 'int'
    statement   := return_stmt
    return_stmt := 'return' expression NEWLINE?
    expression   := binary_expr
    binary_expr  := unary_expr (BIN_OP binary_expr)*   [precedence climbing]
    unary_expr   := ('-' | '~' | '!') unary_expr
                   | primary_expr
    primary_expr := NUMBER
                   | '(' expression ')'

unary_expr is recursive (rather than a single optional prefix) so that
chained unary operators parse correctly -- e.g. `~-2`, `!!flag`, `--x`.
Each recurses on itself, which makes the operators right-associative:
`- - 2` parses as Unary(-, Unary(-, Constant(2))), i.e. "negate (negate
2)", which is the reading you'd expect.

binary_expr uses precedence climbing (see parse_binary and the
_BINARY_OPS table below) so operator precedence and associativity are
both driven by a small data table rather than by a cascade of
precedence-level methods (parse_additive/parse_multiplicative/etc). That
makes both properties -- including per-operator associativity, e.g. a
future right-associative exponentiation operator -- a one-line change
when a new operator shows up, instead of a restructuring.

primary_expr also accepts a parenthesized sub-expression. This wasn't
explicitly requested, but it's what lets precedence actually be
*overridden* (e.g. `(1 + 2) * 3`), which is the main reason anyone
reaches for explicit precedence handling in the first place -- without
it there'd be no way to test, or use, most of what precedence climbing
is for. It reuses the OPEN_PAREN/CLOSE_PAREN tokens already used for
function declarations; there's no ambiguity since the two uses occur in
disjoint grammar positions.

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


class BinaryOp(Enum):
    ADD = auto()       # '+'
    SUBTRACT = auto()  # '-'
    MULTIPLY = auto()  # '*'
    DIVIDE = auto()    # '/'

    def symbol(self) -> str:
        return {
            BinaryOp.ADD: '+',
            BinaryOp.SUBTRACT: '-',
            BinaryOp.MULTIPLY: '*',
            BinaryOp.DIVIDE: '/',
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
class Binary(Node):
    op: BinaryOp
    left: Node
    right: Node

    def pretty(self) -> str:
        # Binary has two children, so the linear "A -> B" chain style
        # used elsewhere (Return, Unary, Function) doesn't quite fit --
        # this bracketed form keeps it a single readable line while still
        # showing the tree shape, e.g. for `(1 + 2) * 3`:
        #   Binary(op: *) -> [Binary(op: +) -> [Constant(value: 1), Constant(value: 2)], Constant(value: 3)]
        return f"Binary(op: {self.op.symbol()}) -> [{self.left.pretty()}, {self.right.pretty()}]"


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


class Associativity(Enum):
    LEFT = auto()
    RIGHT = auto()


@dataclass(frozen=True)
class OperatorInfo:
    op: BinaryOp
    precedence: int
    associativity: Associativity


# TokenType -> parsing metadata for each binary (infix) operator: which
# BinaryOp it produces, its precedence (higher binds tighter), and
# whether it's left- or right-associative. parse_binary()'s
# precedence-climbing loop reads entirely from this table, so adding a
# new binary operator -- including a right-associative one -- is just
# adding a row here, not restructuring the parser.
#
# For example, right-associative exponentiation (so `2 ** 3 ** 2` parses
# as `2 ** (3 ** 2)`, not `(2 ** 3) ** 2`) would slot in above STAR/SLASH
# at a higher precedence once the lexer grows a token for it:
#
#   TokenType.STAR_STAR: OperatorInfo(BinaryOp.POWER, precedence=3, associativity=Associativity.RIGHT),
#
_BINARY_OPS = {
    TokenType.PLUS:  OperatorInfo(BinaryOp.ADD,      precedence=1, associativity=Associativity.LEFT),
    TokenType.MINUS: OperatorInfo(BinaryOp.SUBTRACT, precedence=1, associativity=Associativity.LEFT),
    TokenType.STAR:  OperatorInfo(BinaryOp.MULTIPLY, precedence=2, associativity=Associativity.LEFT),
    TokenType.SLASH: OperatorInfo(BinaryOp.DIVIDE,   precedence=2, associativity=Associativity.LEFT),
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
        return self.parse_binary()

    def parse_binary(self, min_prec: int = 0) -> Node:
        """Precedence-climbing parse of a binary expression.

        Starts with a single unary/primary operand, then repeatedly folds
        in further `operand OP operand` pairs as long as the next
        operator's precedence is high enough to bind at this level
        (`>= min_prec`).

        Associativity is entirely a matter of what minimum precedence
        gets passed to the recursive call that parses the right-hand
        side:
          - LEFT-associative operators recurse with `precedence + 1`,
            which stops that recursive call from also consuming another
            operator at the *same* precedence -- so that next operator
            instead gets picked up by *this* call's own loop, producing
            left-leaning nesting: `1 - 2 - 3` -> `(1 - 2) - 3`.
          - RIGHT-associative operators recurse with `precedence`
            unchanged, which lets that recursive call keep consuming
            further same-precedence operators itself, producing
            right-leaning nesting: `2 ^ 3 ^ 2` -> `2 ^ (3 ^ 2)`.
        """
        left = self.parse_unary()
        while True:
            op_info = _BINARY_OPS.get(self.current().type)
            if op_info is None or op_info.precedence < min_prec:
                break
            self.advance()  # consume the operator token
            next_min_prec = (
                op_info.precedence + 1
                if op_info.associativity == Associativity.LEFT
                else op_info.precedence
            )
            right = self.parse_binary(next_min_prec)
            left = Binary(op=op_info.op, left=left, right=right)
        return left

    def parse_unary(self) -> Node:
        if self.check(*_UNARY_OPS):
            op_tok = self.advance()
            # Recurse on parse_unary (not parse_primary) so operators
            # chain: `~-2` is COMPLEMENT applied to (NEGATE applied to 2).
            operand = self.parse_unary()
            return Unary(op=_UNARY_OPS[op_tok.type], operand=operand)
        return self.parse_primary()

    def parse_primary(self) -> Node:
        if self.check(TokenType.NUMBER):
            tok = self.advance()
            value = float(tok.val) if '.' in tok.val else int(tok.val)
            return Constant(value=value)
        if self.match(TokenType.OPEN_PAREN):
            expr = self.parse_expression()
            self.expect(TokenType.CLOSE_PAREN, "Expected ')' to close grouped expression")
            return expr
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
