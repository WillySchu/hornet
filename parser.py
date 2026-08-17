"""Parser

Recursive-descent parser that turns the token stream produced by Lexer
into an Abstract Syntax Tree (AST).

Grammar supported (matches what the current lexer can produce):

    program     := function* EOF
    function    := 'def' type IDENTIFIER '(' ')' ':' NEWLINE block
    type        := 'int' | 'bool'
    block       := INDENT statement+ DEDENT
    statement   := decl_stmt
                 | assign_stmt
                 | return_stmt
                 | if_stmt
                 | while_stmt
                 | break_stmt
                 | continue_stmt
                 | expr_stmt
    decl_stmt   := type IDENTIFIER ('=' expression)? NEWLINE
    assign_stmt := IDENTIFIER '=' expression NEWLINE
    return_stmt := 'return' expression NEWLINE?
    if_stmt     := 'if' expression ':' NEWLINE block
                    ('elif' expression ':' NEWLINE block)*
                    ('else' ':' NEWLINE block)?
    while_stmt   := 'while' expression ':' NEWLINE block
    break_stmt   := 'break' NEWLINE?
    continue_stmt := 'continue' NEWLINE?
    expr_stmt   := expression NEWLINE
    expression   := binary_expr
    binary_expr  := unary_expr (BIN_OP binary_expr)*   [precedence climbing]
    unary_expr   := ('-' | '~' | 'not') unary_expr
                   | primary_expr
    primary_expr := NUMBER
                   | 'true' | 'false'
                   | IDENTIFIER
                   | '(' expression ')'

unary_expr is recursive (rather than a single optional prefix) so that
chained unary operators parse correctly -- e.g. `~-2`, `not not flag`,
`--x`. Each recurses on itself, which makes the operators
right-associative: `- - 2` parses as Unary(-, Unary(-, Constant(2))),
i.e. "negate (negate 2)", which is the reading you'd expect.

binary_expr uses precedence climbing (see parse_binary and the
_BINARY_OPS table below) so operator precedence and associativity are
both driven by a small data table rather than by a cascade of
precedence-level methods (parse_additive/parse_multiplicative/etc). That
makes both properties -- including per-operator associativity, e.g. a
future right-associative exponentiation operator -- a one-line change
when a new operator shows up, instead of a restructuring.

Operators currently supported by binary_expr, tightest to loosest
binding: `* / %` , then `+ -` , then `<< >>` , then `< > <= >=` , then
`== !=` , then `&` , then `^` , then `|` , then `and` , then `or`. This
is the classic C precedence ladder (see the comment above _BINARY_OPS
for why it's worth keeping even though it reproduces one of C's better-
known surprises around bitwise operators and equality). All are
left-associative -- including 'and'/'or', which matters beyond just
parenthesization: parsing `a and b and c` left-associatively as `(a and
b) and c` is what makes chained short-circuiting fall out naturally in
codegen (see codegen.py), since evaluating the outer node left-to-right
evaluates `a and b` first and only reaches `c` if that was true.

primary_expr also accepts a parenthesized sub-expression. This wasn't
explicitly requested, but it's what lets precedence actually be
*overridden* (e.g. `(1 + 2) * 3`), which is the main reason anyone
reaches for explicit precedence handling in the first place -- without
it there'd be no way to test, or use, most of what precedence climbing
is for. It reuses the OPEN_PAREN/CLOSE_PAREN tokens already used for
function declarations; there's no ambiguity since the two uses occur in
disjoint grammar positions.

STATEMENT DISPATCH
-------------------
decl_stmt, assign_stmt, and expr_stmt all start with a token that could,
on its own, also start something else -- INT also starts a function's
return type, IDENTIFIER also starts any expression that happens to begin
with a variable, and so on. parse_statement resolves this with one
token of lookahead (peek(1)) rather than backtracking: INT always means
a declaration (a type can't appear anywhere else a statement could
start), but IDENTIFIER is ambiguous between "the start of an assignment"
(`a = ...`) and "the start of some other expression that just happens to
reference `a`" (`a + 1`, or `a` alone) -- so parse_statement peeks one
token ahead and only commits to assign_stmt if that next token is
ASSIGN, falling through to expr_stmt otherwise.

Declaring a variable twice in the same function, referencing one that
was never declared, and every type error (mismatched assignment,
wrong-typed operands, a `return` that doesn't match the function's
declared type, ...) are all caught by a dedicated semantic-analysis pass
(see semantic.py) that runs after parsing and before codegen -- not by
the parser itself, and not deferred to codegen anymore either. The
parser's job stays purely syntactic: it doesn't know or care whether
`int` or `bool` is the "right" type for `var_type` -- it just requires
one of the two type keywords and hands the resulting string on.

One thing that changes as a side effect: semantic.py walks a function's
statements in program order, building up its scope as it goes (see its
module docstring), rather than pre-scanning the whole body up front the
way codegen.py's stack-layout pass does. So declare-before-use in
textual order is now actually enforced -- `a = 1` above `int a` is a
semantic error -- even though codegen's own pre-scan (which only cares
about sizing the stack frame, not validity) still wouldn't catch it on
its own if it somehow ran without semantic analysis first.

NOTE ON INDENTATION (RESOLVED)
--------------------------------
Earlier versions of this parser had no reliable signal for "where a
block ends" beyond "the next 'def' or end of file", because the lexer's
SKIP rule swallowed all whitespace uniformly, including leading
indentation. That was fine as long as every block was flat, but it
could never work for nested blocks (if/while/etc.), since two blocks at
different indentation levels are indistinguishable from one flat block
once whitespace is stripped -- this is exactly the limitation that
blocked `if` statements from being addable at all until now.

The lexer now tracks an indentation stack and synthesizes INDENT/DEDENT
tokens (the classic Python-style approach; see lexer.py's tokenize()).
parse_block() consumes exactly `INDENT statement+ DEDENT`, and -- unlike
the old "scan for def" hack -- this generalizes to any nesting depth for
free: a function's top-level body and an if/elif/else's body both go
through the same parse_block(), so an if nested inside an if nested
inside a function just falls out of ordinary recursion.

NOTE ON '!' -> 'not' (RESOLVED)
----------------------------------
Logical NOT used to be spelled `!` (the BANG token). The lexer has since
dropped BANG entirely -- there's no single-character `!` token anymore,
only the two-character `!=` (NOT_EQUAL, a completely separate token that
was never affected by this) -- and logical NOT is now spelled with the
`not` keyword instead (the NOT token, matched via the ordinary
IDENTIFIER-then-keyword-lookup path, same as `and`/`or`/`return`/etc.).
_UNARY_OPS below maps TokenType.NOT (not BANG) to UnaryOp.NOT; nothing
else about how NOT parses changed -- it's still an ordinary prefix
unary operator, chainable the same way `~`/`-` are (`not not flag`).
"""

import argparse
from dataclasses import dataclass, field
from enum import auto, Enum
from typing import List, Optional, Union

from lexer import Token, TokenType, lex


# ---------------------------------------------------------------------------
# AST Nodes
# ---------------------------------------------------------------------------

class UnaryOp(Enum):
    NEGATE = auto()      # '-'    arithmetic negation
    COMPLEMENT = auto()  # '~'    bitwise complement
    NOT = auto()         # 'not'  logical not

    def symbol(self) -> str:
        return {
            UnaryOp.NEGATE: '-',
            UnaryOp.COMPLEMENT: '~',
            UnaryOp.NOT: 'not',
        }[self]


class BinaryOp(Enum):
    ADD = auto()       # '+'
    SUBTRACT = auto()  # '-'
    MULTIPLY = auto()  # '*'
    DIVIDE = auto()    # '/'
    MODULO = auto()    # '%'

    SHIFT_LEFT = auto()   # '<<'
    SHIFT_RIGHT = auto()  # '>>'

    LESS_THAN = auto()              # '<'
    GREATER_THAN = auto()           # '>'
    LESS_THAN_OR_EQUAL = auto()     # '<='
    GREATER_THAN_OR_EQUAL = auto()  # '>='

    EQUAL = auto()      # '=='
    NOT_EQUAL = auto()  # '!='

    BITWISE_AND = auto()  # '&'
    BITWISE_XOR = auto()  # '^'
    BITWISE_OR = auto()   # '|'

    AND = auto()  # 'and'
    OR = auto()   # 'or'

    def symbol(self) -> str:
        return {
            BinaryOp.ADD: '+',
            BinaryOp.SUBTRACT: '-',
            BinaryOp.MULTIPLY: '*',
            BinaryOp.DIVIDE: '/',
            BinaryOp.MODULO: '%',
            BinaryOp.SHIFT_LEFT: '<<',
            BinaryOp.SHIFT_RIGHT: '>>',
            BinaryOp.LESS_THAN: '<',
            BinaryOp.GREATER_THAN: '>',
            BinaryOp.LESS_THAN_OR_EQUAL: '<=',
            BinaryOp.GREATER_THAN_OR_EQUAL: '>=',
            BinaryOp.EQUAL: '==',
            BinaryOp.NOT_EQUAL: '!=',
            BinaryOp.BITWISE_AND: '&',
            BinaryOp.BITWISE_XOR: '^',
            BinaryOp.BITWISE_OR: '|',
            BinaryOp.AND: 'and',
            BinaryOp.OR: 'or',
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
class BoolLiteral(Node):
    """`true` or `false`. Kept as its own node rather than folded into
    Constant -- Python's bool is a subclass of int, and overloading
    Constant.value to sometimes hold one would make "is this actually an
    int or a bool" ambiguous exactly where semantic.py most needs it to
    be unambiguous."""
    value: bool

    def pretty(self) -> str:
        return f"BoolLiteral(value: {'true' if self.value else 'false'})"


@dataclass
class StringLiteral(Node):
    """`'...'`. `value` holds the string's *actual* content -- quotes
    already stripped and escape sequences already resolved (`\\'` -> `'`,
    `\\n` -> a real newline, etc.) by parse_primary, not the raw source
    text. That mirrors how Constant already works for numbers (the
    parser turns `tok.val` -- the raw '2' or '2.5' text -- into a real
    Python int/float once, rather than every downstream pass re-parsing
    the source string itself)."""
    value: str

    def pretty(self) -> str:
        return f"StringLiteral(value: {self.value!r})"


@dataclass
class Variable(Node):
    """A reference to a local variable, e.g. the `a` in `a + 1`."""
    name: str

    def pretty(self) -> str:
        return f"Variable(name: {self.name})"


@dataclass
class Call(Node):
    """`name(arg1, arg2, ...)` -- a function call, used as an ordinary
    expression. There's no separate "call statement" concept: `foo(1)`
    alone on its own line already parses as an expr_stmt wrapping this
    (see ExprStmt), exactly the same way any other expression can be a
    bare statement -- a call just happens to be one whose value people
    often want to discard."""
    name: str
    args: List[Node] = field(default_factory=list)

    def pretty(self) -> str:
        args_str = ', '.join(a.pretty() for a in self.args)
        return f"Call(name: {self.name}) -> [{args_str}]"


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
class VarDecl(Node):
    """`int a` (init=None) or `int a = 1` (init=the initializer expression)."""
    name: str
    var_type: str
    init: Optional[Node] = None

    def pretty(self) -> str:
        head = f"VarDecl(name: {self.name}, type: {self.var_type})"
        return head if self.init is None else f"{head} -> {self.init.pretty()}"


@dataclass
class Assign(Node):
    """`a = <value>`, assigning to an already-declared variable."""
    name: str
    value: Node

    def pretty(self) -> str:
        return f"Assign(name: {self.name}) -> {self.value.pretty()}"


@dataclass
class ExprStmt(Node):
    """A bare expression used as a full statement, e.g. `2 + 2` on its
    own line -- evaluated for any side effects (there are none yet, but
    this is also how a future function-call-as-statement would look) and
    then discarded."""
    expr: Node

    def pretty(self) -> str:
        return f"ExprStmt -> {self.expr.pretty()}"


@dataclass
class If(Node):
    """`if cond: <then_body> [elif cond: ...]* [else: <else_body>]?`.

    An `elif` isn't its own AST concept -- parse_if desugars it into a
    single-element else_body containing one more If node, i.e. `elif c:
    b` is represented exactly like `else: if c: b`. That means
    else_body is always just `Optional[List[Node]]`, the same shape as
    then_body, whether it came from a real `else` block or a chain of
    elifs -- semantic.py and codegen.py both consume it uniformly and
    never need to know which case they're looking at, and an elif chain
    of any length falls out of ordinary nesting rather than needing
    dedicated handling anywhere else in the pipeline.
    """
    condition: Node
    then_body: List[Node]
    else_body: Optional[List[Node]] = None

    def pretty(self) -> str:
        then_str = '; '.join(stmt.pretty() for stmt in self.then_body)
        head = f"If({self.condition.pretty()}) -> [{then_str}]"
        if self.else_body is None:
            return head
        else_str = '; '.join(stmt.pretty() for stmt in self.else_body)
        return f"{head} else -> [{else_str}]"


@dataclass
class While(Node):
    """`while cond: <body>`. Loops as long as `cond` evaluates to true,
    re-checking it before every iteration including the first (so a
    false condition means the body never runs at all)."""
    condition: Node
    body: List[Node]

    def pretty(self) -> str:
        body_str = '; '.join(stmt.pretty() for stmt in self.body)
        return f"While({self.condition.pretty()}) -> [{body_str}]"


@dataclass
class Break(Node):
    """`break` -- exits the *innermost* enclosing loop immediately.
    Only valid inside a while body; semantic.py rejects one that isn't
    (see its loop_depth tracking)."""

    def pretty(self) -> str:
        return "Break"


@dataclass
class Continue(Node):
    """`continue` -- skips the rest of the current iteration of the
    *innermost* enclosing loop and jumps straight to re-checking its
    condition. Same "only valid inside a while" rule as Break."""

    def pretty(self) -> str:
        return "Continue"


@dataclass
class Param(Node):
    """A single `type name` entry in a function's parameter list, e.g.
    the `int a` in `def int add(int a, int b):`. Not part of the
    function's own statement/expression tree -- it's a declaration
    record attached to Function, conceptually the same role VarDecl
    plays for a local, just without an initializer (a parameter's
    "initial value" is whatever the caller passed)."""
    name: str
    type: str

    def pretty(self) -> str:
        return f"Param(name: {self.name}, type: {self.type})"


@dataclass
class Function(Node):
    name: str
    return_type: str
    params: List[Param] = field(default_factory=list)
    body: List[Node] = field(default_factory=list)

    def pretty(self) -> str:
        # Statements are joined with "; " rather than the "->" used
        # everywhere else, specifically so statement boundaries stay
        # visually distinct from the "->" each statement already uses
        # internally for its own children (e.g. VarDecl's initializer).
        # With a single Return per function this was never ambiguous;
        # with multiple statements it would be.
        params_str = ', '.join(p.pretty() for p in self.params)
        body_str = '; '.join(stmt.pretty() for stmt in self.body)
        return f"Function(name: {self.name}, params: [{params_str}]) -> {body_str}"


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


# Escape sequences recognized inside a STRING literal's raw text. Keyed
# by the character *after* the backslash.
_ESCAPE_SEQUENCES = {
    'n': '\n',
    't': '\t',
    'r': '\r',
    '0': '\0',
    "'": "'",
    '"': '"',
    '\\': '\\',
}


def _unescape_string_literal(raw: str) -> str:
    """Converts a STRING token's raw text (still including its
    surrounding quotes, e.g. the 7 characters `'it\\'s'`) into the
    string's actual content: quotes stripped, and each backslash escape
    resolved via _ESCAPE_SEQUENCES.

    This mirrors how Constant's value is only ever computed once, here
    in the parser, rather than every later pass re-parsing `tok.val`
    itself -- codegen.py just needs the real bytes to embed as a
    string constant, not the original source syntax for them.

    An escape not in the table (`\\x` for some `x` the lexer's own
    STRING regex still happily matched, since it accepts a backslash
    followed by *any* single character) is treated leniently: the
    backslash is dropped and `x` is kept as-is, rather than raising.
    """
    inner = raw[1:-1]  # strip the surrounding single quotes
    chars = []
    i = 0
    while i < len(inner):
        ch = inner[i]
        if ch == '\\' and i + 1 < len(inner):
            nxt = inner[i + 1]
            chars.append(_ESCAPE_SEQUENCES.get(nxt, nxt))
            i += 2
        else:
            chars.append(ch)
            i += 1
    return ''.join(chars)


# Maps a prefix-operator token straight to the UnaryOp it represents.
_UNARY_OPS = {
    TokenType.MINUS: UnaryOp.NEGATE,
    TokenType.TILDE: UnaryOp.COMPLEMENT,
    TokenType.NOT: UnaryOp.NOT,
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
# Precedence levels, tightest to loosest -- this is the classic C
# ladder (https://en.cppreference.com/w/c/language/operator_precedence),
# adopted deliberately rather than invented fresh, since it's what
# anyone coming from a C-family language already expects:
#   10: *  /  %
#    9: +  -
#    8: <<  >>
#    7: <  >  <=  >=
#    6: ==  !=
#    5: &
#    4: ^
#    3: |
#    2: and
#    1: or
#
# Putting the bitwise operators *below* equality (rather than, say,
# right next to the other int-only arithmetic operators) reproduces a
# well-known C surprise: `a & b == c` parses as `a & (b == c)`, not
# `(a & b) == c`, because == binds tighter than &. In C that's a classic
# footgun -- it silently compiles into something you probably didn't
# mean. Here it isn't: `b == c` is bool, `&` requires int, so
# semantic.py rejects it outright as a type error rather than silently
# accepting the "wrong" grouping. Strong typing turns a subtle
# precedence trap into a compile error -- see TestSemanticErrors'
# test_bitwise_and_equality_precedence_is_a_type_error for exactly this
# case.
#
# For example, right-associative exponentiation (so `2 ** 3 ** 2` parses
# as `2 ** (3 ** 2)`, not `(2 ** 3) ** 2`) would slot in above STAR/SLASH
# at a higher precedence once the lexer grows a token for it:
#
#   TokenType.STAR_STAR: OperatorInfo(BinaryOp.POWER, precedence=11, associativity=Associativity.RIGHT),
#
_BINARY_OPS = {
    TokenType.STAR:    OperatorInfo(BinaryOp.MULTIPLY, precedence=10, associativity=Associativity.LEFT),
    TokenType.SLASH:   OperatorInfo(BinaryOp.DIVIDE,   precedence=10, associativity=Associativity.LEFT),
    TokenType.PERCENT: OperatorInfo(BinaryOp.MODULO,   precedence=10, associativity=Associativity.LEFT),

    TokenType.PLUS:  OperatorInfo(BinaryOp.ADD,      precedence=9, associativity=Associativity.LEFT),
    TokenType.MINUS: OperatorInfo(BinaryOp.SUBTRACT, precedence=9, associativity=Associativity.LEFT),

    TokenType.SHIFT_LEFT:  OperatorInfo(BinaryOp.SHIFT_LEFT,  precedence=8, associativity=Associativity.LEFT),
    TokenType.SHIFT_RIGHT: OperatorInfo(BinaryOp.SHIFT_RIGHT, precedence=8, associativity=Associativity.LEFT),

    TokenType.LESS_THAN:             OperatorInfo(BinaryOp.LESS_THAN,             precedence=7, associativity=Associativity.LEFT),
    TokenType.GREATER_THAN:          OperatorInfo(BinaryOp.GREATER_THAN,          precedence=7, associativity=Associativity.LEFT),
    TokenType.LESS_THAN_OR_EQUAL:    OperatorInfo(BinaryOp.LESS_THAN_OR_EQUAL,    precedence=7, associativity=Associativity.LEFT),
    TokenType.GREATER_THAN_OR_EQUAL: OperatorInfo(BinaryOp.GREATER_THAN_OR_EQUAL, precedence=7, associativity=Associativity.LEFT),

    TokenType.EQUAL:     OperatorInfo(BinaryOp.EQUAL,     precedence=6, associativity=Associativity.LEFT),
    TokenType.NOT_EQUAL: OperatorInfo(BinaryOp.NOT_EQUAL, precedence=6, associativity=Associativity.LEFT),

    TokenType.AMPERSAND: OperatorInfo(BinaryOp.BITWISE_AND, precedence=5, associativity=Associativity.LEFT),
    TokenType.CARET:     OperatorInfo(BinaryOp.BITWISE_XOR, precedence=4, associativity=Associativity.LEFT),
    TokenType.PIPE:      OperatorInfo(BinaryOp.BITWISE_OR,  precedence=3, associativity=Associativity.LEFT),

    TokenType.AND: OperatorInfo(BinaryOp.AND, precedence=2, associativity=Associativity.LEFT),

    TokenType.OR: OperatorInfo(BinaryOp.OR, precedence=1, associativity=Associativity.LEFT),
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
        params = self.parse_params()
        self.expect(TokenType.CLOSE_PAREN, "Expected ')' after parameter list")
        self.expect(TokenType.COLON, "Expected ':' to start the function body")
        self.expect(TokenType.NEWLINE, "Expected a newline after ':'")

        body = self.parse_block()
        return Function(name=name_tok.val, return_type=return_type, params=params, body=body)

    def parse_params(self) -> List[Param]:
        """Parses a comma-separated parameter list -- zero or more
        `type IDENTIFIER` entries -- stopping *without* consuming
        CLOSE_PAREN, so the caller (parse_function) still owns matching
        it. An empty list (`()`) is valid and just returns []."""
        params = []
        if self.check(TokenType.CLOSE_PAREN):
            return params
        params.append(self.parse_param())
        while self.match(TokenType.COMMA):
            params.append(self.parse_param())
        return params

    def parse_param(self) -> Param:
        param_type = self.parse_type()
        name_tok = self.expect(TokenType.IDENTIFIER, "Expected a parameter name")
        return Param(name=name_tok.val, type=param_type)

    def parse_type(self) -> str:
        # 'int', 'bool', or 'str' -- semantic.py is what actually knows
        # what to do with the resulting string; the parser just needs to
        # accept one of the three type keywords here.
        if self.check(TokenType.INT, TokenType.BOOL, TokenType.STR):
            return self.advance().val
        tok = self.current()
        raise ParseError(
            f"Expected a type ('int', 'bool', or 'str'), got {tok.type} "
            f"('{tok.val}') at line {tok.line}, column {tok.col}"
        )

    def parse_block(self) -> List[Node]:
        """Parses an indented block: INDENT statement+ DEDENT.

        This is the one routine every block in the language goes
        through -- a function's top-level body and an if/elif/else's
        body alike -- which is exactly what makes nesting work for
        free: parse_if calls this same method for its own then/else
        bodies, so an if inside an if inside a function just falls out
        of ordinary recursion, with no separate "nested block" concept
        anywhere. skip_newlines() before the INDENT handles any blank
        lines between the block-opening NEWLINE (from `:`) and the
        block's first real line; skip_newlines() inside the loop does
        the same between statements.
        """
        self.skip_newlines()
        self.expect(TokenType.INDENT, "Expected an indented block")
        self.skip_newlines()
        statements = []
        while not self.check(TokenType.DEDENT) and not self.at_end():
            statements.append(self.parse_statement())
            self.skip_newlines()
        self.expect(TokenType.DEDENT, "Expected the end of an indented block")
        if not statements:
            raise ParseError("Expected at least one statement in this block")
        return statements

    def parse_statement(self) -> Node:
        if self.check(TokenType.INT, TokenType.BOOL, TokenType.STR):
            return self.parse_var_decl()
        if self.check(TokenType.RETURN):
            return self.parse_return()
        if self.check(TokenType.IF):
            return self.parse_if()
        if self.check(TokenType.WHILE):
            return self.parse_while()
        if self.check(TokenType.BREAK):
            return self.parse_break()
        if self.check(TokenType.CONTINUE):
            return self.parse_continue()
        if self.check(TokenType.IDENTIFIER) and self.peek(1).type == TokenType.ASSIGN:
            return self.parse_assign()
        return self.parse_expr_stmt()

    def parse_while(self) -> While:
        self.expect(TokenType.WHILE, "Expected 'while'")
        condition = self.parse_expression()
        self.expect(TokenType.COLON, "Expected ':' to start the while body")
        self.expect(TokenType.NEWLINE, "Expected a newline after ':'")
        body = self.parse_block()
        return While(condition=condition, body=body)

    def parse_break(self) -> Break:
        self.expect(TokenType.BREAK, "Expected 'break'")
        return Break()

    def parse_continue(self) -> Continue:
        self.expect(TokenType.CONTINUE, "Expected 'continue'")
        return Continue()

    def parse_if(self) -> If:
        self.expect(TokenType.IF, "Expected 'if'")
        return self._parse_if_body()

    def parse_elif_as_if(self) -> If:
        # See If's docstring: an elif is parsed as an ordinary If, just
        # nested one level inside the enclosing if's else_body.
        self.expect(TokenType.ELIF, "Expected 'elif'")
        return self._parse_if_body()

    def _parse_if_body(self) -> If:
        """Shared by parse_if and parse_elif_as_if -- both are just
        `KEYWORD expression ':' NEWLINE block`, differing only in which
        keyword the caller already consumed. Handles an arbitrarily
        long elif chain by recursing into parse_elif_as_if, and an
        optional trailing else.
        """
        condition = self.parse_expression()
        self.expect(TokenType.COLON, "Expected ':' to start the if body")
        self.expect(TokenType.NEWLINE, "Expected a newline after ':'")
        then_body = self.parse_block()

        else_body = None
        self.skip_newlines()
        if self.check(TokenType.ELIF):
            else_body = [self.parse_elif_as_if()]
        elif self.match(TokenType.ELSE):
            self.expect(TokenType.COLON, "Expected ':' to start the else body")
            self.expect(TokenType.NEWLINE, "Expected a newline after ':'")
            else_body = self.parse_block()

        return If(condition=condition, then_body=then_body, else_body=else_body)

    def parse_var_decl(self) -> VarDecl:
        var_type = self.parse_type()
        name_tok = self.expect(TokenType.IDENTIFIER, "Expected a variable name")
        init = None
        if self.match(TokenType.ASSIGN):
            init = self.parse_expression()
        return VarDecl(name=name_tok.val, var_type=var_type, init=init)

    def parse_assign(self) -> Assign:
        name_tok = self.expect(TokenType.IDENTIFIER)
        self.expect(TokenType.ASSIGN, "Expected '=' in assignment")
        value = self.parse_expression()
        return Assign(name=name_tok.val, value=value)

    def parse_return(self) -> Return:
        self.expect(TokenType.RETURN)
        value = self.parse_expression()
        return Return(value=value)

    def parse_expr_stmt(self) -> ExprStmt:
        return ExprStmt(expr=self.parse_expression())

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
        if self.check(TokenType.TRUE, TokenType.FALSE):
            tok = self.advance()
            return BoolLiteral(value=(tok.type == TokenType.TRUE))
        if self.check(TokenType.STRING):
            tok = self.advance()
            return StringLiteral(value=_unescape_string_literal(tok.val))
        if self.check(TokenType.IDENTIFIER):
            # One token of lookahead disambiguates a call (`foo(...)`)
            # from a bare variable reference (`foo`), the same way
            # parse_statement already peeks ahead to tell an assignment
            # apart from any other expression starting with a name.
            if self.peek(1).type == TokenType.OPEN_PAREN:
                return self.parse_call()
            tok = self.advance()
            return Variable(name=tok.val)
        if self.match(TokenType.OPEN_PAREN):
            expr = self.parse_expression()
            self.expect(TokenType.CLOSE_PAREN, "Expected ')' to close grouped expression")
            return expr
        tok = self.current()
        raise ParseError(
            f"Expected an expression, got {tok.type} ('{tok.val}') "
            f"at line {tok.line}, column {tok.col}"
        )

    def parse_call(self) -> Call:
        name_tok = self.expect(TokenType.IDENTIFIER)
        self.expect(TokenType.OPEN_PAREN, "Expected '(' to start a call's argument list")
        args = []
        if not self.check(TokenType.CLOSE_PAREN):
            args.append(self.parse_expression())
            while self.match(TokenType.COMMA):
                args.append(self.parse_expression())
        self.expect(TokenType.CLOSE_PAREN, "Expected ')' to close a call's argument list")
        return Call(name=name_tok.val, args=args)


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
