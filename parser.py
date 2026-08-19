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
    assign_stmt := IDENTIFIER assign_op expression NEWLINE
    assign_op   := '=' | '+=' | '-=' | '*=' | '/=' | '%='
                 | '&=' | '|=' | '^=' | '<<=' | '>>='
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
token of lookahead (peek(1)) rather than backtracking, with one
exception: IDENTIFIER is ambiguous between "the start of an assignment"
(`a = ...`) and "the start of some other expression that just happens to
reference `a`" (`a + 1`, or `a` alone) -- so parse_statement peeks one
token ahead and only commits to assign_stmt if that next token is
ASSIGN, falling through to expr_stmt otherwise.

A type-starting token (INT, BOOL, STR, or OPEN_BRACKET) no longer
unconditionally means a declaration the way it once did: it's also how
a bare, fully-typed array-literal statement starts (`[3]int[1, 2, 3]`
alone, with no assignment -- see ArrayLiteral's own docstring for why
that form, unlike the plain `[1, 2, 3]` one, is a genuine, self-
describing expression rather than something restricted to a VarDecl's
own initializer). Both shapes begin by parsing the exact same type, so
parse_statement parses it once, up front, and only THEN decides which
this is, based on what immediately follows: an IDENTIFIER (a variable
name) means a declaration; an ArrayTypeExpr immediately followed by
another OPEN_BRACKET means the literal's own elements are starting
right there instead. Committing to parse_type() first here, rather
than using bounded lookahead the way parse_primary's own, separate
disambiguation for this same shape does (see _looks_like_typed_array_
literal), is safe specifically because every statement starting with
one of these tokens was already required to start with a full type,
even before typed array literals existed -- there's no valid program
where committing to parse_type() here could turn out to be wrong.

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

COMPOUND ASSIGNMENT
---------------------
`+= -= *= /= %= &= |= ^= <<= >>=` are all handled entirely inside
parse_assign, by desugaring: `a += b` is parsed directly into
Assign(name='a', value=Binary(op=ADD, left=Variable(name='a'),
right=b)) -- the exact same tree a hand-written `a = a + b` would
produce, not a dedicated CompoundAssign node carrying the original
syntax through separately.

This is deliberate, not just convenient. It only works this cleanly
because every assignment target in this language is a bare variable
name -- there's no array-index or struct-field l-value where "evaluate
the target once" would actually matter, and reading a bare variable has
no side effect to worry about duplicating. Given that, the desugared
tree isn't an approximation of what `+=` means, it just *is* what `+=`
means, exactly as precisely as if the person had written the long form
themselves. The payoff: semantic.py and codegen.py need zero changes to
support any of these ten operators. check_binary already knows every
type rule each underlying operator has (including str's `+` overload
for concatenation), and codegen.py's gen_binary_into/
gen_string_concat_into already handle every one of these AST shapes
correctly -- including the concatenation memory-freeing optimization,
which correctly does *not* fire here, since the left operand is a
Variable node (a named, possibly-still-referenced value), never a
fresh Binary(ADD, ...) result.

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
from typing import Any, List, Optional, Union

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
    # Set by semantic.py's check_expr after type-checking, not at parse
    # time -- see this field's fuller explanation on StringLiteral
    # below, which was the first node to need it documented in detail.
    resolved_type: Optional[Any] = None

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
    resolved_type: Optional[Any] = None

    def pretty(self) -> str:
        return f"BoolLiteral(value: {'true' if self.value else 'false'})"


@dataclass
class NoneLiteral(Node):
    """`none` -- Hornet's nil-style zero value, analogous to Go's own
    `nil`, but deliberately narrower: Go's nil has no fixed type of its
    own at all, adapting to whatever nilable type context expects it
    via Go's general untyped-constant mechanism (the same one numeric
    literals use there). Hornet has no untyped-constant mechanism for
    ANY literal yet, so building one just for `none` would be a much
    bigger structural change than adding a value -- every expression's
    type is currently derivable purely from itself and its children,
    with no context needed, and an untyped node would break that.

    Instead, NoneLiteral resolves to one single, fixed, purely-internal
    type, Type.NONE (see semantic.py) -- checked for COMPATIBILITY,
    not equality, specifically wherever a value flows into a slice-
    typed context (a VarDecl initializer, an Assign, a function
    argument, a return value, or one side of ==/!=) -- see semantic.py's
    _types_compatible. From the outside this behaves like Go's nil for
    everything usable today; only the internal mechanism is narrower.

    Only slices are nilable so far -- none is NOT compatible with
    int/bool/str/array, even though str is also a pointer under the
    hood at the machine level. Extending this to other composite/
    reference types, if any come along later, is real, separable
    follow-up work, not implemented yet.

    At the machine level, none becomes the {ptr: 0, len: 0} slice
    descriptor (see codegen.py's gen_none_into) -- the same shape
    Go's own nil slice has: a valid, safely-indexable-into-nothing
    slice with no backing array, not a special, separately-tracked
    null flag. Every existing slice operation (indexing, printing,
    re-slicing) already handles a zero-length slice correctly -- see
    TestSliceBoundsChecking's own positive control for `arr[5:5]` in
    test_compiler.py -- so a none-valued slice needs no new mechanism
    for those, only for producing the {0, 0} descriptor in the first
    place, and for comparing a slice against none directly. That
    comparison (`s == none`) checks specifically the descriptor's
    `ptr` field against 0, matching Go's own nil-vs-empty-slice
    distinction: a real, zero-length slice sliced from a real array
    (e.g. `arr[5:5]`) has a non-null pointer and is NOT `== none`,
    even though both are equally safe, equally zero-length slices for
    every other purpose."""
    resolved_type: Optional[Any] = None

    def pretty(self) -> str:
        return "NoneLiteral"


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
    # `resolved_type` is None right after parsing -- it only gets a real
    # value once semantic.py's check_expr has actually type-checked the
    # node (see its module docstring's TYPES section). It's typed as
    # Optional[Any] -- not Optional[semantic.Type] -- specifically so
    # parser.py never has to import semantic.py, which already imports
    # *from* parser.py; a Type import here would be circular. What's
    # ACTUALLY stored, once semantic.py sets it, is a full semantic.Type
    # instance (not a string -- that changed when array types were
    # added, since a type now needs to carry an element type and size,
    # not just a name). codegen.py reads this directly (see its
    # _type_of) instead of re-deriving an expression's type itself, and
    # is free to inspect the Type object's own fields (.kind,
    # .element_type, .size) since it already imports Type from
    # semantic.py -- only parser.py has the import-direction
    # restriction that Any works around.
    resolved_type: Optional[Any] = None

    def pretty(self) -> str:
        return f"StringLiteral(value: {self.value!r})"


@dataclass
class Variable(Node):
    """A reference to a local variable, e.g. the `a` in `a + 1`."""
    name: str
    resolved_type: Optional[Any] = None

    def pretty(self) -> str:
        return f"Variable(name: {self.name})"


@dataclass
class ArrayLiteral(Node):
    """`[e1, e2, ...]` in EXPRESSION position -- an array literal, e.g.
    the value side of `[3]int arr = [1, 2, 3]`. An element can itself
    be another ArrayLiteral for a multi-dimensional literal (e.g.
    `[[1,2,3],[4,5,6]]` for a [2][3]int value) -- there's no special
    casing for this in the parser at all; parse_expression naturally
    recurses into a nested `[...]` the same way it would parse any
    other nested expression, since ArrayLiteral is just one more
    primary-expression shape. Elements don't have to be constants --
    any expression is valid per element (see codegen.py's ARRAYS
    section for how each one gets evaluated and stored).

    type_expr is None for the plain, untyped form above -- which only
    ever type-checks where an expected type is already known from
    context (see semantic.py's check_array_literal for why that's true
    regardless: it's still a real, self-contained inference over the
    elements themselves, not something borrowed from outside, so this
    form is restricted to a VarDecl's own initializer or an Assign's
    own value -- the only two places codegen currently has anywhere to
    WRITE the result, whether that's an ordinary array's own slot, or
    (see the module docstring's SLICE LITERALS section in codegen.py)
    a freshly-created, heap-allocated one backing a slice, when the
    target being initialized/assigned is slice-typed instead of
    array-typed). type_expr is set for the fully-typed form, `[3]int[1,
    2, 3]` -- an ArrayTypeExpr, parsed exactly like any other type via
    parse_type() -- which makes the literal entirely self-describing
    and usable as a genuine, general expression: a bare statement, a
    VarDecl initializer (redundantly restating a type the declaration
    already gives, but that's allowed, not an error), or anywhere else
    an expression is valid. See parse_primary's own bounded-lookahead
    disambiguation (_looks_like_typed_literal) for how `[3]int[...]` is
    told apart from a plain `[N, ...]` despite both starting with
    OPEN_BRACKET NUMBER.

    A SLICE literal, `[]int[1, 2, 3]`, is NOT its own node type at all
    -- it's sugar, entirely resolved right here in the parser: see
    _parse_bracketed_literal, which wraps an ArrayLiteral like this one
    (type_expr synthesized as `[3]int`, the element count derived from
    how many elements were actually parsed -- a slice has no size of
    its own; its backing array does) in an implicit, whole-array Slice
    node (low=None, high=None -- "the whole thing", exactly like
    `arr[:]` already means for a named array). That's what lets
    check_slice, gen_slice_into, and gen_indexable_base_into handle a
    slice literal almost entirely via machinery that already existed
    for slicing a NAMED array -- the one genuinely new piece needed was
    teaching gen_indexable_base_into that an ArrayLiteral base means
    "allocate a fresh one," not "find an existing one" (see codegen.py's
    gen_array_literal_heap_alloc_into)."""
    elements: List[Node] = field(default_factory=list)
    type_expr: Optional['ArrayTypeExpr'] = None
    resolved_type: Optional[Any] = None

    def pretty(self) -> str:
        elems_str = ', '.join(e.pretty() for e in self.elements)
        prefix = "" if self.type_expr is None else self.type_expr.pretty() + " "
        return f"{prefix}ArrayLiteral -> [{elems_str}]"


@dataclass
class Index(Node):
    """`array[index]` -- reads a single element (or, for a
    multi-dimensional array not yet fully indexed, a sub-array) out of
    `array`. Multi-dimensional indexing `matrix[i][j]` is represented
    as NESTED Index nodes -- Index(array=Index(array=Variable('matrix'),
    index=i), index=j) -- one per bracket pair, matching how the TYPE
    itself is structured (an array of arrays): reading the outer Index
    first yields a whole row (itself array-typed), and the outer
    bracket pair indexes into THAT. See parse_postfix for how this gets
    built left-to-right off of however many `[...]` pairs follow a
    primary expression.

    `array` can itself be Slice-typed, not just Array-typed -- indexing
    into a slice (`s[i]`) uses this exact same node; nothing about
    Index's own shape needs to know or care which kind of thing it's
    indexing into (see semantic.py's indexable-and-index check, which
    accepts either)."""
    array: Node
    index: Node
    resolved_type: Optional[Any] = None

    def pretty(self) -> str:
        return f"Index -> [{self.array.pretty()}, {self.index.pretty()}]"


@dataclass
class Slice(Node):
    """`array[low:high]` -- a VIEW into `array` spanning [low, high):
    low inclusive, high exclusive, matching Go's own convention (which
    this whole feature is deliberately modeled on). Produces a Slice-
    typed VALUE (conceptually a {pointer, length} descriptor -- see
    codegen.py's eventual SLICES section for the concrete
    representation), not a copy of the underlying elements: a slice is
    a genuine alias into its base's own backing storage, unlike plain
    array assignment (`arr2 = arr1`), which already copies. This is
    exactly the aliasing surface that makes size-based stack safety's
    heap-promotion policy load-bearing rather than optional once
    slicing exists -- see codegen.py's ARRAYS section for the
    existing mechanism this reuses (a second trigger for
    is_heap_allocated: "is this array ever sliced", not just "is it
    over the size threshold").

    `array` can be EITHER Array-typed or Slice-typed -- slicing a
    slice (`s2 = s[1:3]`) and slicing the outer dimension of a multi-
    dimensional array (`matrix[0:2]`, yielding a slice of ROWS, type
    `[][3]int`) are both valid, and both go through this same node;
    see semantic.py's check_slice for the shared base-type check.

    `low`/`high` are both OPTIONAL, independently -- `arr[:]` omits
    both, `arr[2:]` omits high, `arr[:5]` omits low. Represented as
    None here rather than synthesizing a default value at parse time:
    low's default (0) could technically be filled in here, since it
    never depends on anything the parser doesn't already know, but
    high's default (the base's own length) can't be -- for a Slice
    base specifically, that length is a runtime value read out of the
    descriptor, not something the parser could ever compute. Rather
    than defaulting one bound at parse time and deferring the other,
    both stay None uniformly, resolved together wherever a Slice is
    actually generated.

    Deliberately NOT a valid assignment target -- `arr[1:3] = ...`
    doesn't parse as anything at all (parse_expr_stmt_or_index_assign's
    existing isinstance(expr, Index) check already excludes this
    automatically, simply because Slice is a different class; no
    changes were needed there to get this for free). This matches Go,
    where slicing produces a value, not an addressable location: you
    can assign to a single indexed element (`s[i] = x`) but never to a
    sliced range.
    """
    array: Node
    low: Optional[Node] = None
    high: Optional[Node] = None
    resolved_type: Optional[Any] = None

    def pretty(self) -> str:
        low_str = self.low.pretty() if self.low is not None else ''
        high_str = self.high.pretty() if self.high is not None else ''
        return f"Slice -> [{self.array.pretty()}, {low_str}:{high_str}]"


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
    resolved_type: Optional[Any] = None

    def pretty(self) -> str:
        args_str = ', '.join(a.pretty() for a in self.args)
        return f"Call(name: {self.name}) -> [{args_str}]"


@dataclass
class Unary(Node):
    op: UnaryOp
    operand: Node
    resolved_type: Optional[Any] = None

    def pretty(self) -> str:
        return f"Unary(op: {self.op.symbol()}) -> {self.operand.pretty()}"


@dataclass
class Binary(Node):
    op: BinaryOp
    left: Node
    right: Node
    resolved_type: Optional[Any] = None

    def pretty(self) -> str:
        # Binary has two children, so the linear "A -> B" chain style
        # used elsewhere (Return, Unary, Function) doesn't quite fit --
        # this bracketed form keeps it a single readable line while still
        # showing the tree shape, e.g. for `(1 + 2) * 3`:
        #   Binary(op: *) -> [Binary(op: +) -> [Constant(value: 1), Constant(value: 2)], Constant(value: 3)]
        return f"Binary(op: {self.op.symbol()}) -> [{self.left.pretty()}, {self.right.pretty()}]"


@dataclass
class Return(Node):
    """`return <expr>`, or a bare `return` with no expression at all
    (value=None) -- valid inside a function with no declared return
    type (see Function's own docstring), the same way C/Go/Python all
    let a void-like function exit early with a bare `return`. Not
    represented as, say, a sentinel Node wrapping "nothing": `None`
    here mirrors exactly how Function.return_type itself represents
    "no declared type" -- both ends of the same absence, checked the
    same way (`is None`) wherever either is consumed."""
    value: Optional[Node] = None

    def pretty(self) -> str:
        return "Return" if self.value is None else f"Return -> {self.value.pretty()}"


@dataclass
class ArrayTypeExpr(Node):
    """`[size]element_type` in TYPE position -- e.g. the `[3]int` in
    `[3]int arr = ...`, or the `[2][3]int` in `[2][3]int matrix = ...`
    (parsed as ArrayTypeExpr(size=2, element_type=ArrayTypeExpr(size=3,
    element_type='int'))) -- an array of 2 arrays of 3 ints, matching
    row-major layout: the OUTERMOST dimension is listed first, closest
    to the brackets. Wherever a plain type keyword string
    ('int'/'bool'/'str') was previously the ONLY valid value for
    VarDecl.var_type / Param.type / Function.return_type, those fields
    now accept a plain string, one of these, OR a SliceTypeExpr (see
    its own docstring), recursed arbitrarily deep -- hence
    `element_type` being typed Union[str, 'ArrayTypeExpr',
    'SliceTypeExpr'] rather than just str.

    `size` is required to be a positive integer LITERAL (see
    Parser.parse_type) -- not an arbitrary constant expression like
    `[2+3]int` -- kept deliberately simple for this first pass, and
    validated at parse time (unlike most validation in this file, which
    is left to semantic.py) since an array's size isn't really an
    ordinary expression the way a VarDecl's initializer is; it's closer
    to syntax, the same way a type keyword itself is validated directly
    by the parser rather than deferred.
    """
    size: int
    element_type: Union[str, 'ArrayTypeExpr', 'SliceTypeExpr']

    def pretty(self) -> str:
        et = self.element_type if isinstance(self.element_type, str) else self.element_type.pretty()
        return f"[{self.size}]{et}"


@dataclass
class SliceTypeExpr(Node):
    """`[]element_type` in TYPE position -- e.g. the `[]int` in
    `[]int s = arr[:]`. Sibling to ArrayTypeExpr, distinguished by the
    single token of lookahead right after `[` (see Parser.parse_type):
    a NUMBER means an array's size, an immediate `]` means a slice with
    no size at all.

    That's the essential difference from ArrayTypeExpr, not just a
    detail of parsing it: a slice's length is a RUNTIME property of the
    slice VALUE itself (part of what gets stored at a slice's own
    `{pointer, length}` representation -- see codegen.py's eventual
    SLICES section), not part of its TYPE the way an array's size is.
    Two slices can have the exact same type ([]int) while holding
    completely different lengths at runtime; two arrays with different
    sizes ([3]int vs [4]int) are, and always were, different types
    entirely (see semantic.Type's own structural equality).

    `element_type` recurses exactly like ArrayTypeExpr's does -- a
    slice of arrays (`[][3]int`) and a slice of slices (`[][]int`) are
    both valid element types, parsed with no special-casing beyond
    what parse_type's own recursive structure already provides.
    """
    element_type: Union[str, ArrayTypeExpr, 'SliceTypeExpr']

    def pretty(self) -> str:
        et = self.element_type if isinstance(self.element_type, str) else self.element_type.pretty()
        return f"[]{et}"


@dataclass
class VarDecl(Node):
    """`int a` (init=None) or `int a = 1` (init=the initializer expression).
    `var_type` is a plain type keyword string ('int'/'bool'/'str'), an
    ArrayTypeExpr, or a SliceTypeExpr (see their own docstrings) for an
    array- or slice-typed decl."""
    name: str
    var_type: Union[str, ArrayTypeExpr, SliceTypeExpr]
    init: Optional[Node] = None

    def pretty(self) -> str:
        vt = self.var_type if isinstance(self.var_type, str) else self.var_type.pretty()
        head = f"VarDecl(name: {self.name}, type: {vt})"
        return head if self.init is None else f"{head} -> {self.init.pretty()}"


@dataclass
class Assign(Node):
    """`a = <value>`, assigning to an already-declared variable."""
    name: str
    value: Node

    def pretty(self) -> str:
        return f"Assign(name: {self.name}) -> {self.value.pretty()}"


@dataclass
class IndexAssign(Node):
    """`array[index] = value` -- writes a single array element.
    `array` is the expression being indexed: a bare Variable for
    `arr[i] = v`, or itself an Index node for the OUTER dimensions of a
    multi-dimensional assignment like `matrix[i][j] = v`, which parses
    as IndexAssign(array=Index(array=Variable('matrix'), index=i),
    index=j, value=v) -- "index INTO matrix to get a row, then
    index-ASSIGN into that row at the given column". This mirrors
    exactly how Index itself nests for multi-dimensional READS (see
    Index's own docstring) -- one node per bracket pair, either way.

    Compound index-assignment (`arr[i] += 1`) is NOT supported yet --
    see parse_statement's own handling -- since it would need the index
    expression evaluated exactly once and reused for both the read and
    the write, which the simple desugaring used for `x += y` on a bare
    variable (see the module docstring's COMPOUND ASSIGNMENT section)
    doesn't guarantee if the index expression isn't side-effect-free.
    """
    array: Node
    index: Node
    value: Node

    def pretty(self) -> str:
        return f"IndexAssign(index: {self.index.pretty()}) -> [{self.array.pretty()}, {self.value.pretty()}]"


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
    "initial value" is whatever the caller passed). `type` is a plain
    type keyword string, an ArrayTypeExpr, or a SliceTypeExpr, exactly
    like VarDecl.var_type (see their own docstrings)."""
    name: str
    type: Union[str, ArrayTypeExpr, SliceTypeExpr]

    def pretty(self) -> str:
        t = self.type if isinstance(self.type, str) else self.type.pretty()
        return f"Param(name: {self.name}, type: {t})"


@dataclass
class Function(Node):
    """`def type NAME(params):` (return_type set), or `def NAME(params):`
    with the type omitted entirely (return_type=None) -- a function
    with no declared return type. Not a `none`/void keyword-typed
    declaration -- there is no such keyword -- just the absence of a
    type before the name, disambiguated from the ordinary case by a
    single token of lookahead in parse_function (a type keyword or '['
    starts a type; an IDENTIFIER, which is what a function name always
    starts with instead, never does). Such a function may fall off the
    end of its body without an explicit `return` at all, or exit early
    via a bare `return` (see Return's own docstring) -- semantic.py
    deliberately does not require every path to return explicitly the
    way every other function's declared type does (see its own
    always_returns skip for this case)."""
    name: str
    return_type: Optional[Union[str, ArrayTypeExpr, SliceTypeExpr]]
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
        rt = "(none)" if self.return_type is None else (
            self.return_type if isinstance(self.return_type, str) else self.return_type.pretty()
        )
        return f"Function(name: {self.name}, return_type: {rt}, params: [{params_str}]) -> {body_str}"


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


# TokenType -> the BinaryOp a compound-assignment operator desugars
# into. See parse_assign: `x += y` is parsed directly into the exact
# same AST shape as `x = x + y` (Assign wrapping a Binary), rather than
# introducing a dedicated CompoundAssign node -- see the module
# docstring's COMPOUND ASSIGNMENT section for why that's not a
# shortcut so much as the actually-correct representation here, given
# this language's assignment targets are always a bare name.
_COMPOUND_ASSIGN_OPS = {
    TokenType.PLUS_ASSIGN:      BinaryOp.ADD,
    TokenType.MINUS_ASSIGN:     BinaryOp.SUBTRACT,
    TokenType.STAR_ASSIGN:      BinaryOp.MULTIPLY,
    TokenType.SLASH_ASSIGN:     BinaryOp.DIVIDE,
    TokenType.PERCENT_ASSIGN:   BinaryOp.MODULO,
    TokenType.AMPERSAND_ASSIGN: BinaryOp.BITWISE_AND,
    TokenType.PIPE_ASSIGN:      BinaryOp.BITWISE_OR,
    TokenType.CARET_ASSIGN:     BinaryOp.BITWISE_XOR,
    TokenType.SHIFT_LEFT_ASSIGN:  BinaryOp.SHIFT_LEFT,
    TokenType.SHIFT_RIGHT_ASSIGN: BinaryOp.SHIFT_RIGHT,
}

# Every token that can start the operator position of an assignment
# statement: plain '=' plus every compound form. parse_statement uses
# this set for its one-token lookahead (IDENTIFIER followed by one of
# these means "this is an assignment statement", exactly the same way
# it already used to check for TokenType.ASSIGN alone).
_ASSIGNMENT_TOKENS = {TokenType.ASSIGN, *_COMPOUND_ASSIGN_OPS.keys()}


class Parser:
    def __init__(self, tokens: List[Token]):
        if len(tokens) == 0:
            raise ValueError('tokens must have non zero length')
        if tokens[-1].type != TokenType.EOF:
            raise ValueError('tokens must be terminated by an EOF')
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
        # A type keyword or '[' starts a return type; anything else --
        # in practice always IDENTIFIER, since that's what a function
        # name always starts with -- means the return type was omitted
        # entirely: this function has no declared return type at all
        # (see Function's own docstring). One token of lookahead is
        # enough to tell these apart unambiguously: a function name can
        # never itself BE a type keyword, since those are reserved.
        if self.check(TokenType.INT, TokenType.BOOL, TokenType.STR, TokenType.OPEN_BRACKET):
            return_type = self.parse_type()
        else:
            return_type = None
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

    def parse_type(self) -> Union[str, ArrayTypeExpr, SliceTypeExpr]:
        # 'int', 'bool', or 'str' -- semantic.py is what actually knows
        # what to do with the resulting string; the parser just needs to
        # accept one of the three type keywords here. OR: '[' NUMBER ']'
        # followed by another type, recursively, for an array
        # (ArrayTypeExpr -- this is what lets `[2][3]int` parse at all,
        # each bracket pair peeling off one more ArrayTypeExpr wrapping
        # whatever parse_type() returns for the rest). OR: '[' ']'
        # followed by another type, recursively, for a slice
        # (SliceTypeExpr) -- distinguished from the array case by a
        # single token of lookahead right after '[': a NUMBER means an
        # array's size, an immediate ']' means "no size at all, this is
        # a slice" (see SliceTypeExpr's own docstring for why that's a
        # meaningful distinction, not just a syntax difference). The
        # array size, when present, has to be a literal, positive,
        # whole NUMBER token -- checked and rejected right here, unlike
        # most validation in this file (which is semantic.py's job): an
        # array's size isn't really an expression the way a VarDecl
        # initializer is, it's closer to syntax, the same way a type
        # keyword itself is validated directly rather than deferred.
        if self.check(TokenType.OPEN_BRACKET):
            self.advance()
            if self.check(TokenType.CLOSE_BRACKET):
                self.advance()
                element_type = self.parse_type()
                return SliceTypeExpr(element_type=element_type)
            size_tok = self.expect(
                TokenType.NUMBER,
                "Expected an array size (a positive integer literal), or ']' for a slice type",
            )
            if '.' in size_tok.val:
                raise ParseError(
                    f"Array size must be a whole number, got '{size_tok.val}' "
                    f"at line {size_tok.line}, column {size_tok.col}"
                )
            size = int(size_tok.val)
            if size <= 0:
                raise ParseError(
                    f"Array size must be positive, got {size} "
                    f"at line {size_tok.line}, column {size_tok.col}"
                )
            self.expect(TokenType.CLOSE_BRACKET, "Expected ']' after array size")
            element_type = self.parse_type()
            return ArrayTypeExpr(size=size, element_type=element_type)
        if self.check(TokenType.INT, TokenType.BOOL, TokenType.STR):
            return self.advance().val
        tok = self.current()
        raise ParseError(
            f"Expected a type ('int', 'bool', 'str', '[size]type', or "
            f"'[]type'), got {tok.type} ('{tok.val}') at line {tok.line}, "
            f"column {tok.col}"
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
        if self.check(TokenType.INT, TokenType.BOOL, TokenType.STR, TokenType.OPEN_BRACKET):
            # A type-starting token could mean EITHER a VarDecl
            # (`[3]int arr = ...`) or a bare, fully-typed array- or
            # slice-literal expression statement (`[3]int[1, 2, 3]`,
            # `[]int[1, 2, 3]`) -- both start by parsing the exact
            # same type, so parse it once, up front, and let whatever
            # follows decide which this actually is: an IDENTIFIER (a
            # variable name) means a VarDecl; another OPEN_BRACKET,
            # when the type just parsed is specifically an
            # ArrayTypeExpr or a SliceTypeExpr (not a bare scalar
            # name, which can't validly be followed by one at all),
            # means the literal's own elements are starting right here
            # instead. Unlike parse_primary's own, separate
            # disambiguation for this same shape (see
            # _looks_like_typed_literal), committing to parse_type()
            # first is safe here specifically because every statement
            # starting with one of these tokens was ALREADY required
            # to start with a full type, even before typed literals
            # existed -- there's no risk of misparsing an UNTYPED
            # array literal this way, since one can never validly
            # start a statement at all (see ArrayLiteral's own
            # docstring in this file for why that form stays
            # restricted to a VarDecl's/Assign's own value).
            parsed_type = self.parse_type()
            if isinstance(parsed_type, (ArrayTypeExpr, SliceTypeExpr)) and self.check(TokenType.OPEN_BRACKET):
                literal = self._parse_bracketed_literal(parsed_type)
                return ExprStmt(expr=literal)
            return self.parse_var_decl(var_type=parsed_type)
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
        if self.check(TokenType.IDENTIFIER) and self.peek(1).type in _ASSIGNMENT_TOKENS:
            return self.parse_assign()
        return self.parse_expr_stmt_or_index_assign()

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

    def parse_var_decl(self, var_type: Optional[Union[str, 'ArrayTypeExpr', 'SliceTypeExpr']] = None) -> VarDecl:
        """`type NAME` or `type NAME = <expr>`. `var_type`, when
        already supplied, is a type parse_statement did itself before
        realizing this is a declaration and not a bare, fully-typed
        array-literal statement (`[3]int[1, 2, 3]`) -- both start by
        parsing the exact same type, so parse_statement parses it once
        and hands it here rather than this method parsing it again."""
        if var_type is None:
            var_type = self.parse_type()
        name_tok = self.expect(TokenType.IDENTIFIER, "Expected a variable name")
        init = None
        if self.match(TokenType.ASSIGN):
            init = self.parse_expression()
        return VarDecl(name=name_tok.val, var_type=var_type, init=init)

    def parse_assign(self) -> Assign:
        """`a = <expr>` or a compound form (`a += <expr>`, etc).

        A compound form is desugared right here into the exact same
        Assign(name, Binary(op, Variable(name), value)) shape a
        hand-written `a = a + <expr>` would already produce -- not into
        some dedicated CompoundAssign node. That's deliberate: reading
        `a` to combine with the new value has no side effect in this
        language (a bare variable reference never does), so there's
        nothing lost by representing "read a, combine, write back" as
        literally that, and everything downstream -- every type rule in
        check_binary (including str's `+` overload), and every codegen
        path (including the string-concatenation memory-freeing
        optimization) -- already handles this AST shape correctly with
        no changes needed anywhere else.
        """
        name_tok = self.expect(TokenType.IDENTIFIER)
        op_tok = self.advance()  # one of _ASSIGNMENT_TOKENS -- already confirmed by parse_statement's lookahead
        value = self.parse_expression()

        if op_tok.type == TokenType.ASSIGN:
            return Assign(name=name_tok.val, value=value)

        binary_op = _COMPOUND_ASSIGN_OPS[op_tok.type]
        desugared_value = Binary(op=binary_op, left=Variable(name=name_tok.val), right=value)
        return Assign(name=name_tok.val, value=desugared_value)

    def parse_return(self) -> Return:
        """`return <expr>` or a bare `return` (see Return's own
        docstring). A NEWLINE immediately after 'return' unambiguously
        signals the bare form: every statement in this grammar is
        NEWLINE-terminated (including the last one in a file with no
        trailing newline of its own -- the lexer synthesizes one, see
        its own tokenize() docstring), and no expression can itself
        start with NEWLINE, so this single check never needs to
        backtrack or look further ahead."""
        self.expect(TokenType.RETURN)
        if self.check(TokenType.NEWLINE):
            return Return(value=None)
        value = self.parse_expression()
        return Return(value=value)

    def parse_expr_stmt_or_index_assign(self) -> Node:
        """Handles two statement shapes that -- unlike plain `a = ...`
        -- can't be told apart by a single token of lookahead: a bare
        expression statement (`arr[i] + 1`, or just `foo()`), and an
        index-assignment (`arr[i] = value`, or `matrix[i][j] = value`
        for deeper nesting). The number of `[...]` pairs on the left
        varies, so instead of trying to look ahead through however many
        of them there might be, this just parses the leading expression
        through the ordinary machinery first (which already builds
        nested Index nodes for however many brackets follow -- see
        parse_postfix), then decides based on what comes next.

        Only plain `=` is handled here -- `arr[i] += 1` and the other
        compound forms are deliberately rejected with a clear error
        rather than silently mis-parsed (see IndexAssign's own
        docstring for why compound index-assignment isn't supported
        yet at all, not just here)."""
        expr = self.parse_expression()
        if self.check(TokenType.ASSIGN):
            if not isinstance(expr, Index):
                tok = self.current()
                raise ParseError(
                    f"Left-hand side of '=' is not assignable "
                    f"at line {tok.line}, column {tok.col}"
                )
            self.advance()
            value = self.parse_expression()
            return IndexAssign(array=expr.array, index=expr.index, value=value)
        if isinstance(expr, Index) and self.current().type in _COMPOUND_ASSIGN_OPS:
            tok = self.current()
            raise ParseError(
                f"Compound assignment to an array element ('{tok.val}') "
                f"is not supported yet -- write it as a plain '=' instead "
                f"at line {tok.line}, column {tok.col}"
            )
        return ExprStmt(expr=expr)

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
        return self.parse_postfix()

    def parse_postfix(self) -> Node:
        """Wraps a primary expression with zero or more `[...]`
        suffixes -- each one either an index (`matrix[i][j]`, parsed
        here as `parse_primary` producing the `matrix` Variable, then
        this loop wrapping it in two nested Index nodes, one per
        bracket pair -- see Index's own docstring for why that nesting,
        rather than a single Index carrying a list of indices, is the
        right shape) or a slice (`arr[low:high]`, see
        _parse_index_or_slice for how the two are told apart within one
        `[...]` pair). Either shape can chain with the other --
        `s[1:3][0]` (index into a slice) and `matrix[0:2][1]` (index
        into a slice of rows) both fall out of this same loop with no
        special-casing, exactly like chained indexing already did.

        Sits between parse_unary and parse_primary specifically so
        indexing/slicing binds TIGHTER than a prefix unary operator:
        `-arr[0]` has to mean `-(arr[0])`, not `(-arr)[0]` (which
        wouldn't even type-check, since unary '-' requires an int
        operand, not an array) -- and it does, here, since parse_unary's
        own base case (no unary operator present) calls straight into
        this method rather than parse_primary directly.
        """
        expr = self.parse_primary()
        while self.check(TokenType.OPEN_BRACKET):
            self.advance()
            expr = self.parse_index_or_slice(expr)
        return expr

    def parse_index_or_slice(self, array_expr: Node) -> Node:
        """Parses the CONTENT of one `[...]` pair, immediately following
        an already-consumed OPEN_BRACKET, and returns either an Index
        (`a[i]`) or a Slice (`a[low:high]`, with either bound optionally
        omitted -- see Slice's own docstring) wrapping `array_expr`.

        A leading ':' unambiguously signals a slice with an omitted low
        bound (`a[:...]`) -- ':' can't start any expression in this
        language, so seeing it immediately after '[' can only mean
        this. Otherwise, an expression is parsed first; if a ':'
        follows THAT, this is a slice too (with high optionally omitted
        -- `a[low:]`), and if it doesn't, this was a plain index all
        along, and the parsed expression was the index itself.
        """
        if self.check(TokenType.COLON):
            self.advance()
            high = None if self.check(TokenType.CLOSE_BRACKET) else self.parse_expression()
            self.expect(TokenType.CLOSE_BRACKET, "Expected ']' to close a slice expression")
            return Slice(array=array_expr, low=None, high=high)

        first = self.parse_expression()

        if self.match(TokenType.COLON):
            high = None if self.check(TokenType.CLOSE_BRACKET) else self.parse_expression()
            self.expect(TokenType.CLOSE_BRACKET, "Expected ']' to close a slice expression")
            return Slice(array=array_expr, low=first, high=high)

        self.expect(TokenType.CLOSE_BRACKET, "Expected ']' after array index")
        return Index(array=array_expr, index=first)

    def parse_primary(self) -> Node:
        if self.check(TokenType.NUMBER):
            tok = self.advance()
            value = float(tok.val) if '.' in tok.val else int(tok.val)
            return Constant(value=value)
        if self.check(TokenType.TRUE, TokenType.FALSE):
            tok = self.advance()
            return BoolLiteral(value=(tok.type == TokenType.TRUE))
        if self.check(TokenType.NONE):
            self.advance()
            return NoneLiteral()
        if self.check(TokenType.STRING):
            tok = self.advance()
            return StringLiteral(value=_unescape_string_literal(tok.val))
        if self._looks_like_typed_literal():
            parsed_type = self.parse_type()
            return self._parse_bracketed_literal(parsed_type)
        if self.check(TokenType.OPEN_BRACKET):
            return self.parse_array_literal()
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

    def _looks_like_typed_literal(self) -> bool:
        """True if the CURRENT position starts a fully-typed array or
        slice literal (`[3]int[1, 2, 3]` or `[]int[1, 2, 3]`) rather
        than a plain, untyped array literal (`[1, 2, 3]`) or a single-
        element one (`[5]`) -- both of which also start with
        OPEN_BRACKET, and, for a size-1 case like `[5]`, even start
        with the identical OPEN_BRACKET NUMBER CLOSE_BRACKET prefix an
        array type's own size bracket does.

        Resolved with BOUNDED lookahead alone -- three or four tokens
        depending on which shape it turns out to be, no backtracking
        -- rather than speculatively parsing a type and rewinding on
        failure, which this parser has consistently avoided elsewhere
        (see parse_statement's own one-token dispatch). The two shapes
        that are genuinely unambiguous: an array type's own size
        bracket is `[` NUMBER `]` immediately followed by a type-
        starting token, and a slice type's own (empty) bracket pair is
        `[` `]` immediately followed by one -- a type keyword, or
        another `[` for a nested array/slice element type
        (`[2][2]int[...]`, `[][]int[...]`). Neither an untyped
        literal's closing `]` (array) nor a bare `[]` (which, on its
        own, parses as a valid but always semantically-rejected empty
        array literal -- see check_array_literal) is ever immediately
        followed by one of those in any valid program (nothing can
        validly follow a complete expression that way), so checking
        for either specific shape is enough to always tell them apart
        from their own untyped counterparts, including the array
        side's size-1 edge case: `[5]` alone has nothing of that shape
        following its `]`, so it's correctly left to parse as a
        single-element array literal instead.
        """
        if not self.check(TokenType.OPEN_BRACKET):
            return False
        if self.peek(1).type == TokenType.CLOSE_BRACKET:
            return self.peek(2).type in (TokenType.INT, TokenType.BOOL, TokenType.STR, TokenType.OPEN_BRACKET)
        if self.peek(1).type == TokenType.NUMBER and self.peek(2).type == TokenType.CLOSE_BRACKET:
            return self.peek(3).type in (TokenType.INT, TokenType.BOOL, TokenType.STR, TokenType.OPEN_BRACKET)
        return False

    def _parse_bracketed_literal(self, parsed_type: Union[str, 'ArrayTypeExpr', 'SliceTypeExpr']) -> Node:
        """Given an already-parsed type (from either parse_primary's
        own _looks_like_typed_literal-triggered path, or
        parse_statement's own "parse the type first, then decide"
        dispatch for the same shape at statement-start), parses the
        literal's own bracketed elements and returns the appropriate
        node: an ArrayLiteral directly for an ArrayTypeExpr
        (`[3]int[1, 2, 3]`), or one wrapped in an implicit, whole-array
        Slice for a SliceTypeExpr (`[]int[1, 2, 3]`) -- see Slice's own
        docstring for why an omitted low/high already means "the whole
        thing", which is exactly what a slice literal's own freshly-
        created backing array needs: every element, no sub-range.

        For the slice form, the WRAPPED ArrayLiteral's own type_expr is
        synthesized from the SliceTypeExpr's element_type and the
        ACTUAL number of elements parsed -- `[]int[1, 2, 3]`'s inner
        array is `[3]int`, not `[]int`; a slice has no size of its own,
        only its backing array does. Set on the node AFTER
        construction, once the element count is known, rather than
        threaded through as a constructor argument, so parse_array_
        literal's own signature stays exactly what it already was for
        the plain, typed-array-literal case -- ArrayLiteral is an
        ordinary, freely-mutable dataclass (unlike semantic.py's own
        Type, which is deliberately frozen), so this is a normal,
        unremarkable pattern here, not a workaround."""
        if isinstance(parsed_type, SliceTypeExpr):
            array_literal = self.parse_array_literal()
            array_literal.type_expr = ArrayTypeExpr(
                size=len(array_literal.elements),
                element_type=parsed_type.element_type,
            )
            return Slice(array=array_literal, low=None, high=None)
        return self.parse_array_literal(type_expr=parsed_type)

    def parse_array_literal(self, type_expr: Optional['ArrayTypeExpr'] = None) -> ArrayLiteral:
        """`[e1, e2, ...]`, optionally preceded by an already-parsed
        type (`type_expr`, from _parse_bracketed_literal) for the
        fully-typed form, `[3]int[1, 2, 3]` -- see ArrayLiteral's own
        docstring for why no special handling is needed here for a
        multi-dimensional literal like `[[1,2,3],[4,5,6]]`: each
        element is just parsed via the ordinary parse_expression,
        which naturally recurses back into this same method (with
        type_expr staying None, since only the OUTERMOST literal in
        `[2][3]int[[1,2,3],[4,5,6]]` is ever preceded by an explicit
        type -- an inner row is still just `[1,2,3]`, untyped, exactly
        like the untyped form's own nested rows already were)."""
        self.expect(TokenType.OPEN_BRACKET)
        elements = []
        if not self.check(TokenType.CLOSE_BRACKET):
            elements.append(self.parse_expression())
            while self.match(TokenType.COMMA):
                elements.append(self.parse_expression())
        self.expect(TokenType.CLOSE_BRACKET, "Expected ']' to close array literal")
        return ArrayLiteral(elements=elements, type_expr=type_expr)

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
