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
from dataclasses import dataclass, field, fields
from enum import auto, Enum
from typing import Any, List, Optional, Tuple, Union

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


_PRETTY_MAX_WIDTH = 88  # a compact-but-not-cramped line budget, matching
                         # black's own default -- not chosen to match any
                         # property of Hornet source itself, just a
                         # reasonable width for a human to scan
_PRETTY_INDENT = "    "  # four spaces per nesting level


def _pretty_scalar(value: Any) -> str:
    """Renders a single non-Node, non-list field value. UnaryOp/
    BinaryOp render as their own readable .symbol() (`+`, `not`, ...)
    rather than a raw `BinaryOp.ADD`-style enum repr -- the one place
    this whole mechanism still special-cases a specific type, since an
    operator's own symbol is genuinely more scannable here than its
    enum member name, and there are only these two enums to ever reach
    this function at all (every other field is a plain str/int/float/
    bool/None, or a Node/list, both handled elsewhere). Still passed
    through repr() same as everything else, though, not returned bare
    -- an unquoted symbol glued directly onto its own `op=` prefix
    with no delimiter is genuinely ambiguous for a multi-character
    operator (`op===` for `==`, easy to misread as a typo or a
    different operator entirely); quoting it like any other string
    value (`op='=='`) removes that ambiguity the same way quoting
    already does for every other string-valued field."""
    if isinstance(value, (UnaryOp, BinaryOp)):
        value = value.symbol()
    return repr(value)


def _pretty_value(value: Any, indent: int) -> str:
    """Renders any field value -- a Node, a list, or a scalar -- at
    nesting depth `indent`. Returned text's own FIRST line never has
    leading whitespace of its own (every caller is about to place it
    right after a `name=` prefix, or as a bare list entry, which
    already provides whatever comes before it on that line); every
    line AFTER the first, if there are any, is already indented
    exactly `indent` levels deep -- so a caller embedding a multi-line
    child never has to re-indent it itself, only decide where its own
    first line goes."""
    if isinstance(value, Node):
        return _pretty_node(value, indent)
    if isinstance(value, list):
        return _pretty_list(value, indent)
    return _pretty_scalar(value)


def _pretty_list(items: list, indent: int) -> str:
    """Renders a list field -- Function.body, Call.args, and so on.
    Empty is always the bare `[]`, regardless of context. A non-empty
    list tries one line first (`[Constant(value=1), Constant(value=2)]`)
    the same way _pretty_node does, falling back to one indented item
    per line, each with its own trailing comma, only if that doesn't
    fit within _PRETTY_MAX_WIDTH -- so a short param list or a small
    handful of simple statements stays compact, while a real function
    body reliably unfolds into a scannable, one-statement-per-line
    block."""
    if not items:
        return "[]"
    rendered = [_pretty_value(item, indent + 1) for item in items]
    if not any('\n' in r for r in rendered):
        candidate = f"[{', '.join(rendered)}]"
        if indent * len(_PRETTY_INDENT) + len(candidate) <= _PRETTY_MAX_WIDTH:
            return candidate
    inner = ",\n".join(f"{_PRETTY_INDENT * (indent + 1)}{r}" for r in rendered)
    return f"[\n{inner},\n{_PRETTY_INDENT * indent}]"


def _pretty_node(node: 'Node', indent: int) -> str:
    """Renders one Node as `ClassName(field=value, field=value, ...)`,
    driven entirely by dataclasses.fields(node) -- this works
    identically for every Node subclass with no per-type code at all,
    since a dataclass's own fields() are introspectable generically
    regardless of which specific subclass an instance is. `resolved_
    type` is skipped everywhere it appears (see Node.pretty's own
    docstring for why). A node with no fields left to show (Break,
    Continue, or any node whose only field was resolved_type) renders
    as a bare `ClassName()`.

    Tries the whole node on one line first, exactly like _pretty_list
    does, falling back to one indented `field=value` line per field --
    each value recursively rendered the SAME way, so a field whose own
    value is short still collapses to one line for itself even when
    the enclosing node as a whole didn't fit -- only when it doesn't
    fit within _PRETTY_MAX_WIDTH. This one rule, applied uniformly and
    recursively, is the entire mechanism: a small, simple node (a bare
    Constant, a Variable) always renders compactly regardless of how
    deep it's nested, while a large, genuinely nested structure (a
    chain of Binary operations, a Function with a real body) naturally
    unfolds into an indented tree, with no per-node-type layout
    decisions anywhere in this file."""
    class_name = type(node).__name__
    field_names = [f.name for f in fields(node) if f.name != 'resolved_type']
    if not field_names:
        return f"{class_name}()"

    rendered = {name: _pretty_value(getattr(node, name), indent + 1) for name in field_names}
    if not any('\n' in r for r in rendered.values()):
        candidate = f"{class_name}({', '.join(f'{n}={rendered[n]}' for n in field_names)})"
        if indent * len(_PRETTY_INDENT) + len(candidate) <= _PRETTY_MAX_WIDTH:
            return candidate

    inner = ",\n".join(f"{_PRETTY_INDENT * (indent + 1)}{n}={rendered[n]}" for n in field_names)
    return f"{class_name}(\n{inner},\n{_PRETTY_INDENT * indent})"


class Node:
    """Base class for all AST nodes.

    pretty() is implemented ONCE, here, generically -- via dataclasses.
    fields() introspection (see _pretty_node/_pretty_list/_pretty_value
    above), rather than as 29 individual hand-written methods, one per
    subclass, the way this used to work. Every node renders as
    `ClassName(field=value, field=value, ...)`, with Node-valued and
    list-valued fields recursing through that exact same machinery,
    falling back from a single compact line to an indented, one-
    field-per-line block whenever a node (or a list of them) doesn't
    fit comfortably within one line -- inspired by astpretty
    (https://github.com/asottile/astpretty), which pretty-prints
    stdlib Python ASTs via this identical "one line if it fits, an
    indented tree if it doesn't" rule, rather than ast.dump's own
    single unbroken line for an entire tree regardless of size.

    A newly added Node subclass needs no pretty() of its own at all to
    be correctly, consistently rendered -- a real advantage over the
    old scheme once this AST had grown past a couple dozen node types:
    each one's own bespoke method (a mix of "->"-arrows, ad hoc
    brackets, and "; "-joins, hand-picked per node to keep a single
    line readable) had become exactly the kind of thing that gets
    harder to visually parse as the underlying structures do, and
    harder still to keep looking consistent across every node as more
    are added.

    `resolved_type` is deliberately never shown: it's None on every
    node before semantic analysis runs, so printing `resolved_type=
    None` on every single leaf would be pure noise for the most common
    use of this (inspecting a freshly parsed tree); once semantic
    analysis has run, it holds a real semantic.Type, which anything
    that specifically needs it can already read directly off the node
    rather than needing it spelled out in a debug print.
    """

    def pretty(self) -> str:
        return _pretty_node(self, indent=0)


@dataclass
class Constant(Node):
    value: Union[int, float]
    # Set by semantic.py's check_expr after type-checking, not at parse
    # time -- see this field's fuller explanation on StringLiteral
    # below, which was the first node to need it documented in detail.
    resolved_type: Optional[Any] = None


@dataclass
class BoolLiteral(Node):
    """`true` or `false`. Kept as its own node rather than folded into
    Constant -- Python's bool is a subclass of int, and overloading
    Constant.value to sometimes hold one would make "is this actually an
    int or a bool" ambiguous exactly where semantic.py most needs it to
    be unambiguous."""
    value: bool
    resolved_type: Optional[Any] = None


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


@dataclass
class Variable(Node):
    """A reference to a local variable, e.g. the `a` in `a + 1`."""
    name: str
    resolved_type: Optional[Any] = None


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


@dataclass
class Slice(Node):
    """`array[low:high]` -- a VIEW into `array` spanning [low, high):
    low inclusive, high exclusive, matching Go's own convention (which
    this whole feature is deliberately modeled on). Produces a Slice-
    typed VALUE (conceptually a {pointer, length, capacity} descriptor
    -- see codegen.py's SLICES section for the concrete representation),
    not a copy of the underlying elements: a slice is a genuine alias
    into its base's own backing storage, unlike plain array assignment
    (`arr2 = arr1`), which already copies. This is exactly the aliasing
    surface that makes stack safety genuinely load-bearing once slicing
    exists, not merely a nice-to-have: a slice that outlives the stack
    frame its own backing array lives in becomes a dangling pointer the
    moment that frame is torn down -- see codegen.py's own analyze_
    array_escapes for the actual mechanism (an escape analysis, not
    just a size check) that decides which arrays need to survive that.

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
    doesn't parse as anything at all (parse_expr_stmt_or_assign's
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


@dataclass
class Call(Node):
    """`name(arg1, arg2, ...)` -- a function call, used as an ordinary
    expression. There's no separate "call statement" concept: `foo(1)`
    alone on its own line already parses as an expr_stmt wrapping this
    (see ExprStmt), exactly the same way any other expression can be a
    bare statement -- a call just happens to be one whose value people
    often want to discard.

    Also `Name(x=1, y='a')` -- NAMED-field construction for a struct
    literal (`kwargs` populated, `args` left empty), as an alternative
    to positional construction (`args` populated, `kwargs` left None
    -- the ORIGINAL, unchanged representation every ordinary function
    call and every positional struct literal still uses). The two are
    mutually exclusive by construction, never both populated on the
    same node: parse_call enforces "no mixing positional and named
    arguments in one call" as a hard GRAMMAR rule (a ParseError,
    raised immediately at the offending argument), not a semantic one
    -- matching Python's own identical restriction being a SyntaxError
    rather than something deferred to runtime. This is a pure syntax-
    shape question, independent of what `name` actually refers to
    (unlike almost everything else about a Call, which the parser
    deliberately leaves to semantic.py to resolve -- see check_call's
    own struct-vs-function dispatch): whether an argument list mixes
    `expr` and `name=expr` entries is visible from the tokens alone,
    with no need to know if `name` names a struct, a function, or
    nothing declared at all.

    kwargs is a List[Tuple[str, Node]], not a Dict[str, Node], and
    deliberately preserves the ORDER each field was written in --
    duplicate names (`A(x=1, x=2)`) are consequently NOT rejected here
    either, for the identical reason mixing isn't checked here: the
    parser has no symbol table and doesn't know `x` is meant to name a
    real, unique struct field at all, only that it's an identifier
    followed by '='. Both duplicate-name and unknown-field-name
    rejection are semantic.py's job (see check_struct_literal), the
    same "parser accepts the shape, semantic.py validates the
    meaning" split this file already uses for the struct/function
    name itself.

    NAMED construction is deliberately scoped to struct literals only,
    not a general "call functions with named arguments" feature -- an
    ordinary function call written with named arguments (`foo(x=1)`)
    parses into exactly the same shape (kwargs populated) as a named
    struct literal does, syntactically indistinguishable at parse
    time, and is rejected by check_call once it resolves `foo` to an
    ordinary function rather than a struct (see its own docstring).
    Partial construction (omitting a field entirely) leaves that
    field's own storage genuinely uninitialized, matching every other
    place this language already treats uninitialized memory as real,
    unwritten UB rather than implicitly zero -- not, for now, filled
    with a zero value the way Go's own struct literals would; that's
    an intentionally separate, larger piece of follow-up work.

    Also `receiver.name(args)` -- a METHOD call (`receiver` populated;
    None for every other shape above). Parsed by parse_postfix, not
    parse_call: the receiver is whatever expression already preceded
    the '.', built by the SAME postfix loop that already builds a
    Field chain, so `a.b.method(1)` (a method call at the end of a
    longer field-access chain) already falls out with no special
    casing needed beyond "does '(' follow the name". Arguments are
    always positional here -- see parse_postfix's own note on why
    named arguments don't extend to method calls (at least not yet).

    Only alive as a distinct shape for the DURATION of semantic
    analysis: check_call's own _check_method_call resolves the
    receiver's type, looks up the matching method, and then REWRITES
    this exact node in place -- prepending the receiver into args as
    an ordinary first argument, replacing `name` with a mangled,
    guaranteed-collision-free symbol name, and setting receiver back
    to None -- so that by the time codegen.py (or any of the many
    isinstance(expr, Call) checks throughout this file) ever sees it,
    a method call is completely indistinguishable from an ordinary
    call to that mangled function. This in-place rewrite is exactly
    the same "mutate the node semantic.py already holds a reference
    to" pattern check_expr's own resolved_type annotation already
    uses, just touching two more fields -- deliberately NOT a new,
    separate AST node type kept alive all the way through codegen,
    which would otherwise mean auditing every existing isinstance(
    expr, Call) check across this codebase (there are many: array/
    struct/slice value production, argument materialization, print,
    append, ...) to also recognize it."""
    name: str
    args: List[Node] = field(default_factory=list)
    kwargs: Optional[List[Tuple[str, Node]]] = None
    resolved_type: Optional[Any] = None
    receiver: Optional[Node] = None


@dataclass
class Unary(Node):
    op: UnaryOp
    operand: Node
    resolved_type: Optional[Any] = None


@dataclass
class Cast(Node):
    """`TYPE(expr)` -- an explicit numeric cast, e.g. `int8(x)`,
    `uint8(255)`, `int(someByteValue)`. Syntactically identical in
    SHAPE to an ordinary function call (a name immediately followed by
    a single parenthesized argument), but distinguished at PARSE time
    rather than left to semantic.py the way Call's own struct-vs-
    function ambiguity is: target_type here is always one of the five
    built-in scalar type KEYWORDS (int, int8, uint8, bool, str -- see
    parse_primary's own dispatch), a lexically distinct token kind
    from IDENTIFIER -- so there's no genuine ambiguity with a struct
    literal or an ordinary function call to disambiguate later the way
    Call needs to: a struct name or a type ALIAS name is always an
    IDENTIFIER token, never one of these five keywords, so neither can
    ever produce a Cast node, no matter what it resolves to.

    Deliberately scoped to a SINGLE argument, unlike Call's own list --
    a cast only ever converts one value, so there's no positional/
    named-argument shape to disambiguate here at all.

    Only int/int8/uint8 are actually SUPPORTED on either side of a
    cast right now (see semantic.py's check_cast) -- bool and str are
    still accepted HERE, at the parse level, since the parser has no
    reason to know which specific (source, target) pairs are valid;
    that's semantic.py's job, matching this file's consistent "parser
    accepts the shape, semantic.py validates the meaning" split seen
    throughout (e.g. Call's own struct-vs-function resolution, or a
    named argument's field name never being checked against a real
    struct here either).

    Casting to a STRUCT name or a type ALIAS name (`MyByte(x)`, where
    `type MyByte = int8`) is NOT supported by this node at all -- such
    an expression parses as an ordinary Call instead (see parse_
    primary's own IDENTIFIER branch), which check_call has no cast-
    aware case for, so it's rejected as an undeclared-function or a
    struct-literal error, whichever check_call reaches first. A
    separate, narrower gap, matching the identical, already-documented
    one for constructing a struct via its own alias name (see
    _collect_type_aliases's own docstring)."""
    target_type: str
    expr: Node
    resolved_type: Optional[Any] = None


@dataclass
class Binary(Node):
    op: BinaryOp
    left: Node
    right: Node
    resolved_type: Optional[Any] = None


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


@dataclass
class VarDecl(Node):
    """`int a` (init=None) or `int a = 1` (init=the initializer expression).
    `var_type` is a plain type keyword string ('int'/'bool'/'str'), an
    ArrayTypeExpr, or a SliceTypeExpr (see their own docstrings) for an
    array- or slice-typed decl."""
    name: str
    var_type: Union[str, ArrayTypeExpr, SliceTypeExpr]
    init: Optional[Node] = None


@dataclass
class Assign(Node):
    """`a = <value>`, assigning to an already-declared variable."""
    name: str
    value: Node


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


@dataclass
class Field(Node):
    """`base.name` -- reads a single field out of a struct-typed
    `base`. Unlike Index, there's no separate "nested" shape to worry
    about for a multi-level chain (`a.b.c`): it's just Field(base=
    Field(base=Variable('a'), name='b'), name='c'), one node per '.',
    exactly mirroring how Index nests for `matrix[i][j]` -- and the
    two kinds of suffix chain together freely too (`a.b[0]`, `arr[0].f`,
    `a.b.c[0].d`), since parse_postfix's own loop builds both in a
    single pass with no special-casing for which comes first (see its
    own docstring)."""
    base: Node
    name: str
    resolved_type: Optional[Any] = None


@dataclass
class FieldAssign(Node):
    """`base.name = value` -- writes a single struct field. Mirrors
    IndexAssign exactly, one level over: `base` is the expression
    being accessed -- a bare Variable for `s.f = v`, or itself a
    Field or Index node for a longer chain (`s.inner.f = v`,
    `s.rows[0].f = v`, `rows[0].f.g = v`) -- built the same way
    IndexAssign's own multi-dimensional target is, by parsing the
    whole left-hand expression first and reinterpreting it as an
    assignment target afterward (see parse_expr_stmt_or_assign).

    Like IndexAssign, compound assignment (`s.f += 1`) is deliberately
    rejected with a clear error rather than silently mis-parsed or
    silently accepted -- for the identical reason IndexAssign's own
    docstring gives: a target like `s.rows[i()].f` could contain a
    side-effecting sub-expression that a naive desugaring into
    `s.rows[i()].f = s.rows[i()].f + 1` would evaluate twice."""
    base: Node
    name: str
    value: Node


@dataclass
class StructField(Node):
    """One field declaration inside a struct body: `type name`, with
    no initializer -- a struct has no per-field default VALUES the way
    a VarDecl can have; every field simply starts at its own type's
    zero value (0 for int, false for bool, '' for str, a zeroed
    backing for an array, ...) until explicitly assigned, the same way
    an uninitialized local variable already does."""
    name: str
    field_type: Union[str, 'ArrayTypeExpr', 'SliceTypeExpr']


@dataclass
class MethodDef(Node):
    """A method declared inside a struct body: `def [type] name(receiver,
    param2, ...):` -- syntactically an ordinary `def`, except its FIRST
    parameter (the receiver) is written as a bare, untyped identifier
    rather than `type name` -- the enclosing struct's own name is what
    implicitly gives it its type, the same way `self`/`this` in other
    languages doesn't need its own type written out. Every parameter
    AFTER the receiver uses ordinary Param syntax, and return_type
    follows Function's own identical rule (None means no declared
    return type at all, not a void keyword).

    Never reaches semantic.py as its own concept for long: analyze()'s
    own struct-method-collection pass (_collect_methods) immediately
    synthesizes an ordinary Function from each MethodDef -- the
    receiver becomes an ordinary first Param, typed as the enclosing
    struct -- and appends it directly to Program.functions, with a
    mangled name (StructName.methodName, using '.', a character that
    can never appear inside a single Hornet IDENTIFIER token -- see
    the lexer's own IDENTIFIER regex, r'[a-zA-Z_]\\w*' -- so a mangled
    name can never collide with anything a user could actually write,
    with no explicit collision check needed anywhere) to avoid
    colliding with an unrelated method of the same name on a different
    struct, or with an ordinary free function of the same name. From
    that point on, every later pass (signature collection, body-
    checking) and all of codegen.py treat it as an entirely ordinary
    function; neither ever needs to know MethodDef existed at all."""
    receiver_name: str
    name: str
    return_type: Optional[Union[str, ArrayTypeExpr, SliceTypeExpr]]
    params: List['Param'] = field(default_factory=list)
    body: List[Node] = field(default_factory=list)


@dataclass
class StructDef(Node):
    """`struct Name: <field-or-method>+` -- declares a new, NOMINAL type
    (see semantic.py's own struct-registry pass for what nominal typing
    means here and why it falls out naturally from how Type's own
    equality already works). Field order is preserved exactly as
    written, since it determines both codegen's own memory layout
    (fields are laid out at sequential byte offsets in declaration
    order, with no reordering) and print's own field-printing order.

    Fields and methods can be freely interleaved -- parse_struct_def's
    own body loop just checks, line by line, whether the next token is
    `def` (a method) or a type (a field), with no ordering requirement
    between the two kinds. This isn't a deliberately permissive design
    choice so much as the natural one: unlike a field, a method never
    participates in the struct's own memory layout at all (it's fully
    lowered away into an ordinary top-level function before codegen
    ever runs -- see MethodDef's own docstring), so there's no layout
    reason to require methods to come after every field, and enforcing
    an ordering restriction anyway would need its own extra state and
    its own error message for no actual benefit. At least one FIELD is
    still required (see parse_struct_def's own check) -- a struct with
    zero fields isn't supported at all yet -- but methods are entirely
    optional, and a struct with none at all is exactly what every
    struct looked like before this feature existed."""
    name: str
    fields: List[StructField] = field(default_factory=list)
    methods: List[MethodDef] = field(default_factory=list)


@dataclass
class ExprStmt(Node):
    """A bare expression used as a full statement, e.g. `2 + 2` on its
    own line -- evaluated for any side effects (there are none yet, but
    this is also how a future function-call-as-statement would look) and
    then discarded."""
    expr: Node


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


@dataclass
class While(Node):
    """`while cond: <body>`. Loops as long as `cond` evaluates to true,
    re-checking it before every iteration including the first (so a
    false condition means the body never runs at all)."""
    condition: Node
    body: List[Node]


@dataclass
class Break(Node):
    """`break` -- exits the *innermost* enclosing loop immediately.
    Only valid inside a while body; semantic.py rejects one that isn't
    (see its loop_depth tracking)."""


@dataclass
class Continue(Node):
    """`continue` -- skips the rest of the current iteration of the
    *innermost* enclosing loop and jumps straight to re-checking its
    condition. Same "only valid inside a while" rule as Break."""


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


@dataclass
class TypeAlias(Node):
    """`type Name = TargetType` -- a top-level declaration introducing
    `Name` as an alternate spelling for an existing type, interchangeable
    with it everywhere (an ALIAS, not a Go-style newtype/"defined type"
    -- `type Name TargetType`, with no '=', would be the newtype form;
    deliberately not what this represents, and not currently supported
    at all). `target_type` is parsed via the ordinary parse_type(), the
    exact same grammar a VarDecl/Param/struct field's own type uses --
    syntactically, an alias's target can be anything parse_type()
    accepts, including an array or slice type expression or a struct
    name; semantic.py's own _collect_type_aliases is what actually
    narrows this down to int/bool/str or another alias for now (see its
    own docstring for exactly why, and what would need to change to
    lift that restriction later). This mirrors how struct field types
    are parsed broadly and validated precisely elsewhere in this file
    -- the parser accepts the general shape, semantic.py enforces the
    specific rule.

    Resolved once, centrally, by threading a new `aliases` registry
    through type_from_name (the one function every OTHER type-name
    resolution in this codebase -- VarDecl, Param, a struct field, a
    function's own return type -- already calls) -- so every one of
    those call sites gains alias support automatically, with no
    changes needed at any of them beyond passing that registry through
    like they already do for the struct registry."""
    name: str
    target_type: Union[str, ArrayTypeExpr, SliceTypeExpr]


@dataclass
class Program(Node):
    functions: List[Function] = field(default_factory=list)
    structs: List[StructDef] = field(default_factory=list)
    type_aliases: List[TypeAlias] = field(default_factory=list)

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
        structs = []
        type_aliases = []
        self.skip_newlines()
        while not self.at_end():
            if self.check(TokenType.STRUCT):
                structs.append(self.parse_struct_def())
            elif self.check(TokenType.TYPE):
                type_aliases.append(self.parse_type_alias())
            else:
                functions.append(self.parse_function())
            self.skip_newlines()
        return Program(functions=functions, structs=structs, type_aliases=type_aliases)

    def parse_type_alias(self) -> TypeAlias:
        """`type Name = TargetType` -- a single-line, top-level
        declaration; no body, no indented block, unlike struct/function/
        method definitions. `Name` is an ordinary IDENTIFIER (never one
        of the reserved type keywords -- 'int'/'bool'/'str' are their
        own token types, never tokenized as IDENTIFIER at all, so
        `type int = ...` is rejected by the very next `expect` call,
        not by a special check here). TargetType reuses parse_type()
        directly -- see TypeAlias's own docstring for why the PARSER
        accepts its full generality here even though semantic.py
        currently narrows what's actually allowed."""
        self.expect(TokenType.TYPE, "Expected 'type' to start a type alias")
        name_tok = self.expect(TokenType.IDENTIFIER, "Expected a name for this type alias")
        self.expect(TokenType.ASSIGN, "Expected '=' in a type alias declaration")
        target_type = self.parse_type()
        self.expect(TokenType.NEWLINE, "Expected a newline after a type alias declaration")
        return TypeAlias(name=name_tok.val, target_type=target_type)

    def parse_struct_def(self) -> StructDef:
        """`struct Name: <field-or-method>+` -- a top-level declaration,
        parsed the same general way a function is: header line, then an
        indented block, with the lexer's own INDENT/DEDENT tokens (see
        lexer.py's tokenize()) already doing the real work of finding
        where the body starts and ends, regardless of what's actually
        being indented. Each FIELD line is just `type name` -- no
        initializer, no assignment -- so this reuses parse_type()
        directly rather than parse_var_decl (which exists specifically
        to also handle an optional `= value` a struct field can't
        have). Each METHOD line starts with `def`, unambiguously --
        disambiguated from a field with a single token of lookahead,
        since a field's own type can never start with the `def`
        keyword -- and is delegated to parse_method_def entirely; see
        StructDef's own docstring for why fields and methods can freely
        interleave rather than needing methods to come after every
        field."""
        self.expect(TokenType.STRUCT, "Expected 'struct'")
        name_tok = self.expect(TokenType.IDENTIFIER, "Expected a struct name")
        self.expect(TokenType.COLON, "Expected ':' to start the struct body")
        self.expect(TokenType.NEWLINE, "Expected a newline after ':'")
        self.skip_newlines()
        self.expect(TokenType.INDENT, "Expected an indented struct body")
        self.skip_newlines()
        fields: List[StructField] = []
        methods: List[MethodDef] = []
        while not self.check(TokenType.DEDENT) and not self.at_end():
            if self.check(TokenType.DEF):
                methods.append(self.parse_method_def())
            else:
                field_type = self.parse_type()
                field_name_tok = self.expect(TokenType.IDENTIFIER, "Expected a field name")
                self.expect(TokenType.NEWLINE, "Expected a newline after a field declaration")
                fields.append(StructField(name=field_name_tok.val, field_type=field_type))
            self.skip_newlines()
        self.expect(TokenType.DEDENT, "Expected a dedent to end the struct body")
        if not fields:
            raise ParseError(
                f"Expected at least one field in struct '{name_tok.val}'"
            )
        return StructDef(name=name_tok.val, fields=fields, methods=methods)

    def _check_starts_with_return_type(self) -> bool:
        """True if the CURRENT position starts an optional return type
        before a def's own name -- shared by parse_function and parse_
        method_def, since both need the identical one-or-two-token
        lookahead to tell `def Point make(...)` (a struct-typed return,
        an IDENTIFIER immediately followed by ANOTHER identifier) apart
        from `def make(...)` (no declared return type at all -- a
        single IDENTIFIER, the def's own name, with no return type
        before it).

        A type keyword or '[' starts a return type unambiguously with
        just one token of lookahead -- a function or method name can
        never itself be one of those, since they're reserved. A
        struct-typed return is the one case that genuinely needs a
        SECOND token of lookahead rather than just the first:
        IDENTIFIER alone is ambiguous between "this names a struct
        return type" and "this is the def's own name", resolved by
        peeking one token further -- a SECOND identifier immediately
        after can only mean the first one was a type name, since a
        function or method's own name is always immediately followed
        by '(', never another identifier. Exactly the same two-vs-one-
        IDENTIFIER disambiguation parse_statement's own struct-typed-
        VarDecl check already needs, for the identical underlying
        reason (struct names are ordinary identifiers, not reserved
        keywords, so there's no way to tell "this identifier is a
        type" from "this identifier is something else" with only one
        token of lookahead)."""
        return self.check(TokenType.INT, TokenType.INT8, TokenType.UINT8, TokenType.BOOL, TokenType.STR, TokenType.OPEN_BRACKET) or (
            self.check(TokenType.IDENTIFIER) and self.peek(1).type == TokenType.IDENTIFIER
        )

    def parse_function(self) -> Function:
        self.expect(TokenType.DEF, "Expected 'def' to start a function definition")
        return_type = self.parse_type() if self._check_starts_with_return_type() else None
        name_tok = self.expect(TokenType.IDENTIFIER, "Expected a function name")
        self.expect(TokenType.OPEN_PAREN, "Expected '(' after function name")
        params = self.parse_params()
        self.expect(TokenType.CLOSE_PAREN, "Expected ')' after parameter list")
        self.expect(TokenType.COLON, "Expected ':' to start the function body")
        self.expect(TokenType.NEWLINE, "Expected a newline after ':'")

        body = self.parse_block()
        return Function(name=name_tok.val, return_type=return_type, params=params, body=body)

    def parse_method_def(self) -> MethodDef:
        """`def [type] name(receiver, param2, ...):` -- mirrors parse_
        function closely (same optional-return-type lookahead, same
        body parsing), except the FIRST entry in the parameter list is
        always the receiver, written as a bare, untyped IDENTIFIER with
        no type-disambiguation logic needed at all: unlike an ordinary
        Param, there's no ambiguity to resolve here -- the very next
        token after '(' is unconditionally the receiver's own name,
        since a method is required to have exactly one (see MethodDef's
        own docstring for why: no pointers yet means no reference
        receiver, and no receiver-less "static" methods are supported
        in this first cut either). Every parameter AFTER the receiver
        is ordinary Param syntax, reusing parse_param directly."""
        self.expect(TokenType.DEF, "Expected 'def' to start a method definition")
        return_type = self.parse_type() if self._check_starts_with_return_type() else None
        name_tok = self.expect(TokenType.IDENTIFIER, "Expected a method name")
        self.expect(TokenType.OPEN_PAREN, "Expected '(' after method name")
        receiver_tok = self.expect(TokenType.IDENTIFIER, "Expected a receiver name as a method's first parameter")
        params: List[Param] = []
        while self.match(TokenType.COMMA):
            params.append(self.parse_param())
        self.expect(TokenType.CLOSE_PAREN, "Expected ')' after parameter list")
        self.expect(TokenType.COLON, "Expected ':' to start the method body")
        self.expect(TokenType.NEWLINE, "Expected a newline after ':'")

        body = self.parse_block()
        return MethodDef(receiver_name=receiver_tok.val, name=name_tok.val, return_type=return_type, params=params, body=body)

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
        if self.check(TokenType.INT, TokenType.INT8, TokenType.UINT8, TokenType.BOOL, TokenType.STR):
            return self.advance().val
        if self.check(TokenType.IDENTIFIER):
            # A struct type reference (`MyStruct`) -- the parser has no
            # symbol table and doesn't know or care whether this name
            # actually names a declared struct; it just accepts any
            # identifier here and hands the bare string on, exactly
            # like it already does for 'int'/'int8'/'uint8'/'bool'/
            # 'str' above. semantic.py's own struct-registry pass is
            # what actually validates the name (see type_from_name).
            return self.advance().val
        tok = self.current()
        raise ParseError(
            f"Expected a type ('int', 'int8', 'uint8', 'bool', 'str', "
            f"a struct name, '[size]type', or '[]type'), got {tok.type} "
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
        if self.check(TokenType.INT, TokenType.INT8, TokenType.UINT8, TokenType.BOOL, TokenType.STR) and self.peek(1).type == TokenType.OPEN_PAREN:
            # A cast expression used as a bare statement (`int8(x)`,
            # discarding its own result -- unusual, but not something
            # to special-case rejecting just because it's unusual),
            # not the start of a VarDecl -- looks IDENTICAL to the
            # type-starting branch just below up to this exact point
            # (both start with one of these five scalar keywords), but
            # a scalar type keyword is never itself immediately
            # followed by '(' in any valid VarDecl (that position
            # always holds the variable's own NAME, an IDENTIFIER
            # token, never an open paren) -- so this one token of
            # lookahead unambiguously tells the two apart before ever
            # committing to parse_type() below. See Cast's own
            # docstring for why a struct name or a type alias name
            # never has this same ambiguity to resolve at all: neither
            # is ever one of these five keyword token types, so
            # `MyStruct(...)`/`MyAlias(...)` was already, correctly,
            # routed through parse_call's own IDENTIFIER branch, with
            # nothing here needing to change for either.
            return self.parse_expr_stmt_or_assign()
        if self.check(TokenType.INT, TokenType.INT8, TokenType.UINT8, TokenType.BOOL, TokenType.STR, TokenType.OPEN_BRACKET):
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
        if self.check(TokenType.IDENTIFIER) and self.peek(1).type == TokenType.IDENTIFIER:
            # Two consecutive identifiers can only mean a struct-typed
            # VarDecl (`MyStruct x = ...`, or `MyStruct x` with no
            # initializer -- structs have no literal syntax yet to
            # disambiguate against, unlike the INT/BOOL/STR/
            # OPEN_BRACKET case above, so there's no second check
            # needed here the way there is there). A bare IDENTIFIER
            # alone is NOT enough to signal this -- it's otherwise
            # ambiguous with a variable reference, a function call, a
            # field access, ... -- so this needs the SECOND token to
            # disambiguate before parse_type() is ever called.
            parsed_type = self.parse_type()
            return self.parse_var_decl(var_type=parsed_type)
        if self.check(TokenType.IDENTIFIER) and self.peek(1).type in _ASSIGNMENT_TOKENS:
            return self.parse_assign()
        return self.parse_expr_stmt_or_assign()

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

    def parse_expr_stmt_or_assign(self) -> Node:
        """Handles three statement shapes that -- unlike plain
        `a = ...` -- can't be told apart by a single token of
        lookahead: a bare expression statement (`arr[i] + 1`, or just
        `foo()`), an index-assignment (`arr[i] = value`, or
        `matrix[i][j] = value` for deeper nesting), and a field-
        assignment (`s.f = value`, or `s.inner.f = value` /
        `s.rows[0].f = value` for a longer chain). The number and mix
        of `[...]`/`.name` suffixes on the left varies, so instead of
        trying to look ahead through however many of them there might
        be, this just parses the leading expression through the
        ordinary machinery first (which already builds nested Index
        and Field nodes for however many suffixes follow, in whatever
        order -- see parse_postfix), then decides based on what kind of
        node came out and what comes next.

        Only plain `=` is handled for either assignable shape --
        `arr[i] += 1` and `s.f += 1` are both deliberately rejected
        with a clear error rather than silently mis-parsed (see
        IndexAssign's and FieldAssign's own docstrings for why compound
        assignment isn't supported yet at all, not just here)."""
        expr = self.parse_expression()
        if self.check(TokenType.ASSIGN):
            if isinstance(expr, Index):
                self.advance()
                value = self.parse_expression()
                return IndexAssign(array=expr.array, index=expr.index, value=value)
            if isinstance(expr, Field):
                self.advance()
                value = self.parse_expression()
                return FieldAssign(base=expr.base, name=expr.name, value=value)
            tok = self.current()
            raise ParseError(
                f"Left-hand side of '=' is not assignable "
                f"at line {tok.line}, column {tok.col}"
            )
        if isinstance(expr, (Index, Field)) and self.current().type in _COMPOUND_ASSIGN_OPS:
            tok = self.current()
            article_and_kind = "an array element" if isinstance(expr, Index) else "a struct field"
            raise ParseError(
                f"Compound assignment to {article_and_kind} ('{tok.val}') "
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
        """Wraps a primary expression with zero or more `[...]`,
        `.name`, or `.name(...)` suffixes -- `[...]` is either an index
        (`matrix[i][j]`, parsed here as `parse_primary` producing the
        `matrix` Variable, then this loop wrapping it in two nested
        Index nodes, one per bracket pair -- see Index's own docstring
        for why that nesting, rather than a single Index carrying a
        list of indices, is the right shape) or a slice (`arr[low:
        high]`, see _parse_index_or_slice for how the two are told
        apart within one `[...]` pair); `.name` alone is a field access
        (`s.f`, wrapped in a Field node the same nesting way -- see
        Field's own docstring), while `.name(...)` -- the SAME `.name`
        immediately followed by '(' -- is a METHOD CALL instead,
        wrapped in an ordinary Call node with `receiver` set to
        whatever expression preceded the '.' (see Call's own docstring
        for why this reuses Call rather than introducing a whole new
        node type). All four shapes chain together freely, in any
        order -- `a.b[0]`, `arr[0].f`, `s[1:3][0]`, `a.b.c[0].d`,
        `a.b.method(1)[0]` -- falling out of this one loop with no
        special-casing for which suffix comes first or how many of
        each follow, exactly like chained indexing alone already did
        before Field (and now method calls) existed.

        Sits between parse_unary and parse_primary specifically so
        indexing/slicing/field-access/method-calls all bind TIGHTER
        than a prefix unary operator: `-arr[0]` has to mean `-(arr[0])`,
        not `(-arr)[0]` (which wouldn't even type-check, since unary
        '-' requires an int operand, not an array) -- and it does,
        here, since parse_unary's own base case (no unary operator
        present) calls straight into this method rather than
        parse_primary directly.
        """
        expr = self.parse_primary()
        while self.check(TokenType.OPEN_BRACKET, TokenType.DOT):
            if self.match(TokenType.DOT):
                name_tok = self.expect(TokenType.IDENTIFIER, "Expected a field name after '.'")
                if self.match(TokenType.OPEN_PAREN):
                    args = self.parse_positional_call_args()
                    self.expect(TokenType.CLOSE_PAREN, "Expected ')' after method call arguments")
                    expr = Call(name=name_tok.val, args=args, receiver=expr)
                else:
                    expr = Field(base=expr, name=name_tok.val)
            else:
                self.advance()
                expr = self.parse_index_or_slice(expr)
        return expr

    def parse_positional_call_args(self) -> List[Node]:
        """Parses a plain, comma-separated, purely positional argument
        list for a method call -- unlike parse_call's own argument-
        list parsing, this has no named-argument (`name=value`)
        support at all: that was built specifically for struct
        literals (see Call's own docstring for why it's scoped that
        way), and method calls -- which desugar into an ordinary call
        to a mangled function, see semantic.py's own _check_method_
        call -- don't currently extend that support to their own
        arguments. A natural, separately-scoped follow-up if it's ever
        wanted, matching how named construction itself started narrow
        (struct literals only) and grew position by position."""
        args: List[Node] = []
        if self.check(TokenType.CLOSE_PAREN):
            return args
        args.append(self.parse_expression())
        while self.match(TokenType.COMMA):
            args.append(self.parse_expression())
        return args

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
        if self.check(TokenType.INT, TokenType.INT8, TokenType.UINT8, TokenType.BOOL, TokenType.STR) and self.peek(1).type == TokenType.OPEN_PAREN:
            return self.parse_cast()
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
        slice literal (`[3]int[1, 2, 3]`, `[]int[1, 2, 3]`, or -- since
        struct literals started giving struct names their own literal
        syntax to disambiguate against -- `[3]Point[Point(1,2), ...]`,
        `[]Point[...]`) rather than a plain, untyped array literal
        (`[1, 2, 3]`) or a single-element one (`[5]`) -- both of which
        also start with OPEN_BRACKET, and, for a size-1 case like
        `[5]`, even start with the identical OPEN_BRACKET NUMBER
        CLOSE_BRACKET prefix an array type's own size bracket does.

        Resolved with BOUNDED lookahead alone -- three or four tokens
        depending on which shape it turns out to be, no backtracking
        -- rather than speculatively parsing a type and rewinding on
        failure, which this parser has consistently avoided elsewhere
        (see parse_statement's own one-token dispatch). The two shapes
        that are genuinely unambiguous: an array type's own size
        bracket is `[` NUMBER `]` immediately followed by a type-
        starting token, and a slice type's own (empty) bracket pair is
        `[` `]` immediately followed by one -- a type keyword, an
        IDENTIFIER (a struct name -- see parse_type's own identical
        acceptance of one there), or another `[` for a nested
        array/slice element type (`[2][2]int[...]`, `[][]int[...]`,
        `[2][2]Point[...]`). Neither an untyped literal's closing `]`
        (array) nor a bare `[]` (which, on its own, parses as a valid
        but always semantically-rejected empty array literal -- see
        check_array_literal) is ever immediately followed by one of
        those in any valid program (nothing can validly follow a
        complete expression that way), so checking for either specific
        shape is enough to always tell them apart from their own
        untyped counterparts, including the array side's size-1 edge
        case: `[5]` alone has nothing of that shape following its `]`,
        so it's correctly left to parse as a single-element array
        literal instead.

        IDENTIFIER was added to both lookahead checks only once struct
        literals existed at all -- before that, a struct name here
        could only ever have meant an ordinary VarDecl's own type
        (`Point p`, no literal following it possible), so there was
        nothing for this method to disambiguate in that case and no
        reason to check for it. A BARE, fully-typed literal STATEMENT
        with a struct element type (`[3]Point[Point(1,2), ...]` alone,
        no assignment) never actually needed this fix at all: parse_
        statement's own OPEN_BRACKET branch commits to parse_type()
        directly before this method is ever consulted (see its own
        comment on why that's safe), and parse_type already accepted
        an IDENTIFIER as an element type from the very beginning -- the
        gap this fixes is specifically for a typed struct literal used
        as an EXPRESSION (a VarDecl initializer, an Assign value, a
        function argument, ...), which reaches this method via
        parse_primary instead."""
        if not self.check(TokenType.OPEN_BRACKET):
            return False
        if self.peek(1).type == TokenType.CLOSE_BRACKET:
            return self.peek(2).type in (TokenType.INT, TokenType.INT8, TokenType.UINT8, TokenType.BOOL, TokenType.STR, TokenType.IDENTIFIER, TokenType.OPEN_BRACKET)
        if self.peek(1).type == TokenType.NUMBER and self.peek(2).type == TokenType.CLOSE_BRACKET:
            return self.peek(3).type in (TokenType.INT, TokenType.INT8, TokenType.UINT8, TokenType.BOOL, TokenType.STR, TokenType.IDENTIFIER, TokenType.OPEN_BRACKET)
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
        """`name(arg1, arg2, ...)` or `name(f1=v1, f2=v2, ...)` -- see
        Call's own docstring for the full design; this just needs to
        tell the two argument-list SHAPES apart and refuse to mix them,
        with no idea yet (and no need to know) whether `name` ends up
        being a struct or a function.

        Disambiguated per-argument with exactly one token of lookahead
        past the argument's own start: IDENTIFIER immediately followed
        by ASSIGN ('=', never EQUAL '==' -- the lexer already tokenizes
        those as two different token types, so there's no risk of
        `A(x==1)` -- an ordinary positional argument that happens to
        be a boolean expression -- being mistaken for a named one)
        means `name=value`; anything else is parsed as an ordinary
        positional expression. `start_tok` is captured before parsing
        either shape purely so a mixing error, if one turns out to be
        needed, points at the argument that actually broke the
        pattern, not wherever the token stream happens to be after
        parsing its value."""
        name_tok = self.expect(TokenType.IDENTIFIER)
        self.expect(TokenType.OPEN_PAREN, "Expected '(' to start a call's argument list")
        args: List[Node] = []
        kwargs: Optional[List[Tuple[str, Node]]] = None
        if not self.check(TokenType.CLOSE_PAREN):
            while True:
                start_tok = self.current()
                if self.check(TokenType.IDENTIFIER) and self.peek(1).type == TokenType.ASSIGN:
                    field_name = self.advance().val
                    self.advance()  # consume '='
                    value = self.parse_expression()
                    if args:
                        raise ParseError(
                            f"Cannot mix positional and named arguments in "
                            f"a call -- '{field_name}=...' follows a "
                            f"positional argument at line {start_tok.line}, "
                            f"column {start_tok.col}"
                        )
                    if kwargs is None:
                        kwargs = []
                    kwargs.append((field_name, value))
                else:
                    value = self.parse_expression()
                    if kwargs is not None:
                        raise ParseError(
                            f"Cannot mix positional and named arguments in "
                            f"a call -- a positional argument follows a "
                            f"named one at line {start_tok.line}, column "
                            f"{start_tok.col}"
                        )
                    args.append(value)
                if not self.match(TokenType.COMMA):
                    break
        self.expect(TokenType.CLOSE_PAREN, "Expected ')' to close a call's argument list")
        return Call(name=name_tok.val, args=args, kwargs=kwargs)

    def parse_cast(self) -> Cast:
        """`TYPE(expr)` -- see Cast's own docstring for the full
        design. The caller (parse_primary) has already confirmed the
        current token is one of the five scalar type keywords AND the
        next one is '(', so this just needs to consume both, parse the
        single argument expression in between, and close it -- no
        shape ambiguity left to resolve here at all, unlike parse_
        call's own positional-vs-named dispatch."""
        type_tok = self.advance()
        self.expect(TokenType.OPEN_PAREN, "Expected '(' to start a cast's argument")
        expr = self.parse_expression()
        self.expect(TokenType.CLOSE_PAREN, "Expected ')' to close a cast's argument")
        return Cast(target_type=type_tok.val, expr=expr)


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
