"""Parser

Recursive-descent parser turning the Lexer's token stream into an AST.
Grammar isn't restated here in full (it would drift out of sync the
way an earlier, now-removed sketch did) -- parse_statement/parse_
primary/parse_type are the source of truth for exactly what's
accepted.

Expression parsing: parse_unary recurses on itself (not parse_primary),
so prefix operators chain and are right-associative (`~-2` is
COMPLEMENT(NEGATE(2))). parse_binary uses precedence climbing, driven
by the _BINARY_OPS table (TokenType -> BinaryOp, precedence,
associativity) rather than a cascade of parse_additive/parse_
multiplicative/etc methods -- adding an operator, including a right-
associative one, is a table row, not a restructuring. See the comment
above _BINARY_OPS for the precedence ladder and why bitwise sits below
equality (a deliberate, C-inherited footgun that semantic.py's type
checking turns into a compile error instead of a silent surprise).
parse_postfix sits between parse_unary and parse_primary so indexing/
slicing/field-access/method-calls bind tighter than a prefix operator
(`-arr[0]` is `-(arr[0])`).

Statement dispatch uses one token of lookahead (parse_statement), with
one exception: IDENTIFIER is ambiguous between the start of an
assignment and the start of any other expression referencing that
name, resolved by peeking one token further for an assignment
operator. A type-starting token (INT/BOOL/STR/OPEN_BRACKET) is also
ambiguous, between a VarDecl and a bare, fully-typed array/slice
literal statement (`[3]int[1, 2, 3]`) -- both start with the same
type, so parse_statement parses it once and decides from what follows.

Compound assignment (`+= -= *= /= %= &= |= ^= <<= >>=`) is desugared
directly in parse_assign: `a += b` builds the exact same tree as
`a = a + b` (Assign wrapping Binary), not a dedicated CompoundAssign
node. This is exact, not approximate, specifically because every
assignment target in this language is a bare variable name -- reading
one back has no side effect to worry about duplicating -- so semantic.
py and codegen.py need no changes to support any of the ten operators.

The parser is purely syntactic: it doesn't validate that a referenced
variable was declared, that types match, or that a function's return
matches its declared type -- all of that is semantic.py's job, which
runs after parsing and (unlike the parser) walks statements in program
order, so declare-before-use is enforced there, not here.
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
    """Renders a non-Node, non-list field value. UnaryOp/BinaryOp
    render as their own .symbol() (`+`, `not`, ...), quoted like any
    other string -- unquoted, a multi-character symbol glued onto its
    own `op=` prefix would be ambiguous (`op===` for `==`)."""
    if isinstance(value, (UnaryOp, BinaryOp)):
        value = value.symbol()
    return repr(value)


def _pretty_value(value: Any, indent: int) -> str:
    """Renders a Node, list, or scalar field value at nesting depth
    `indent`. The first line of the result never has leading
    whitespace (the caller places it after a `name=` prefix or as a
    bare list entry); every later line is already indented `indent`
    levels, so embedding a multi-line child needs no re-indenting."""
    if isinstance(value, Node):
        return _pretty_node(value, indent)
    if isinstance(value, list):
        return _pretty_list(value, indent)
    return _pretty_scalar(value)


def _pretty_list(items: list, indent: int) -> str:
    """Renders a list field (Function.body, Call.args, ...). Empty is
    always `[]`. Non-empty tries one line first, falling back to one
    indented item per line (each with a trailing comma) only if that
    doesn't fit within _PRETTY_MAX_WIDTH."""
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
    """Renders one Node as `ClassName(field=value, ...)`, driven by
    dataclasses.fields(node) -- works for every subclass generically,
    with no per-type code. `resolved_type` is always skipped (see
    Node.pretty). A node with no fields left to show renders as a bare
    `ClassName()`.

    Tries the whole node on one line first, like _pretty_list, falling
    back to one indented `field=value` line per field -- each
    recursively rendered the same way -- only when it doesn't fit
    within _PRETTY_MAX_WIDTH."""
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

    pretty() is implemented once, here, generically via dataclasses.
    fields() introspection (see _pretty_node/_pretty_list/_pretty_value
    above) -- a new Node subclass needs no pretty() of its own to be
    correctly rendered. Inspired by astpretty
    (https://github.com/asottile/astpretty): one line if it fits, an
    indented tree if it doesn't, rather than ast.dump's single
    unbroken line regardless of size.

    `resolved_type` is never shown: it's None on every node before
    semantic analysis runs (so printing it would be pure noise for the
    common case of inspecting a freshly parsed tree), and once set,
    anything that needs it reads it directly off the node instead.
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
    """`true` or `false`. Its own node rather than folded into Constant
    -- Python's bool is a subclass of int, and overloading Constant.
    value to sometimes hold one would make "int or bool" ambiguous
    exactly where semantic.py needs it unambiguous."""
    value: bool
    resolved_type: Optional[Any] = None


@dataclass
class NoneLiteral(Node):
    """`none` -- Hornet's nil-style zero value, analogous to Go's nil
    but narrower: it resolves to one single, fixed, internal type,
    Type.NONE (see semantic.py), checked for COMPATIBILITY (not
    equality) wherever a value flows into a slice-typed context (see
    semantic.py's _types_compatible) -- there's no general untyped-
    constant mechanism here the way Go's nil relies on.

    Only slices are nilable -- none is not compatible with int/bool/
    str/array. At the machine level it becomes the {ptr: 0, len: 0}
    slice descriptor (see codegen.py's gen_none_into), the same shape
    Go's own nil slice has -- every existing slice operation already
    handles a zero-length slice correctly, so this only needs to
    produce that descriptor and support comparing a slice against
    none directly. That comparison checks the descriptor's `ptr`
    field against 0, matching Go's nil-vs-empty-slice distinction: a
    real zero-length slice sliced from a real array (`arr[5:5]`) has
    a non-null pointer and is NOT `== none`."""
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
    # None until semantic.py's check_expr type-checks this node, then a
    # full semantic.Type (not a string, since a type can carry an
    # element type and size). Typed Optional[Any], not Optional[
    # semantic.Type], purely to avoid a circular import -- semantic.py
    # already imports from parser.py. codegen.py reads this directly
    # instead of re-deriving an expression's type.
    resolved_type: Optional[Any] = None


@dataclass
class Variable(Node):
    """A reference to a local variable, e.g. the `a` in `a + 1`."""
    name: str
    resolved_type: Optional[Any] = None


@dataclass
class ArrayLiteral(Node):
    """`[e1, e2, ...]` in expression position, e.g. the value side of
    `[3]int arr = [1, 2, 3]`. An element can itself be another
    ArrayLiteral for a multi-dimensional literal (`[[1,2,3],[4,5,6]]`)
    -- no special casing needed; parse_expression just recurses into
    the nested `[...]` like any other expression. Elements don't have
    to be constants.

    type_expr is None for this plain, untyped form, which only type-
    checks where an expected type is already known from context (see
    semantic.py's check_array_literal) -- so it's restricted to a
    VarDecl initializer or an Assign value. type_expr is set for the
    fully-typed form, `[3]int[1, 2, 3]` (an ArrayTypeExpr), making the
    literal self-describing and usable as a general expression
    anywhere -- see parse_primary's _looks_like_typed_literal for how
    that's told apart from a plain `[N, ...]`.

    A SLICE literal, `[]int[1, 2, 3]`, isn't its own node type -- it's
    sugar resolved entirely in the parser: see _parse_bracketed_
    literal, which wraps an ArrayLiteral like this one in an implicit,
    whole-array Slice node (low=None, high=None, meaning "the whole
    thing", like `arr[:]`). That lets slicing machinery already built
    for a named array handle a slice literal too, with only
    gen_indexable_base_into needing to learn that an ArrayLiteral base
    means "allocate a fresh one," not "find an existing one"."""
    elements: List[Node] = field(default_factory=list)
    type_expr: Optional['ArrayTypeExpr'] = None
    resolved_type: Optional[Any] = None


@dataclass
class Index(Node):
    """`array[index]` -- reads a single element (or, for a multi-
    dimensional array not yet fully indexed, a sub-array). Multi-
    dimensional indexing `matrix[i][j]` is NESTED Index nodes, one per
    bracket pair (see parse_postfix), matching the type's own
    structure: the outer Index yields a whole (array-typed) row, which
    the outer bracket then indexes into.

    `array` can be Slice-typed too -- indexing into a slice (`s[i]`)
    uses this same node (see semantic.py's indexable-and-index check,
    which accepts either)."""
    array: Node
    index: Node
    resolved_type: Optional[Any] = None


@dataclass
class Slice(Node):
    """`array[low:high]` -- a VIEW into `array` spanning [low, high),
    matching Go's convention. Produces a Slice-typed value (a
    {pointer, length, capacity} descriptor -- see codegen.py's SLICES
    section), not a copy: a genuine alias into the base's own backing
    storage, unlike plain array assignment. This aliasing is what
    makes stack safety load-bearing once slicing exists -- a slice
    outliving its backing array's stack frame becomes a dangling
    pointer; see codegen.py's analyze_array_escapes for the mechanism
    that prevents it.

    `array` can be either Array- or Slice-typed -- slicing a slice and
    slicing the outer dimension of a multi-dimensional array both use
    this node (see semantic.py's check_slice).

    `low`/`high` are independently optional (`arr[:]`, `arr[2:]`,
    `arr[:5]`), represented as None rather than a default filled in at
    parse time: low's default (0) could be, but high's (the base's own
    length) can't be for a Slice base, since that's a runtime value --
    so both stay None uniformly, resolved together downstream.

    Deliberately not a valid assignment target -- `arr[1:3] = ...`
    doesn't parse (Slice being a different class than Index already
    excludes it in parse_expr_stmt_or_assign). Matches Go: slicing
    produces a value, not an addressable location.
    """
    array: Node
    low: Optional[Node] = None
    high: Optional[Node] = None
    resolved_type: Optional[Any] = None


@dataclass
class Call(Node):
    """`name(arg1, arg2, ...)` -- an ordinary function call expression.
    No separate "call statement" concept: `foo(1)` alone parses as an
    ExprStmt wrapping this, like any other expression used as a bare
    statement.

    Also `Name(x=1, y='a')` -- named-field struct construction (kwargs
    populated, args empty), as an alternative to positional (args
    populated, kwargs None). The two are mutually exclusive by
    construction: parse_call rejects mixing them as a ParseError, a
    pure syntax-shape rule independent of what `name` resolves to
    (semantic.py's check_call handles struct-vs-function dispatch).

    kwargs is a List[Tuple[str, Node]], not a dict, to preserve
    written order -- duplicate names (`A(x=1, x=2)`) aren't rejected
    here either, since the parser has no symbol table; both duplicate-
    name and unknown-field-name rejection are check_struct_literal's
    job. Named construction is scoped to struct literals only: an
    ordinary call written with named arguments parses into the same
    shape and is rejected by check_call once `name` resolves to a
    function rather than a struct. Omitting a field leaves it
    genuinely uninitialized, matching this language's usual treatment
    of uninitialized memory.

    Also `receiver.name(args)` -- a method call (`receiver` populated).
    Parsed by parse_postfix, not parse_call, from whatever expression
    preceded the '.'; arguments are always positional here.

    Only alive as a distinct shape during semantic analysis: check_
    call's _check_method_call rewrites this node in place -- prepending
    the receiver into args, replacing `name` with a mangled, collision-
    free symbol, and clearing receiver -- so by the time codegen.py
    sees it, a method call is indistinguishable from an ordinary call
    to that mangled function. Deliberately an in-place rewrite, not a
    separate AST node kept alive through codegen, which would mean
    auditing every isinstance(expr, Call) check in this codebase."""
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
    """`TYPE(expr)` -- an explicit numeric cast, e.g. `int8(x)`. Same
    surface shape as a function call, but distinguished at PARSE time:
    target_type is always one of the six scalar type keywords, a
    lexically distinct token from IDENTIFIER, so a struct name or type
    alias (always IDENTIFIER) can never produce a Cast node -- unlike
    Call's struct-vs-function ambiguity, there's nothing to resolve
    later. Scoped to a single argument, unlike Call's list.

    Only int/int8/uint8/int64 are actually supported on either side
    (see semantic.py's check_cast); bool/str are still accepted here
    at parse time, matching this file's "parser accepts the shape,
    semantic.py validates the meaning" split.

    Casting to a struct or type-alias name (`MyByte(x)`) isn't
    supported by this node -- it parses as an ordinary Call instead,
    which check_call has no cast-aware case for, so it's rejected as
    an undeclared-function or struct-literal error."""
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
    """`return <expr>`, or a bare `return` (value=None) -- valid inside
    a function with no declared return type. `None` here mirrors how
    Function.return_type represents "no declared type"."""
    value: Optional[Node] = None


@dataclass
class ArrayTypeExpr(Node):
    """`[size]element_type` in type position, e.g. `[3]int`, or
    `[2][3]int` (ArrayTypeExpr(size=2, element_type=ArrayTypeExpr(
    size=3, element_type='int'))) -- row-major, outermost dimension
    first. `element_type` is Union[str, ArrayTypeExpr, SliceTypeExpr],
    recursed arbitrarily deep.

    `size` must be a positive integer LITERAL (see Parser.parse_type),
    not an expression like `[2+3]int` -- validated at parse time,
    unlike most validation in this file, since an array's size is
    closer to syntax than to an ordinary expression."""
    size: int
    element_type: Union[str, 'ArrayTypeExpr', 'SliceTypeExpr']


@dataclass
class SliceTypeExpr(Node):
    """`[]element_type` in type position, e.g. `[]int`. Sibling to
    ArrayTypeExpr, distinguished by one token of lookahead after `[`
    (NUMBER means an array's size; immediate `]` means a slice).

    Not just a parsing detail: a slice's length is a RUNTIME property
    of the value itself, not part of its type the way an array's size
    is -- two slices of type []int can hold different lengths; two
    arrays of different sizes are different types entirely.

    `element_type` recurses like ArrayTypeExpr's does -- `[][3]int`
    and `[][]int` are both valid."""
    element_type: Union[str, ArrayTypeExpr, 'SliceTypeExpr']


@dataclass
class VarDecl(Node):
    """`int a` (init=None) or `int a = 1`. `var_type` is a type
    keyword string, an ArrayTypeExpr, or a SliceTypeExpr."""
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
    `array` is a bare Variable for `arr[i] = v`, or itself an Index
    for the outer dimensions of `matrix[i][j] = v`, mirroring how
    Index nests for reads.

    Compound index-assignment (`arr[i] += 1`) isn't supported: it
    would need the index expression evaluated once and reused for both
    read and write, which the simple `x += y` desugaring doesn't
    guarantee if the index isn't side-effect-free."""
    array: Node
    index: Node
    value: Node


@dataclass
class Field(Node):
    """`base.name` -- reads a field out of a struct-typed `base`. A
    multi-level chain (`a.b.c`) is nested Field nodes, one per '.',
    mirroring Index -- and the two chain together freely (`a.b[0]`,
    `arr[0].f`), since parse_postfix builds both in one pass."""
    base: Node
    name: str
    resolved_type: Optional[Any] = None


@dataclass
class FieldAssign(Node):
    """`base.name = value` -- writes a struct field. Mirrors
    IndexAssign: `base` can itself be a Field or Index for a longer
    chain (`s.inner.f = v`), built by parsing the whole left-hand
    expression first and reinterpreting it as a target. Compound
    assignment (`s.f += 1`) is rejected for the same reason
    IndexAssign's is -- a side-effecting sub-expression in the target
    could get evaluated twice."""
    base: Node
    name: str
    value: Node


@dataclass
class StructField(Node):
    """One field declaration inside a struct body: `type name`, no
    initializer -- every field starts at its type's zero value until
    explicitly assigned, like an uninitialized local."""
    name: str
    field_type: Union[str, 'ArrayTypeExpr', 'SliceTypeExpr']


@dataclass
class MethodDef(Node):
    """A method inside a struct body: `def [type] name(receiver,
    param2, ...):` -- an ordinary `def`, except the first parameter
    (the receiver) is a bare, untyped identifier; the enclosing
    struct's name implicitly gives it its type, the way `self`/`this`
    doesn't need one written out elsewhere.

    Never reaches semantic.py as its own concept for long: analyze()'s
    _collect_methods immediately synthesizes an ordinary Function from
    each MethodDef -- receiver becomes a typed first Param, and the
    name is mangled to `StructName.methodName` ('.' can't appear in a
    Hornet identifier, so no collision check is needed). From there,
    every later pass and all of codegen.py treat it as an ordinary
    function."""
    receiver_name: str
    name: str
    return_type: Optional[Union[str, ArrayTypeExpr, SliceTypeExpr]]
    params: List['Param'] = field(default_factory=list)
    body: List[Node] = field(default_factory=list)


@dataclass
class StructDef(Node):
    """`struct Name: <field-or-method>+` -- declares a new, nominal
    type. Field order is preserved exactly as written, since it
    determines both codegen's memory layout and print's field order.

    Fields and methods can freely interleave -- parse_struct_def just
    checks, per line, whether the next token is `def` or a type, with
    no ordering requirement, since a method never participates in the
    struct's own memory layout (it's fully lowered to a top-level
    function before codegen runs). At least one field is required;
    methods are entirely optional."""
    name: str
    fields: List[StructField] = field(default_factory=list)
    methods: List[MethodDef] = field(default_factory=list)


@dataclass
class ExprStmt(Node):
    """A bare expression used as a full statement, e.g. `2 + 2` alone
    on its own line -- evaluated and its value discarded."""
    expr: Node


@dataclass
class If(Node):
    """`if cond: <then_body> [elif cond: ...]* [else: <else_body>]?`.

    An `elif` isn't its own AST concept -- parse_if desugars it into a
    single-element else_body containing one more If node (`elif c: b`
    is `else: if c: b`). else_body is always just Optional[List[Node]]
    either way, so semantic.py/codegen.py consume it uniformly and an
    elif chain of any length falls out of ordinary nesting.
    """
    condition: Node
    then_body: List[Node]
    else_body: Optional[List[Node]] = None


@dataclass
class While(Node):
    """`while cond: <body>`. Re-checks `cond` before every iteration
    including the first, so a false condition never runs the body."""
    condition: Node
    body: List[Node]


@dataclass
class Break(Node):
    """`break` -- exits the *innermost* enclosing loop immediately.
    Only valid inside a while body; semantic.py rejects one that isn't."""


@dataclass
class Continue(Node):
    """`continue` -- skips to re-checking the *innermost* enclosing
    loop's condition. Same "only valid inside a while" rule as Break."""


@dataclass
class Param(Node):
    """A single `type name` entry in a function's parameter list.
    Not part of the function's own statement/expression tree -- a
    declaration record attached to Function, like VarDecl for a local
    but with no initializer. `type` is a type keyword string, an
    ArrayTypeExpr, or a SliceTypeExpr."""
    name: str
    type: Union[str, ArrayTypeExpr, SliceTypeExpr]


@dataclass
class Function(Node):
    """`def type NAME(params):`, or `def NAME(params):` with the type
    omitted entirely (return_type=None) -- no declared return type,
    not a void keyword (there is none). Disambiguated by one token of
    lookahead: a type keyword or '[' starts a type; an IDENTIFIER
    (what a function name always starts with) never does. Such a
    function may fall off the end of its body with no explicit
    `return`, or exit early via a bare one -- semantic.py doesn't
    require every path to return explicitly in this case."""
    name: str
    return_type: Optional[Union[str, ArrayTypeExpr, SliceTypeExpr]]
    params: List[Param] = field(default_factory=list)
    body: List[Node] = field(default_factory=list)


@dataclass
class TypeAlias(Node):
    """`type Name = TargetType` -- introduces `Name` as an alternate
    spelling for an existing type, interchangeable with it everywhere
    (an ALIAS, not a Go-style newtype -- `type Name TargetType`, no
    '=', isn't supported). `target_type` uses the ordinary parse_type()
    -- syntactically anything parse_type() accepts, though semantic.
    py's _collect_type_aliases currently narrows what's actually
    allowed to int/bool/str or another alias.

    Resolved once, centrally, by threading an `aliases` registry
    through type_from_name -- the one function every other type-name
    resolution already calls -- so every call site gains alias support
    automatically."""
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
    """Converts a STRING token's raw text (still quoted, e.g. `'it\\'s'`)
    into its actual content: quotes stripped, backslash escapes
    resolved via _ESCAPE_SEQUENCES. An escape not in the table (the
    lexer's STRING regex accepts a backslash followed by any single
    character) is treated leniently -- the backslash is dropped and
    the character kept as-is, rather than raising."""
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


# TokenType -> parsing metadata for each binary operator: which
# BinaryOp it produces, its precedence (higher binds tighter), and
# associativity. parse_binary()'s precedence-climbing loop reads
# entirely from this table, so a new operator is a row here, not a
# restructuring.
#
# Precedence, tightest to loosest -- the classic C ladder:
#   10: *  /  %          6: ==  !=
#    9: +  -              5: &
#    8: <<  >>            4: ^
#    7: <  >  <=  >=      3: |
#                         2: and    1: or
#
# Bitwise sits below equality deliberately, reproducing a well-known C
# footgun: `a & b == c` parses as `a & (b == c)`. Here that's harmless
# -- `b == c` is bool, `&` requires int, so semantic.py rejects it as a
# type error rather than silently accepting the "wrong" grouping (see
# TestSemanticErrors.test_bitwise_and_equality_precedence_is_a_type_error).
#
# A future right-associative operator (e.g. exponentiation) would slot
# in with associativity=Associativity.RIGHT and a chosen precedence --
# see STAR's own row for the shape.
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
# into (see parse_assign and the module docstring's compound-
# assignment paragraph).
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

# Every token that can start an assignment operator: '=' plus every
# compound form. parse_statement uses this for its one-token lookahead.
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
        declaration, no body. `Name` is an ordinary IDENTIFIER (a type
        keyword is its own token type, never tokenized as IDENTIFIER,
        so `type int = ...` is rejected by the next `expect` call).
        TargetType reuses parse_type() directly -- see TypeAlias's own
        docstring for why the parser accepts more here than semantic.py
        currently allows."""
        self.expect(TokenType.TYPE, "Expected 'type' to start a type alias")
        name_tok = self.expect(TokenType.IDENTIFIER, "Expected a name for this type alias")
        self.expect(TokenType.ASSIGN, "Expected '=' in a type alias declaration")
        target_type = self.parse_type()
        self.expect(TokenType.NEWLINE, "Expected a newline after a type alias declaration")
        return TypeAlias(name=name_tok.val, target_type=target_type)

    def parse_struct_def(self) -> StructDef:
        """`struct Name: <field-or-method>+` -- header line then an
        indented block, like a function. Each FIELD line is `type
        name` (no initializer), reusing parse_type() directly rather
        than parse_var_decl. Each METHOD line starts with `def`,
        unambiguous with one token of lookahead, delegated to parse_
        method_def -- see StructDef's own docstring for why fields and
        methods can freely interleave."""
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
        """True if the current position starts an optional return type
        before a def's name -- shared by parse_function/parse_method_
        def. A type keyword or '[' starts a return type unambiguously
        with one token of lookahead. A struct-typed return needs a
        SECOND token: IDENTIFIER alone is ambiguous between "a struct
        return type" and "the def's own name", resolved by a second
        identifier immediately after (a name is always followed by
        '(', never another identifier) -- the same two-vs-one-
        IDENTIFIER disambiguation parse_statement's struct-typed-
        VarDecl check needs, for the same reason (struct names aren't
        reserved keywords)."""
        return self.check(TokenType.INT, TokenType.INT8, TokenType.UINT8, TokenType.INT64, TokenType.BOOL, TokenType.STR, TokenType.OPEN_BRACKET) or (
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
        function, except the first parameter is always the receiver, a
        bare untyped IDENTIFIER (a method requires exactly one -- see
        MethodDef's own docstring). Every later parameter is ordinary
        Param syntax."""
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
        """Comma-separated `type IDENTIFIER` entries, stopping without
        consuming CLOSE_PAREN (the caller matches it). `()` is valid,
        returning []."""
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
        # A type keyword ('int'/'bool'/'str'/...), OR '[' NUMBER ']'
        # followed by another type recursively (ArrayTypeExpr -- each
        # bracket pair peels off one more wrapping whatever parse_type
        # returns for the rest), OR '[' ']' followed by another type
        # recursively (SliceTypeExpr) -- an immediate ']' after '['
        # means no size, i.e. a slice (see SliceTypeExpr's own
        # docstring for why that's meaningful, not just syntax). The
        # array size, when present, must be a literal positive whole
        # NUMBER -- validated here, unlike most validation in this
        # file, since a size is closer to syntax than an expression.
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
        if self.check(TokenType.INT, TokenType.INT8, TokenType.UINT8, TokenType.INT64, TokenType.BOOL, TokenType.STR):
            return self.advance().val
        if self.check(TokenType.IDENTIFIER):
            # A struct type reference -- the parser has no symbol table
            # and just accepts any identifier, handing the bare string
            # on; semantic.py's struct-registry pass validates it (see
            # type_from_name).
            return self.advance().val
        tok = self.current()
        raise ParseError(
            f"Expected a type ('int', 'int8', 'uint8', 'int64', 'bool', "
            f"'str', a struct name, '[size]type', or '[]type'), got "
            f"{tok.type} ('{tok.val}') at line {tok.line}, column "
            f"{tok.col}"
        )

    def parse_block(self) -> List[Node]:
        """Parses an indented block: INDENT statement+ DEDENT. The one
        routine every block goes through -- a function's body and an
        if/elif/else's body alike -- so nesting falls out of ordinary
        recursion with no separate "nested block" concept. skip_
        newlines() before the INDENT handles blank lines after the
        block-opening NEWLINE; the one inside the loop does the same
        between statements.
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
        if self.check(TokenType.INT, TokenType.INT8, TokenType.UINT8, TokenType.INT64, TokenType.BOOL, TokenType.STR) and self.peek(1).type == TokenType.OPEN_PAREN:
            # A cast used as a bare statement (`int8(x)`), not a
            # VarDecl -- a scalar type keyword is never immediately
            # followed by '(' in a valid VarDecl (that position always
            # holds the variable's own NAME), so one token of lookahead
            # tells the two apart before committing to parse_type()
            # below. A struct/alias name never has this ambiguity at
            # all (see Cast's own docstring) -- neither is one of
            # these six keyword token types.
            return self.parse_expr_stmt_or_assign()
        if self.check(TokenType.INT, TokenType.INT8, TokenType.UINT8, TokenType.INT64, TokenType.BOOL, TokenType.STR, TokenType.OPEN_BRACKET):
            # A type-starting token could mean a VarDecl or a bare,
            # fully-typed array/slice-literal statement (`[3]int[1, 2,
            # 3]`) -- both start with the same type, so parse it once
            # and decide from what follows: an IDENTIFIER means a
            # VarDecl; another OPEN_BRACKET (only possible when the
            # type is an ArrayTypeExpr/SliceTypeExpr) means the
            # literal's elements start here instead. Committing to
            # parse_type() first, rather than parse_primary's bounded-
            # lookahead approach for this same shape (_looks_like_
            # typed_literal), is safe here because every statement
            # starting with one of these tokens already required a
            # full type before typed literals existed.
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
            # VarDecl (structs have no literal syntax to disambiguate
            # against, unlike the type-keyword case above). A bare
            # IDENTIFIER alone is ambiguous with a variable reference,
            # call, or field access, so the second token is needed
            # before parse_type() is called at all.
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
        """Shared by parse_if/parse_elif_as_if -- both are `KEYWORD
        expression ':' NEWLINE block`, differing only in which keyword
        the caller already consumed. Recurses into parse_elif_as_if
        for an arbitrarily long elif chain, plus an optional else.
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
        already supplied, is a type parse_statement already parsed
        before realizing this is a declaration rather than a bare,
        fully-typed array-literal statement -- passed in rather than
        parsed twice."""
        if var_type is None:
            var_type = self.parse_type()
        name_tok = self.expect(TokenType.IDENTIFIER, "Expected a variable name")
        init = None
        if self.match(TokenType.ASSIGN):
            init = self.parse_expression()
        return VarDecl(name=name_tok.val, var_type=var_type, init=init)

    def parse_assign(self) -> Assign:
        """`a = <expr>` or a compound form (`a += <expr>`, etc),
        desugared right here into the same Assign(name, Binary(op,
        Variable(name), value)) shape a hand-written `a = a + <expr>`
        would produce -- see the module docstring's compound-assignment
        paragraph for why that's exact, not approximate."""
        name_tok = self.expect(TokenType.IDENTIFIER)
        op_tok = self.advance()  # one of _ASSIGNMENT_TOKENS -- already confirmed by parse_statement's lookahead
        value = self.parse_expression()

        if op_tok.type == TokenType.ASSIGN:
            return Assign(name=name_tok.val, value=value)

        binary_op = _COMPOUND_ASSIGN_OPS[op_tok.type]
        desugared_value = Binary(op=binary_op, left=Variable(name=name_tok.val), right=value)
        return Assign(name=name_tok.val, value=desugared_value)

    def parse_return(self) -> Return:
        """`return <expr>` or a bare `return`. A NEWLINE immediately
        after 'return' unambiguously signals the bare form: every
        statement is NEWLINE-terminated and no expression can start
        with one, so this never needs to look further ahead."""
        self.expect(TokenType.RETURN)
        if self.check(TokenType.NEWLINE):
            return Return(value=None)
        value = self.parse_expression()
        return Return(value=value)

    def parse_expr_stmt_or_assign(self) -> Node:
        """Handles three shapes that can't be told apart by one token
        of lookahead: a bare expression statement (`foo()`), an
        index-assignment (`arr[i] = value`), and a field-assignment
        (`s.f = value`). Rather than look ahead through however many
        `[...]`/`.name` suffixes the left side has, this parses the
        leading expression through ordinary machinery first (already
        building nested Index/Field nodes -- see parse_postfix), then
        decides from what kind of node came out and what follows.

        Only plain `=` is handled for either assignable shape --
        `arr[i] += 1` and `s.f += 1` are rejected with a clear error
        (see IndexAssign's/FieldAssign's own docstrings for why
        compound assignment isn't supported there at all)."""
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
        """Precedence-climbing parse of a binary expression: starts
        with one unary/primary operand, then folds in further
        `operand OP operand` pairs as long as the next operator's
        precedence is high enough to bind here (`>= min_prec`).

        Associativity is purely a matter of the min-precedence passed
        to the recursive right-hand-side call:
          - LEFT-associative recurses with `precedence + 1`, so that
            call can't also consume another same-precedence operator
            -- it falls to THIS call's own loop instead, producing
            left-leaning nesting: `1 - 2 - 3` -> `(1 - 2) - 3`.
          - RIGHT-associative recurses with `precedence` unchanged, so
            that call keeps consuming further same-precedence
            operators itself: `2 ^ 3 ^ 2` -> `2 ^ (3 ^ 2)`.
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
        `.name`, or `.name(...)` suffixes. `[...]` is an index
        (`matrix[i][j]`, nested Index nodes -- see Index's own
        docstring) or a slice (see _parse_index_or_slice); `.name`
        alone is a Field access; `.name(...)` is a method call, an
        ordinary Call with `receiver` set to whatever preceded the '.'.
        All four chain together freely (`a.b[0]`, `a.b.method(1)[0]`)
        with no special-casing for order.

        Sits between parse_unary and parse_primary so these all bind
        TIGHTER than a prefix operator: `-arr[0]` means `-(arr[0])`,
        since parse_unary's base case calls straight into this method.
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
        """A plain, comma-separated, purely positional argument list
        for a method call -- unlike parse_call, no named-argument
        support: that's scoped to struct literals (see Call's own
        docstring), and method calls don't currently extend to it."""
        args: List[Node] = []
        if self.check(TokenType.CLOSE_PAREN):
            return args
        args.append(self.parse_expression())
        while self.match(TokenType.COMMA):
            args.append(self.parse_expression())
        return args

    def parse_index_or_slice(self, array_expr: Node) -> Node:
        """Parses the content of one `[...]` pair (OPEN_BRACKET
        already consumed), returning an Index (`a[i]`) or a Slice
        (`a[low:high]`, either bound optionally omitted) wrapping
        `array_expr`.

        A leading ':' unambiguously signals a slice with an omitted
        low bound -- ':' can't start any expression here. Otherwise an
        expression is parsed first; a ':' after it means a slice (high
        optionally omitted); no ':' means this was a plain index all
        along.
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
        if self.check(TokenType.INT, TokenType.INT8, TokenType.UINT8, TokenType.INT64, TokenType.BOOL, TokenType.STR) and self.peek(1).type == TokenType.OPEN_PAREN:
            return self.parse_cast()
        if self._looks_like_typed_literal():
            parsed_type = self.parse_type()
            return self._parse_bracketed_literal(parsed_type)
        if self.check(TokenType.OPEN_BRACKET):
            return self.parse_array_literal()
        if self.check(TokenType.IDENTIFIER):
            # One token of lookahead tells a call (`foo(...)`) apart
            # from a bare variable reference.
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
        """True if the current position starts a fully-typed array or
        slice literal (`[3]int[1, 2, 3]`, `[]Point[...]`) rather than a
        plain untyped one (`[1, 2, 3]`) or a single-element one (`[5]`)
        -- both of which also start with OPEN_BRACKET, and for `[5]`,
        the identical OPEN_BRACKET NUMBER CLOSE_BRACKET prefix an
        array type's size bracket has.

        Resolved with bounded lookahead (3-4 tokens), no backtracking:
        an array type's size bracket is `[` NUMBER `]` immediately
        followed by a type-starting token, and a slice type's empty
        bracket pair is `[` `]` immediately followed by one -- a type
        keyword, an IDENTIFIER (a struct name), or another `[` for a
        nested element type. Nothing can validly follow a complete
        untyped literal or bare `[]` that way, so checking for either
        shape always tells them apart, including `[5]`'s single-
        element-array edge case.

        IDENTIFIER was added to both checks once struct literals
        existed -- before that, a struct name here could only mean an
        ordinary VarDecl's type, with nothing to disambiguate."""
        if not self.check(TokenType.OPEN_BRACKET):
            return False
        if self.peek(1).type == TokenType.CLOSE_BRACKET:
            return self.peek(2).type in (TokenType.INT, TokenType.INT8, TokenType.UINT8, TokenType.INT64, TokenType.BOOL, TokenType.STR, TokenType.IDENTIFIER, TokenType.OPEN_BRACKET)
        if self.peek(1).type == TokenType.NUMBER and self.peek(2).type == TokenType.CLOSE_BRACKET:
            return self.peek(3).type in (TokenType.INT, TokenType.INT8, TokenType.UINT8, TokenType.INT64, TokenType.BOOL, TokenType.STR, TokenType.IDENTIFIER, TokenType.OPEN_BRACKET)
        return False

    def _parse_bracketed_literal(self, parsed_type: Union[str, 'ArrayTypeExpr', 'SliceTypeExpr']) -> Node:
        """Given an already-parsed type (from parse_primary's
        _looks_like_typed_literal path, or parse_statement's "parse
        the type first" dispatch), parses the literal's bracketed
        elements and returns an ArrayLiteral for an ArrayTypeExpr
        (`[3]int[1, 2, 3]`), or one wrapped in an implicit, whole-array
        Slice for a SliceTypeExpr (`[]int[1, 2, 3]`) -- see Slice's
        own docstring for why omitted low/high means "the whole
        thing", exactly what a fresh backing array needs.

        For the slice form, the wrapped ArrayLiteral's type_expr is
        synthesized from the actual element count -- `[]int[1, 2, 3]`'s
        inner array is `[3]int`, since a slice has no size of its own.
        Set after construction, once the count is known, so parse_
        array_literal's own signature stays unchanged for the plain
        typed-array case."""
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
        type for the fully-typed form. A multi-dimensional literal
        (`[[1,2,3],[4,5,6]]`) needs no special handling: each element
        just recurses back into this method via parse_expression, with
        type_expr staying None (only the outermost literal is ever
        preceded by an explicit type)."""
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
        Call's own docstring; this just tells the two shapes apart and
        refuses to mix them, with no idea yet whether `name` is a
        struct or a function.

        Disambiguated per-argument with one token of lookahead:
        IDENTIFIER immediately followed by ASSIGN ('=', never EQUAL
        '==', a distinct token) means `name=value`; anything else is
        an ordinary positional expression. `start_tok` is captured so
        a mixing error points at the argument that broke the pattern."""
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
        """`TYPE(expr)` -- see Cast's own docstring. The caller (parse_
        primary) already confirmed a scalar type keyword followed by
        '(', so this just consumes both, parses the argument, and
        closes it -- no shape ambiguity to resolve."""
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
