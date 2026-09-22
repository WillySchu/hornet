"""Semantic analysis

Walks a parsed Program (see parser.py) and rejects it before codegen
ever runs if it's not well-formed: an undeclared variable is
referenced, a variable is declared twice, or any expression's type
doesn't match what the surrounding context requires.

    lex (lexer.py) -> parse (parser.py) -> analyze (semantic.py) -> codegen (codegen.py)

Putting a real pass here (rather than catching name/type problems
incidentally inside codegen, as a side effect of a stack-offset
lookup) means codegen only ever runs on programs already known valid.

THE TYPE SYSTEM
-----------------
A genuinely strong static type system: no implicit conversion in
either direction between any two types (see Type for the full set --
int, bool, str, array, slice, struct, plus the internal-only VOID/
NONE below). A bool is not a 0-or-1 int; `not`/`and`/`or` all require
real bool operands (`not 0` is a type error, not "not true").

Operator typing, precisely:
  - Arithmetic (+ - * /) and unary (- ~): int operand(s), int result.
  - Ordering (< > <= >=): int operands only, bool result.
  - Equality (== !=): both operands the *same* type, bool result --
    int vs bool is a type error even though both are "just numbers".
  - Logical (not/and/or): bool operands, bool result.
  - An if/elif condition must be bool -- no int-as-truthy shortcut.
  - A VarDecl initializer, Assign value, or Return value must match
    the relevant declared type exactly.

Number literals are always `int` -- there's no float type yet, even
though the lexer's NUMBER rule matches decimals; check_constant
rejects a non-whole-number literal as a type error (codegen has no
fractional-immediate instruction).

SCOPING
--------
self.scopes is a List[Dict[str, Type]], one dict per open block
(if/elif/else, while), the function's top-level body as the bottom
entry. A block's own scope is pushed before walking its statements
and popped after (analyze_if/analyze_while) -- a static fact about
the program text, independent of how many times a while body actually
runs. Shadowing is allowed: declaration only checks the *current*
scope for a collision (_declare); a reference resolves by walking
outward from innermost to outermost (_lookup). An if's then/else
branches get independent scopes, since they're mutually exclusive.

Statements are walked in program order, adding each variable to scope
only once its own VarDecl is processed, so declare-before-use and
rejection of `int a = a` both fall out for free, per-scope. This
differs from codegen.py's own local handling, which pre-scans a whole
function body up front just to size the stack frame (layout, not
validity) -- by the time it runs, this pass has already guaranteed
well-formedness.

LOOPS: break/continue validity
---------------------------------
Tracked with a counter, self.loop_depth, incremented/decremented
around a while's body (analyze_while) -- survives nesting inside an
if (a break inside an if inside a while is fine) while still
resetting once a nested while's own body finishes, so an outer loop's
break/continue is never validated by an unrelated inner loop.
codegen.py mirrors this with its own (start_label, end_label) stack,
for the same reason: break/continue always target the *innermost*
enclosing loop.

FUNCTIONS
----------
self.functions (name -> (param types, return type)) is a single,
program-wide flat namespace, separate from self.scopes -- so a
variable and a function can share a name. Built in a dedicated first
pass over every function (analyze()) before any body is checked, so
call order never matters: forward references and (mutual) recursion
both just work, since every signature is already present by the time
any body looks one up.

Parameters are bound into the function's own scope at the start of
analyze_function, like already-declared locals -- so a duplicate
parameter name is caught by the ordinary _declare collision check,
with no separate check needed.

FUNCTIONS WITH NO DECLARED RETURN TYPE
-------------------------------------------
`def NAME(params):`, the type before the name omitted entirely (not a
void/none keyword) -- Function.return_type is None. Internally this
gets a real singleton, Type.VOID, kept distinct from resolved_type's
own None (which means "not yet type-checked") so a legitimately void
expression is never confused with one analysis hasn't reached yet.

Such a function's body may fall off the end with no explicit `return`
(a bare `return` is valid exactly when return_type is Type.VOID -- see
analyze_return), and always_returns is skipped for it entirely (see
ALL PATHS RETURN below). Type.VOID flowing where a real value is
expected (a VarDecl initializer, an argument, ...) is already rejected
by the ordinary type-mismatch check at each site, since none of those
ever compares against something a user could write as "void" --
except print's argument check and binary equality, which needed an
explicit case apiece (print used to accept anything unconditionally;
`Type.VOID == Type.VOID` is trivially true by structural equality, the
same way any type equals itself). print itself is Type.VOID now.

NONE
-----
`none` -- see NoneLiteral's own docstring in parser.py for the full
reasoning. Resolves to one fixed internal type, Type.NONE, checked for
COMPATIBILITY (not equality, via _types_compatible) wherever a value
flows into a slice-typed context (VarDecl initializer, Assign,
IndexAssign, argument, return). Only slices are nilable. Equality has
no fixed "target" side (either operand could be the none one), so
check_binary checks for a slice-vs-none pair directly before falling
through to its array/slice/void rejection, which NONE now also joins
(`none == none` would otherwise trivially type-check). codegen.py
holds none's actual runtime representation ({ptr: 0, len: 0}) and the
ptr-only `== none` comparison.

BUILTINS
---------
print/len/append are builtins, not ordinary user-defined functions --
check_call special-cases each before ever consulting self.functions,
and analyze()'s signature-collection pass rejects a user function
whose name collides with one (_BUILTIN_FUNCTION_NAMES).

check_print_call accepts one argument of any real type and is itself
Type.VOID. check_len_call is the near-opposite: only array or slice
(str explicitly rejected with its own message, a separable follow-up,
not folded into the generic rejection), always returning Type.INT.
check_append_call requires a slice and a value matching its element
type (via _check_value_flowing_into, the same recursive treatment a
VarDecl/Assign/IndexAssign value gets), always returning that same
slice type back.

TYPES: ANNOTATING THE AST FOR codegen.py
-------------------------------------------
check_expr, after computing an expression's Type, stores it on the
node (expr.resolved_type = result) before returning -- the one place
this happens; every check_* method stays a pure type-computation
function, and every recursive call anywhere in this file for an
operand/argument/condition already goes through check_expr rather
than a check_* method directly. So every expression node ends up
annotated automatically, with nothing to remember wiring up as this
file grows. resolved_type holds the actual Type object (not a name
string, since an array type needs its own element_type/size too).

codegen.py reads this directly instead of re-deriving a type with its
own independent logic, which is what it used to do via a method called
_infer_type -- a real liability: adding `print` needed a Call case
added there separately from this file's own check_call, and the six
int-only operators needed a separate addition to its own int-producing
branch. Both omissions were easy to make and were only caught by
manual testing. Annotating here and having codegen.py read the
annotation removes that second, independently-maintained copy
entirely. codegen.py's own scope-stack (offset AND type per local,
for resolving which of several same-named declarations a reference
means) remains a separate, unrelated exception -- an expression's type
alone can never say which stack slot it resolves to.

ALL PATHS RETURN
------------------
analyze_function's last step, once every statement is already known
well-typed: always_returns(fn.body), for every function with a real
declared return type. Not just a nicety -- a function whose generated
code falls through with no `ret` corrupts the calling function's own
stack (the return address from `call` is never popped). A VOID
function is the one exception, skipped entirely (see FUNCTIONS WITH NO
DECLARED RETURN TYPE above).

A simple, conservative "terminating statement" check (like Go's own
spec), not full flow analysis: scans a statement list front-to-back
for the first one that, on its own, guarantees a return -- a Return
itself; an If with a non-None else_body where both branches guarantee
one (handling any elif chain for free, since elif desugars into a
nested If); or a `while true` loop with no reachable break. The while
case is the subtle one: only the literal `true` counts (checked
structurally, not "prove this expression is always true"), and even
then only if there's no `break` to escape through -- contains_
reachable_break finds one anywhere in the loop's own body, including
nested inside if/elif/else, but deliberately not inside a *nested*
while's own body (same reasoning as break's own validity/loop_depth,
and codegen.py's loop_labels stack): a break inside an inner loop
belongs to that loop, not whatever encloses it.

ERROR REPORTING
-----------------
Raises SemanticError on the *first* problem and stops, matching
ParseError/CodegenError elsewhere in this pipeline, rather than
collecting every error. Every AST node carries the line/column of its
own start token (Node.line/col, set once during parsing -- see
parser.py), so SemanticError can name a real source position, not just
the offending variable/operator/type -- see SemanticError's own
docstring for how.
"""

import argparse
from dataclasses import dataclass
from enum import auto, Enum
from typing import Dict, List, Optional, Set, Tuple

from lexer import lex
from desugar import mangle_method_name
from parser import (
    ArrayLiteral,
    ArrayTypeExpr,
    Assign,
    Binary,
    BinaryOp,
    BoolLiteral,
    Break,
    Call,
    Cast,
    Constant,
    Continue,
    DerefAssign,
    ExprStmt,
    ExternFunctionDecl,
    Field,
    FieldAssign,
    Function,
    If,
    Index,
    IndexAssign,
    IsCheck,
    Node,
    NoneLiteral,
    Param,
    Parser,
    PointerTypeExpr,
    Program,
    Return,
    Slice,
    SliceTypeExpr,
    StringLiteral,
    StructDef,
    SumTypeDef,
    TypeAlias,
    Unary,
    UnaryOp,
    VarDecl,
    Variable,
    While,
)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class TypeKind(Enum):
    INT = auto()
    INT8 = auto()
    UINT8 = auto()
    INT64 = auto()
    BOOL = auto()
    STR = auto()
    ARRAY = auto()
    SLICE = auto()
    STRUCT = auto()
    SUM = auto()   # see SumTypeInfo's own docstring
    POINTER = auto()  # see Type's own docstring below
    VOID = auto()  # see Type.VOID's own docstring below -- purely internal
    NONE = auto()  # see Type.NONE's own docstring below -- user-writable
                   # (via the `none` literal), but never as a DECLARED type


@dataclass(frozen=True)
class Type:
    """A type: one of the three-plus scalars (kind alone), an array
    (kind=ARRAY, element_type one level down, size that dimension's
    fixed length), a slice (kind=SLICE, element_type one level down,
    size always None -- a slice's length is a runtime property of the
    VALUE, not its type), a struct (kind=STRUCT, struct_name the
    declared name, element_type/size both None -- field layout lives
    in the struct registry, not duplicated here), a sum type (kind=
    SUM, sum_type_name the declared name -- variant list lives in the
    sum-type registry, the same split STRUCT already has), or a
    pointer (kind=POINTER, element_type one level down -- REUSING
    ARRAY/SLICE's own field, since "the type this points to" is the
    identical shape as "the type this contains"; no dedicated field of
    its own, the same way SLICE doesn't get one just because it isn't
    called "element_type" in its own vocabulary).

    Frozen to get structural equality/hashing for free: `Type(ARRAY,
    Type.INT, 3) == Type(ARRAY, Type.INT, 3)` is correctly True for
    two separate objects, and `!= Type(ARRAY, Type.INT, 4)` is
    correctly True too, recursing to arbitrary depth since element_
    type is itself a Type -- no special-casing needed anywhere that
    already does `left_type != right_type` (check_binary, analyze_
    var_decl, check_call, ...). This is also what gives struct (and
    sum) types NOMINAL equality essentially for free: struct_name (or
    sum_type_name) is just one more field this same machinery
    compares, so two structs -- or sum types -- with identical shapes
    but different names are correctly different types, with no field-
    or variant-by-variant comparison involved. A pointer's own equality
    is structural on its OWN element_type too, which is exactly what
    makes `*Circle == *Circle` a real, meaningful check (are these
    pointers even the SAME kind of pointer) distinct from the runtime
    question check_binary's equality branch answers separately (do
    these two same-typed pointers hold the same address).
    """
    kind: TypeKind
    element_type: Optional['Type'] = None  # set when kind == ARRAY, SLICE, or POINTER
    size: Optional[int] = None             # only set when kind == ARRAY
    struct_name: Optional[str] = None      # only set when kind == STRUCT
    sum_type_name: Optional[str] = None    # only set when kind == SUM

    def __str__(self) -> str:
        if self.kind == TypeKind.ARRAY:
            return f"[{self.size}]{self.element_type}"
        if self.kind == TypeKind.SLICE:
            return f"[]{self.element_type}"
        if self.kind == TypeKind.STRUCT:
            return self.struct_name
        if self.kind == TypeKind.SUM:
            return self.sum_type_name
        if self.kind == TypeKind.POINTER:
            return f"*{self.element_type}"
        return self.kind.name.lower()


# Singleton instances -- class attributes assigned after the class
# body, not instance fields, so every Type.INT/BOOL/STR/... reference
# throughout this file works unchanged. `frozen=True` only prevents
# mutating an instance's own fields; it says nothing about adding
# attributes to the Type class object itself.
Type.INT = Type(TypeKind.INT)
Type.INT8 = Type(TypeKind.INT8)
Type.UINT8 = Type(TypeKind.UINT8)
Type.INT64 = Type(TypeKind.INT64)
Type.BOOL = Type(TypeKind.BOOL)
Type.STR = Type(TypeKind.STR)
# VOID and NONE are both kept deliberately out of _TYPE_NAMES below --
# neither is reachable by parsing an ordinary type expression from
# source. VOID is purely internal bookkeeping (the "return type" of a
# function with no declared one -- see analyze_function/check_call);
# there's no keyword for it and no way to declare a void-typed
# anything. NONE, despite `none` being a real, user-writable keyword,
# is only ever reached through parse_primary's NoneLiteral production,
# never through parse_type -- `none` is a VALUE (see NoneLiteral's own
# docstring in parser.py), never a type annotation.
Type.VOID = Type(TypeKind.VOID)
Type.NONE = Type(TypeKind.NONE)


_TYPE_NAMES = {
    'int': Type.INT,
    'int8': Type.INT8,
    'uint8': Type.UINT8,
    # 'byte' is a built-in alias for uint8 -- the same Type object, not
    # a third TypeKind -- so it's interchangeable everywhere down to
    # codegen, and leaves no trace in an error message or print()
    # output (both always say "uint8").
    'byte': Type.UINT8,
    'int64': Type.INT64,
    'bool': Type.BOOL,
    'str': Type.STR,
}


@dataclass
class StructInfo:
    """Everything semantic analysis (and, via Program.struct_registry,
    codegen) needs about one declared struct: its name and its fields,
    an ordinary dict from field name to resolved Type. Field order is
    preserved (a plain dict already does this) since it determines
    codegen's memory layout and print's field order."""
    name: str
    fields: Dict[str, Type]


@dataclass
class SumTypeInfo:
    """Everything semantic analysis (and, eventually, codegen) needs
    about one declared sum type: its name and its variants, an
    ordered list of already-declared STRUCT names (never a scalar, an
    alias, or another sum type -- see _resolve_sum_types). Order is
    preserved and load-bearing: it's what decides each variant's own
    discriminant (its index in this list), the same way a struct's own
    field order decides codegen's memory layout.

    Deliberately just names, not resolved Types the way StructInfo's
    own fields are -- a variant's real shape is already available by
    looking it up in the struct registry directly (self.structs),
    exactly the same struct a bare `Circle(5)` literal already
    resolves against; duplicating that here would just be two sources
    of truth for the same thing."""
    name: str
    variants: List[str]


def type_from_name(
    type_expr,
    structs: Dict[str, StructInfo],
    aliases: Dict[str, Type],
    node: Optional[Node] = None,
    sum_types: Dict[str, SumTypeInfo] = None,
) -> Type:
    """Converts a parsed type expression (VarDecl.var_type/Function.
    return_type/Param.type/StructField.field_type) into a Type.
    `type_expr` is a plain str (scalar, struct name, sum-type name, or
    alias name), an ArrayTypeExpr, a SliceTypeExpr, or a
    PointerTypeExpr (see their own docstrings in parser.py) -- handled
    by recursing on element_type/pointee_type, bottoming out at a
    scalar/struct/sum-type/alias name with no depth limit.

    `structs` and `aliases` are both required parameters, not defaulted
    to empty dicts, so a call site that forgets to pass one fails
    loudly (TypeError) rather than silently misresolving a struct- or
    alias-typed declaration as unknown. Both registries are already
    fully built by the time this is called with a name needing them
    (see analyze()'s ordering: aliases, then structs, then function
    signatures) -- resolving either is a single dict lookup, never a
    recursive re-resolution.

    `sum_types`, unlike those two, DOES default -- to nothing resolvable
    (see below) -- because unlike structs/aliases, which every call
    site needs to recognize, a sum type is deliberately NOT allowed
    EVERYWHERE yet: not as a struct field's type, at any depth of array
    wrapping (see _resolve_struct_fields's own call site). That would
    need real cycle-detection work this doesn't have yet -- a struct
    field embeds its value inline, directly or through an array (the
    same reason _check_struct_contains already walks through arrays
    looking for an embedded struct), so a sum-typed field, or an array-
    of-sum-type-typed one, could form the identical infinite-size cycle
    a self-containing struct already can't. Every OTHER position (a
    VarDecl's own type, a Param, a return type, a fully-typed array
    literal's own type_expr, even a Cast's target -- rejected there
    anyway, just less specifically, by the int-family check right
    after) is safe regardless of array/slice wrapping, since none of
    them embed a value inside another type's own fixed layout the way
    a struct field does. Omitting the argument (or passing `None`,
    same thing) is how a call site says "sum types aren't valid here"
    -- deliberately the SAFE default, unlike structs/aliases:
    forgetting to pass it just means a sum-type name reports as
    unknown here, not that it silently resolves somewhere it
    shouldn't.

    `node`, purely for error attribution (see SemanticError's own
    docstring), is the declaration OWNING type_expr (a Param, VarDecl,
    Function, ...) -- type_expr itself is often a bare string with no
    position of its own, so this is threaded through unchanged rather
    than re-derived at each recursive call.

    Only fails for a program that isn't syntactically valid, or
    references an undeclared struct/alias/sum-type name (or a sum-type
    name where one currently isn't allowed) -- parse_type() already
    restricts everything else at parse time."""
    if isinstance(type_expr, ArrayTypeExpr):
        element = type_from_name(type_expr.element_type, structs, aliases, node, sum_types)
        return Type(TypeKind.ARRAY, element_type=element, size=type_expr.size)
    if isinstance(type_expr, SliceTypeExpr):
        element = type_from_name(type_expr.element_type, structs, aliases, node, sum_types)
        return Type(TypeKind.SLICE, element_type=element)
    if isinstance(type_expr, PointerTypeExpr):
        # Pointer-to-pointer (`**int`) parses fine -- PointerTypeExpr
        # itself places no restriction on nesting (see its own
        # docstring in parser.py) -- but is rejected HERE, for now:
        # the easier of the two ways to disallow it, given genuine
        # pointer-to-pointer support is planned for later, once this
        # first slice of pointer support is proven out. No struct/
        # array-field cycle concern the way a sum-typed field has --
        # a pointer is always a fixed 8 bytes regardless of what it
        # points to -- this restriction is purely "not yet", not "not
        # safe".
        pointee = type_from_name(type_expr.pointee_type, structs, aliases, node, sum_types)
        if pointee.kind == TypeKind.POINTER:
            raise SemanticError(
                "Pointer-to-pointer types aren't supported yet -- "
                "planned for later, once single-level pointers are proven out",
                node,
            )
        return Type(TypeKind.POINTER, element_type=pointee)
    if type_expr in _TYPE_NAMES:
        return _TYPE_NAMES[type_expr]
    if type_expr in aliases:
        return aliases[type_expr]
    if type_expr in structs:
        return Type(TypeKind.STRUCT, struct_name=type_expr)
    if sum_types is not None and type_expr in sum_types:
        return Type(TypeKind.SUM, sum_type_name=type_expr)
    raise SemanticError(f"Unknown type '{type_expr}'", node)


def always_returns(statements: List[Node]) -> bool:
    """Does every execution path through this list of statements reach
    a `return` before falling off the end? See the module docstring's
    ALL PATHS RETURN section for the full reasoning. Scans front-to-
    back for the first statement that, on its own, guarantees a return
    (everything after it is irrelevant to this question); reaching the
    end without finding one means False.
    """
    for stmt in statements:
        if isinstance(stmt, Return):
            return True
        if isinstance(stmt, If):
            # A match with no explicit trailing else relies entirely
            # on its own exhaustiveness (already verified by analyze_
            # if by the time this ever runs -- always_returns is only
            # ever called after a function's whole body has already
            # been analyzed) rather than an else at all, so it needs
            # its own check, walked the SAME match_arm_count-bounded
            # way _check_match_exhaustiveness itself walks (see If's
            # own docstring for why "as long as else_body looks like
            # a nested If" isn't safe here either -- the identical
            # ambiguity with a hand-written `if NAME is Type:` as an
            # explicit else's own sole statement applies to this walk
            # too).
            if stmt.is_match and _match_always_returns(stmt):
                return True
            # Only counts with an else where BOTH sides guarantee a
            # return -- an if with no else can always just not run its
            # body.
            if stmt.else_body is not None and always_returns(stmt.then_body) and always_returns(stmt.else_body):
                return True
        if isinstance(stmt, While):
            # A while's body might run zero times, so it can never
            # guarantee a return on its own -- unless it's a genuine
            # `while true` with no reachable break, in which case
            # nothing after it is reachable at all.
            is_infinite = isinstance(stmt.condition, BoolLiteral) and stmt.condition.value is True
            if is_infinite and not contains_reachable_break(stmt.body):
                return True
        # VarDecl, Assign, Break, Continue, ExprStmt: none of these
        # guarantee a return; move on to the next statement.
    return False


def _match_always_returns(stmt: If) -> bool:
    """Whether a match-desugared If chain (stmt is its outermost node
    -- see If's own docstring) guarantees a return on every one of its
    arms, INCLUDING an explicit trailing else if it has one.

    Walks EXACTLY stmt.match_arm_count steps through else_body[0] to
    find the chain's own last arm -- never "as long as else_body looks
    like a nested If" -- for the identical reason _check_match_
    exhaustiveness's own walk needs that same bound: an ordinary,
    hand-written `if NAME is Type:` can legally be the sole statement
    inside this SAME match's own explicit `else:` block, and looks
    exactly like one more synthesized arm by shape alone.

    Only ever called with stmt.is_match already confirmed true, from
    always_returns' own If case -- and only after analyze_if has
    already run (always_returns is only ever called once a function's
    whole body has already been analyzed), so this trusts, rather than
    re-verifies, that the chain was already proven exhaustive if it
    has no trailing else at all."""
    current = stmt
    for i in range(stmt.match_arm_count):
        if not always_returns(current.then_body):
            return False
        if i < stmt.match_arm_count - 1:
            current = current.else_body[0]
    # current is now the LAST arm.
    if current.else_body is None:
        return True  # no explicit else -- relies on already-verified exhaustiveness
    return always_returns(current.else_body)


def contains_reachable_break(statements: List[Node]) -> bool:
    """Does this list of statements contain a `break` belonging to
    *this* loop (not already claimed by a nested one)? Recurses into
    if/elif/else bodies, but deliberately not into a nested While's own
    body -- same scoping break already has (analyze_break/loop_depth)
    and codegen.py's loop_labels stack. Only used by always_returns.
    """
    for stmt in statements:
        if isinstance(stmt, Break):
            return True
        if isinstance(stmt, If):
            if contains_reachable_break(stmt.then_body):
                return True
            if stmt.else_body is not None and contains_reachable_break(stmt.else_body):
                return True
        # While: deliberately not recursed into -- see docstring above.
    return False


# Builtins, not ordinary user-definable functions -- see check_call.
# Kept as a set so adding another is "a name plus its own check_*/
# gen_* pair", not a search-and-replace.
_BUILTIN_FUNCTION_NAMES = {'print', 'len', 'append'}


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class SemanticError(Exception):
    """Raised on the first semantic problem found: an undeclared or
    re-declared variable, or a type mismatch anywhere in the program.

    `node`, when given, is the AST node most responsible for the
    problem -- its position is appended once, here, the same way
    ParseError's own messages already end ("at line L, column C"),
    rather than repeated at every one of this file's ~80 raise sites.
    Silently omitted for a node with no real position (line and col
    both still 0 -- e.g. one built by hand in a test, with no actual
    token behind it) instead of printing a misleading "line 0, column
    0"."""
    def __init__(self, message: str, node: Optional[Node] = None):
        if node is not None and (node.line or node.col):
            message = f"{message} at line {node.line}, column {node.col}"
        super().__init__(message)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

# BinaryOp -> which typing rule applies (see the module docstring for
# what each category requires). ADD is deliberately NOT here -- it's
# overloaded (int+int arithmetic, str+str concatenation), handled as
# its own case in check_binary. Named _INT_ONLY rather than
# "_ARITHMETIC" since it also covers modulo, bitwise, and shifts, all
# sharing the same rule (both operands int, result int).
_INT_ONLY_BINARY_OPS = {
    BinaryOp.SUBTRACT, BinaryOp.MULTIPLY, BinaryOp.DIVIDE, BinaryOp.MODULO,
    BinaryOp.BITWISE_AND, BinaryOp.BITWISE_OR, BinaryOp.BITWISE_XOR,
    BinaryOp.SHIFT_LEFT, BinaryOp.SHIFT_RIGHT,
}
_ORDERING_OPS = {BinaryOp.LESS_THAN, BinaryOp.GREATER_THAN,
                  BinaryOp.LESS_THAN_OR_EQUAL, BinaryOp.GREATER_THAN_OR_EQUAL}
_EQUALITY_OPS = {BinaryOp.EQUAL, BinaryOp.NOT_EQUAL}
_LOGICAL_OPS = {BinaryOp.AND, BinaryOp.OR}

# Every integer type arithmetic/ordering/unary operators accept --
# mixing (int8 + uint8, int8 + int, ...) is rejected exactly like
# bool + int already is.
_INTEGER_TYPES = {Type.INT, Type.INT8, Type.UINT8, Type.INT64}

# int8/uint8's own literal ranges (two's-complement / unsigned, both
# 256 values wide) -- used only by _check_value_flowing_into's literal-
# range check, a compile-time check on a literal's written value, not
# runtime arithmetic wrapping (codegen's job).
_NARROW_INT_RANGES = {
    Type.INT8: (-128, 127),
    Type.UINT8: (0, 255),
}


class SemanticAnalyzer:
    """Type-checks and scope-checks a Program. Call analyze() once per
    Program; analyze_function() resets internal scope state, so a fresh
    SemanticAnalyzer isn't required per function, only per full run if
    you want to be safe against reuse across unrelated programs."""

    def __init__(self):
        self.scopes: List[Dict[str, Type]] = []
        self.loop_depth = 0  # how many enclosing `while` loops we're currently inside
        self.functions: Dict[str, tuple] = {}  # name -> (List[Type] param types, Type return type)
        self.structs: Dict[str, StructInfo] = {}  # name -> resolved fields; see _reserve_struct_names/_resolve_struct_fields
        self.methods: Dict[Tuple[str, str], Tuple[List[Type], Type, str]] = {}  # (struct, method) -> (param types, return type, mangled name); see _collect_methods
        self.type_aliases: Dict[str, Type] = {}  # name -> resolved target Type; see _collect_type_aliases
        self.sum_types: Dict[str, SumTypeInfo] = {}  # name -> resolved variants; see _resolve_sum_types
        self._narrowed_names: set = set()  # currently-narrowed variable names; see analyze_if/analyze_assign

    def analyze(self, program: Program) -> None:
        # Pass order matters and is load-bearing:
        # 1. Reserve every struct's NAME only (not fields yet) -- makes
        #    room for alias resolution next, since an alias can target
        #    a struct name (`type PointAlias = Point`), needing only
        #    the name to exist.
        struct_registry = self._reserve_struct_names(program.structs)

        # 2. Resolve every type alias (target can be int/bool/str, a
        #    struct name, or another alias). Must finish before struct
        #    field resolution, since a field's own type can itself be
        #    an alias.
        self.type_aliases = self._collect_type_aliases(program.type_aliases, struct_registry)
        program.type_alias_registry = self.type_aliases  # stashed for codegen.py's own use, mirroring struct_registry

        # 3. Resolve every struct's field types, then check for cycles.
        self.structs = self._resolve_struct_fields(program.structs, struct_registry)
        program.struct_registry = self.structs  # stashed for codegen.py's own use

        # 3.5. Resolve every sum type: each variant against the struct
        #    registry just finished above (needs REAL struct types, not
        #    just reserved names, so this can't run any earlier). Must
        #    finish before method/function-signature collection below,
        #    so a signature can already reference a sum type.
        self.sum_types = self._resolve_sum_types(program.sum_types, self.structs)
        program.sum_type_registry = self.sum_types  # stashed for codegen.py's own future use, mirroring struct_registry

        # 3.6. Collect struct methods, immediately lowering each into
        #    an ordinary mangled-name Function appended to program.
        #    functions -- see _collect_methods. Must run after struct
        #    fields are resolved (a receiver or param might need a
        #    real struct type) but before function-signature collection
        #    below, so that pass sees the synthesized functions too.
        self.methods = self._collect_methods(program)

        # 4. Collect every function's signature before checking any
        #    body -- what makes call order not matter (forward
        #    references, recursion). Also registers each synthesized
        #    method-function's mangled name here, harmlessly -- never
        #    looked up through self.functions (method calls resolve
        #    via self.methods instead), but a mangled name can't
        #    collide with anything else here regardless.
        self.functions = {}
        for fn in program.functions:
            if fn.name in _BUILTIN_FUNCTION_NAMES:
                raise SemanticError(
                    f"'{fn.name}' is a builtin and can't be redefined as "
                    f"a function",
                    fn,
                )
            if fn.name in self.structs:
                raise SemanticError(
                    f"Function '{fn.name}' collides with a struct of the "
                    f"same name -- struct and function names share one "
                    f"namespace and can never be the same, since "
                    f"'{fn.name}(...)' would otherwise be ambiguous "
                    f"between a call and a struct literal",
                    fn,
                )
            if fn.name in self.type_aliases:
                raise SemanticError(
                    f"Function '{fn.name}' collides with a type alias "
                    f"of the same name -- function and type-alias "
                    f"names share one namespace and can never be the "
                    f"same",
                    fn,
                )
            if fn.name in self.sum_types:
                raise SemanticError(
                    f"Function '{fn.name}' collides with a sum type "
                    f"of the same name -- function and sum-type names "
                    f"share one namespace and can never be the same",
                    fn,
                )
            if fn.name in self.functions:
                raise SemanticError(f"Function '{fn.name}' is already declared", fn)
            param_types = [type_from_name(p.type, self.structs, self.type_aliases, p, self.sum_types) for p in fn.params]
            return_type = Type.VOID if fn.return_type is None else type_from_name(fn.return_type, self.structs, self.type_aliases, fn, self.sum_types)
            self.functions[fn.name] = (param_types, return_type)

        # 4.5. Collect every extern function's signature into this
        #    SAME registry -- check_call's own lookup (self.functions)
        #    doesn't distinguish an extern declaration from an
        #    ordinary one at all once this runs, which is exactly the
        #    point: calling one looks identical to calling the other
        #    from here on. The one thing that IS extern-specific:
        #    every param and the return type must be a scalar or
        #    pointer kind -- see check_extern_function_decl's own
        #    docstring for why that's a v1 scope line, not a
        #    permanent restriction.
        for ext in program.extern_functions:
            self.check_extern_function_decl(ext)
        program.function_registry = self.functions  # stashed for codegen.py's own use, mirroring struct_registry -- see ir/scalars.py's own argument-widening use

        # 5. Check each function's own body, including every
        #    synthesized method-function's, via the same analyze_
        #    function -- a method's receiver is just its first Param by
        #    now, indistinguishable from an ordinary function.
        for fn in program.functions:
            self.analyze_function(fn)

    def _resolve_sum_types(self, sum_type_defs: List[SumTypeDef], structs: Dict[str, StructInfo]) -> Dict[str, SumTypeInfo]:
        """Resolves every sum type: a name-collision check against
        everything already established by this point (builtins,
        structs, aliases -- the same "check the new name against
        everything established so far" pattern _collect_type_aliases
        already follows for aliases-vs-structs), then each variant
        name against the struct registry `_resolve_struct_fields` just
        finished -- a variant must already be a declared struct, never
        a scalar, an alias, or another sum type. At least two variants
        is already guaranteed by the parser (parse_type_declaration's
        own _parse_sum_type_body); a DUPLICATE variant name within one
        sum type is this method's own job, the same split a struct's
        duplicate-field-name check already draws against parse_struct_
        def's own "at least one field" check.

        No cycle check needed here, unlike _resolve_struct_fields: a
        sum type can never be reached again once you've stepped into a
        struct's own field graph, since struct field resolution never
        allows a sum type as a field's own type in the first place
        (type_from_name's own sum_types parameter, omitted at that call
        site on purpose -- see its docstring). Nothing can ever
        "contain" a sum type transitively, so the self-containment
        cycle _check_struct_contains guards structs against simply
        can't arise here."""
        registry: Dict[str, SumTypeInfo] = {}
        for std in sum_type_defs:
            if std.name in _BUILTIN_FUNCTION_NAMES:
                raise SemanticError(
                    f"'{std.name}' is a builtin and can't be used as a "
                    f"sum type name",
                    std,
                )
            if std.name in structs:
                raise SemanticError(
                    f"Sum type '{std.name}' collides with a struct of "
                    f"the same name -- struct and sum-type names share "
                    f"one namespace and can never be the same",
                    std,
                )
            if std.name in self.type_aliases:
                raise SemanticError(
                    f"Sum type '{std.name}' collides with a type alias "
                    f"of the same name -- type-alias and sum-type names "
                    f"share one namespace and can never be the same",
                    std,
                )
            if std.name in registry:
                raise SemanticError(f"Sum type '{std.name}' is already declared", std)

            seen_variants: Set[str] = set()
            for variant_name in std.variants:
                if variant_name not in structs:
                    raise SemanticError(
                        f"Sum type '{std.name}' names '{variant_name}' as "
                        f"a variant, but '{variant_name}' is not a "
                        f"declared struct -- a sum type's variants must "
                        f"each be an already-declared struct name",
                        std,
                    )
                if variant_name in seen_variants:
                    raise SemanticError(
                        f"Sum type '{std.name}' lists '{variant_name}' "
                        f"as a variant more than once",
                        std,
                    )
                seen_variants.add(variant_name)

            registry[std.name] = SumTypeInfo(name=std.name, variants=list(std.variants))
        return registry

    def _collect_methods(self, program: Program) -> Dict[Tuple[str, str], Tuple[List[Type], Type, str]]:
        """For every struct's methods: reject a duplicate name on the
        SAME struct (fine across different structs), independently
        re-deriving the identical mangled name desugar_methods already
        gave this method's own synthesized Function (see mangle_
        method_name, shared by both so they can never disagree).

        Returns a (struct_name, method_name) -> (param types excluding
        the receiver, return type, mangled name) lookup for check_
        call's _check_method_call to resolve and rewrite a call site.
        The synthesized Function itself -- and program.functions
        already containing it -- is desugar_methods' own job, run
        before semantic.analyze() is ever called (see its own module
        docstring for why this can't happen here, or any later than
        here): resolving each method's own parameter/return types into
        real Type objects, which desugar_methods has no struct
        registry available to do yet, is this method's own reason to
        exist independently of it."""
        methods: Dict[Tuple[str, str], Tuple[List[Type], Type, str]] = {}
        for sd in program.structs:
            seen_names: Set[str] = set()
            for md in sd.methods:
                if md.name in seen_names:
                    raise SemanticError(
                        f"Method '{md.name}' is already declared on "
                        f"struct '{sd.name}'",
                        md,
                    )
                seen_names.add(md.name)
                param_types = [type_from_name(p.type, self.structs, self.type_aliases, p, self.sum_types) for p in md.params]
                return_type = Type.VOID if md.return_type is None else type_from_name(md.return_type, self.structs, self.type_aliases, md, self.sum_types)
                methods[(sd.name, md.name)] = (param_types, return_type, mangle_method_name(sd.name, md.name))
        return methods

    def _collect_type_aliases(self, alias_defs: List[TypeAlias], structs: Dict[str, StructInfo]) -> Dict[str, Type]:
        """Builds this program's alias registry: name -> the alias's
        fully-resolved Type, resolved all the way down once here
        rather than left as indirection for every type_from_name call
        to re-chase. Runs after struct names are reserved but before
        struct fields are resolved -- an alias can target a struct
        name (needs only the name), and a struct field can target an
        alias (needs aliases fully resolved first).

        Two passes, mirroring _reserve_struct_names/_resolve_struct_
        fields one level down: (1) reserve every name first, which is
        what lets one alias forward-reference another declared later;
        (2) resolve each target via a small memoized resolver with its
        own cycle guard (`type A = B; type B = A;` must be rejected,
        not infinitely recurse) -- the same shape _check_struct_
        contains uses for a cyclic struct, over a flat name chain
        instead of a field graph. A cycle is still caught even through
        array/slice wrapping (`type A = []B; type B = []A;`), since
        resolving a bare alias name always re-enters the same resolver
        regardless of how many array/slice layers wrap it -- unlike a
        struct field, which can safely self-reference through a slice,
        an alias whose entire definition is just "a slice of X" has no
        other structure to bottom out at.

        Struct literal construction (`PointAlias(1, 2)`) is the one
        place alias-to-struct interchangeability doesn't extend: check_
        call's own membership check (`expr.name in self.structs`) never
        matches an alias name -- a separate, narrower gap, not fixed
        here."""
        # Pass 1: reserve names, rejecting a duplicate, a builtin-
        # function-name collision, or a struct-name collision. No
        # check needed for a builtin TYPE keyword ('int'/'bool'/'str'):
        # those are their own token types, never tokenized as an
        # IDENTIFIER, so the parser could never produce a TypeAlias
        # with one of those names.
        seen: Dict[str, TypeAlias] = {}
        for ad in alias_defs:
            if ad.name in _BUILTIN_FUNCTION_NAMES:
                raise SemanticError(
                    f"'{ad.name}' is a builtin and can't be used as a "
                    f"type alias name",
                    ad,
                )
            if ad.name in structs:
                raise SemanticError(
                    f"Type alias '{ad.name}' collides with a struct of "
                    f"the same name -- struct and type-alias names "
                    f"share one namespace and can never be the same",
                    ad,
                )
            if ad.name in seen:
                raise SemanticError(f"Type alias '{ad.name}' is already declared", ad)
            seen[ad.name] = ad

        resolved: Dict[str, Type] = {}
        resolving: Set[str] = set()

        def resolve(name: str) -> Type:
            if name in resolved:
                return resolved[name]
            if name in resolving:
                raise SemanticError(
                    f"Type alias '{name}' is defined in terms of "
                    f"itself (a cycle)",
                    seen[name],
                )
            resolving.add(name)
            result = resolve_target(seen[name].target_type, seen[name])
            resolving.discard(name)
            resolved[name] = result
            return result

        def resolve_target(target, alias_node: TypeAlias) -> Type:
            """`alias_node` is threaded through purely for error
            attribution -- a bare name has no position of its own, so
            an "unknown type" here is blamed on the alias declaration
            containing it, however deep target's own array/slice
            nesting goes."""
            if isinstance(target, ArrayTypeExpr):
                return Type(TypeKind.ARRAY, element_type=resolve_target(target.element_type, alias_node), size=target.size)
            if isinstance(target, SliceTypeExpr):
                return Type(TypeKind.SLICE, element_type=resolve_target(target.element_type, alias_node))
            # target is a bare name from here on -- int/bool/str, an
            # alias (possibly not yet resolved -- recurse into resolve
            # itself, which is what makes forward references and cycle
            # detection work), or a struct name.
            if target in _TYPE_NAMES:
                return _TYPE_NAMES[target]
            if target in seen:
                return resolve(target)
            if target in structs:
                return Type(TypeKind.STRUCT, struct_name=target)
            raise SemanticError(
                f"Unknown type '{target}' in a type alias's own target "
                f"-- expected int, bool, str, a struct name, or "
                f"another type alias",
                alias_node,
            )

        for ad in alias_defs:
            resolve(ad.name)
        return resolved

    def _reserve_struct_names(self, struct_defs: List[StructDef]) -> Dict[str, StructInfo]:
        """Reserves every struct's NAME up front (a None placeholder
        in the registry), rejecting a duplicate or builtin-colliding
        name. Split from field resolution (_resolve_struct_fields) so
        type-alias resolution can run in between them -- an alias can
        target a struct name (needs only the name), while a struct
        field can target an alias (needs aliases resolved first); the
        opposite orderings are satisfied by splitting struct collection
        into two passes with alias resolution between them.

        The alias-vs-struct-name collision check lives in _collect_
        type_aliases instead (a new alias name against an already-
        reserved struct name) -- by the time this runs, no alias
        exists yet to collide with."""
        registry: Dict[str, StructInfo] = {}
        for sd in struct_defs:
            if sd.name in _BUILTIN_FUNCTION_NAMES:
                raise SemanticError(
                    f"'{sd.name}' is a builtin and can't be used as a "
                    f"struct name",
                    sd,
                )
            if sd.name in registry:
                raise SemanticError(f"Struct '{sd.name}' is already declared", sd)
            registry[sd.name] = None
        return registry

    def _resolve_struct_fields(self, struct_defs: List[StructDef], registry: Dict[str, StructInfo]) -> Dict[str, StructInfo]:
        """Resolves every struct's field types, then checks for
        cycles. `registry` already has every struct's NAME reserved (a
        None placeholder), which is what lets a forward reference work
        (`type A struct: B b` before `type B struct: ...` is
        declared) -- type_from_name's struct-name check only needs
        NAME membership.

        1. Resolve each struct's field types (rejecting a duplicate
           field name), replacing its None placeholder with a real
           StructInfo. A field's type can be anything, including a
           slice (see codegen.py's analyze_array_escapes for how a
           slice-typed field's backing array gets the same escape
           treatment as other slice-holding shapes).
        2. Only once every struct's fields are fully resolved, check
           each for a cycle (_check_struct_contains) -- needs real,
           resolved types to walk, so it's a separate pass after 1."""
        for sd in struct_defs:
            fields: Dict[str, Type] = {}
            for f in sd.fields:
                if f.name in fields:
                    raise SemanticError(
                        f"Field '{f.name}' is already declared in struct '{sd.name}'",
                        f,
                    )
                fields[f.name] = type_from_name(f.field_type, registry, self.type_aliases, f)
            registry[sd.name] = StructInfo(name=sd.name, fields=fields)

        by_name = {sd.name: sd for sd in struct_defs}
        for sd in struct_defs:
            self._check_struct_contains(sd.name, registry, path=[], by_name=by_name)

        return registry

    def _check_struct_contains(self, name: str, registry: Dict[str, StructInfo], path: List[str], by_name: Dict[str, StructDef]) -> None:
        """DFS over the struct-containment graph -- X has an edge to Y
        if X has a field of type Y, directly or through any depth of
        array wrapping (an array embeds its element inline, N times
        over, so an array of a struct that contains X is exactly as
        size-infinite as X containing itself). A SLICE field
        deliberately doesn't count: its backing storage is a separate,
        runtime-sized allocation, not embedded inline -- `type A
        struct: []A elements` is a genuinely supported pattern (a tree
        or linked structure), not merely tolerated.

        `path` is the visited chain, for a readable error message --
        struct counts are small enough that this doesn't need
        memoization against already-explored, cycle-free structs.
        `by_name` is only for error attribution -- StructInfo (registry's
        own value type) has no position of its own, so the original
        StructDef is looked up separately, by whichever name closes
        the cycle."""
        if name in path:
            cycle = ' -> '.join(path + [name])
            raise SemanticError(
                f"Struct '{name}' cannot contain itself, directly or "
                f"transitively: {cycle}",
                by_name[name],
            )
        info = registry[name]
        for field_type in info.fields.values():
            contained = self._directly_embedded_struct_name(field_type)
            if contained is not None:
                self._check_struct_contains(contained, registry, path + [name], by_name)

    @staticmethod
    def _directly_embedded_struct_name(field_type: Type) -> Optional[str]:
        """If `field_type` is a struct, or an array (at any depth) of
        one, returns that struct's name -- see _check_struct_contains
        for why arrays count and slices don't. None for a scalar field,
        a slice-typed field (of anything), or an array of scalars."""
        while field_type.kind == TypeKind.ARRAY:
            field_type = field_type.element_type
        return field_type.struct_name if field_type.kind == TypeKind.STRUCT else None

    @staticmethod
    def _contains_sum_type_at_any_array_depth(t: Type) -> bool:
        """True if `t` is itself a sum type, or an array (at any
        depth) of one -- the same array-unwrapping _directly_embedded_
        struct_name already does for structs, but asking a narrower
        question (just "is a sum type anywhere in here", not "which
        struct"). Used only to decide whether a VarDecl's own zero-
        value case is even well-defined (see analyze_var_decl) -- a
        sum type has none, so neither does an array of them. A SLICE
        of sum types is deliberately NOT unwrapped the way ARRAY is:
        its own zero value (the nil slice) never actually touches an
        individual element."""
        while t.kind == TypeKind.ARRAY:
            t = t.element_type
        return t.kind == TypeKind.SUM

    def check_extern_function_decl(self, ext: ExternFunctionDecl) -> None:
        """Registers an `extern` declaration into self.functions, the
        exact same registry ordinary Function signatures already live
        in -- check_call's own lookup never needs to know which kind
        of function it found. Mirrors the collision checks the
        ordinary-function loop just above already does (builtin,
        struct, alias, sum-type, already-declared name), plus one more
        extern-specific check with no ordinary-function counterpart:
        every param and the return type must be a scalar or pointer
        kind.

        That restriction is a v1 SCOPE line, not a permanent one --
        struct-by-value across an FFI boundary raises a real question
        this doesn't answer yet (Hornet's own struct layout has no
        padding or alignment at all, unlike C's, so the two can
        silently disagree the moment a struct mixes int8/uint8 fields
        with wider ones -- see this feature's own design discussion).
        Array and slice are excluded for the same reason a struct is
        (an array is just as layout-sensitive, and a slice's own
        three-word descriptor has no C equivalent to line up against
        at all), and sum type for the additional reason that its own
        runtime tag has no meaning to C code regardless of layout.
        Scalar and pointer are excluded from this restriction because
        they're the one shape with an unambiguous, single, already-
        agreed-on C representation on both sides: a scalar's own
        width already matches its C counterpart's (an `int` here is
        already 4 bytes, `int64` already 8, ...), and a pointer is
        just an 8-byte address regardless of what it points at,
        exactly like C's own pointer types are."""
        if ext.name in _BUILTIN_FUNCTION_NAMES:
            raise SemanticError(
                f"'{ext.name}' is a builtin and can't be redefined as "
                f"an extern function",
                ext,
            )
        if ext.name in self.structs:
            raise SemanticError(
                f"Extern function '{ext.name}' collides with a struct "
                f"of the same name -- struct and function names share "
                f"one namespace and can never be the same, since "
                f"'{ext.name}(...)' would otherwise be ambiguous "
                f"between a call and a struct literal",
                ext,
            )
        if ext.name in self.type_aliases:
            raise SemanticError(
                f"Extern function '{ext.name}' collides with a type "
                f"alias of the same name -- function and type-alias "
                f"names share one namespace and can never be the same",
                ext,
            )
        if ext.name in self.sum_types:
            raise SemanticError(
                f"Extern function '{ext.name}' collides with a sum "
                f"type of the same name -- function and sum-type "
                f"names share one namespace and can never be the same",
                ext,
            )
        if ext.name in self.functions:
            raise SemanticError(f"Function '{ext.name}' is already declared", ext)

        param_types = [type_from_name(p.type, self.structs, self.type_aliases, p, self.sum_types) for p in ext.params]
        return_type = Type.VOID if ext.return_type is None else type_from_name(ext.return_type, self.structs, self.type_aliases, ext, self.sum_types)

        for p, p_type in zip(ext.params, param_types):
            if p_type.kind in (TypeKind.ARRAY, TypeKind.SLICE, TypeKind.STRUCT, TypeKind.SUM):
                raise SemanticError(
                    f"Extern function '{ext.name}''s parameter '{p.name}' has "
                    f"type {p_type} -- only scalar and pointer types are "
                    f"supported in an extern function's signature for now "
                    f"(array/slice/struct/sum-typed parameters aren't yet)",
                    p,
                )
        if return_type.kind in (TypeKind.ARRAY, TypeKind.SLICE, TypeKind.STRUCT, TypeKind.SUM):
            raise SemanticError(
                f"Extern function '{ext.name}' returns {return_type} -- "
                f"only scalar and pointer types are supported as an "
                f"extern function's own return type for now "
                f"(array/slice/struct/sum aren't yet)",
                ext,
            )

        self.functions[ext.name] = (param_types, return_type)

    def analyze_function(self, fn: Function) -> None:
        self.scopes = [{}]  # fresh, single-level scope stack per function
        self.loop_depth = 0
        # Parameters act like already-declared locals from the body's
        # point of view -- _declare here also gets duplicate-parameter-
        # name checking for free (`def int f(int a, int a):` collides in
        # this same scope exactly like `int a` twice in a row would).
        for p in fn.params:
            self._declare(p.name, type_from_name(p.type, self.structs, self.type_aliases, p, self.sum_types), p)
        return_type = Type.VOID if fn.return_type is None else type_from_name(fn.return_type, self.structs, self.type_aliases, fn, self.sum_types)
        for stmt in fn.body:
            self.analyze_statement(stmt, return_type)
        # Checked last, after every statement is individually known
        # well-typed -- see the module docstring's ALL PATHS RETURN
        # section. A function with no declared return type is the one
        # exception: falling off the end is exactly how it's expected
        # to exit (codegen.py's gen_function still guarantees a real
        # `ret` either way, via an unconditional trailing epilogue).
        if return_type != Type.VOID and not always_returns(fn.body):
            raise SemanticError(
                f"Function '{fn.name}' (declared to return {return_type}) "
                f"does not return a value on all code paths",
                fn,
            )

    # -- scope stack ------------------------------------------------------

    def _push_scope(self) -> None:
        self.scopes.append({})

    def _pop_scope(self) -> None:
        self.scopes.pop()

    def _declare(self, name: str, type_: Type, node: Optional[Node] = None) -> None:
        """Adds `name` to the *current* (innermost) scope. Only checks
        that scope for a collision -- a name already declared in an
        enclosing scope is fine to shadow, it's only a re-declaration
        error if it collides with something in this same block."""
        if name in self.scopes[-1]:
            raise SemanticError(f"Variable '{name}' is already declared in this scope", node)
        self.scopes[-1][name] = type_

    def _lookup(self, name: str, node: Optional[Node] = None) -> Type:
        """Resolves `name` by walking outward from the innermost scope
        to the outermost, returning the type from the first (nearest
        enclosing) match."""
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        raise SemanticError(f"Reference to undeclared variable '{name}'", node)

    # -- statements ---------------------------------------------------

    def analyze_statement(self, stmt: Node, return_type: Type) -> None:
        if isinstance(stmt, VarDecl):
            self.analyze_var_decl(stmt)
        elif isinstance(stmt, Assign):
            self.analyze_assign(stmt)
        elif isinstance(stmt, IndexAssign):
            self.analyze_index_assign(stmt)
        elif isinstance(stmt, FieldAssign):
            self.analyze_field_assign(stmt)
        elif isinstance(stmt, DerefAssign):
            self.analyze_deref_assign(stmt)
        elif isinstance(stmt, Return):
            self.analyze_return(stmt, return_type)
        elif isinstance(stmt, If):
            self.analyze_if(stmt, return_type)
        elif isinstance(stmt, While):
            self.analyze_while(stmt, return_type)
        elif isinstance(stmt, Break):
            self.analyze_break(stmt)
        elif isinstance(stmt, Continue):
            self.analyze_continue(stmt)
        elif isinstance(stmt, ExprStmt):
            self._check_expr_allowing_struct_literal(stmt.expr)  # evaluated for validity; result unused
        else:
            raise SemanticError(f"No semantic rule for statement: {stmt!r}", stmt)

    def _types_compatible(self, value_type: Type, target_type: Type) -> bool:
        """True if a value of `value_type` can be used where
        `target_type` is expected -- ordinary equality, or one of
        three exceptions this language allows: Type.NONE is compatible
        with ANY slice type OR any pointer type (both share the same
        "absent" zero/nil value), and a struct is compatible with a
        sum type that lists it as one of its own variants (the one and
        only way a sum-typed value ever gets its value at all, there
        being no separate variant-constructor syntax -- see
        SumTypeDef's own docstring). Deliberately narrow otherwise --
        not int/bool/str/array, even though str is also a pointer
        under the hood, and NOT sum-type-to-sum-type even when their
        variant lists happen to overlap -- only a bare struct widens
        into a sum type, not a value that's already been widened into
        a different one.

        Shared by every site with a clear "this is the expected type"
        side (a VarDecl initializer, Assign, IndexAssign, argument,
        return value), so `none` and struct-to-sum-type both become
        valid at all of them uniformly. Equality (`==`/`!=`) has no
        such fixed side and is checked separately, directly in
        check_binary."""
        if value_type == target_type:
            return True
        if value_type == Type.NONE and target_type.kind in (TypeKind.SLICE, TypeKind.POINTER):
            return True
        if value_type.kind == TypeKind.STRUCT and target_type.kind == TypeKind.SUM:
            return value_type.struct_name in self.sum_types[target_type.sum_type_name].variants
        return False

    def _as_folded_int_literal(self, expr: Node) -> Optional[int]:
        """If `expr` is a compile-time integer literal -- a bare
        Constant, or a Unary NEGATE wrapping one -- returns its folded
        value; None otherwise (a variable, call, arithmetic
        expression, ...). Checking Unary(NEGATE, Constant) matters:
        `-100` always parses as that shape (the lexer's NUMBER rule has
        no minus sign), never as a Constant already holding -100 -- so
        without this case, `int8 x = -100` would fail the int8/uint8
        range check by never being recognized as a literal at all.
        Doesn't fold anything deeper (`- -100`, `100 + 1`) -- this is
        for the two shapes source actually produces, not general
        constant folding."""
        if isinstance(expr, Constant):
            return expr.value
        if isinstance(expr, Unary) and expr.op == UnaryOp.NEGATE and isinstance(expr.operand, Constant):
            return -expr.operand.value
        return None

    def _check_value_flowing_into(self, expr: Node, target_type: Type) -> Type:
        """Type-checks `expr` as a value flowing into an already-typed
        slot, returning its type for the caller's _types_compatible
        check -- almost always just check_expr, with three exceptions:

        1. An UNTYPED array literal flowing into a SLICE- or ARRAY-
           typed target is checked against target_type's own element
           type directly (via check_array_literal's expected_element_
           type), not purely inferred and then matched -- what makes
           the untyped form behave identically to the fully-typed one,
           and what lets it correctly produce int8/uint8 elements (an
           independently-inferred element would land on plain int and
           simply fail to match). Returns target_type itself for a
           SLICE target (an actual, named array is deliberately NOT
           given this treatment -- it still needs an explicit `arr[:]`
           to become one); returns the literal's own computed array
           type for an ARRAY target, so a real size mismatch still
           surfaces as an ordinary _types_compatible failure.

        2. A compile-time integer LITERAL (see _as_folded_int_literal)
           flowing into an int8/uint8 target is checked against that
           type's own range instead of requiring an exact type match
           -- the only way to produce one at all, with no casting yet.
           An arbitrary int-typed EXPRESSION does NOT get this
           treatment (`int8 x = someIntVariable` is a real mismatch,
           matching this language's explicit-over-implicit stance).
           Annotates target_type onto expr directly, overwriting
           check_expr's own Type.INT, and returns target_type so the
           compatibility check trivially succeeds.

        3. The same literal shape flowing into int64 gets the same
           treatment minus the range check (int64's range is a strict
           superset of int's) -- still scoped to a literal only, not
           an arbitrary expression, for the same explicit-over-
           implicit consistency, even though widening would be safe.

        Cases 2/3 bypass check_expr's dispatch since it has no way to
        receive an expected type. Shared by analyze_var_decl/analyze_
        assign -- the two places with a clear "expected type" side
        (unlike `==`/`!=`, handled separately in check_binary)."""
        if isinstance(expr, ArrayLiteral) and expr.type_expr is None and target_type.kind in (TypeKind.SLICE, TypeKind.ARRAY):
            array_type = self.check_array_literal(expr, expected_element_type=target_type.element_type)
            expr.resolved_type = array_type
            return target_type if target_type.kind == TypeKind.SLICE else array_type
        value_type = self.check_expr(expr)
        if value_type == Type.INT and target_type in _NARROW_INT_RANGES:
            literal_value = self._as_folded_int_literal(expr)
            if literal_value is not None:
                lo, hi = _NARROW_INT_RANGES[target_type]
                if not (lo <= literal_value <= hi):
                    raise SemanticError(
                        f"{literal_value} is out of range for "
                        f"{target_type} ({lo} to {hi})",
                        expr,
                    )
                self._annotate_literal_resolved_type(expr, target_type)
                return target_type
        if value_type == Type.INT and target_type == Type.INT64:
            if self._as_folded_int_literal(expr) is not None:
                self._annotate_literal_resolved_type(expr, target_type)
                return target_type
        return value_type

    def _annotate_literal_resolved_type(self, expr: Node, target_type: Type) -> None:
        """Sets expr.resolved_type = target_type -- and, if expr is a
        Unary NEGATE wrapping a Constant (the `-100` shape), ALSO sets
        the inner Constant's own resolved_type. The second part is a
        real, previously-found bug fix, not defensive: check_expr's
        earlier pass already set the inner Constant to plain Type.INT,
        and codegen.py's gen_expr_into reads a Constant's own resolved_
        type directly to decide its width -- INT64 needs a full 64-bit
        MovQ (the value can exceed 32-bit range), while INT8/UINT8/INT
        stay an ordinary 32-bit Mov. Leaving the inner annotation stale
        silently truncated a large int64 literal's own immediate
        before the outer negation ever ran on the correct value."""
        expr.resolved_type = target_type
        if isinstance(expr, Unary) and expr.op == UnaryOp.NEGATE and isinstance(expr.operand, Constant):
            expr.operand.resolved_type = target_type

    def _check_expr_allowing_struct_literal(self, expr: Node) -> Type:
        """check_expr, except a struct literal is recognized and
        routed through check_struct_literal instead of falling into
        check_call's rejection of it. Shared by every position that
        allows a struct literal with no "already-typed slot" to flow
        into: a call argument, a return value, a nested struct-literal
        argument, and an array-literal element with no target element
        type yet. See _check_value_flowing_into_allowing_struct_
        literal, its sibling below, for positions that also need the
        untyped-array-into-slice treatment on top of this.

        Deliberately NOT used by analyze_index_assign/analyze_field_
        assign -- a struct literal as an IndexAssign/FieldAssign value
        remains a separate, not-yet-covered follow-up."""
        if isinstance(expr, Call) and expr.name in self.structs:
            return self.check_struct_literal(expr)
        return self.check_expr(expr)

    def _check_value_flowing_into_allowing_struct_literal(self, expr: Node, target_type: Type) -> Type:
        """The _check_value_flowing_into counterpart to _check_expr_
        allowing_struct_literal above, for positions that also need
        the untyped-array-into-slice treatment: analyze_var_decl/
        analyze_assign, and check_array_literal's typed/expected-
        element-type branches. Struct-literal detection is checked
        first, before target_type is consulted, since a struct
        literal's type comes entirely from its own name -- a mismatch
        is still caught afterward by the caller's ordinary _types_
        compatible check."""
        if isinstance(expr, Call) and expr.name in self.structs:
            return self.check_struct_literal(expr)
        return self._check_value_flowing_into(expr, target_type)

    def analyze_var_decl(self, stmt: VarDecl) -> None:
        declared_type = type_from_name(stmt.var_type, self.structs, self.type_aliases, stmt, self.sum_types)
        if stmt.init is None and self._contains_sum_type_at_any_array_depth(declared_type):
            # Unlike every other type, a sum type has no natural zero
            # value -- no variant is privileged as "the default", and
            # picking one implicitly (e.g. always the first-declared)
            # would be exactly the kind of implicit behavior this
            # language avoids everywhere else. So, unlike a struct or
            # array (every field/element zeroed) or a scalar (0/false/
            # ''), a sum-typed declaration must be explicitly
            # initialized -- and so, transitively, must an ARRAY of
            # them: `[3]Shape shapes` (no initializer) would need to
            # zero-fill 3 slots with a value that doesn't exist, same
            # as a bare `Shape s` would. A SLICE of sum types has no
            # such problem and isn't checked here -- its own zero
            # value is the nil slice (ptr=0, len=0, cap=0), with no
            # individual Shape ever actually written.
            raise SemanticError(
                f"'{stmt.name}' (declared {declared_type}) has no "
                f"initializer -- a sum type has no natural zero value, "
                f"so one is required here",
                stmt,
            )
        if stmt.init is not None:
            # Checked before `stmt.name` is added to scope below, so a
            # self-referential initializer (`int a = a`) correctly fails
            # as "undeclared variable" rather than reading itself.
            init_type = self._check_value_flowing_into_allowing_struct_literal(stmt.init, declared_type)
            if not self._types_compatible(init_type, declared_type):
                raise SemanticError(
                    f"Cannot initialize '{stmt.name}' (declared {declared_type}) "
                    f"with a value of type {init_type}",
                    stmt,
                )
        self._declare(stmt.name, declared_type, stmt)

    def analyze_assign(self, stmt: Assign) -> None:
        if stmt.name in self._narrowed_names:
            raise SemanticError(
                f"Cannot reassign '{stmt.name}' while it's narrowed by "
                f"an enclosing 'is' check -- assign to a different "
                f"variable instead",
                stmt,
            )
        declared_type = self._lookup(stmt.name, stmt)  # may resolve to an enclosing scope
        value_type = self._check_value_flowing_into_allowing_struct_literal(stmt.value, declared_type)
        if not self._types_compatible(value_type, declared_type):
            raise SemanticError(
                f"Cannot assign a value of type {value_type} to '{stmt.name}' "
                f"(declared {declared_type})",
                stmt,
            )

    def analyze_index_assign(self, stmt: IndexAssign) -> None:
        """`array[index] = value` -- value flows into the element type
        via _check_value_flowing_into_allowing_struct_literal, not a
        plain check_expr, so an untyped array/slice literal or a
        struct literal assigned into a slice- or struct-typed element
        gets the same treatment every other already-typed slot does.
        Needed no codegen changes: gen_index_assign's SLICE/STRUCT
        branches already handle every shape this can produce."""
        element_type = self._check_indexable_and_index(stmt.array, stmt.index)
        value_type = self._check_value_flowing_into_allowing_struct_literal(stmt.value, element_type)
        if not self._types_compatible(value_type, element_type):
            raise SemanticError(
                f"Cannot assign a value of type {value_type} to an array "
                f"element of type {element_type}",
                stmt,
            )

    def analyze_field_assign(self, stmt: FieldAssign) -> None:
        """`base.name = value` -- mirrors analyze_index_assign one
        level over, for the identical reasons."""
        field_type = self._check_struct_and_field(stmt.base, stmt.name)
        value_type = self._check_value_flowing_into_allowing_struct_literal(stmt.value, field_type)
        if not self._types_compatible(value_type, field_type):
            raise SemanticError(
                f"Cannot assign a value of type {value_type} to field "
                f"'{stmt.name}' of type {field_type}",
                stmt,
            )

    def analyze_deref_assign(self, stmt: DerefAssign) -> None:
        """`*pointer = value` -- writes through a pointer, mirroring
        analyze_field_assign/analyze_index_assign one level over: check
        `pointer` is pointer-typed, then that `value` is compatible
        with its element_type (the pointee's own type, what actually
        gets overwritten)."""
        pointer_type = self.check_expr(stmt.pointer)
        if pointer_type.kind != TypeKind.POINTER:
            raise SemanticError(
                f"Cannot dereference a value of type {pointer_type} for "
                f"assignment -- '*' requires a pointer operand",
                stmt.pointer,
            )
        pointee_type = pointer_type.element_type
        value_type = self._check_value_flowing_into_allowing_struct_literal(stmt.value, pointee_type)
        if not self._types_compatible(value_type, pointee_type):
            raise SemanticError(
                f"Cannot assign a value of type {value_type} through a "
                f"pointer to {pointee_type}",
                stmt,
            )

    def _check_indexable_and_index(self, base_expr: Node, index_expr: Node) -> Type:
        """Shared by check_index (`base[index]`) and analyze_index_
        assign (`base[index] = value`): validates `base_expr` is
        array- or slice-typed and `index_expr` is int-typed, returning
        the element type. Recurses correctly for multi-dimensional
        access for free: for `matrix[i][j]`, the outer call's base_expr
        is itself an Index node, so checking it via check_expr runs
        this same method again, returning the row's own element type.

        Named for what it accepts, not just arrays -- `s[i]` on a
        Slice uses this same check, since indexing a slice works
        identically to indexing an array from this file's point of
        view; only codegen differs in where it finds the address."""
        base_type = self.check_expr(base_expr)
        if base_type.kind not in (TypeKind.ARRAY, TypeKind.SLICE):
            raise SemanticError(
                f"Cannot index into a value of type {base_type} -- "
                f"only arrays and slices support indexing",
                base_expr,
            )
        index_type = self.check_expr(index_expr)
        if index_type != Type.INT:
            raise SemanticError(f"Index must be int, got {index_type}", index_expr)
        return base_type.element_type

    def check_slice(self, expr: Slice) -> Type:
        """`array[low:high]`. `array` must be array- or slice-typed,
        the same acceptance _check_indexable_and_index uses, since
        slicing a slice and slicing a multi-dimensional array's outer
        dimension are both valid. Either bound, if present, must be
        int; an omitted bound needs no check here -- its default is
        resolved later, at codegen time.

        The result is ALWAYS Type(SLICE, element_type=...) regardless
        of what's being sliced -- a slice expression's own type never
        depends on its bounds, only on the element type of whatever's
        being sliced, matching how check_index's own result never
        depends on WHICH index was used."""
        base_type = self.check_expr(expr.array)
        if base_type.kind not in (TypeKind.ARRAY, TypeKind.SLICE):
            raise SemanticError(
                f"Cannot slice a value of type {base_type} -- only "
                f"arrays and slices support slicing",
                expr,
            )
        if expr.low is not None:
            low_type = self.check_expr(expr.low)
            if low_type != Type.INT:
                raise SemanticError(f"Slice low bound must be int, got {low_type}", expr.low)
        if expr.high is not None:
            high_type = self.check_expr(expr.high)
            if high_type != Type.INT:
                raise SemanticError(f"Slice high bound must be int, got {high_type}", expr.high)
        return Type(TypeKind.SLICE, element_type=base_type.element_type)

    def analyze_return(self, stmt: Return, return_type: Type) -> None:
        """`return <expr>` or a bare `return` (Return's own docstring
        in parser.py). A bare return is valid exactly when the
        enclosing function has no declared return type. Returning a
        value from such a function is checked AFTER type-checking that
        value, not before, so a genuine error inside the value itself
        is reported rather than masked.

        A struct literal returned directly is checked via _check_expr_
        allowing_struct_literal. An array literal needs no equivalent
        special-casing -- array literals were never restricted, so
        plain check_expr already handles one correctly; this asymmetry
        is purely because struct literals are the only kind check_call
        rejects outside a short allow-list."""
        if stmt.value is None:
            if return_type != Type.VOID:
                raise SemanticError(
                    f"Function is declared to return {return_type}, but "
                    f"this bare 'return' returns nothing",
                    stmt,
                )
            return
        value_type = self._check_value_flowing_into_allowing_struct_literal(stmt.value, return_type)
        if return_type == Type.VOID:
            raise SemanticError(
                f"Function has no declared return type and cannot "
                f"return a value (got {value_type}) -- use a bare "
                f"'return' instead",
                stmt,
            )
        if not self._types_compatible(value_type, return_type):
            raise SemanticError(
                f"Function is declared to return {return_type}, but this "
                f"'return' statement returns {value_type}",
                stmt,
            )

    def analyze_if(self, stmt: If, return_type: Type) -> None:
        condition_type = self.check_expr(stmt.condition)
        if condition_type != Type.BOOL:
            raise SemanticError(
                f"'if' condition must be bool, got {condition_type} "
                f"(no implicit int-to-bool conversion -- try `x != 0` "
                f"instead of `x`)",
                stmt.condition,
            )

        if stmt.is_match:
            self._check_match_exhaustiveness(stmt)

        self._push_scope()
        # An IsCheck condition narrows its own variable_name to
        # type_name for exactly this then_body -- check_is_check has
        # already confirmed variable_name is a sum-typed variable and
        # type_name one of its own declared variants, so re-declaring
        # it here, in the scope just pushed, is a plain, ordinary
        # shadow (see _declare's own docstring: "a name already
        # declared in an enclosing scope is fine to shadow"), ordinary
        # innermost-scope lookup doing the rest for every reference
        # inside. Deliberately NOT done for else_body -- see IsCheck's
        # own docstring for why (no nameable "not Circle" type once a
        # sum type has more than two variants, so there's no single
        # rule that would apply consistently either way).
        #
        # _narrowed_names records the name for analyze_assign's own
        # reassignment check for exactly as long as this then_body
        # (and anything nested inside it) is being analyzed -- added
        # right before, removed right after, symmetric with the scope
        # push/pop themselves. A flat set, not a stack keyed to scope
        # depth: two DIFFERENT names narrowed in a nested `if shape1
        # is Circle: if shape2 is Square: ...` both stay valid
        # entries simultaneously with no conflict, and re-narrowing
        # the SAME name can't arise at all -- once narrowed, that
        # name's own type is already the plain struct variant, not a
        # sum type any more, so check_is_check's own first check would
        # already reject a second `shape is ...` on it before this
        # would ever matter.
        narrowed_name = None
        if isinstance(stmt.condition, IsCheck):
            narrowed_name = stmt.condition.variable_name
            self._declare(narrowed_name, Type(TypeKind.STRUCT, struct_name=stmt.condition.type_name), stmt.condition)
            self._narrowed_names.add(narrowed_name)
        for s in stmt.then_body:
            self.analyze_statement(s, return_type)
        if narrowed_name is not None:
            self._narrowed_names.discard(narrowed_name)
        self._pop_scope()

        # then/else get independent scopes (module docstring), so a
        # name in one is never visible in the other. An elif's else_
        # body is a single nested If (parser.py's If docstring);
        # analyze_if just recurses into it like any other statement.
        if stmt.else_body is not None:
            self._push_scope()
            for s in stmt.else_body:
                self.analyze_statement(s, return_type)
            self._pop_scope()

    def _check_match_exhaustiveness(self, stmt: If) -> None:
        """Walks a match-desugared If chain (stmt is its outermost
        node -- see If's own docstring) checking two things: no
        variant is tested more than once, and, if the chain has no
        explicit trailing `else:`, every one of the subject's own sum
        type's declared variants is covered by some arm.

        Walks EXACTLY stmt.match_arm_count steps through else_body[0]
        -- never "as long as else_body looks like a single nested If"
        -- since an ordinary, hand-written `if NAME is Type:` can
        legally be the sole statement inside this SAME match's own
        explicit `else:` block, indistinguishable by shape alone from
        one more synthesized arm (identical IsCheck condition shape,
        identical single-statement else_body). match_arm_count, fixed
        at desugaring time, is what makes this walk unambiguous
        regardless of what the user wrote in their own else block.

        Called from analyze_if AFTER check_expr(stmt.condition) --
        i.e. check_is_check -- has already confirmed the FIRST arm's
        own variable_name is a sum-typed variable and its type_name a
        real variant, so _lookup here is safe to assume succeeds."""
        subject_name = stmt.condition.variable_name
        subject_type = self._lookup(subject_name, stmt.condition)
        sum_type_info = self.sum_types[subject_type.sum_type_name]

        seen: Dict[str, IsCheck] = {}
        current = stmt
        for i in range(stmt.match_arm_count):
            arm_condition = current.condition
            if arm_condition.type_name in seen:
                raise SemanticError(
                    f"'{arm_condition.type_name}' is tested more than once in "
                    f"this match on '{subject_name}'",
                    arm_condition,
                )
            seen[arm_condition.type_name] = arm_condition
            if i < stmt.match_arm_count - 1:
                current = current.else_body[0]
        # current is now the LAST arm.

        if current.else_body is not None:
            return  # an explicit trailing else -- exhaustiveness not required

        missing = [v for v in sum_type_info.variants if v not in seen]
        if missing:
            raise SemanticError(
                f"This match on '{subject_name}' (declared {subject_type}) "
                f"doesn't cover every variant -- missing: {', '.join(missing)} "
                f"(add an arm for each, or an 'else:' to cover the rest)",
                stmt,
            )

    def analyze_while(self, stmt: While, return_type: Type) -> None:
        condition_type = self.check_expr(stmt.condition)
        if condition_type != Type.BOOL:
            raise SemanticError(
                f"'while' condition must be bool, got {condition_type} "
                f"(no implicit int-to-bool conversion -- try `x != 0` "
                f"instead of `x`)",
                stmt.condition,
            )

        # loop_depth, not the scope stack, is what break/continue check
        # against -- a counter, not a boolean, so a nested while's own
        # push/pop doesn't make an outer loop's break/continue invalid.
        self.loop_depth += 1
        self._push_scope()
        for s in stmt.body:
            self.analyze_statement(s, return_type)
        self._pop_scope()
        self.loop_depth -= 1

    def analyze_break(self, stmt: Break) -> None:
        if self.loop_depth == 0:
            raise SemanticError("'break' outside of a loop", stmt)

    def analyze_continue(self, stmt: Continue) -> None:
        if self.loop_depth == 0:
            raise SemanticError("'continue' outside of a loop", stmt)

    # -- expressions ----------------------------------------------------
    # Every check_* method both validates its node and returns its Type,
    # so callers (including other check_* methods, for operands) get
    # both in one call rather than needing a separate inference pass.

    def check_expr(self, expr: Node) -> Type:
        """Type-checks `expr` and, as a side effect, annotates it with
        the result (expr.resolved_type = result) -- the ONE place this
        happens; see the module docstring's TYPES section for the full
        design and why this replaced codegen.py's old, independently-
        duplicated _infer_type. Stores the actual Type object, not a
        name string, since an array type needs its own element_type/
        size too."""
        if isinstance(expr, Constant):
            result = self.check_constant(expr)
        elif isinstance(expr, BoolLiteral):
            result = Type.BOOL
        elif isinstance(expr, NoneLiteral):
            result = Type.NONE
        elif isinstance(expr, StringLiteral):
            result = Type.STR
        elif isinstance(expr, Variable):
            result = self.check_variable(expr)
        elif isinstance(expr, ArrayLiteral):
            result = self.check_array_literal(expr)
        elif isinstance(expr, Index):
            result = self.check_index(expr)
        elif isinstance(expr, Field):
            result = self.check_field(expr)
        elif isinstance(expr, Slice):
            result = self.check_slice(expr)
        elif isinstance(expr, Call):
            result = self.check_call(expr)
        elif isinstance(expr, Unary):
            result = self.check_unary(expr)
        elif isinstance(expr, Cast):
            result = self.check_cast(expr)
        elif isinstance(expr, Binary):
            result = self.check_binary(expr)
        elif isinstance(expr, IsCheck):
            result = self.check_is_check(expr)
        else:
            raise SemanticError(f"No semantic rule for expression: {expr!r}", expr)
        expr.resolved_type = result
        return result

    def check_array_literal(self, expr: ArrayLiteral, expected_element_type: Optional[Type] = None) -> Type:
        """`[e1, e2, ...]`, or the fully-typed `[N]TYPE[e1, e2, ...]`
        (expr.type_expr set). Every element must be the same type --
        no heterogeneous arrays.

        UNTYPED form (type_expr and expected_element_type both None):
        the type is inferred entirely from elements, each checked via
        check_expr and compared to the first. A ragged literal (e.g.
        `[[1,2,3],[4,5]]`) is rejected by this same check for free,
        since the two rows' types ([3]int vs [2]int) are simply
        different types once Type is structurally comparable. Needs at
        least one element -- nothing to infer from otherwise.

        TYPED form (an explicit type_expr, or a supplied expected_
        element_type from an untyped literal flowing into an already-
        typed VarDecl/Assign value): the declared type is resolved
        first and used as the standard every element is checked
        AGAINST, via _check_value_flowing_into_allowing_struct_literal
        (not plain check_expr) -- the same recursive treatment a top-
        level value gets. This is what makes genuinely nested slice
        construction (a slice of slices, `[][]int rows = [][]int[[1,
        2], [3, 4]]`) fall out for free rather than only working one
        level deep: a plain check_expr would infer each inner literal
        as an ordinary array, which would then fail to match the
        expected slice type (`[][2]int` happened to work by
        coincidence before this, since array-vs-array equality was all
        that case needed -- which is exactly what masked the gap until
        a genuinely nested slice was tried). Either typed path also
        correctly allows zero elements, unlike the untyped path -- a
        real, externally-known type exists even with nothing to infer.

        A struct-typed element (a Variable/Field/Index/Call, or a
        struct literal directly) needs no special case here -- check_
        expr/_check_value_flowing_into already handle it like any
        other type. codegen's gen_array_literal_into needed the actual
        fix, for a struct-typed element with no literal involved at
        all."""
        if expr.type_expr is not None:
            declared_type = type_from_name(expr.type_expr, self.structs, self.type_aliases, expr, self.sum_types)
            if len(expr.elements) != declared_type.size:
                raise SemanticError(
                    f"Array literal declares type {declared_type} (size "
                    f"{declared_type.size}), but has {len(expr.elements)} "
                    f"element(s)",
                    expr,
                )
            for i, element in enumerate(expr.elements, start=1):
                element_type = self._check_value_flowing_into_allowing_struct_literal(element, declared_type.element_type)
                if not self._types_compatible(element_type, declared_type.element_type):
                    raise SemanticError(
                        f"Array literal declares element type "
                        f"{declared_type.element_type}, but element {i} "
                        f"is {element_type}",
                        element,
                    )
            return declared_type

        if expected_element_type is not None:
            for i, element in enumerate(expr.elements, start=1):
                element_type = self._check_value_flowing_into_allowing_struct_literal(element, expected_element_type)
                if not self._types_compatible(element_type, expected_element_type):
                    raise SemanticError(
                        f"Array literal's elements must all be "
                        f"{expected_element_type} (to match the "
                        f"declared element type), but element {i} "
                        f"is {element_type}",
                        element,
                    )
            return Type(TypeKind.ARRAY, element_type=expected_element_type, size=len(expr.elements))

        if len(expr.elements) == 0:
            raise SemanticError("Array literals must have at least one element", expr)
        element_types = [self._check_expr_allowing_struct_literal(e) for e in expr.elements]
        first = element_types[0]
        for i, t in enumerate(element_types[1:], start=2):
            if t != first:
                raise SemanticError(
                    f"Array literal elements must all be the same type -- "
                    f"element 1 is {first}, element {i} is {t}",
                    expr.elements[i - 1],
                )
        return Type(TypeKind.ARRAY, element_type=first, size=len(expr.elements))

    def check_index(self, expr: Index) -> Type:
        return self._check_indexable_and_index(expr.array, expr.index)

    def check_field(self, expr: Field) -> Type:
        return self._check_struct_and_field(expr.base, expr.name)

    def _check_struct_and_field(self, base_expr: Node, field_name: str) -> Type:
        """Shared by check_field (`base.name`) and analyze_field_
        assign (`base.name = value`), mirroring _check_indexable_and_
        index one level over: check base_expr is struct-typed, look up
        field_name in its registered field list, and return the
        field's type -- or raise a clear error for whichever went
        wrong.

        A POINTER-to-struct base auto-dereferences here too, Go-style:
        `p.field` (and `p.field = value`, through this SAME shared
        helper) works directly whether `p` is a Circle or a *Circle,
        with no explicit `(*p).field` needed. This is the ONE place
        that decision needs making -- both callers, and every method-
        call receiver too (see check_method_call's own, separate
        auto-deref, mirroring this one for the identical reason: `.`
        should mean the same thing whether it's a field or a method),
        go through code that ultimately resolves a struct-typed base
        this same way."""
        base_type = self.check_expr(base_expr)
        if base_type.kind == TypeKind.POINTER and base_type.element_type.kind == TypeKind.STRUCT:
            base_type = base_type.element_type
        if base_type.kind != TypeKind.STRUCT:
            raise SemanticError(
                f"Cannot access field '{field_name}' on non-struct type {base_type}",
                base_expr,
            )
        struct_info = self.structs[base_type.struct_name]
        if field_name not in struct_info.fields:
            raise SemanticError(
                f"Struct '{base_type.struct_name}' has no field '{field_name}'",
                base_expr,
            )
        return struct_info.fields[field_name]

    def check_struct_literal(self, expr: Call) -> Type:
        """`Name(arg1, arg2, ...)` -- a struct literal, e.g. `A a =
        A(6, 'hello')`. Disambiguated from an ordinary function call
        purely by registry membership (`Name` is a struct, not a
        function) -- no dedicated parser syntax; analyze()'s struct/
        function collision check guarantees a name can never be both.

        Positional and exhaustive: exactly one argument per field, in
        declaration order -- no partial construction with an implicit
        zero value, matching this language's explicit-over-implicit
        stance.

        Each argument is checked via _check_expr_allowing_struct_
        literal, not plain check_expr, so a nested struct-literal
        argument (`A(B(1, 2), 3)`) recurses back into this same
        method -- arbitrary nesting depth for free. Reached from every
        position that allows a struct literal directly: analyze_var_
        decl/analyze_assign/analyze_index_assign/analyze_field_assign
        (via the shared helper), check_call's own argument loop,
        analyze_return, check_array_literal's own element loop, and
        this method's own argument loop recursively. Every OTHER
        position (a Binary operand, a Field-access base, ...) funnels
        through check_expr's ordinary dispatch into check_call, which
        rejects a struct-name Call -- the entire mechanism keeping
        struct literals scoped narrower than an ordinary call.
        Annotates expr.resolved_type directly, bypassing check_expr's
        own dispatch and its annotation step.

        NAMED construction (`A(x=1, y='a')`, expr.kwargs populated) is
        delegated to _check_named_struct_literal, which also supports
        PARTIAL construction (omitting a field, given its type's
        implicit zero value -- see gen_struct_literal_into in
        codegen.py). Positional construction stays exhaustive; only
        the named form can be partial."""
        struct_info = self.structs[expr.name]
        field_items = list(struct_info.fields.items())
        if expr.kwargs is not None:
            return self._check_named_struct_literal(expr, struct_info, field_items)
        if len(expr.args) != len(field_items):
            field_names = ', '.join(name for name, _ in field_items)
            raise SemanticError(
                f"Struct literal for '{expr.name}' expects "
                f"{len(field_items)} argument(s) (one per field, in "
                f"declaration order: {field_names}), got {len(expr.args)}",
                expr,
            )
        for i, (arg, (field_name, field_type)) in enumerate(zip(expr.args, field_items), start=1):
            arg_type = self._check_value_flowing_into_allowing_struct_literal(arg, field_type)
            if not self._types_compatible(arg_type, field_type):
                raise SemanticError(
                    f"Argument {i} to struct literal '{expr.name}' "
                    f"(field '{field_name}') should be {field_type}, "
                    f"got {arg_type}",
                    arg,
                )
        result = Type(TypeKind.STRUCT, struct_name=expr.name)
        expr.resolved_type = result
        return result

    def _check_named_struct_literal(self, expr: Call, struct_info: StructInfo, field_items: list) -> Type:
        """`A(x=1, y='a')`, or a PARTIAL `A(x=1)`. Unlike the
        positional form, not required to be exhaustive -- an unmentioned
        field gets its implicit zero value (gen_struct_literal_into),
        the same value an ordinary `A a` VarDecl with no initializer
        gives every field.

        Each name is checked against real field membership and against
        being specified more than once (`A(x=1, x=2)`) -- both genuine
        semantic questions the parser can't answer itself. Each value
        is checked via _check_expr_allowing_struct_literal, same as
        the positional form -- a nested struct literal as a field's
        value recurses the identical way."""
        field_types = struct_info.fields
        valid_names = ', '.join(name for name, _ in field_items)
        seen = set()
        for field_name, value in expr.kwargs:
            if field_name not in field_types:
                raise SemanticError(
                    f"Struct literal for '{expr.name}' has no field "
                    f"'{field_name}' -- valid fields are: {valid_names}",
                    expr,
                )
            if field_name in seen:
                raise SemanticError(
                    f"Field '{field_name}' specified more than once in "
                    f"struct literal for '{expr.name}'",
                    expr,
                )
            seen.add(field_name)
            value_type = self._check_value_flowing_into_allowing_struct_literal(value, field_types[field_name])
            expected_type = field_types[field_name]
            if not self._types_compatible(value_type, expected_type):
                raise SemanticError(
                    f"Field '{field_name}' of struct literal '{expr.name}' "
                    f"should be {expected_type}, got {value_type}",
                    value,
                )
        result = Type(TypeKind.STRUCT, struct_name=expr.name)
        expr.resolved_type = result
        return result

    def _check_method_call(self, expr: Call) -> Type:
        """`receiver.name(args)` -- resolves and validates a method
        call, then rewrites `expr` in place into an ordinary call to
        the matching mangled function (see Call's own docstring in
        parser.py for why an in-place rewrite).

        The receiver's type is checked via plain check_expr, not
        _check_expr_allowing_struct_literal -- a struct literal used
        directly as a receiver is already rejected by check_call's own
        guard; a struct-returning call as a receiver type-checks fine
        here but is later rejected by codegen's gen_struct_address_into
        (the same "assign to a variable first" restriction several
        other unnamed-struct positions already have), gotten for free
        by simply not special-casing the receiver.

        Argument checking mirrors check_call's ordinary-function loop:
        exact count, each checked via _check_expr_allowing_struct_
        literal against the method's declared parameter types -- the
        receiver is never counted, since it's never in the written
        argument list."""
        receiver_type = self.check_expr(expr.receiver)
        if receiver_type.kind == TypeKind.POINTER and receiver_type.element_type.kind == TypeKind.STRUCT:
            # Go-style auto-deref, mirroring _check_struct_and_field's
            # own identical decision for field access -- `.` means the
            # same thing here whether the receiver is a Circle or a
            # *Circle.
            receiver_type = receiver_type.element_type
        if receiver_type.kind != TypeKind.STRUCT:
            raise SemanticError(
                f"Cannot call method '{expr.name}' on a value of type "
                f"{receiver_type} -- methods are only defined on structs",
                expr.receiver,
            )
        key = (receiver_type.struct_name, expr.name)
        if key not in self.methods:
            raise SemanticError(
                f"Struct '{receiver_type.struct_name}' has no method "
                f"'{expr.name}'",
                expr,
            )
        param_types, return_type, mangled_name = self.methods[key]
        if len(expr.args) != len(param_types):
            raise SemanticError(
                f"Method '{expr.name}' on '{receiver_type.struct_name}' "
                f"expects {len(param_types)} argument(s), got "
                f"{len(expr.args)}",
                expr,
            )
        for i, (arg, expected_type) in enumerate(zip(expr.args, param_types), start=1):
            actual_type = self._check_value_flowing_into_allowing_struct_literal(arg, expected_type)
            if not self._types_compatible(actual_type, expected_type):
                raise SemanticError(
                    f"Argument {i} to method '{expr.name}' on "
                    f"'{receiver_type.struct_name}' should be "
                    f"{expected_type}, got {actual_type}",
                    arg,
                )
        expr.args = [expr.receiver] + expr.args
        expr.name = mangled_name
        expr.receiver = None
        expr.resolved_type = return_type
        return return_type

    def check_call(self, expr: Call) -> Type:
        if expr.receiver is not None:
            # Checked first: expr.name here is a METHOD name, which
            # could coincidentally match a struct or builtin name (no
            # shared namespace, so this is legal) -- receiver presence
            # takes priority to avoid a false-positive match below.
            return self._check_method_call(expr)
        if expr.name in self.structs:
            raise SemanticError(
                f"'{expr.name}(...)' is a struct literal, which is only "
                f"allowed as a variable's initializer, a plain "
                f"assignment's value, a direct function-call argument, "
                f"a direct return value, an array literal's own "
                f"element, an IndexAssign's own element, a "
                f"FieldAssign's own field, or a bare statement -- not "
                f"most other kinds of expressions (a Binary operand, a "
                f"Field-access base, ...); assign it to a variable "
                f"first if you need it in one of those positions",
                expr,
            )
        if expr.kwargs is not None:
            # Named arguments parse into the same shape a named struct
            # literal does -- this is the one place that distinction
            # is made, now that expr.name is known not to be a struct.
            # Named construction is scoped to struct literals only.
            raise SemanticError(
                f"'{expr.name}(...)' uses named arguments, which are "
                f"only supported for struct literals, not function calls",
                expr,
            )
        if expr.name == 'print':
            return self.check_print_call(expr)
        if expr.name == 'len':
            return self.check_len_call(expr)
        if expr.name == 'append':
            return self.check_append_call(expr)
        if expr.name not in self.functions:
            raise SemanticError(f"Call to undeclared function '{expr.name}'", expr)
        param_types, return_type = self.functions[expr.name]

        if len(expr.args) != len(param_types):
            raise SemanticError(
                f"Function '{expr.name}' expects {len(param_types)} "
                f"argument(s), got {len(expr.args)}",
                expr,
            )
        for i, (arg, expected_type) in enumerate(zip(expr.args, param_types), start=1):
            # _check_value_flowing_into_allowing_struct_literal, not
            # just _check_expr_allowing_struct_literal, so an untyped
            # array literal argument can flow into a slice-typed
            # parameter, and an int8/uint8-typed parameter can accept
            # a range-checked literal directly -- both previously fell
            # through to a plain mismatch here.
            actual_type = self._check_value_flowing_into_allowing_struct_literal(arg, expected_type)
            if not self._types_compatible(actual_type, expected_type):
                raise SemanticError(
                    f"Argument {i} to '{expr.name}' should be "
                    f"{expected_type}, got {actual_type}",
                    arg,
                )
        return return_type

    def check_print_call(self, expr: Call) -> Type:
        """`print` takes exactly one argument, of any REAL type -- not
        tied to one fixed parameter type, since every Hornet type is
        printable. "Real" excludes Type.VOID (the result of calling a
        no-declared-return-type function) specifically. An array or
        slice argument formats as `TYPE[elem, elem, ...]`, the type
        prefix appearing once at the outermost level (see codegen.py's
        _gen_print_collection); a str element is quoted inside a
        collection even though a bare str argument prints unquoted.

        print itself is Type.VOID -- Hornet's first, and so far only,
        builtin with no meaningful value to return. Nothing changes in
        codegen.py for this: gen_print_call_into still leaves something
        in %eax at the end of every path, but nothing reads it, same
        as any other void call's leftover register value."""
        if len(expr.args) != 1:
            raise SemanticError(
                f"'print' expects exactly 1 argument, got {len(expr.args)}",
                expr,
            )
        arg_type = self.check_expr(expr.args[0])
        if arg_type == Type.VOID:
            raise SemanticError(
                "'print' cannot be called with the result of a function "
                "that has no declared return type -- there's no value there to print",
                expr.args[0],
            )
        if arg_type == Type.NONE:
            raise SemanticError(
                "'print' cannot be called with a bare 'none' -- store it "
                "in a slice-typed variable first (e.g. `[]int s = none`), "
                "then print that",
                expr.args[0],
            )
        return Type.VOID

    def check_len_call(self, expr: Call) -> Type:
        """`len(x)`: x must be array- or slice-typed -- str isn't
        supported yet (a real, separable follow-up, not an oversight);
        every other type is rejected by this same, single check, where
        print's own much more permissive one needs several carve-outs.

        x is fully type-checked via check_expr regardless of whether
        codegen needs its computed value (an array's length is a
        compile-time constant, never actually read -- see gen_len_
        call_into), so an invalid expression buried inside x is still
        caught here.

        Always returns int -- a real, useful value, unlike print's
        VOID, so `len(x)` works as an ordinary expression."""
        if len(expr.args) != 1:
            raise SemanticError(
                f"'len' expects exactly 1 argument, got {len(expr.args)}",
                expr,
            )
        arg_type = self.check_expr(expr.args[0])
        if arg_type == Type.STR:
            raise SemanticError(
                "'len' does not support str arguments yet",
                expr.args[0],
            )
        if arg_type.kind not in (TypeKind.ARRAY, TypeKind.SLICE):
            raise SemanticError(
                f"'len' requires an array or slice argument, got {arg_type}",
                expr.args[0],
            )
        return Type.INT

    def check_append_call(self, expr: Call) -> Type:
        """`append(s, value)`, Hornet's third builtin -- Go-style:
        returns a NEW slice rather than mutating s in place (see
        codegen.py's APPEND BUILTIN section).

        s must be slice-typed; value must match its element type,
        checked via _check_value_flowing_into_allowing_struct_literal,
        the same recursive treatment a VarDecl/Assign/IndexAssign value
        gets -- so appending an untyped array literal into a slice-of-
        slices correctly constructs a fresh nested slice, and a struct
        literal appended into a slice of structs is recognized the
        same way any other allowed position is.

        Always returns s's own slice type -- append never changes what
        a slice is a slice OF. Doesn't restrict what kind of expression
        s is, matching gen_append_call_into's own generality on the
        codegen side, unlike print's/len's argument-shape restrictions
        (real, still enforced only at the codegen layer for those two).
        """
        if len(expr.args) != 2:
            raise SemanticError(
                f"'append' expects exactly 2 arguments, got {len(expr.args)}",
                expr,
            )
        slice_arg, value_arg = expr.args
        slice_type = self.check_expr(slice_arg)
        if slice_type.kind != TypeKind.SLICE:
            raise SemanticError(
                f"'append' requires a slice as its first argument, "
                f"got {slice_type}",
                slice_arg,
            )
        value_type = self._check_value_flowing_into_allowing_struct_literal(value_arg, slice_type.element_type)
        if not self._types_compatible(value_type, slice_type.element_type):
            raise SemanticError(
                f"'append' cannot append a value of type {value_type} "
                f"to a {slice_type} (element type "
                f"{slice_type.element_type})",
                value_arg,
            )
        return slice_type

    def check_constant(self, expr: Constant) -> Type:
        if isinstance(expr.value, float) and not expr.value.is_integer():
            raise SemanticError(
                f"'{expr.value}' is not a whole number -- this language has "
                f"no floating-point type; only int and bool exist",
                expr,
            )
        return Type.INT

    def check_variable(self, expr: Variable) -> Type:
        return self._lookup(expr.name, expr)

    def check_is_check(self, expr: IsCheck) -> Type:
        """`NAME is TypeName` -- an if/elif condition's own special
        shape (see IsCheck's own docstring in parser.py), not a
        general expression: checked here like any other, so it gets
        an ordinary resolved_type=Type.BOOL annotation and analyze_
        if's own "condition must be bool" check needs no special-
        casing for it at all. The actual NARROWING this enables --
        rebinding NAME to TypeName within the if's own then_body --
        happens separately, in analyze_if, the only caller that still
        has stmt.condition itself in hand (check_expr's own dispatch,
        here, only ever returns a bare Type).

        Three checks, in order: NAME must already be an in-scope, SUM-
        typed variable (not a struct, not a scalar -- there's nothing
        to narrow otherwise); TypeName must be a declared struct at
        all; and, more specifically, TypeName must be one of NAME's
        own sum type's declared variants -- not just any struct, since
        `shape is Rectangle`, Rectangle never one of Shape's own
        variants, can never be true, and letting it silently type-
        check as an always-false check would hide what's almost
        certainly a mistake -- the same reasoning _types_compatible
        already applies to a struct widening into an unrelated sum
        type."""
        variable_type = self._lookup(expr.variable_name, expr)
        if variable_type.kind != TypeKind.SUM:
            raise SemanticError(
                f"'{expr.variable_name}' (declared {variable_type}) is "
                f"not a sum type -- 'is' only narrows a sum-typed "
                f"variable to one of its own declared variants",
                expr,
            )
        if expr.type_name not in self.structs:
            raise SemanticError(f"'{expr.type_name}' is not a declared struct", expr)
        sum_type_info = self.sum_types[variable_type.sum_type_name]
        if expr.type_name not in sum_type_info.variants:
            raise SemanticError(
                f"'{expr.type_name}' is not one of {variable_type}'s own "
                f"declared variants ({', '.join(sum_type_info.variants)})",
                expr,
            )
        return Type.BOOL

    def check_unary(self, expr: Unary) -> Type:
        operand_type = self.check_expr(expr.operand)
        if expr.op in (UnaryOp.NEGATE, UnaryOp.COMPLEMENT):
            if operand_type not in _INTEGER_TYPES:
                raise SemanticError(
                    f"'{expr.op.symbol()}' requires an int, int8, "
                    f"uint8, or int64 operand, got {operand_type}",
                    expr,
                )
            # Stays the operand's own type -- -int8 is int8, not
            # promoted to int -- exactly the same "narrow stays
            # narrow" rule check_binary's own arithmetic operators
            # follow, one operand instead of two.
            return operand_type
        if expr.op == UnaryOp.NOT:
            if operand_type != Type.BOOL:
                raise SemanticError(
                    f"'not' requires a bool operand, got {operand_type} "
                    f"(no implicit int-to-bool conversion -- try "
                    f"`not (x == 0)` instead of `not x`)",
                    expr,
                )
            return Type.BOOL
        if expr.op == UnaryOp.ADDRESS_OF:
            # Restricted to a bare Variable for this first slice of
            # pointer support -- see PointerTypeExpr's own docstring
            # for the "widen later" framing this restriction shares
            # with pointer-to-pointer's own. `&s.field`/`&arr[i]` are
            # the natural next step (escape analysis already has a
            # "slot" concept for aggregate members, from slices), not
            # ruled out for a structural reason the way, say, `&(x +
            # 1)` (no variable, nothing to take the address OF) would
            # be -- just not built yet.
            if not isinstance(expr.operand, Variable):
                raise SemanticError(
                    f"'&' can only take the address of a bare variable "
                    f"for now, not {type(expr.operand).__name__} -- "
                    f"struct fields and array/slice elements are planned, "
                    f"not yet supported",
                    expr,
                )
            return Type(TypeKind.POINTER, element_type=operand_type)
        if expr.op == UnaryOp.DEREFERENCE:
            if operand_type.kind != TypeKind.POINTER:
                raise SemanticError(
                    f"'*' requires a pointer operand, got {operand_type}",
                    expr,
                )
            pointee_type = operand_type.element_type
            if pointee_type.kind in (TypeKind.ARRAY, TypeKind.SLICE, TypeKind.STRUCT, TypeKind.SUM):
                # Restricted to a SCALAR pointee for this first slice
                # of pointer support -- not a structural limitation
                # (unlike, say, `&` requiring a bare Variable, which
                # reflects what escape analysis can currently reason
                # about): reading a whole composite value out of a
                # dereferenced pointer as a SOURCE (`Circle c = *p`,
                # `someFunc(*p)`, `return *p`) would need every ir/
                # statements.py call site that currently recognizes
                # Variable/Field/Index as a composite-addressable
                # shape to also recognize this one -- a dozen call
                # sites across four files, a substantially larger
                # change than everything else in this pointer slice
                # combined. `p.field` (auto-deref, no explicit '*'
                # needed) and `*p = value` (DerefAssign, overwriting
                # the whole pointee) both already work regardless of
                # the pointee's own kind -- this restriction is
                # specifically about READING a composite value out
                # through an explicit '*', nothing else.
                raise SemanticError(
                    f"'*' on a pointer to {pointee_type} (a composite type) "
                    f"isn't supported yet as a value -- write through it "
                    f"with '*p = value', or access a field directly "
                    f"(auto-deref already handles 'p.field')",
                    expr,
                )
            return pointee_type
        raise SemanticError(f"No semantic rule for unary operator: {expr.op}", expr)

    def check_cast(self, expr: Cast) -> Type:
        """`TYPE(expr)` -- an explicit numeric cast. Resolves target_
        type via type_from_name, the same choke point every other
        type-name resolution in this file uses.

        The source is checked via plain check_expr, not the target-
        aware _check_value_flowing_into a VarDecl/Assign uses -- a
        cast's point is converting an ALREADY-typed value, unlike that
        special case, which exists specifically because a cast didn't
        yet exist as an alternative way to produce an int8/uint8 value.
        A literal argument still gets ordinary Type.INT treatment here,
        then converts like any other int-typed expression -- `int8(
        200)` wraps to -56 with no compile-time range check, since
        range-checking only makes sense for the "no other way to
        produce this value" case _check_value_flowing_into covers.

        Only int/int8/uint8/int64 are supported on either side; bool
        and str are syntactically valid targets but rejected here
        (bool is non-numeric everywhere else; str conversion is a
        fundamentally different kind of operation, a separate, later
        feature). A cast always produces exactly the type it names,
        unlike arithmetic (where int8 stays int8) -- there's no
        operand-dependent result to derive."""
        target_type = type_from_name(expr.target_type, self.structs, self.type_aliases, expr, self.sum_types)
        source_type = self.check_expr(expr.expr)
        if target_type not in _INTEGER_TYPES or source_type not in _INTEGER_TYPES:
            raise SemanticError(
                f"Cannot cast {source_type} to {target_type} -- casting "
                f"is only supported between int, int8, uint8, and int64 "
                f"right now",
                expr,
            )
        return target_type

    def _is_comparable_type(self, t: Type) -> bool:
        """Whether '==' is defined for a value of type `t` -- true for
        int/bool/str; recursively true for an ARRAY whose element type
        is itself comparable (this method IS the recursion, unwrapping
        one ARRAY level per call); recursively true for a STRUCT whose
        fields are all comparable; always false for a SLICE (slice
        equality beyond `s == none` isn't defined).

        Used by check_binary's ARRAY-vs-ARRAY and STRUCT-vs-STRUCT
        branches alike -- an array's own comparability already depends
        on this question applied to its element type, which can now
        legitimately be a struct, needing the identical recursive
        check. This is also what makes an array of comparable structs
        work with no extra plumbing: check_binary's ARRAY branch calls
        this on the whole array type, which unwraps down
        to the STRUCT case, which recurses into Point's own fields,
        exactly like it would for a bare Point-vs-Point comparison."""
        if t.kind == TypeKind.ARRAY:
            return self._is_comparable_type(t.element_type)
        if t.kind == TypeKind.STRUCT:
            struct_info = self.structs[t.struct_name]
            return all(self._is_comparable_type(field_type) for field_type in struct_info.fields.values())
        if t.kind == TypeKind.SLICE:
            return False
        return True  # INT, BOOL, STR, POINTER

    def check_binary(self, expr: Binary) -> Type:
        left_type = self.check_expr(expr.left)
        right_type = self.check_expr(expr.right)
        op = expr.op

        if op == BinaryOp.ADD:
            # Overloaded: int-family+int-family (both the SAME one,
            # never mixed) is arithmetic, str+str is concatenation.
            # Anything else is a type error. Checked explicitly here,
            # not via _require_same_integer_type, since str+str is a
            # second, entirely different valid shape that helper
            # knows nothing about.
            if left_type in _INTEGER_TYPES and left_type == right_type:
                return left_type
            if left_type == Type.STR and right_type == Type.STR:
                return Type.STR
            raise SemanticError(
                f"'+' requires two operands of the same integer type "
                f"(int, int8, uint8, or int64) or two str operands, "
                f"got {left_type} and {right_type}",
                expr,
            )

        if op in _INT_ONLY_BINARY_OPS:
            # Stays whichever integer type both operands were -- int8
            # + int8 is int8, never promoted to int (unlike C's own
            # integer-promotion rules).
            return self._require_same_integer_type(left_type, right_type, op, expr)

        if op in _ORDERING_OPS:
            self._require_same_integer_type(left_type, right_type, op, expr)
            return Type.BOOL

        if op in _EQUALITY_OPS:
            # A slice OR a pointer compared to `none` (either order)
            # is checked first, since it's meaningful and allowed --
            # both share the same "absent" zero/nil value, and this is
            # the one place equality doesn't have a fixed "target"
            # side the way _types_compatible's other callers do, so
            # its own none-vs-slice/pointer carve-out is checked
            # directly here rather than through that shared helper.
            none_vs_nilable = (
                (left_type == Type.NONE and right_type.kind in (TypeKind.SLICE, TypeKind.POINTER)) or
                (right_type == Type.NONE and left_type.kind in (TypeKind.SLICE, TypeKind.POINTER))
            )
            if none_vs_nilable:
                return Type.BOOL

            # ARRAY vs ARRAY: valid when both sides are the exact same
            # array type (Type's own structural equality already
            # checks size and element type together) AND the array is
            # comparable (_is_comparable_type) -- which now includes
            # an array of comparable structs, needing no changes here
            # since this branch already applies the check to the
            # whole array type rather than a fixed set of leaf kinds.
            if left_type.kind == TypeKind.ARRAY and right_type.kind == TypeKind.ARRAY:
                if left_type != right_type:
                    raise SemanticError(
                        f"Cannot compare {left_type} to {right_type} with "
                        f"'{op.symbol()}' -- arrays must have the same "
                        f"length and element type",
                        expr,
                    )
                if not self._is_comparable_type(left_type):
                    raise SemanticError(
                        f"'{op.symbol()}' does not support {left_type} "
                        f"operands -- array equality isn't defined yet "
                        f"when the elements are (or contain) a slice, "
                        f"which has no '==' defined for it yet",
                        expr,
                    )
                return Type.BOOL

            # STRUCT vs STRUCT: valid when both sides are the exact
            # same struct type AND every field, at any nesting depth,
            # is itself comparable. A slice-typed field anywhere has
            # no well-defined field-by-field comparison, same reason
            # an array of slices doesn't just above.
            if left_type.kind == TypeKind.STRUCT and right_type.kind == TypeKind.STRUCT:
                if left_type != right_type:
                    raise SemanticError(
                        f"Cannot compare {left_type} to {right_type} with "
                        f"'{op.symbol()}' -- structs must be the exact "
                        f"same type",
                        expr,
                    )
                if not self._is_comparable_type(left_type):
                    raise SemanticError(
                        f"'{op.symbol()}' does not support {left_type} "
                        f"operands -- struct equality isn't defined yet "
                        f"when a field (directly, or nested inside "
                        f"another struct or an array field) is a slice, "
                        f"which has no '==' defined for it yet",
                        expr,
                    )
                return Type.BOOL

            # A bare slice-vs-slice comparison is rejected outright:
            # codegen has no slice comparison logic, and it isn't even
            # well-defined yet (compare elements, like array equality
            # now does, or the pointer/length/cap triple?) -- a real
            # feature to consider later, not implemented yet. Sum-type
            # equality is rejected for the identical reason -- codegen
            # has no lowering for it yet either (tag equality, then
            # conditional payload equality only if the tags match, is
            # the obvious shape it would eventually take, mirroring
            # how STRUCT vs STRUCT above already works) -- not because
            # two Shapes being equal is nonsensical the way comparing
            # a void result would be.
            #
            # VOID is rejected because it's structurally nonsensical,
            # not a missing feature: `foo() == bar()`, neither with a
            # declared return type, would otherwise trivially type-
            # check (Type.VOID == Type.VOID), comparing two "nothing"s.
            # Every other place a void result might flow is already
            # rejected by never matching a real, user-declared type --
            # equality is the one place two void operands could match
            # each other instead.
            #
            # NONE is rejected for the same reason, except the case
            # already handled above -- equality has no fixed target
            # side the way _types_compatible's other callers do.
            if left_type.kind in (TypeKind.SLICE, TypeKind.VOID, TypeKind.NONE, TypeKind.SUM) or right_type.kind in (TypeKind.SLICE, TypeKind.VOID, TypeKind.NONE, TypeKind.SUM):
                raise SemanticError(
                    f"'{op.symbol()}' does not support slice, void, sum "
                    f"type, or none operands, except comparing a slice "
                    f"or pointer to none",
                    expr,
                )
            if left_type != right_type:
                raise SemanticError(
                    f"Cannot compare {left_type} to {right_type} with "
                    f"'{op.symbol()}' -- both sides must be the same type",
                    expr,
                )
            return Type.BOOL

        if op in _LOGICAL_OPS:
            self._require_type(left_type, Type.BOOL, op, expr)
            self._require_type(right_type, Type.BOOL, op, expr)
            return Type.BOOL

        raise SemanticError(f"No semantic rule for binary operator: {op}", expr)

    def _require_type(self, actual: Type, expected: Type, op, node: Optional[Node] = None) -> None:
        if actual != expected:
            raise SemanticError(
                f"'{op.symbol()}' requires {expected} operands, got {actual}",
                node,
            )

    def _require_same_integer_type(self, left_type: Type, right_type: Type, op, node: Optional[Node] = None) -> Type:
        """Requires left_type and right_type to be the exact SAME
        integer type -- never a mix, even between two different-but-
        both-integer types (`int8 + uint8` is rejected like `bool +
        int` already is). Returns that shared type as the operator's
        own result -- used directly by check_binary's _INT_ONLY_
        BINARY_OPS and ordering-operator cases. `node`, as everywhere
        else in this file, is purely for error attribution -- passed
        through unchanged from check_binary's own Binary node."""
        if left_type not in _INTEGER_TYPES or left_type != right_type:
            raise SemanticError(
                f"'{op.symbol()}' requires two operands of the same "
                f"integer type (int, int8, uint8, or int64), got "
                f"{left_type} and {right_type}",
                node,
            )
        return left_type


# ---------------------------------------------------------------------------
# Convenience entry points
# ---------------------------------------------------------------------------

def analyze(program: Program) -> None:
    SemanticAnalyzer().analyze(program)


def analyze_source(filename: str) -> Program:
    """Runs lex -> parse -> analyze on a file and returns the (now
    known-valid) Program, for callers that want the checked AST rather
    than just a pass/fail."""
    tokens = lex(filename)
    program = Parser(tokens).parse_program()
    analyze(program)
    return program


def main():
    arg_parser = argparse.ArgumentParser(description='Semantic analyzer')
    arg_parser.add_argument('file', type=str, help='File to check.')
    args = arg_parser.parse_args()
    analyze_source(args.file)
    print("OK: no semantic errors found")


if __name__ == '__main__':
    main()
