"""Semantic analysis

Walks a parsed Program (see parser.py) and rejects it before codegen
ever runs if it's not well-formed: an undeclared variable is
referenced, a variable is declared twice, or any expression's type
doesn't match what the surrounding context requires.

This is a genuinely new *phase* in the pipeline, not an extension of an
existing one:

    lex (lexer.py) -> parse (parser.py) -> analyze (semantic.py) -> codegen (codegen.py)

Before this pass existed, undeclared/double-declared variables were
only caught incidentally, deep inside codegen, when a variable's stack
offset was looked up. That worked, but it meant a name-resolution
problem only ever surfaced as a side effect of code generation, with no
place for a *type* system to hook in at all. Putting a real pass here
instead means codegen can go back to only ever being asked to generate
code for programs already known to be valid -- it doesn't need to
(and after this, mostly doesn't) defend against malformed input itself.

THE TYPE SYSTEM
-----------------
`int` and `bool` were the first two types (see semantic.Type for the
full, current set -- `str`, fixed-size arrays, and slices have all
been added since, each with their own typing rules documented at
their own check_* method rather than repeated here). This is a
genuinely *strong* static type system in the traditional PL sense:
there is no implicit conversion between them in either direction. A
bool is not a 0-or-1 int that happens to print differently -- it's a
distinct type, and using one where the other is expected is a type
error, full stop. In particular (and this is the part most likely to
surprise someone coming from C): `not`, `and`, and `or` all require
real `bool` operands. `not 0` is a type error, not "not true". Write
`not (x == 0)` or `not false` instead.

The operator typing rules, precisely:
  - Arithmetic (+ - * /) and the two purely-numeric unary operators
    (- ~): both/the operand(s) must be `int`; result is `int`.
  - Ordering comparisons (< > <= >=): both operands must be `int`;
    result is `bool`. (There's no inherent ordering on bool, so these
    don't accept bool operands.)
  - Equality (== !=): both operands must be the *same* type (either
    both int or both bool); result is `bool`. Comparing an int to a
    bool is a type error even though both are "just numbers" underneath
    -- that's exactly the kind of mismatch strong typing exists to
    catch.
  - Logical (not/and/or): all operands must be `bool`; result is `bool`.
  - An `if`/`elif` condition must be `bool` -- same rule as everywhere
    else: no int-as-truthy shortcut, write `x != 0` or similar.
  - A VarDecl's initializer, an Assign's value, and a Return's value
    must all match the relevant declared type (the variable's declared
    type, or the function's declared return type) exactly.

Number literals are always `int`. There's no float type in this
language yet, even though the lexer's NUMBER rule matches decimals (a
holdover from before there was any type checking to catch this) --
check_constant rejects a non-whole-number literal as a type error
rather than silently truncating or miscompiling it (codegen has no
instruction for a fractional immediate; before this pass existed, a
literal like `2.5` would have produced literally invalid assembly,
`movl $2.5, %eax`, with no clear error pointing at why).

SCOPING
--------
Blocks now nest (if/elif/else, and while), so scope tracking is a real
stack: self.scopes is a List[Dict[str, Type]], one dict per
currently-open block, with the function's own top-level body as the
bottom entry. Two rules fall out of using a stack rather than one flat
dict per function:
  - A variable declared inside an `if` (or `else`, or a `while` body)
    is only visible for the rest of that block -- analyze_if/
    analyze_while push a fresh scope before walking their statements
    and pop it again afterward, so the name simply isn't there anymore
    once the block ends. This applies uniformly whether the block runs
    zero times, once, or (for a while body) many times -- the scope is
    a *static* fact about the program text, pushed and popped exactly
    once during analysis, regardless of how many times the block might
    actually execute at runtime.
  - Shadowing is allowed: a block can declare a variable with the same
    name as one in an enclosing scope. Declaration only checks the
    *current* (innermost) scope for a collision (see _declare), while
    a reference or assignment resolves by walking outward from
    innermost to outermost until it finds a match (see _lookup) -- the
    same lookup order used for the classic "nearest enclosing
    declaration wins" semantics most block-scoped languages use.
  - The `then` and `else` branches of one `if` get *independent*
    scopes (each is its own push/pop), since they're mutually
    exclusive at runtime -- a name declared in `then` has no business
    being visible in `else`, and vice versa.

Aside from that, analyze_function/analyze_if/analyze_while still walk
statements in *program order*, adding each variable to its scope only
once its own VarDecl has actually been processed, so declare-before-use
in textual order and rejection of self-referential initializers
(`int a = a`) both still fall out for free, exactly as before -- they
just now apply per-scope rather than per-function.

This is notably different from codegen.py's own local-variable
handling, which pre-scans a whole function body (recursively, into
every if/else branch and while body) for VarDecls up front purely to
size the stack frame before emitting any instructions -- that pass is
about layout, not validity, and by the time it runs this one has
already guaranteed the program is well-formed. codegen.py also ends up
needing its own scope-stack, for a different reason: see its LOCAL
VARIABLES section.

LOOPS: break/continue VALIDITY
---------------------------------
`break` and `continue` are each only meaningful inside a loop -- there's
nothing to break out of, or skip the rest of an iteration of, at the
top level of a function. This is tracked with a simple counter,
self.loop_depth, incremented before analyzing a while's body and
decremented after (analyze_while), rather than anything scope-related:
it needs to survive being nested inside an `if` (a `break` inside an
`if` that's inside a `while` is fine -- loop_depth doesn't care about
intervening non-loop blocks) while still correctly resetting once a
nested while's own body finishes being analyzed, so an outer loop's
break/continue isn't accidentally validated by an inner loop that has
nothing to do with it. codegen.py mirrors this with its own stack of
(start_label, end_label) pairs -- see its LOOPS section -- for the same
underlying reason: break/continue always target the *innermost*
enclosing loop, never an outer one.

FUNCTIONS
----------
self.functions (name -> (param types, return type)) is deliberately a
single, program-wide, flat namespace -- completely separate from
self.scopes (variable names). That's what lets a variable and a
function share a name without colliding, and it's built in a dedicated
first pass over *every* function in the Program, before any function's
own body is checked (see analyze()). Doing it in two passes rather than
building each function's signature just before checking that function
is what makes call order not matter at all: a function can call one
defined later in the file, and a function can call itself (or two
functions can call each other) recursively, since by the time
analyze_function ever looks anything up in self.functions, every
signature -- including the current function's own -- is already there.

Parameters are bound into the function's own scope right at the start
of analyze_function, before any statement is walked, exactly like
already-declared locals -- which, as a side effect, means a duplicate
parameter name (`def int f(int a, int a):`) is caught by the ordinary
_declare collision check, with no separate check needed for it.

BUILTINS
---------
`print` is the first (only, so far) builtin -- a callable that isn't an
ordinary user-defined function and doesn't go through self.functions at
all. check_call special-cases it before ever consulting self.functions,
and analyze()'s signature-collection pass rejects any user function
whose name collides with a builtin (_BUILTIN_FUNCTION_NAMES), so there's
no ambiguity about which one wins -- a program simply can't define its
own `print`.

check_print_call accepts exactly one argument of *any* type (int, bool,
and str are all printable, and there's no reason to force a caller to
pick a differently-named builtin per type the way an ordinary function's
fixed parameter types would require), and always "returns" int. That
return type is a bit of a formality -- Hornet has no void type, and
`print(x)` is almost always used as a bare expression statement whose
value is thrown away -- but giving it *some* real type is what lets it
flow through the exact same ExprStmt path every other call already
uses, with nothing print-specific needed anywhere else in this pass.
codegen.py is where print's actual behavior (which underlying libc call
per argument type, and that it always evaluates to a clean 0 rather
than leaking through whatever puts/printf themselves return) lives --
see its BUILTINS section.

TYPES: ANNOTATING THE AST FOR codegen.py
-------------------------------------------
check_expr does one thing beyond type-checking: after computing an
expression's Type, it stores that result on the node itself
(expr.resolved_type = str(result)) before returning. Every check_*
method below stays a pure type-computation function with no knowledge
of this -- the annotation happens in exactly one place, check_expr's
own dispatch, and every recursive call for an operand, argument, or
condition anywhere in this file already goes through check_expr rather
than some check_* method directly (see analyze_var_decl, check_binary,
check_unary, check_call, and every other caller). That means every
expression node anywhere in a program, no matter how deeply nested,
ends up annotated automatically, with no separate wiring needed and no
per-node-type case to remember adding as this pass grows.

resolved_type is a plain string ('int' / 'bool' / 'str'), matching
str(Type.X), rather than a Type enum value directly -- parser.py
defines these dataclass fields and must not import semantic.py, which
already imports *from* parser.py; a Type-typed field would be
circular. This also happens to match codegen.py's own established
plain-string type representation exactly, so no translation is needed
on the read side either.

codegen.py reads this directly (see its _type_of) instead of
re-deriving an expression's type with its own independent logic, which
is what it used to do, via a method called _infer_type. That older
approach was a real liability, not just an aesthetic one: adding
`print` needed a Call case added to _infer_type separately from this
file's own check_call, and adding the six int-only operators (%  &  |
^  <<  >>) needed them added to _infer_type's own int-producing branch
separately from this file's own _INT_ONLY_BINARY_OPS. Both omissions
were easy to make and were only caught by manual testing, not by
anything that would have failed loudly on its own. Annotating the AST
here and having codegen.py read the annotation removes that second,
independently-maintained copy of the logic entirely -- whatever this
file already decided is just read directly downstream, whatever it
happens to be, with nothing left elsewhere to fall out of sync.

codegen.py's own scope-stack (offset AND type per local variable, kept
for resolving which of possibly-several same-named declarations a
Variable reference means -- see its LOCAL VARIABLES section) is a
deliberate, separate exception to this: it still exists after this
change, unrelated to resolved_type, since an expression's type alone
can never tell codegen *which stack slot* a variable reference resolves
to. That's a distinct kind of duplication (of scope/offset resolution,
not of type inference) that this annotation mechanism doesn't attempt
to address.

ALL PATHS RETURN
------------------
analyze_function's last step, after every statement in a function's
body is already known to be individually well-typed, is
always_returns(fn.body): does every execution path through this
function's body reach a `return` before falling off the end? This
applies to every function regardless of declared return type, since
this language has no void -- and it's not just a correctness nicety.
Once functions could call each other (see codegen.py's FUNCTIONS
section), a function whose generated code falls through to whatever
comes after it with no `ret` ever executed doesn't just return garbage
to its caller -- it corrupts the *calling* function's own stack, since
there's a real return address sitting on the stack from the `call` that
invoked it, with nothing left to pop it and jump there.

This is deliberately modeled as a simple, conservative "terminating
statement" check (the same shape Go's specification uses for this exact
problem) rather than a fully general flow analysis: always_returns scans
a list of statements front-to-back for the first one that, *on its
own*, guarantees a return, and stops there (anything after it doesn't
matter to this question -- dead/unreachable code is a separate concern
this doesn't address). A statement guarantees a return if it's a Return
itself; an If with a non-None else_body where both branches themselves
guarantee a return (which, since elif desugars into a nested If in
else_body, handles an elif chain of any length for free); or a `while
true` loop with no reachable break anywhere in its body.

That last case is the one genuinely subtle piece here. In general a
while loop can't guarantee anything -- its condition might be false
immediately, so its body might run zero times -- except when the
condition is the literal constant `true` (checked structurally, as
`isinstance(condition, BoolLiteral) and condition.value is True`; this
does not try to prove some other expression is always true, e.g. `1 ==
1` -- only the literal keyword counts). Even then, a `while true` loop
only guarantees a return if there's no way to escape it other than
returning: if it also contains a `break`, that break could fire and
fall through to whatever comes after the loop, so the loop stops
counting as guaranteeing anything on its own, and something has to
catch that path explicitly (typically a return placed right after the
loop). contains_reachable_break finds a break anywhere in a loop's own
body, including nested arbitrarily deep inside if/elif/else -- but
deliberately does NOT recurse into a *nested* while loop's own body, on
the same reasoning break already has for its own validity (see
analyze_break/loop_depth) and at the codegen level (see codegen.py's
loop_labels stack): a break inside an inner loop belongs to that inner
loop, not whatever loop encloses it.

ERROR REPORTING
-----------------
This raises SemanticError on the *first* problem found and stops,
matching how ParseError and CodegenError already behave elsewhere in
this pipeline, rather than collecting every error in the program and
reporting them all at once. Neither AST nodes nor this pass currently
track source positions (that information exists only transiently, on
Tokens, during parsing), so error messages name the offending variable,
operator, or type mismatch as specifically as possible without being
able to point at a line/column -- the same limitation CodegenError
already had. Adding position tracking to AST nodes would be a good,
fairly contained follow-up if these messages need to get more precise.
"""

import argparse
from dataclasses import dataclass
from enum import auto, Enum
from typing import Dict, List, Optional

from lexer import lex
from parser import (
    ArrayLiteral,
    ArrayTypeExpr,
    Assign,
    Binary,
    BinaryOp,
    BoolLiteral,
    Break,
    Call,
    Constant,
    Continue,
    ExprStmt,
    Function,
    If,
    Index,
    IndexAssign,
    Node,
    Param,
    Parser,
    Program,
    Return,
    Slice,
    SliceTypeExpr,
    StringLiteral,
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
    BOOL = auto()
    STR = auto()
    ARRAY = auto()
    SLICE = auto()


@dataclass(frozen=True)
class Type:
    """A type in this language: one of the three scalars (kind alone,
    element_type/size both None), an array (kind=ARRAY, element_type
    the Type one level down, size that dimension's fixed length), or a
    slice (kind=SLICE, element_type the Type one level down, size
    always None -- a slice's LENGTH is a runtime property of the slice
    VALUE, not part of its type the way an array's size is; see
    SliceTypeExpr's own docstring in parser.py). Two slices of the
    same element type are the same Type regardless of how long either
    one happens to be at runtime, unlike two arrays of different
    sizes, which are different types even with the same element type.

    Frozen specifically to get structural equality and hashing for
    free from the dataclass machinery, rather than writing __eq__ by
    hand -- which is what makes `Type(ARRAY, Type.INT, 3) ==
    Type(ARRAY, Type.INT, 3)` (two SEPARATE Type objects describing the
    same array shape) correctly True, and, just as importantly,
    `Type(ARRAY, Type.INT, 3) != Type(ARRAY, Type.INT, 4)` correctly
    True too -- [3]int and [4]int are different types, exactly like
    [3]int and [3]bool are, with no special-casing needed anywhere
    that already just does `left_type != right_type` (see
    check_binary's equality handling, analyze_var_decl,
    analyze_assign, check_call's argument checking -- none of them
    needed to change at all to correctly handle arrays, once Type
    itself became structurally comparable). This recurses correctly to
    arbitrary nesting depth for free too, since element_type is itself
    just another Type -- and the same structural-equality machinery is
    what makes `Type(SLICE, Type.INT) == Type(SLICE, Type.INT)` true
    regardless of which two separate Type objects produced it, with no
    additional code needed for SLICE specifically.
    """
    kind: TypeKind
    element_type: Optional['Type'] = None  # set when kind == ARRAY or SLICE
    size: Optional[int] = None             # only set when kind == ARRAY

    def __str__(self) -> str:
        if self.kind == TypeKind.ARRAY:
            return f"[{self.size}]{self.element_type}"
        if self.kind == TypeKind.SLICE:
            return f"[]{self.element_type}"
        return self.kind.name.lower()


# Singleton instances for the three scalar kinds -- assigned as class
# attributes after the class body (not instance fields set via
# __init__), so every existing `Type.INT`/`Type.BOOL`/`Type.STR`
# reference throughout this file keeps working completely unchanged.
# `frozen=True` only prevents mutating an INSTANCE's own fields after
# construction; it has nothing to say about adding attributes to the
# Type CLASS object itself, which is all this is doing.
Type.INT = Type(TypeKind.INT)
Type.BOOL = Type(TypeKind.BOOL)
Type.STR = Type(TypeKind.STR)


_TYPE_NAMES = {
    'int': Type.INT,
    'bool': Type.BOOL,
    'str': Type.STR,
}


def type_from_name(type_expr) -> Type:
    """Converts a parsed type expression (VarDecl.var_type /
    Function.return_type / Param.type, straight from parser.py) into a
    Type. `type_expr` is a plain str ('int'/'bool'/'str') for a scalar
    type, an ArrayTypeExpr for an array type, or a SliceTypeExpr for a
    slice type (see their own docstrings in parser.py) -- handled here
    by recursing on element_type, which is itself a plain str,
    another ArrayTypeExpr, or another SliceTypeExpr, naturally
    bottoming out at a scalar and handling arbitrarily-nested types
    (`[2][3]int`, `[][]int`, `[][3]int`, ...) with no depth limit or
    special-casing for "how many dimensions" or "which mix of array
    and slice".

    Only ever fails for a program that isn't syntactically valid in
    the first place -- parse_type() already restricts a scalar
    type_expr to 'int'/'bool'/'str', and already validates an
    ArrayTypeExpr's size is a positive whole number at parse time --
    so the KeyError case here is a defensive check, not a user-facing
    validation path."""
    if isinstance(type_expr, ArrayTypeExpr):
        element = type_from_name(type_expr.element_type)
        return Type(TypeKind.ARRAY, element_type=element, size=type_expr.size)
    if isinstance(type_expr, SliceTypeExpr):
        element = type_from_name(type_expr.element_type)
        return Type(TypeKind.SLICE, element_type=element)
    try:
        return _TYPE_NAMES[type_expr]
    except KeyError:
        raise SemanticError(f"Unknown type '{type_expr}'")


def always_returns(statements: List[Node]) -> bool:
    """Does every execution path through this list of statements reach
    a `return` before falling off the end? Used to enforce that every
    function returns on every code path -- see the module docstring's
    ALL PATHS RETURN section for the full reasoning and what this
    deliberately does and doesn't try to prove.

    Scans front-to-back for the first statement that, on its own,
    guarantees a return; if one is found, everything after it is
    irrelevant to *this* question (dead code is a separate concern this
    function doesn't address). Reaching the end without finding one
    means False -- there's some path through this block that falls
    through without returning.
    """
    for stmt in statements:
        if isinstance(stmt, Return):
            return True
        if isinstance(stmt, If):
            # Only counts if there's an else at all, and *both* sides
            # are themselves guaranteed to return -- an if with no else
            # can always just not run its body, so it can never by
            # itself guarantee anything about what happens next.
            if stmt.else_body is not None and always_returns(stmt.then_body) and always_returns(stmt.else_body):
                return True
        if isinstance(stmt, While):
            # A `while <cond>: ...` loop's body might run zero times
            # (whenever cond isn't literally the constant `true`), so in
            # general a while loop can never by itself guarantee a
            # return -- *unless* it's a genuine `while true` with no way
            # to break out of it, in which case it never falls through
            # to whatever comes after it at all (it either returns from
            # inside, or loops forever) -- either way, nothing after it
            # is reachable, which vacuously satisfies "never falls off
            # the end without returning". See contains_reachable_break
            # for why a break anywhere inside changes this.
            is_infinite = isinstance(stmt.condition, BoolLiteral) and stmt.condition.value is True
            if is_infinite and not contains_reachable_break(stmt.body):
                return True
        # VarDecl, Assign, Break, Continue, ExprStmt: none of these can
        # themselves guarantee a return, and none of them stop the scan
        # -- move on to the next statement.
    return False


def contains_reachable_break(statements: List[Node]) -> bool:
    """Does this list of statements contain a `break` that refers to
    *this* loop -- i.e., one not already claimed by a nested loop?
    Recurses into if/elif/else bodies (a break inside an if that's
    directly in this loop's body still belongs to this loop), but
    deliberately does NOT recurse into a nested While's own body -- a
    break there refers to that inner loop, not this one, exactly the
    same scoping break already has at the semantic-error-checking level
    (see analyze_break/loop_depth) and at the codegen level (see
    codegen.py's loop_labels stack). Only used by always_returns, to
    decide whether a `while true` loop is genuinely inescapable-except-
    by-return or not.
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


# Names that are builtins rather than ordinary user-definable functions
# -- see check_call and the module docstring's BUILTINS section. Kept as
# a set (not hardcoded string comparisons scattered around) so a second
# builtin later is "add a name here plus its own check_*/gen_* pair",
# not a search-and-replace.
_BUILTIN_FUNCTION_NAMES = {'print'}


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class SemanticError(Exception):
    """Raised on the first semantic problem found: an undeclared or
    re-declared variable, or a type mismatch anywhere in the program."""


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

# BinaryOp -> which typing rule applies. See the module docstring for
# what each category actually requires; this table is only "which
# bucket does this operator fall into", kept separate from check_binary
# so adding an operator later is "add it to the right set" rather than
# another branch of if/elif. ADD is deliberately NOT in this set -- it's
# overloaded (int+int is arithmetic, str+str is concatenation), so
# check_binary handles it as its own case rather than lumping it in with
# operators that only ever mean one thing.
#
# Named _INT_ONLY_BINARY_OPS rather than "_ARITHMETIC" now that it
# covers modulo, the bitwise operators, and the shifts too -- all of
# which share the exact same rule (both operands int, result int) as
# the original arithmetic operators, even though "arithmetic" isn't
# really the right word for, say, bitwise XOR.
_INT_ONLY_BINARY_OPS = {
    BinaryOp.SUBTRACT, BinaryOp.MULTIPLY, BinaryOp.DIVIDE, BinaryOp.MODULO,
    BinaryOp.BITWISE_AND, BinaryOp.BITWISE_OR, BinaryOp.BITWISE_XOR,
    BinaryOp.SHIFT_LEFT, BinaryOp.SHIFT_RIGHT,
}
_ORDERING_OPS = {BinaryOp.LESS_THAN, BinaryOp.GREATER_THAN,
                  BinaryOp.LESS_THAN_OR_EQUAL, BinaryOp.GREATER_THAN_OR_EQUAL}
_EQUALITY_OPS = {BinaryOp.EQUAL, BinaryOp.NOT_EQUAL}
_LOGICAL_OPS = {BinaryOp.AND, BinaryOp.OR}


class SemanticAnalyzer:
    """Type-checks and scope-checks a Program. Call analyze() once per
    Program; analyze_function() resets internal scope state, so a fresh
    SemanticAnalyzer isn't required per function, only per full run if
    you want to be safe against reuse across unrelated programs."""

    def __init__(self):
        self.scopes: List[Dict[str, Type]] = []
        self.loop_depth = 0  # how many enclosing `while` loops we're currently inside
        self.functions: Dict[str, tuple] = {}  # name -> (List[Type] param types, Type return type)

    def analyze(self, program: Program) -> None:
        # First pass: collect every function's signature before checking
        # any function's body. This is what makes call order not matter
        # -- a function can call one defined later in the file, or call
        # itself recursively -- since by the time analyze_function ever
        # looks anything up in self.functions, every signature is
        # already there. See the module docstring's FUNCTIONS section.
        self.functions = {}
        for fn in program.functions:
            if fn.name in _BUILTIN_FUNCTION_NAMES:
                raise SemanticError(
                    f"'{fn.name}' is a builtin and can't be redefined as "
                    f"a function"
                )
            if fn.name in self.functions:
                raise SemanticError(f"Function '{fn.name}' is already declared")
            param_types = [type_from_name(p.type) for p in fn.params]
            return_type = type_from_name(fn.return_type)
            self.functions[fn.name] = (param_types, return_type)

        # Second pass: now check each function's own body.
        for fn in program.functions:
            self.analyze_function(fn)

    def analyze_function(self, fn: Function) -> None:
        self.scopes = [{}]  # fresh, single-level scope stack per function
        self.loop_depth = 0
        # Parameters act like already-declared locals from the body's
        # point of view -- _declare here also gets duplicate-parameter-
        # name checking for free (`def int f(int a, int a):` collides in
        # this same scope exactly like `int a` twice in a row would).
        for p in fn.params:
            self._declare(p.name, type_from_name(p.type))
        return_type = type_from_name(fn.return_type)
        for stmt in fn.body:
            self.analyze_statement(stmt, return_type)
        # Checked last, after every statement is individually known to
        # be well-typed -- see the module docstring's ALL PATHS RETURN
        # section. Every function needs this regardless of return type,
        # since this language has no void: falling off the end of a
        # function's generated code was always wrong, but it became a
        # real safety issue once functions could call each other (see
        # codegen.py's FUNCTIONS section) -- control falling through
        # with no `ret` executed corrupts the calling function's own
        # stack, not just the callee's exit code.
        if not always_returns(fn.body):
            raise SemanticError(
                f"Function '{fn.name}' (declared to return {return_type}) "
                f"does not return a value on all code paths"
            )

    # -- scope stack ------------------------------------------------------

    def _push_scope(self) -> None:
        self.scopes.append({})

    def _pop_scope(self) -> None:
        self.scopes.pop()

    def _declare(self, name: str, type_: Type) -> None:
        """Adds `name` to the *current* (innermost) scope. Only checks
        that scope for a collision -- a name already declared in an
        enclosing scope is fine to shadow, it's only a re-declaration
        error if it collides with something in this same block."""
        if name in self.scopes[-1]:
            raise SemanticError(f"Variable '{name}' is already declared in this scope")
        self.scopes[-1][name] = type_

    def _lookup(self, name: str) -> Type:
        """Resolves `name` by walking outward from the innermost scope
        to the outermost, returning the type from the first (nearest
        enclosing) match."""
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        raise SemanticError(f"Reference to undeclared variable '{name}'")

    # -- statements ---------------------------------------------------

    def analyze_statement(self, stmt: Node, return_type: Type) -> None:
        if isinstance(stmt, VarDecl):
            self.analyze_var_decl(stmt)
        elif isinstance(stmt, Assign):
            self.analyze_assign(stmt)
        elif isinstance(stmt, IndexAssign):
            self.analyze_index_assign(stmt)
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
            self.check_expr(stmt.expr)  # evaluated for validity; result unused
        else:
            raise SemanticError(f"No semantic rule for statement: {stmt!r}")

    def analyze_var_decl(self, stmt: VarDecl) -> None:
        declared_type = type_from_name(stmt.var_type)
        if stmt.init is not None:
            # Checked before `stmt.name` is added to scope below, so a
            # self-referential initializer (`int a = a`) correctly fails
            # as "undeclared variable" rather than reading itself.
            init_type = self.check_expr(stmt.init)
            if init_type != declared_type:
                raise SemanticError(
                    f"Cannot initialize '{stmt.name}' (declared {declared_type}) "
                    f"with a value of type {init_type}"
                )
        self._declare(stmt.name, declared_type)

    def analyze_assign(self, stmt: Assign) -> None:
        declared_type = self._lookup(stmt.name)  # may resolve to an enclosing scope
        value_type = self.check_expr(stmt.value)
        if value_type != declared_type:
            raise SemanticError(
                f"Cannot assign a value of type {value_type} to '{stmt.name}' "
                f"(declared {declared_type})"
            )

    def analyze_index_assign(self, stmt: IndexAssign) -> None:
        element_type = self._check_indexable_and_index(stmt.array, stmt.index)
        value_type = self.check_expr(stmt.value)
        if value_type != element_type:
            raise SemanticError(
                f"Cannot assign a value of type {value_type} to an array "
                f"element of type {element_type}"
            )

    def _check_indexable_and_index(self, base_expr: Node, index_expr: Node) -> Type:
        """Shared by check_index (reading `base[index]`) and
        analyze_index_assign (writing `base[index] = value`):
        validates that `base_expr` is actually indexable -- array- or
        slice-typed, see below -- and `index_expr` is int-typed,
        returning the element type: what a successful `[index]`
        operation on it would read or write. Recurses correctly for
        multi-dimensional access for free: for `matrix[i][j]`, the
        outer call's `base_expr` is itself an Index node
        (`matrix[i]`), so checking IT via check_expr recursively runs
        this same method again, returning the row's element type (e.g.
        int, if matrix's rows are [3]int) -- which is exactly the type
        this outer call then needs `base_expr` to have.

        Named for what it actually accepts, not just for arrays
        specifically: `s[i]`, where `s` is Slice-typed, uses this exact
        same check (and was the reason for the rename from this
        method's original _check_array_and_index) -- indexing into a
        slice works identically to indexing into an array from
        semantic.py's point of view, the only difference being where
        codegen eventually finds the address to read from.
        """
        base_type = self.check_expr(base_expr)
        if base_type.kind not in (TypeKind.ARRAY, TypeKind.SLICE):
            raise SemanticError(
                f"Cannot index into a value of type {base_type} -- "
                f"only arrays and slices support indexing"
            )
        index_type = self.check_expr(index_expr)
        if index_type != Type.INT:
            raise SemanticError(f"Index must be int, got {index_type}")
        return base_type.element_type

    def check_slice(self, expr: Slice) -> Type:
        """`array[low:high]`. `array` must be indexable -- array- or
        slice-typed, exactly the same acceptance _check_indexable_and_
        index already uses for ordinary indexing, since slicing a
        slice (`s2 = s[1:3]`) and slicing the outer dimension of a
        multi-dimensional array (`matrix[0:2]`, yielding a slice of
        ROWS, type [][3]int) are both valid. Either bound, if present,
        must be int; an omitted bound (low=None or high=None, see
        Slice's own docstring in parser.py) needs no check at all
        here, since its default value is resolved later, at codegen
        time, not something semantic.py fills in or validates.

        The result is ALWAYS Type(SLICE, element_type=...) regardless
        of what's being sliced -- a slice expression's own type never
        depends on its bounds, only on the element type of whatever's
        being sliced, matching how check_index's own result never
        depends on WHICH index was used."""
        base_type = self.check_expr(expr.array)
        if base_type.kind not in (TypeKind.ARRAY, TypeKind.SLICE):
            raise SemanticError(
                f"Cannot slice a value of type {base_type} -- only "
                f"arrays and slices support slicing"
            )
        if expr.low is not None:
            low_type = self.check_expr(expr.low)
            if low_type != Type.INT:
                raise SemanticError(f"Slice low bound must be int, got {low_type}")
        if expr.high is not None:
            high_type = self.check_expr(expr.high)
            if high_type != Type.INT:
                raise SemanticError(f"Slice high bound must be int, got {high_type}")
        return Type(TypeKind.SLICE, element_type=base_type.element_type)

    def analyze_return(self, stmt: Return, return_type: Type) -> None:
        value_type = self.check_expr(stmt.value)
        if value_type != return_type:
            raise SemanticError(
                f"Function is declared to return {return_type}, but this "
                f"'return' statement returns {value_type}"
            )

    def analyze_if(self, stmt: If, return_type: Type) -> None:
        condition_type = self.check_expr(stmt.condition)
        if condition_type != Type.BOOL:
            raise SemanticError(
                f"'if' condition must be bool, got {condition_type} "
                f"(no implicit int-to-bool conversion -- try `x != 0` "
                f"instead of `x`)"
            )

        self._push_scope()
        for s in stmt.then_body:
            self.analyze_statement(s, return_type)
        self._pop_scope()

        # then/else get independent scopes -- see module docstring --
        # so a name declared in one is never visible in the other. When
        # else_body came from an elif, it's a single nested If (see
        # parser.py's If docstring); analyze_if just recurses into it
        # like any other statement, so the elif gets its own condition
        # check and its own then/else scopes automatically.
        if stmt.else_body is not None:
            self._push_scope()
            for s in stmt.else_body:
                self.analyze_statement(s, return_type)
            self._pop_scope()

    def analyze_while(self, stmt: While, return_type: Type) -> None:
        condition_type = self.check_expr(stmt.condition)
        if condition_type != Type.BOOL:
            raise SemanticError(
                f"'while' condition must be bool, got {condition_type} "
                f"(no implicit int-to-bool conversion -- try `x != 0` "
                f"instead of `x`)"
            )

        # loop_depth (not the scope stack) is what break/continue check
        # against -- see analyze_break/analyze_continue. It has to be a
        # counter rather than a boolean so nested while loops work: the
        # inner loop's own push/pop shouldn't make an outer loop's
        # break/continue look invalid once the inner one's body is done
        # being analyzed.
        self.loop_depth += 1
        self._push_scope()
        for s in stmt.body:
            self.analyze_statement(s, return_type)
        self._pop_scope()
        self.loop_depth -= 1

    def analyze_break(self, stmt: Break) -> None:
        if self.loop_depth == 0:
            raise SemanticError("'break' outside of a loop")

    def analyze_continue(self, stmt: Continue) -> None:
        if self.loop_depth == 0:
            raise SemanticError("'continue' outside of a loop")

    # -- expressions ----------------------------------------------------
    # Every check_* method both validates its node and returns its Type,
    # so callers (including other check_* methods, for operands) get
    # both in one call rather than needing a separate inference pass.

    def check_expr(self, expr: Node) -> Type:
        """Type-checks `expr` and, as a side effect, annotates it with
        the result (expr.resolved_type = result) before returning.
        This is the ONE place that annotation happens -- every check_*
        method below stays a pure type-computation function with no
        knowledge of the annotation step, and every recursive call for
        an operand or argument already goes through check_expr (see
        check_binary/check_unary/check_call), so every expression node
        anywhere in the tree gets annotated automatically, no matter
        how deeply nested, with no risk of a new node type being added
        later and someone forgetting to wire up the annotation for it.
        See the module docstring's TYPES section for why this replaced
        codegen.py's old, independently-duplicated _infer_type.

        Stores the actual Type object here, not str(result) -- that
        changed when array types were added, since a bare name string
        ('int'/'bool'/'str') can no longer represent everything a type
        might be (an array also needs its element type and size).
        codegen.py imports Type from this module directly and compares
        against it (Type.STR, Type.INT, ...) rather than string
        literals, and can freely inspect .kind/.element_type/.size on
        whatever it reads back."""
        if isinstance(expr, Constant):
            result = self.check_constant(expr)
        elif isinstance(expr, BoolLiteral):
            result = Type.BOOL
        elif isinstance(expr, StringLiteral):
            result = Type.STR
        elif isinstance(expr, Variable):
            result = self.check_variable(expr)
        elif isinstance(expr, ArrayLiteral):
            result = self.check_array_literal(expr)
        elif isinstance(expr, Index):
            result = self.check_index(expr)
        elif isinstance(expr, Slice):
            result = self.check_slice(expr)
        elif isinstance(expr, Call):
            result = self.check_call(expr)
        elif isinstance(expr, Unary):
            result = self.check_unary(expr)
        elif isinstance(expr, Binary):
            result = self.check_binary(expr)
        else:
            raise SemanticError(f"No semantic rule for expression: {expr!r}")
        expr.resolved_type = result
        return result

    def check_array_literal(self, expr: ArrayLiteral) -> Type:
        """`[e1, e2, ...]`. Every element must be the same type -- this
        language doesn't support heterogeneous arrays -- checked by
        type-checking each element (via check_expr, so a nested
        ArrayLiteral for a multi-dimensional literal is handled by
        plain recursion, no special-casing needed) and comparing every
        element's type to the first one's.

        A "ragged" literal like `[[1,2,3],[4,5]]` is rejected by this
        same check, with no extra logic needed: the two rows' types
        are [3]int and [2]int, which -- now that Type is structurally
        comparable -- are simply different types, exactly like [3]int
        and [3]bool would be.
        """
        if len(expr.elements) == 0:
            raise SemanticError("Array literals must have at least one element")
        element_types = [self.check_expr(e) for e in expr.elements]
        first = element_types[0]
        for i, t in enumerate(element_types[1:], start=2):
            if t != first:
                raise SemanticError(
                    f"Array literal elements must all be the same type -- "
                    f"element 1 is {first}, element {i} is {t}"
                )
        return Type(TypeKind.ARRAY, element_type=first, size=len(expr.elements))

    def check_index(self, expr: Index) -> Type:
        return self._check_indexable_and_index(expr.array, expr.index)

    def check_call(self, expr: Call) -> Type:
        if expr.name == 'print':
            return self.check_print_call(expr)
        if expr.name not in self.functions:
            raise SemanticError(f"Call to undeclared function '{expr.name}'")
        param_types, return_type = self.functions[expr.name]

        if len(expr.args) != len(param_types):
            raise SemanticError(
                f"Function '{expr.name}' expects {len(param_types)} "
                f"argument(s), got {len(expr.args)}"
            )
        for i, (arg, expected_type) in enumerate(zip(expr.args, param_types), start=1):
            actual_type = self.check_expr(arg)
            if actual_type != expected_type:
                raise SemanticError(
                    f"Argument {i} to '{expr.name}' should be "
                    f"{expected_type}, got {actual_type}"
                )
        return return_type

    def check_print_call(self, expr: Call) -> Type:
        """`print` takes exactly one argument, of *any* scalar type --
        unlike an ordinary function it isn't tied to one fixed
        parameter type, since int/bool/str are all printable and
        there's no reason to force a caller to pick a differently-named
        builtin per type. Arrays and slices are explicitly excluded:
        there's no defined formatting for either (nothing in codegen.py
        knows how to print one), so this rejects them here with a
        clear error rather than letting one through to type-check fine
        and then hit an unhandled case in codegen. Always "returns" int
        (see the module docstring's BUILTINS section for why 0,
        specifically) -- Hornet has no void type, and this keeps
        `print(x)` usable as an ordinary expression statement via the
        same ExprStmt path every other call already goes through, with
        nothing print-specific needed there."""
        if len(expr.args) != 1:
            raise SemanticError(
                f"'print' expects exactly 1 argument, got {len(expr.args)}"
            )
        arg_type = self.check_expr(expr.args[0])
        if arg_type.kind in (TypeKind.ARRAY, TypeKind.SLICE):
            raise SemanticError(f"'print' does not support array or slice arguments (got {arg_type})")
        return Type.INT

    def check_constant(self, expr: Constant) -> Type:
        if isinstance(expr.value, float) and not expr.value.is_integer():
            raise SemanticError(
                f"'{expr.value}' is not a whole number -- this language has "
                f"no floating-point type; only int and bool exist"
            )
        return Type.INT

    def check_variable(self, expr: Variable) -> Type:
        return self._lookup(expr.name)

    def check_unary(self, expr: Unary) -> Type:
        operand_type = self.check_expr(expr.operand)
        if expr.op in (UnaryOp.NEGATE, UnaryOp.COMPLEMENT):
            if operand_type != Type.INT:
                raise SemanticError(
                    f"'{expr.op.symbol()}' requires an int operand, got {operand_type}"
                )
            return Type.INT
        if expr.op == UnaryOp.NOT:
            if operand_type != Type.BOOL:
                raise SemanticError(
                    f"'not' requires a bool operand, got {operand_type} "
                    f"(no implicit int-to-bool conversion -- try "
                    f"`not (x == 0)` instead of `not x`)"
                )
            return Type.BOOL
        raise SemanticError(f"No semantic rule for unary operator: {expr.op}")

    def check_binary(self, expr: Binary) -> Type:
        left_type = self.check_expr(expr.left)
        right_type = self.check_expr(expr.right)
        op = expr.op

        if op == BinaryOp.ADD:
            # Overloaded: int+int is arithmetic addition, str+str is
            # concatenation. Anything else -- mixing the two, or trying
            # to add a bool -- is a type error. This has to be checked
            # explicitly here rather than via _require_type, since
            # there's no single "the" expected type to require.
            if left_type == Type.INT and right_type == Type.INT:
                return Type.INT
            if left_type == Type.STR and right_type == Type.STR:
                return Type.STR
            raise SemanticError(
                f"'+' requires two int operands or two str operands, "
                f"got {left_type} and {right_type}"
            )

        if op in _INT_ONLY_BINARY_OPS:
            self._require_type(left_type, Type.INT, op)
            self._require_type(right_type, Type.INT, op)
            return Type.INT

        if op in _ORDERING_OPS:
            self._require_type(left_type, Type.INT, op)
            self._require_type(right_type, Type.INT, op)
            return Type.BOOL

        if op in _EQUALITY_OPS:
            # Array and slice operands are rejected outright, even when
            # both sides are the exact same type -- codegen.py has no
            # element-wise array-comparison logic (unlike str, which
            # gets a real strcmp-backed comparison), so without this
            # check a same-shaped comparison would type-check fine
            # here and then hit an unhandled case in codegen. Slice
            # equality in particular isn't even well-defined yet
            # without array equality existing first (would `s1 == s2`
            # compare elements, or the underlying pointer/length pair
            # the way Go's own `==` restriction on slices hints at?) --
            # both are real, well-defined features to consider later,
            # just not implemented yet.
            if left_type.kind in (TypeKind.ARRAY, TypeKind.SLICE) or right_type.kind in (TypeKind.ARRAY, TypeKind.SLICE):
                raise SemanticError(f"'{op.symbol()}' does not support array or slice operands")
            if left_type != right_type:
                raise SemanticError(
                    f"Cannot compare {left_type} to {right_type} with "
                    f"'{op.symbol()}' -- both sides must be the same type"
                )
            return Type.BOOL

        if op in _LOGICAL_OPS:
            self._require_type(left_type, Type.BOOL, op)
            self._require_type(right_type, Type.BOOL, op)
            return Type.BOOL

        raise SemanticError(f"No semantic rule for binary operator: {op}")

    def _require_type(self, actual: Type, expected: Type, op) -> None:
        if actual != expected:
            raise SemanticError(
                f"'{op.symbol()}' requires {expected} operands, got {actual}"
            )


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
