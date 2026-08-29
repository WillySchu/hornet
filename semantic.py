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
full, current set -- `str`, fixed-size arrays, slices, and structs
have all been added since, each with their own typing rules documented
at their own check_* method rather than repeated here). This is a
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

FUNCTIONS WITH NO DECLARED RETURN TYPE
-------------------------------------------
`def NAME(params):` -- the type before the name omitted entirely, not
a `void`/`none` keyword -- means this function has no declared return
type at all (Function.return_type is None, see its own docstring in
parser.py). This is deliberately NOT a new user-facing type: there is
no keyword for it, no way to declare a variable/parameter/array or
slice element of it, and no plan to add one. Internally, though, it's
given a real singleton, Type.VOID (see Type's own docstring), rather
than reusing Python's own None for "this expression's type is void" --
resolved_type's OWN None already means "not yet type-checked"
everywhere it's used (see check_expr and codegen.py's own _type_of),
and conflating the two would make a legitimately void expression
indistinguishable from one semantic analysis simply hadn't reached
yet.

Two syntactic consequences follow directly, both documented on their
own AST nodes in parser.py: such a function's body may fall off the
end with no explicit `return` anywhere (Return.value can be None, a
bare `return`, valid exactly when the enclosing function's return_type
is Type.VOID -- see analyze_return), and always_returns is skipped
entirely for it (see the ALL PATHS RETURN section above) -- there's
nothing to guarantee returns, since falling off the end IS how such a
function is expected to exit when it doesn't return early.

Everywhere else Type.VOID might try to flow as a real value -- a
VarDecl initializer, an Assign, a function-call argument, an array/
slice base -- is already rejected for free by the exact same type-
mismatch check that site already had: none of those ever compares
against a type a user could actually write as "void", so Type.VOID
simply never matches. check_print_call and check_binary's equality
handling are the two exceptions that needed an explicit check apiece:
print's argument-type check previously accepted anything unconditionally
(added when arrays/slices became printable, before this feature
existed), and `Type.VOID == Type.VOID` is trivially true by structural
equality alone, the same way any type equals itself -- comparing two
"nothing"s to each other would otherwise silently type-check fine.

print itself is Type.VOID now -- its own docstring used to say
outright that it returned a hardcoded, meaningless int 0 specifically
*because* there was no real void type to give it; see check_print_call.

NONE
-----
`none` -- Hornet's nil-style zero value, analogous to Go's own `nil`,
but deliberately narrower internally: Go's own nil has no fixed type
of its own at all, adapting to whatever nilable type context expects
it via Go's general untyped-constant mechanism (the same one numeric
literals use there). Hornet has no untyped-constant mechanism for ANY
literal yet, so building one just for `none` would be a much bigger
structural change than adding a value -- every expression's type is
currently derivable purely from itself and its children, with no
context needed, and an untyped node would break that invariant
everywhere check_expr is called. See NoneLiteral's own docstring in
parser.py for the full reasoning.

Instead, `none` resolves to one single, fixed, internal type, Type.NONE
(a fifth singleton alongside INT/BOOL/STR/VOID -- see Type.NONE's own
docstring), and _types_compatible checks COMPATIBILITY -- not equality
-- specifically wherever a value flows into a slice-typed context: a
VarDecl initializer, an Assign, an IndexAssign, a function-call
argument, or a return value all go through it now instead of a bare
`!=` comparison. From the outside this behaves like Go's own nil for
everything usable today (`[]int s = none`, `if s == none`); only the
internal mechanism is narrower. Only slices are nilable so far -- none
is NOT compatible with int/bool/str/array, even though str is also a
pointer under the hood at the machine level. Extending this to other
composite/reference types, if any come along later, is real, separable
follow-up work, not implemented here.

Equality (`==`/`!=`) has no such fixed "target" side the way an
assignment does -- either operand could be the none one -- so
check_binary checks for a slice-vs-none pair directly rather than
going through _types_compatible, before falling through to its
existing array/slice/void rejection (which NONE now also joins, for
the same reason VOID is there: `none == none` would otherwise
trivially type-check, comparing two "nothing"s to each other the same
way two void results would).

codegen.py is where `none`'s actual runtime representation (a {ptr: 0,
len: 0} slice descriptor -- Go's own nil slice shape) and the ptr-only
comparison that implements `s == none` correctly (matching Go's own
nil-vs-empty-slice distinction) both live -- see its own NONE: THE
SLICE ZERO VALUE section.

BUILTINS
---------
`print`, `len`, and `append` are builtins -- callables that aren't
ordinary user-defined functions and don't go through self.functions at
all. check_call special-cases each of them before ever consulting
self.functions, and analyze()'s signature-collection pass rejects any
user function whose name collides with a builtin
(_BUILTIN_FUNCTION_NAMES), so there's no ambiguity about which one
wins -- a program simply can't define its own `print`, `len`, or
`append`.

check_print_call accepts exactly one argument of any REAL type (int,
bool, str, array, and slice are all printable, and there's no reason to
force a caller to pick a differently-named builtin per type the way an
ordinary function's fixed parameter types would require) and is itself
Type.VOID -- print's first real user, now that a real (if internal-only)
void type exists at all; see the FUNCTIONS WITH NO DECLARED RETURN TYPE
section below. codegen.py is where print's actual behavior (which
underlying libc call per argument type) lives -- see its own PRINTING
ARRAYS AND SLICES section for the array/slice case specifically.

check_len_call is print's near-opposite in shape: where print accepts
almost every type and carves out VOID/NONE as the only exceptions,
len accepts almost nothing -- only array or slice -- with str explicitly
rejected by its own, specific "not supported yet" message (a real,
separable follow-up, not an oversight) rather than folded into the
same generic rejection every other wrong type gets. Always returns
Type.INT, a real, useful value unlike print's VOID -- `len(x)` works
as an ordinary expression (a loop bound, an operand, ...), not just a
bare statement. codegen.py's gen_len_call_into is where the actual
array-vs-slice split lives (a compile-time constant vs. a runtime
descriptor read); see its own docstring for why the argument itself
is still fully evaluated either way, regardless of whether the
resulting length ends up depending on its runtime value at all.

check_append_call requires a slice as its first argument and a value
matching that slice's own element type as its second, always returning
the same slice type back (append never changes what a slice is a slice
OF, only how many elements are in it). The value flows into the
element type via _check_value_flowing_into, not a plain check_expr --
the same recursive treatment analyze_var_decl/analyze_assign/
analyze_index_assign already give a value flowing into an already-
typed slot, so `append(rows, [5, 6])` on a slice-of-slices correctly
constructs a fresh, nested slice for the new element. Unlike len,
check_append_call doesn't restrict what KIND of expression the first
argument is -- that's a codegen-level restriction (see codegen.py's
gen_append_call_into for why it's deliberately narrower there than
len's own), not a type-checking one. codegen.py's own APPEND BUILTIN
section covers the actual growth-and-aliasing mechanics, which
semantic.py has no need to know anything about.

TYPES: ANNOTATING THE AST FOR codegen.py
-------------------------------------------
check_expr does one thing beyond type-checking: after computing an
expression's Type, it stores that result on the node itself
(expr.resolved_type = result) before returning. Every check_* method
below stays a pure type-computation function with no knowledge of
this -- the annotation happens in exactly one place, check_expr's own
dispatch, and every recursive call for an operand, argument, or
condition anywhere in this file already goes through check_expr rather
than some check_* method directly (see analyze_var_decl, check_binary,
check_unary, check_call, and every other caller). That means every
expression node anywhere in a program, no matter how deeply nested,
ends up annotated automatically, with no separate wiring needed and no
per-node-type case to remember adding as this pass grows.

resolved_type holds the actual Type object directly (Type.INT, an
ARRAY-kind Type with its own element_type/size, Type.VOID, ...), not a
name string -- a bare name like 'int'/'bool'/'str' stopped being able to
represent everything a type might be once arrays existed (an array also
needs its element type and size, which no string alone can carry).
codegen.py imports Type from this module directly and compares against
it (Type.STR, Type.INT, Type.VOID, ...) rather than string literals, and
can freely inspect .kind/.element_type/.size on whatever comes back.

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
applies to every function with a REAL declared return type -- and it's
not just a correctness nicety. Once functions could call each other
(see codegen.py's FUNCTIONS section), a function whose generated code
falls through to whatever comes after it with no `ret` ever executed
doesn't just return garbage to its caller -- it corrupts the *calling*
function's own stack, since there's a real return address sitting on
the stack from the `call` that invoked it, with nothing left to pop it
and jump there.

A function with NO declared return type (return_type is Type.VOID) is
the one deliberate exception, skipped entirely rather than run and
ignored -- see the FUNCTIONS WITH NO DECLARED RETURN TYPE section below
for why falling off such a function's end is exactly how it's expected
to exit, and how codegen.py still guarantees a real `ret` executes
either way.

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
    Field,
    FieldAssign,
    Function,
    If,
    Index,
    IndexAssign,
    Node,
    NoneLiteral,
    Param,
    Parser,
    Program,
    Return,
    Slice,
    SliceTypeExpr,
    StringLiteral,
    StructDef,
    StructField,
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
    STRUCT = auto()
    VOID = auto()  # see Type.VOID's own docstring below -- purely internal
    NONE = auto()  # see Type.NONE's own docstring below -- user-writable
                   # (via the `none` literal), but never as a DECLARED type


@dataclass(frozen=True)
class Type:
    """A type in this language: one of the three scalars (kind alone,
    element_type/size/struct_name all None), an array (kind=ARRAY,
    element_type the Type one level down, size that dimension's fixed
    length), a slice (kind=SLICE, element_type the Type one level
    down, size always None -- a slice's LENGTH is a runtime property
    of the slice VALUE, not part of its type the way an array's size
    is; see SliceTypeExpr's own docstring in parser.py), or a struct
    (kind=STRUCT, struct_name the struct's own declared name, element_
    type/size both None -- a struct's own field LAYOUT lives in the
    struct registry, keyed by this same name, not duplicated onto
    every Type instance that refers to it). Two slices of the same
    element type are the same Type regardless of how long either one
    happens to be at runtime, unlike two arrays of different sizes,
    which are different types even with the same element type.

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

    This is ALSO exactly what gives struct types NOMINAL equality (two
    structs with identical field lists but different declared names
    are different types) essentially for free, rather than needing a
    separate mechanism: struct_name is just one more field this same
    structural-equality machinery already compares, so `Type(STRUCT,
    struct_name='Point') == Type(STRUCT, struct_name='Point')` is True
    (same name, same type) and `!= Type(STRUCT, struct_name='Vector')`
    is True (different name, different type, regardless of whether
    Point and Vector happen to declare the exact same fields) with no
    field-by-field comparison ever entering into it at all.
    """
    kind: TypeKind
    element_type: Optional['Type'] = None  # set when kind == ARRAY or SLICE
    size: Optional[int] = None             # only set when kind == ARRAY
    struct_name: Optional[str] = None      # only set when kind == STRUCT

    def __str__(self) -> str:
        if self.kind == TypeKind.ARRAY:
            return f"[{self.size}]{self.element_type}"
        if self.kind == TypeKind.SLICE:
            return f"[]{self.element_type}"
        if self.kind == TypeKind.STRUCT:
            return self.struct_name
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
# A fourth singleton, kept deliberately OUT of _TYPE_NAMES below (and
# there's no lexer keyword for it either) -- Type.VOID can never be
# reached by parsing a type expression from source, only ever
# produced internally, as the "return type" of a function with no
# declared one (see analyze_function/analyze/check_call). This is
# purely a bookkeeping value, not a user-facing type: there's no way
# to declare a void-typed variable, parameter, or array/slice element,
# and there's no plan to add one -- see the module docstring's
# FUNCTIONS WITH NO DECLARED RETURN TYPE section for why a real
# (if internal-only) Type was still worth it here rather than reusing
# Python's own None for this: None already means "not yet type-
# checked" everywhere resolved_type is used, and conflating the two
# would make a legitimately void expression indistinguishable from
# one semantic analysis simply hadn't reached yet.
Type.VOID = Type(TypeKind.VOID)
# A fifth singleton, ALSO kept deliberately out of _TYPE_NAMES below --
# despite `none` being a real, user-writable keyword (unlike `void`,
# which has none), it's only ever reachable through parse_primary's own
# NoneLiteral production, never through parse_type. There is, and is
# meant to be, no way to write `none x` as a declaration -- `none` is
# a VALUE (Hornet's nil-style zero value for slices, analogous to Go's
# own nil -- see NoneLiteral's own docstring in parser.py), never a
# type annotation. Comparing against Type.NONE directly (rather than
# via _TYPE_NAMES/type_from_name, which stay reserved for real,
# declarable types) is check_expr's own NoneLiteral case's job.
Type.NONE = Type(TypeKind.NONE)


_TYPE_NAMES = {
    'int': Type.INT,
    'bool': Type.BOOL,
    'str': Type.STR,
}


@dataclass
class StructInfo:
    """Everything semantic analysis (and, via Program.struct_registry,
    codegen) needs to know about one declared struct: its own name
    (redundant with whatever key it's stored under in a registry dict,
    but kept here too so a StructInfo is self-describing on its own --
    useful in error messages and anywhere one gets passed around
    without its own dict key close at hand) and its fields, as an
    ordinary dict from field name to that field's own resolved Type.
    Field ORDER matters and is preserved here exactly as declared --
    a plain dict already does this (insertion order, since Python
    3.7), so no separate ordered-list structure is needed alongside
    it -- since it determines both codegen's own memory layout (fields
    are laid out at sequential byte offsets in declaration order) and
    print's own field-printing order."""
    name: str
    fields: Dict[str, Type]


def type_from_name(type_expr, structs: Dict[str, StructInfo]) -> Type:
    """Converts a parsed type expression (VarDecl.var_type /
    Function.return_type / Param.type / StructField.field_type,
    straight from parser.py) into a Type. `type_expr` is a plain str
    for a scalar type OR a struct name (see below), an ArrayTypeExpr
    for an array type, or a SliceTypeExpr for a slice type (see their
    own docstrings in parser.py) -- handled here by recursing on
    element_type, which is itself a plain str, another ArrayTypeExpr,
    or another SliceTypeExpr, naturally bottoming out at a scalar or
    struct name and handling arbitrarily-nested types (`[2][3]int`,
    `[][]int`, `[][3]int`, `[]MyStruct`, ...) with no depth limit or
    special-casing for "how many dimensions" or "which mix of array,
    slice, and struct".

    `structs` is this program's own struct registry (see StructInfo),
    already fully built by the time this is ever called with a
    struct-name type_expr -- see SemanticAnalyzer.analyze's own
    struct-collection pass, which runs before function signatures (and
    therefore before anything that might reference a struct type) are
    resolved at all. A REQUIRED parameter, not one defaulted to an
    empty dict: every call site in this file and in codegen.py was
    updated to pass its own analyzer's or function's struct registry
    through when struct support was added, specifically so a call site
    that got missed fails loudly (a TypeError for a missing argument)
    rather than silently misresolving any struct-typed declaration it
    happens to touch as "unknown type".

    Only ever fails for a program that isn't syntactically valid in
    the first place, OR references a type name that isn't a declared
    struct -- parse_type() already restricts a scalar type_expr to
    'int'/'bool'/'str'/an identifier, and already validates an
    ArrayTypeExpr's size is a positive whole number at parse time, so
    the only genuinely user-facing failure here is an unrecognized
    identifier; the dict lookups are otherwise a defensive check, not
    a user-facing validation path in their own right."""
    if isinstance(type_expr, ArrayTypeExpr):
        element = type_from_name(type_expr.element_type, structs)
        return Type(TypeKind.ARRAY, element_type=element, size=type_expr.size)
    if isinstance(type_expr, SliceTypeExpr):
        element = type_from_name(type_expr.element_type, structs)
        return Type(TypeKind.SLICE, element_type=element)
    if type_expr in _TYPE_NAMES:
        return _TYPE_NAMES[type_expr]
    if type_expr in structs:
        return Type(TypeKind.STRUCT, struct_name=type_expr)
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
# a set (not hardcoded string comparisons scattered around) so adding
# another builtin later is "add a name here plus its own check_*/gen_*
# pair", not a search-and-replace -- print and len were both added this
# way in turn, and append (the third) is what actually exercised that
# claim for the first time.
_BUILTIN_FUNCTION_NAMES = {'print', 'len', 'append'}


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
        self.structs: Dict[str, StructInfo] = {}  # name -> resolved fields; see _collect_structs

    def analyze(self, program: Program) -> None:
        # First pass, ahead of even function signatures: collect every
        # struct definition. Struct types can appear as a parameter or
        # return type, so they have to already be fully resolved before
        # function signatures are -- see _collect_structs's own
        # docstring for why THIS pass itself needs two separate sub-
        # passes internally (struct names first, then field types),
        # the same forward-reference reasoning one level up.
        self.structs = self._collect_structs(program.structs)
        program.struct_registry = self.structs  # stashed for codegen.py's own use

        # Second pass: collect every function's signature before
        # checking any function's body. This is what makes call order
        # not matter -- a function can call one defined later in the
        # file, or call itself recursively -- since by the time
        # analyze_function ever looks anything up in self.functions,
        # every signature is already there. See the module docstring's
        # FUNCTIONS section.
        self.functions = {}
        for fn in program.functions:
            if fn.name in _BUILTIN_FUNCTION_NAMES:
                raise SemanticError(
                    f"'{fn.name}' is a builtin and can't be redefined as "
                    f"a function"
                )
            if fn.name in self.structs:
                raise SemanticError(
                    f"Function '{fn.name}' collides with a struct of the "
                    f"same name -- struct and function names share one "
                    f"namespace and can never be the same, since "
                    f"'{fn.name}(...)' would otherwise be ambiguous "
                    f"between a call and a struct literal"
                )
            if fn.name in self.functions:
                raise SemanticError(f"Function '{fn.name}' is already declared")
            param_types = [type_from_name(p.type, self.structs) for p in fn.params]
            return_type = Type.VOID if fn.return_type is None else type_from_name(fn.return_type, self.structs)
            self.functions[fn.name] = (param_types, return_type)

        # Third pass: now check each function's own body.
        for fn in program.functions:
            self.analyze_function(fn)

    def _collect_structs(self, struct_defs: List[StructDef]) -> Dict[str, StructInfo]:
        """Builds this program's own struct registry (see StructInfo)
        in three separate sub-passes, each depending on the last one
        having already finished for every struct, not just whichever
        one happens to be up next in declaration order:

        1. Reserve every struct's own NAME up front (as a None
           placeholder in the registry dict), rejecting a duplicate
           name immediately. This is what makes a forward reference
           work (`struct A: B b` followed later by `struct B: ...`):
           by the time pass 2 resolves A's own field types, B's name
           is already a recognized key in the registry dict, even
           though B's own fields haven't been filled in yet -- and
           type_from_name's own struct-name check only ever needs
           NAME membership, never the associated value, so a None
           placeholder is exactly as good as a real StructInfo for
           that purpose at this point.
        2. Resolve each struct's own field types (via type_from_name,
           passing this same registry-in-progress), rejecting a
           duplicate field name within one struct, and replace that
           struct's own None placeholder with a real StructInfo. A
           field's own type can be anything, including a slice
           (directly, or through an array or nested struct) -- see
           codegen.py's own analyze_array_escapes, specifically
           field_slot_of and _contains_slice's own STRUCT case, for
           how a slice-typed field's own backing array gets the
           identical escape-analysis treatment array-of-slices and
           slice-of-slices elements already have. This phase used to
           reject a slice-typed field outright, as its own explicit
           pass 4 here, while that escape-analysis extension hadn't
           been built yet; now that it has, there's nothing left for
           this pass to guard against.
        3. Only once EVERY struct's own fields are fully resolved,
           check each one for a cycle (see _check_struct_contains) --
           cycle detection needs the real, resolved field types to
           walk, not just which names exist, so it has to be its own
           pass after 1 and 2 both fully finish, not interleaved with
           either.
        """
        registry: Dict[str, StructInfo] = {}
        for sd in struct_defs:
            if sd.name in _BUILTIN_FUNCTION_NAMES:
                raise SemanticError(
                    f"'{sd.name}' is a builtin and can't be used as a "
                    f"struct name"
                )
            if sd.name in registry:
                raise SemanticError(f"Struct '{sd.name}' is already declared")
            registry[sd.name] = None

        for sd in struct_defs:
            fields: Dict[str, Type] = {}
            for f in sd.fields:
                if f.name in fields:
                    raise SemanticError(
                        f"Field '{f.name}' is already declared in struct '{sd.name}'"
                    )
                fields[f.name] = type_from_name(f.field_type, registry)
            registry[sd.name] = StructInfo(name=sd.name, fields=fields)

        for sd in struct_defs:
            self._check_struct_contains(sd.name, registry, path=[])

        return registry

    def _check_struct_contains(self, name: str, registry: Dict[str, StructInfo], path: List[str]) -> None:
        """DFS over the struct-containment graph -- struct X has an
        edge to struct Y if X has a field whose type is Y, DIRECTLY or
        through any depth of array wrapping (`[5]Y`, `[2][3]Y`, ...),
        since an array embeds its element inline, N times over, so a
        struct containing an array of a struct that (directly or
        transitively) contains the FIRST struct is exactly as size-
        infinite as directly containing itself would be. A SLICE field
        (`[]Y`) deliberately does NOT count as an edge here: a slice's
        own backing storage is a separate, runtime-sized allocation,
        not embedded inline in the containing struct's own layout, so
        `struct A: []A elements` doesn't make A's own size depend on
        itself at all -- it's a real, genuinely supported pattern (a
        tree or linked structure built from slices), not merely
        tolerated: this method's own job is only ever "would this
        create a size-infinite cycle", and a slice field never can, by
        construction, regardless of whether slice-typed fields
        themselves are otherwise allowed.

        `path` is the chain of struct names visited to reach `name`,
        purely for a readable error message -- a real cycle stops this
        DFS from ever needing memoization against already-fully-
        explored, cycle-free structs the way a general-purpose cycle
        detector might for efficiency: struct counts are small enough
        that re-walking a shared, cycle-free dependency from multiple
        starting points costs nothing worth guarding against."""
        if name in path:
            cycle = ' -> '.join(path + [name])
            raise SemanticError(
                f"Struct '{name}' cannot contain itself, directly or "
                f"transitively: {cycle}"
            )
        info = registry[name]
        for field_type in info.fields.values():
            contained = self._directly_embedded_struct_name(field_type)
            if contained is not None:
                self._check_struct_contains(contained, registry, path + [name])

    @staticmethod
    def _directly_embedded_struct_name(field_type: Type) -> Optional[str]:
        """If `field_type` is a struct, or an array (at any nesting
        depth) OF a struct, returns that struct's own name -- see
        _check_struct_contains's own docstring for exactly why arrays
        count here and slices don't. Returns None for a scalar field,
        a slice-typed field (of anything), or an array of scalars."""
        while field_type.kind == TypeKind.ARRAY:
            field_type = field_type.element_type
        return field_type.struct_name if field_type.kind == TypeKind.STRUCT else None

    def analyze_function(self, fn: Function) -> None:
        self.scopes = [{}]  # fresh, single-level scope stack per function
        self.loop_depth = 0
        # Parameters act like already-declared locals from the body's
        # point of view -- _declare here also gets duplicate-parameter-
        # name checking for free (`def int f(int a, int a):` collides in
        # this same scope exactly like `int a` twice in a row would).
        for p in fn.params:
            self._declare(p.name, type_from_name(p.type, self.structs))
        return_type = Type.VOID if fn.return_type is None else type_from_name(fn.return_type, self.structs)
        for stmt in fn.body:
            self.analyze_statement(stmt, return_type)
        # Checked last, after every statement is individually known to
        # be well-typed -- see the module docstring's ALL PATHS RETURN
        # section. Every OTHER function needs this regardless of its
        # return type, since falling off the end of a function's
        # generated code was always wrong, and became a real safety
        # issue once functions could call each other (see codegen.py's
        # FUNCTIONS section) -- control falling through with no `ret`
        # executed corrupts the calling function's own stack, not just
        # the callee's exit code. A function with NO declared return
        # type is the one deliberate exception: falling off the end is
        # exactly how such a function is expected to exit when it
        # doesn't return early (see the module docstring's FUNCTIONS
        # WITH NO DECLARED RETURN TYPE section) -- codegen.py's own
        # gen_function still guarantees a real `ret` executes either
        # way, just via an unconditional trailing epilogue instead of
        # relying on every path having its own explicit one.
        if return_type != Type.VOID and not always_returns(fn.body):
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
        elif isinstance(stmt, FieldAssign):
            self.analyze_field_assign(stmt)
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

    def _types_compatible(self, value_type: Type, target_type: Type) -> bool:
        """True if a value of `value_type` can be used where
        `target_type` is expected -- ordinary type equality, OR the
        one exception this language allows: `none` (Type.NONE) is
        compatible with ANY slice type, representing that slice's
        zero/nil value (see NoneLiteral's own docstring in parser.py).
        Deliberately narrow, at least for now: none is compatible with
        a slice target and nothing else -- not int/bool/str/array,
        even though str is also a pointer under the hood at the
        machine level. Extending this to other composite/reference
        types, if any come along later, is real, separable follow-up
        work, not implemented here.

        Shared by every site a value flows into an already-typed slot
        with a clear "this is the expected type" side -- a VarDecl
        initializer, an Assign, an IndexAssign, a function-call
        argument, a return value -- so `none` becomes valid at all of
        them uniformly, with nothing to remember re-adding per site.
        Equality between two operands with no such fixed side (`==`/
        `!=`) is checked separately, directly in check_binary, since
        neither operand there is "the target" the other must match."""
        if value_type == target_type:
            return True
        return value_type == Type.NONE and target_type.kind == TypeKind.SLICE

    def _check_value_flowing_into(self, expr: Node, target_type: Type) -> Type:
        """Type-checks `expr` as a value flowing into an already-typed
        slot (target_type), returning its own type for the caller's
        own _types_compatible check afterward -- almost always just
        check_expr, with one exception: an UNTYPED array literal
        (isinstance(expr, ArrayLiteral) and expr.type_expr is None)
        flowing directly into a SLICE-typed target (`[]int s = [1, 2,
        3]`) is checked against target_type's own element type
        directly (via check_array_literal's own expected_element_type
        parameter) rather than purely inferring a type from its
        elements that would then need to separately match against
        target_type -- this is what makes the untyped form behave
        identically to the fully-typed `[]int s = []int[1, 2, 3]`,
        just inferring the element count and checking element types
        against the DECLARED element type instead of restating it.

        Returns target_type ITSELF in that one case (not the array
        type check_array_literal actually computed for the literal's
        own elements), so the caller's own _types_compatible check
        against target_type trivially succeeds, the same way it would
        for any other already-slice-typed value -- `[]int s = arr`
        (an ordinary, NAMED array, not a literal) is deliberately NOT
        given this same treatment: only this one, specific expression
        SHAPE is special-cased, not "any array-typed value is
        compatible with a slice target," so an actual array still has
        to be explicitly sliced (`arr[:]`) to become one.

        Bypasses check_expr's own generic dispatch (and does its own
        annotation, manually) for this one case, since check_expr has
        no way to receive an expected type at all; every other kind of
        value goes through check_expr completely unaffected. Shared by
        analyze_var_decl and analyze_assign -- the two places a value
        flows into an already-typed slot with a clear "this is the
        expected type" side (unlike `==`/`!=`, which has no such
        side -- see check_binary's own, separate none-vs-slice
        handling for why that case can't reuse this)."""
        if isinstance(expr, ArrayLiteral) and expr.type_expr is None and target_type.kind == TypeKind.SLICE:
            array_type = self.check_array_literal(expr, expected_element_type=target_type.element_type)
            expr.resolved_type = array_type
            return target_type
        return self.check_expr(expr)

    def _check_expr_allowing_struct_literal(self, expr: Node) -> Type:
        """check_expr, except a struct literal (isinstance(expr, Call)
        and expr.name in self.structs) is recognized and routed
        through check_struct_literal instead of falling into check_
        call's own rejection of it. Shared by every position that
        allows a struct literal to appear directly with no "already-
        typed slot" of its own to flow into: check_call's own
        argument-checking loop (a function-call argument, `foo(A(1,
        2))`), analyze_return (a return value, `return A(1, 2)`),
        check_struct_literal's own argument-checking loop (nested
        inside another struct literal, `A(B(1, 2), 3)`), and check_
        array_literal's fully-untyped inference branch (an array
        literal element, when no target element type exists yet to
        check against). See _check_value_flowing_into_allowing_
        struct_literal, its sibling just below, for the positions that
        ALSO need the untyped-array-literal-into-slice-target
        treatment on top of this same detection.

        Deliberately NOT used by analyze_index_assign/analyze_field_
        assign, which still call _check_value_flowing_into directly,
        unchanged -- a struct literal as an IndexAssign/FieldAssign
        value remains a separate, not-yet-covered follow-up; see
        check_struct_literal's own docstring for the full, current
        list of positions this covers."""
        if isinstance(expr, Call) and expr.name in self.structs:
            return self.check_struct_literal(expr)
        return self.check_expr(expr)

    def _check_value_flowing_into_allowing_struct_literal(self, expr: Node, target_type: Type) -> Type:
        """The _check_value_flowing_into counterpart to _check_expr_
        allowing_struct_literal just above, for positions that ALSO
        need the untyped-array-literal-into-slice-target treatment on
        top of struct-literal detection: analyze_var_decl/analyze_
        assign (a VarDecl initializer or Assign value) and check_
        array_literal's own typed and expected-element-type branches
        (an array literal's own element, when a target element type IS
        already known). Struct-literal detection is checked FIRST,
        before target_type is ever consulted -- exactly like every
        other call site -- since a struct literal's own type comes
        entirely from its own name, never from whatever it's flowing
        into; a mismatch against target_type is still caught
        afterward, by the caller's own ordinary _types_compatible
        check, exactly as if this had gone through plain check_expr."""
        if isinstance(expr, Call) and expr.name in self.structs:
            return self.check_struct_literal(expr)
        return self._check_value_flowing_into(expr, target_type)

    def analyze_var_decl(self, stmt: VarDecl) -> None:
        declared_type = type_from_name(stmt.var_type, self.structs)
        if stmt.init is not None:
            # Checked before `stmt.name` is added to scope below, so a
            # self-referential initializer (`int a = a`) correctly fails
            # as "undeclared variable" rather than reading itself.
            init_type = self._check_value_flowing_into_allowing_struct_literal(stmt.init, declared_type)
            if not self._types_compatible(init_type, declared_type):
                raise SemanticError(
                    f"Cannot initialize '{stmt.name}' (declared {declared_type}) "
                    f"with a value of type {init_type}"
                )
        self._declare(stmt.name, declared_type)

    def analyze_assign(self, stmt: Assign) -> None:
        declared_type = self._lookup(stmt.name)  # may resolve to an enclosing scope
        value_type = self._check_value_flowing_into_allowing_struct_literal(stmt.value, declared_type)
        if not self._types_compatible(value_type, declared_type):
            raise SemanticError(
                f"Cannot assign a value of type {value_type} to '{stmt.name}' "
                f"(declared {declared_type})"
            )

    def analyze_index_assign(self, stmt: IndexAssign) -> None:
        """`array[index] = value`. value flows into the indexed
        element's own type the same way any other value flows into an
        already-typed slot -- via _check_value_flowing_into, not a
        plain check_expr -- so an untyped array literal assigned
        directly into a SLICE-typed element (`rows[0] = [9, 9, 9]`,
        one element of an array OF slices) gets the same recursive
        slice-construction treatment analyze_var_decl/analyze_assign
        already give a VarDecl/Assign's own value (see that method's
        own docstring). Found as the same bug-class in a third
        location, not a hypothetical extension: `rows[0] =
        someNamedSlice` and the explicitly-typed `rows[0] =
        []int[9, 9, 9]` both already worked (their own values already
        carry a real SLICE type by the time they reach here), which is
        exactly what masked this gap until the untyped form specifically
        was tried."""
        element_type = self._check_indexable_and_index(stmt.array, stmt.index)
        value_type = self._check_value_flowing_into(stmt.value, element_type)
        if not self._types_compatible(value_type, element_type):
            raise SemanticError(
                f"Cannot assign a value of type {value_type} to an array "
                f"element of type {element_type}"
            )

    def analyze_field_assign(self, stmt: FieldAssign) -> None:
        """`base.name = value` -- mirrors analyze_index_assign exactly,
        one level over: value flows into the field's own declared type
        via _check_value_flowing_into, not a plain check_expr, so an
        untyped array literal (or slice literal) assigned directly
        into a slice-typed field gets the same recursive slice-
        construction treatment every other already-typed slot (a
        VarDecl, an Assign, an IndexAssign's own element) already
        gives one -- written the general way from the start, exactly
        like analyze_index_assign's own already was, rather than only
        handling the field types a given phase happened to support at
        the time."""
        field_type = self._check_struct_and_field(stmt.base, stmt.name)
        value_type = self._check_value_flowing_into(stmt.value, field_type)
        if not self._types_compatible(value_type, field_type):
            raise SemanticError(
                f"Cannot assign a value of type {value_type} to field "
                f"'{stmt.name}' of type {field_type}"
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
        """`return <expr>` or a bare `return` (stmt.value is None, see
        Return's own docstring in parser.py). A bare return is valid
        exactly when the enclosing function has no declared return
        type (return_type is Type.VOID) -- the reverse direction
        (returning a VALUE from such a function) is checked after
        type-checking that value, not before, so a genuine error
        inside the value expression itself is still reported rather
        than masked by the "this function can't return a value at
        all" rejection.

        A struct literal returned directly (`return A(1, 2)`) is
        checked via _check_expr_allowing_struct_literal, shared by
        every position that allows a struct literal to appear directly
        with no already-typed slot to flow into -- see check_struct_
        literal's own docstring for the full list of positions this
        now covers. An array literal returned directly (`return [1, 2,
        3]`) needs no equivalent special-casing at all: array literals
        were never restricted to begin with, so plain check_expr
        already handles one correctly -- this asymmetry (struct
        literals needing explicit detection, array literals not) is
        purely a consequence of struct literals being the ONLY literal
        kind check_call rejects outside a short, explicit allow-list;
        nothing here treats the two kinds of returned value
        differently on purpose."""
        if stmt.value is None:
            if return_type != Type.VOID:
                raise SemanticError(
                    f"Function is declared to return {return_type}, but "
                    f"this bare 'return' returns nothing"
                )
            return
        value_type = self._check_expr_allowing_struct_literal(stmt.value)
        if return_type == Type.VOID:
            raise SemanticError(
                f"Function has no declared return type and cannot "
                f"return a value (got {value_type}) -- use a bare "
                f"'return' instead"
            )
        if not self._types_compatible(value_type, return_type):
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
        elif isinstance(expr, Binary):
            result = self.check_binary(expr)
        else:
            raise SemanticError(f"No semantic rule for expression: {expr!r}")
        expr.resolved_type = result
        return result

    def check_array_literal(self, expr: ArrayLiteral, expected_element_type: Optional[Type] = None) -> Type:
        """`[e1, e2, ...]`, or the fully-typed `[N]TYPE[e1, e2, ...]`
        (expr.type_expr is not None -- see ArrayLiteral's own
        docstring in parser.py). Either way, every element must be the
        same type -- this language doesn't support heterogeneous
        arrays.

        UNTYPED form, no external context (type_expr is None,
        expected_element_type is None): the array's own type is
        INFERRED entirely from its elements -- checked by type-
        checking each one (via check_expr, so a nested ArrayLiteral
        for a multi-dimensional literal is handled by plain recursion,
        no special-casing needed) and comparing every element's type
        to the first one's. A "ragged" literal like `[[1,2,3],[4,5]]`
        is rejected by this same check, with no extra logic needed:
        the two rows' types are [3]int and [2]int, which -- now that
        Type is structurally comparable -- are simply different types,
        exactly like [3]int and [3]bool would be. Needs at least one
        element -- with nothing else to go on, an empty literal here
        gives no type information at all.

        TYPED form (expr.type_expr is not None): the declared type is
        resolved FIRST (via type_from_name), and used as the standard
        every element (and the literal's own size) is checked AGAINST,
        rather than inferred from them -- the exact same
        _types_compatible check used everywhere else a value flows
        into an already-typed slot (a VarDecl initializer, an
        Assign, ...), so `none` would be just as valid an element here
        as it is anywhere else a slice is expected.

        expected_element_type, when supplied (and type_expr is still
        None): used for an UNTYPED literal flowing directly into an
        already-slice-typed VarDecl/Assign value (`[]int s = [1, 2,
        3]` -- see analyze_var_decl/analyze_assign's own callers,
        which bypass check_expr's generic dispatch specifically to
        pass this through, since check_expr has no way to receive
        context at all). Checked the exact same way the explicitly-
        typed form is, just against a type supplied by the CALLER
        instead of restated in the literal itself.

        Either typed path (an explicit type_expr, or a supplied
        expected_element_type) allows -- and correctly handles -- zero
        elements, unlike the fully-untyped path above: `[]int[]` (see
        parser.py's own Slice-wrapping of a slice literal) or `[]int s
        = []` both have a real, externally-known type to report even
        with nothing to infer from, the same way parse_type() itself
        already allows a slice type with no length embedded in it at
        all -- only the ordinary, standalone array literal has size as
        part of its declared type (enforced at parse time, `parse_
        type`'s own ArrayTypeExpr validation), which is what makes
        zero genuinely uninformative only in that one, fully-untyped
        case.

        Both typed paths check each element via _check_value_flowing_
        into_allowing_struct_literal rather than a plain check_expr --
        not just an ordinary recursive call, but the SAME recursive-
        slice-construction treatment analyze_var_decl/analyze_assign
        already give a top-level value (see that helper's own sibling,
        _check_expr_allowing_struct_literal, for the struct-literal
        detection itself): an untyped ArrayLiteral element flowing into
        a SLICE-kind expected type (e.g. `[][]int rows = [][]int[[1,
        2], [3, 4]]` -- the OUTER literal's own element type is []int,
        a slice, so each INNER `[1, 2]`/`[3, 4]` needs this same
        treatment recursively) is what makes genuinely nested slice
        construction -- a slice of slices, arbitrarily deep -- fall
        out for free, rather than only ever working one level deep.
        A plain check_expr here would infer each inner literal as an
        ordinary ARRAY ([2]int), which would then correctly fail the
        _types_compatible check against the expected SLICE type --
        this was a real, found bug, not a hypothetical one: `[][2]int`
        (array-typed inner elements matching an array-typed expected
        element) happened to still work by coincidence, since ordinary
        type equality was all that case ever needed, which is exactly
        what masked the gap until a genuinely nested slice was tried.

        A STRUCT-typed element -- an ordinary struct value (a
        Variable, Field, Index, or struct-returning Call) or, as of
        this same fix, a struct LITERAL directly (`[Point(1,2),
        Point(3,4)]`) -- is checked the identical way as any other
        element in all three branches below (this method never needed
        its own special case for struct-typed elements at the
        semantic layer; check_expr/_check_value_flowing_into already
        handle a Variable/Field/Index/Call's own type correctly
        regardless of what kind it is). What DID need a real fix was
        codegen: gen_array_literal_into had no STRUCT-typed element
        case at all before this, so a struct-typed array element
        failed even for an ordinary struct VARIABLE, with no literal
        involved -- see its own docstring for the actual fix.
        """
        if expr.type_expr is not None:
            declared_type = type_from_name(expr.type_expr, self.structs)
            if len(expr.elements) != declared_type.size:
                raise SemanticError(
                    f"Array literal declares type {declared_type} (size "
                    f"{declared_type.size}), but has {len(expr.elements)} "
                    f"element(s)"
                )
            for i, element in enumerate(expr.elements, start=1):
                element_type = self._check_value_flowing_into_allowing_struct_literal(element, declared_type.element_type)
                if not self._types_compatible(element_type, declared_type.element_type):
                    raise SemanticError(
                        f"Array literal declares element type "
                        f"{declared_type.element_type}, but element {i} "
                        f"is {element_type}"
                    )
            return declared_type

        if expected_element_type is not None:
            for i, element in enumerate(expr.elements, start=1):
                element_type = self._check_value_flowing_into_allowing_struct_literal(element, expected_element_type)
                if not self._types_compatible(element_type, expected_element_type):
                    raise SemanticError(
                        f"Array literal's elements must all be "
                        f"{expected_element_type} (to match the declared "
                        f"slice type), but element {i} is {element_type}"
                    )
            return Type(TypeKind.ARRAY, element_type=expected_element_type, size=len(expr.elements))

        if len(expr.elements) == 0:
            raise SemanticError("Array literals must have at least one element")
        element_types = [self._check_expr_allowing_struct_literal(e) for e in expr.elements]
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

    def check_field(self, expr: Field) -> Type:
        return self._check_struct_and_field(expr.base, expr.name)

    def _check_struct_and_field(self, base_expr: Node, field_name: str) -> Type:
        """Shared by check_field (reading `base.name`) and
        analyze_field_assign (writing `base.name = value`), mirroring
        _check_indexable_and_index's own shared-helper shape one level
        over: check base_expr's own type is actually a struct, look up
        field_name in that struct's own registered field list, and
        return the field's own type -- or raise a clear error at
        whichever of the two things actually went wrong (base_expr
        isn't struct-typed at all, or it is but this particular struct
        has no field by this name)."""
        base_type = self.check_expr(base_expr)
        if base_type.kind != TypeKind.STRUCT:
            raise SemanticError(
                f"Cannot access field '{field_name}' on non-struct type {base_type}"
            )
        struct_info = self.structs[base_type.struct_name]
        if field_name not in struct_info.fields:
            raise SemanticError(
                f"Struct '{base_type.struct_name}' has no field '{field_name}'"
            )
        return struct_info.fields[field_name]

    def check_struct_literal(self, expr: Call) -> Type:
        """`Name(arg1, arg2, ...)` -- a struct literal, e.g. `A a =
        A(6, 'hello')` for `struct A: int x; str y`. Disambiguated from
        an ordinary function call purely by registry membership --
        `Name` is a struct, not a function -- with no dedicated parser
        syntax at all: `A(6, 'hello')` already parses as an ordinary
        Call node (see parser.py), exactly like any other call. This is
        never ambiguous: analyze()'s own struct/function collision
        check guarantees a name can never be both a struct and a
        function, so a Call's own name membership in self.structs vs.
        self.functions is a clean, mutually-exclusive dispatch.

        Positional and exhaustive: exactly one argument per field, in
        the struct's own declaration order -- no named arguments, no
        partial construction with an implicit zero value for an
        omitted field, matching this language's existing preference
        for explicit over implicit (e.g. no int-to-bool coercion
        anywhere else either).

        Each argument is checked via _check_expr_allowing_struct_
        literal -- the same shared helper every other call site below
        uses -- rather than a plain check_expr, so a NESTED struct
        literal argument (`A(B(1, 2), 3)`) recurses right back into
        THIS method. That's what makes arbitrarily deep nesting work
        with no depth limit and no extra bookkeeping: each level's own
        argument-checking loop is the same recursive call, terminating
        naturally once every argument bottoms out at an ordinary,
        non-struct-literal expression. (`none` flowing into a slice-
        typed field still works fine here, since that's an ordinary
        _types_compatible check with no recursive construction
        involved; an untyped array literal argument flowing into an
        array- or slice-typed field is a smaller, related gap left for
        now, unrelated to struct literals as such.)

        Only ever reached from analyze_var_decl/analyze_assign (via
        _check_value_flowing_into_allowing_struct_literal, for a
        VarDecl initializer or an Assign value), check_call's own
        argument-checking loop (a direct function-call argument,
        `foo(A(1, 2))`), analyze_return (a direct return value,
        `return A(1, 2)`), check_array_literal (an array literal's own
        element, `[A(1, 2), A(3, 4)]` -- see its own docstring for why
        this needed a genuine codegen fix too, not just this same
        semantic detection: gen_array_literal_into had no STRUCT-typed
        element case at all before this, so a struct-typed array
        element failed even for an ordinary struct VARIABLE, with no
        literal involved), and -- recursively -- its OWN argument-
        checking loop (a struct literal nested as an argument to
        another struct literal). Every one of those checks for this
        exact shape (isinstance(expr, Call) and expr.name in self.
        structs) BEFORE calling _check_value_flowing_into/check_expr
        at all -- via one of the two small shared helpers just above
        analyze_var_decl, not duplicated inline at each site anymore.
        Every OTHER place a Call can appear (an IndexAssign/FieldAssign
        value, a bare statement, or most other kinds of expressions --
        a Binary operand, a Field-access base, ...) still funnels
        through check_expr's ordinary dispatch into check_call instead,
        which rejects a struct-name Call there. That's the entire
        mechanism that keeps struct literals scoped to exactly these
        positions, deliberately narrower than where an ordinary
        function call is allowed to appear.

        Annotates expr.resolved_type directly (mirroring _check_value_
        flowing_into's own array-literal-into-slice special case, for
        the identical reason: this bypasses check_expr's generic
        dispatch -- and its own annotation step -- entirely, so
        nothing else would perform it)."""
        struct_info = self.structs[expr.name]
        field_items = list(struct_info.fields.items())
        if len(expr.args) != len(field_items):
            field_names = ', '.join(name for name, _ in field_items)
            raise SemanticError(
                f"Struct literal for '{expr.name}' expects "
                f"{len(field_items)} argument(s) (one per field, in "
                f"declaration order: {field_names}), got {len(expr.args)}"
            )
        for i, (arg, (field_name, field_type)) in enumerate(zip(expr.args, field_items), start=1):
            arg_type = self._check_expr_allowing_struct_literal(arg)
            if not self._types_compatible(arg_type, field_type):
                raise SemanticError(
                    f"Argument {i} to struct literal '{expr.name}' "
                    f"(field '{field_name}') should be {field_type}, "
                    f"got {arg_type}"
                )
        result = Type(TypeKind.STRUCT, struct_name=expr.name)
        expr.resolved_type = result
        return result

    def check_call(self, expr: Call) -> Type:
        if expr.name in self.structs:
            raise SemanticError(
                f"'{expr.name}(...)' is a struct literal, which is only "
                f"allowed as a variable's initializer, a plain "
                f"assignment's value, a direct function-call argument, "
                f"a direct return value, or an array literal's own "
                f"element -- not as an IndexAssign/FieldAssign value, a "
                f"bare statement, or most other kinds of expressions (a "
                f"Binary operand, a Field-access base, ...); assign it "
                f"to a variable first if you need it in one of those "
                f"positions"
            )
        if expr.name == 'print':
            return self.check_print_call(expr)
        if expr.name == 'len':
            return self.check_len_call(expr)
        if expr.name == 'append':
            return self.check_append_call(expr)
        if expr.name not in self.functions:
            raise SemanticError(f"Call to undeclared function '{expr.name}'")
        param_types, return_type = self.functions[expr.name]

        if len(expr.args) != len(param_types):
            raise SemanticError(
                f"Function '{expr.name}' expects {len(param_types)} "
                f"argument(s), got {len(expr.args)}"
            )
        for i, (arg, expected_type) in enumerate(zip(expr.args, param_types), start=1):
            # A struct literal used directly as an argument (`foo(A(1,
            # 2))`) is checked via _check_expr_allowing_struct_literal,
            # shared by every position that allows this shape with no
            # already-typed slot to flow into -- see check_struct_
            # literal's own docstring for the full, current list of
            # positions a struct literal is allowed to appear in
            # directly, and for why every position NOT on that list
            # still funnels through the ordinary check_expr ->
            # check_call dispatch above, which rejects a struct-name
            # Call there.
            actual_type = self._check_expr_allowing_struct_literal(arg)
            if not self._types_compatible(actual_type, expected_type):
                raise SemanticError(
                    f"Argument {i} to '{expr.name}' should be "
                    f"{expected_type}, got {actual_type}"
                )
        return return_type

    def check_print_call(self, expr: Call) -> Type:
        """`print` takes exactly one argument, of any REAL type --
        unlike an ordinary function it isn't tied to one fixed
        parameter type, since every type Hornet has is printable and
        there's no reason to force a caller to pick a differently-
        named builtin per type. "Real" excludes Type.VOID specifically
        -- the result of calling a function with no declared return
        type -- since there's nothing to format for a value that
        doesn't exist; see check_call for how that's already the
        result you'd get calling one of those. An array or slice
        argument is formatted as `TYPE[elem, elem, ...]` -- e.g.
        `[3]int[1, 2, 3]` or `[]int[1, 2, 3]` -- the type prefix
        appearing exactly once, at the outermost level, with no
        repetition for nested rows (see codegen.py's own
        _gen_print_collection); a str element is quoted inside a
        collection (`'alice'`) even though a bare str argument prints
        unquoted -- matching how most languages format a string
        differently in a collection than when printed on its own.

        print itself is Type.VOID -- Hornet's first, and so far only,
        builtin with no meaningful value to return. Nothing has to
        change in codegen.py for this: gen_print_call_into still
        leaves *something* in %eax at the end of every path (a
        harmless leftover from before print had anywhere real to
        return to), but nothing ever reads it anymore, the same way
        nothing reads any OTHER void call's leftover register value
        -- see the module docstring's FUNCTIONS WITH NO DECLARED
        RETURN TYPE section."""
        if len(expr.args) != 1:
            raise SemanticError(
                f"'print' expects exactly 1 argument, got {len(expr.args)}"
            )
        arg_type = self.check_expr(expr.args[0])
        if arg_type == Type.VOID:
            raise SemanticError(
                "'print' cannot be called with the result of a function "
                "that has no declared return type -- there's no value there to print"
            )
        if arg_type == Type.NONE:
            raise SemanticError(
                "'print' cannot be called with a bare 'none' -- store it "
                "in a slice-typed variable first (e.g. `[]int s = none`), "
                "then print that"
            )
        return Type.VOID

    def check_len_call(self, expr: Call) -> Type:
        """`len(x)`: x must be array- or slice-typed -- str isn't
        supported yet (see the module docstring's LEN BUILTIN section
        for why that's a real, separable follow-up rather than an
        oversight), and every other type (int, bool, void, none) is
        rejected by the same, single "must be array or slice" check,
        with no per-type carve-out needed the way print's own, much
        more permissive check needs several: len accepts almost
        nothing, where print accepts almost everything.

        x is still fully type-checked via check_expr regardless of
        whether codegen ends up needing its computed VALUE for
        anything (an array's own length is a compile-time constant,
        never actually read out of the argument at all -- see
        codegen.py's gen_len_call_into) -- so an invalid expression
        buried inside x (an undeclared variable, a type error) is
        still caught here exactly like it would be anywhere else.

        Always returns int -- unlike print, len has a real, useful
        value, so `len(x)` is usable as an ordinary expression (a loop
        bound, an operand, ...), not just a bare statement."""
        if len(expr.args) != 1:
            raise SemanticError(
                f"'len' expects exactly 1 argument, got {len(expr.args)}"
            )
        arg_type = self.check_expr(expr.args[0])
        if arg_type == Type.STR:
            raise SemanticError(
                "'len' does not support str arguments yet"
            )
        if arg_type.kind not in (TypeKind.ARRAY, TypeKind.SLICE):
            raise SemanticError(
                f"'len' requires an array or slice argument, got {arg_type}"
            )
        return Type.INT

    def check_append_call(self, expr: Call) -> Type:
        """`append(s, value)`, Hornet's third builtin -- Go-style:
        returns a NEW slice rather than mutating s in place (see
        codegen.py's own APPEND BUILTIN section for the full growth-
        and-aliasing story).

        s must be slice-typed; value must match its own element type,
        checked via _check_value_flowing_into_allowing_struct_literal
        rather than a plain check_expr -- the same recursive treatment
        analyze_var_decl/analyze_assign/analyze_index_assign already
        give a value flowing into an already-typed slot, so appending
        an untyped array literal into a slice-of-slices (`append(rows,
        [5, 6])`) correctly constructs a fresh, nested slice for the
        new element, exactly like assigning one directly already does
        -- and, as of the same fix that added struct literals as array
        elements, appending a struct literal directly into a slice of
        structs (`append(pts, Point(1, 2))`) is recognized the same
        way any other position allowing this shape already is, rather
        than falling through to check_call's own rejection of it.

        Always returns s's own slice type -- the NEW slice's type is
        identical to the one appended to, obviously, since append
        never changes what a slice is a slice OF, only how many
        elements are in it.

        Doesn't restrict what KIND of expression s itself is (a bare
        Variable, an Index, a re-slice, a whole slice literal, ...) --
        matching gen_append_call_into's own generality on the codegen
        side now (any slice-typed expression materializes into the
        shared unnamed-slice scratch slot if it isn't already a bare
        Variable or `none`), unlike print's and len's own argument-
        shape restrictions, which are real and still enforced only at
        the codegen layer for those two.
        """
        if len(expr.args) != 2:
            raise SemanticError(
                f"'append' expects exactly 2 arguments, got {len(expr.args)}"
            )
        slice_arg, value_arg = expr.args
        slice_type = self.check_expr(slice_arg)
        if slice_type.kind != TypeKind.SLICE:
            raise SemanticError(
                f"'append' requires a slice as its first argument, "
                f"got {slice_type}"
            )
        value_type = self._check_value_flowing_into_allowing_struct_literal(value_arg, slice_type.element_type)
        if not self._types_compatible(value_type, slice_type.element_type):
            raise SemanticError(
                f"'append' cannot append a value of type {value_type} "
                f"to a {slice_type} (element type "
                f"{slice_type.element_type})"
            )
        return slice_type

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
            # A slice compared to `none` (in EITHER order) is the one
            # exception to the array/slice/void/none rejection just
            # below -- `s == none` / `none == s`, checking whether a
            # slice is the nil/zero-value slice (see NoneLiteral's own
            # docstring in parser.py). Checked first, before the
            # general rejection, since it's the one case that's
            # actually meaningful and allowed to proceed rather than
            # be rejected by it.
            none_vs_slice = (
                (left_type == Type.NONE and right_type.kind == TypeKind.SLICE) or
                (right_type == Type.NONE and left_type.kind == TypeKind.SLICE)
            )
            if none_vs_slice:
                return Type.BOOL

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
            #
            # VOID is rejected for a completely different reason: it's
            # not a missing FEATURE, it's structurally nonsensical --
            # `foo() == bar()`, where neither foo nor bar has a
            # declared return type, would otherwise type-check fine
            # here (Type.VOID == Type.VOID is trivially true, same as
            # any other type compared to itself), comparing two
            # "nothing"s to each other. Every OTHER place a void call's
            # result might flow (VarDecl, Assign, a binary operand,
            # function argument, array/slice base) is already rejected
            # for free, just by never matching the real, user-declared
            # type each of those checks compares against -- this is the
            # one place two void operands could accidentally match each
            # other instead of a real type, so it needs its own,
            # explicit check.
            #
            # NONE is rejected here for the SAME reason as VOID
            # (`none == none` would otherwise trivially type-check,
            # comparing two "nothing"s the same way two void results
            # would), EXCEPT for the one case already handled above.
            # Every OTHER place `none` might flow (VarDecl, Assign,
            # function argument, return value) already allows it
            # specifically via _types_compatible, which has a real
            # target type to check none against -- equality has no
            # such fixed target on either side, so this needed its own
            # explicit exception rather than reusing that helper as-is.
            if left_type.kind in (TypeKind.ARRAY, TypeKind.SLICE, TypeKind.VOID, TypeKind.NONE) or right_type.kind in (TypeKind.ARRAY, TypeKind.SLICE, TypeKind.VOID, TypeKind.NONE):
                raise SemanticError(
                    f"'{op.symbol()}' does not support array, slice, void, "
                    f"or none operands, except comparing a slice to none"
                )
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
