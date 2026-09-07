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
collecting every error. Neither AST nodes nor this pass track source
positions (only Tokens do, transiently, during parsing), so messages
name the offending variable/operator/type as specifically as possible
without a line/column, the same limitation CodegenError has.
"""

import argparse
from dataclasses import dataclass
from enum import auto, Enum
from typing import Dict, List, Optional, Set, Tuple

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
    Cast,
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
    VOID = auto()  # see Type.VOID's own docstring below -- purely internal
    NONE = auto()  # see Type.NONE's own docstring below -- user-writable
                   # (via the `none` literal), but never as a DECLARED type


@dataclass(frozen=True)
class Type:
    """A type: one of the three-plus scalars (kind alone), an array
    (kind=ARRAY, element_type one level down, size that dimension's
    fixed length), a slice (kind=SLICE, element_type one level down,
    size always None -- a slice's length is a runtime property of the
    VALUE, not its type), or a struct (kind=STRUCT, struct_name the
    declared name, element_type/size both None -- field layout lives
    in the struct registry, not duplicated here).

    Frozen to get structural equality/hashing for free: `Type(ARRAY,
    Type.INT, 3) == Type(ARRAY, Type.INT, 3)` is correctly True for
    two separate objects, and `!= Type(ARRAY, Type.INT, 4)` is
    correctly True too, recursing to arbitrary depth since element_
    type is itself a Type -- no special-casing needed anywhere that
    already does `left_type != right_type` (check_binary, analyze_
    var_decl, check_call, ...). This is also what gives struct types
    NOMINAL equality essentially for free: struct_name is just one
    more field this same machinery compares, so two structs with
    identical fields but different names are correctly different
    types, with no field-by-field comparison involved.
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


def type_from_name(type_expr, structs: Dict[str, StructInfo], aliases: Dict[str, Type]) -> Type:
    """Converts a parsed type expression (VarDecl.var_type/Function.
    return_type/Param.type/StructField.field_type) into a Type.
    `type_expr` is a plain str (scalar, struct name, or alias name), an
    ArrayTypeExpr, or a SliceTypeExpr (see their own docstrings in
    parser.py) -- handled by recursing on element_type, bottoming out
    at a scalar/struct/alias name with no depth limit.

    `structs` and `aliases` are both required parameters, not defaulted
    to empty dicts, so a call site that forgets to pass one fails
    loudly (TypeError) rather than silently misresolving a struct- or
    alias-typed declaration as unknown. Both registries are already
    fully built by the time this is called with a name needing them
    (see analyze()'s ordering: aliases, then structs, then function
    signatures) -- resolving either is a single dict lookup, never a
    recursive re-resolution.

    Only fails for a program that isn't syntactically valid, or
    references an undeclared struct/alias name -- parse_type() already
    restricts everything else at parse time."""
    if isinstance(type_expr, ArrayTypeExpr):
        element = type_from_name(type_expr.element_type, structs, aliases)
        return Type(TypeKind.ARRAY, element_type=element, size=type_expr.size)
    if isinstance(type_expr, SliceTypeExpr):
        element = type_from_name(type_expr.element_type, structs, aliases)
        return Type(TypeKind.SLICE, element_type=element)
    if type_expr in _TYPE_NAMES:
        return _TYPE_NAMES[type_expr]
    if type_expr in aliases:
        return aliases[type_expr]
    if type_expr in structs:
        return Type(TypeKind.STRUCT, struct_name=type_expr)
    raise SemanticError(f"Unknown type '{type_expr}'")


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
    re-declared variable, or a type mismatch anywhere in the program."""


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

        # 3.5. Collect struct methods, immediately lowering each into
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
            if fn.name in self.type_aliases:
                raise SemanticError(
                    f"Function '{fn.name}' collides with a type alias "
                    f"of the same name -- function and type-alias "
                    f"names share one namespace and can never be the "
                    f"same"
                )
            if fn.name in self.functions:
                raise SemanticError(f"Function '{fn.name}' is already declared")
            param_types = [type_from_name(p.type, self.structs, self.type_aliases) for p in fn.params]
            return_type = Type.VOID if fn.return_type is None else type_from_name(fn.return_type, self.structs, self.type_aliases)
            self.functions[fn.name] = (param_types, return_type)

        # 5. Check each function's own body, including every
        #    synthesized method-function's, via the same analyze_
        #    function -- a method's receiver is just its first Param by
        #    now, indistinguishable from an ordinary function.
        for fn in program.functions:
            self.analyze_function(fn)

    def _collect_methods(self, program: Program) -> Dict[Tuple[str, str], Tuple[List[Type], Type, str]]:
        """For every struct's methods: reject a duplicate name on the
        SAME struct (fine across different structs), synthesize an
        ordinary Function (receiver becomes a typed first Param) and
        append it to program.functions in place.

        The synthesized name is `StructName.methodName` -- '.' can't
        appear in a Hornet IDENTIFIER, so it structurally can't
        collide with any free function, another struct's method, or a
        builtin; no explicit collision check needed.

        Returns a (struct_name, method_name) -> (param types excluding
        the receiver, return type, mangled name) lookup for check_
        call's _check_method_call to resolve and rewrite a call site."""
        methods: Dict[Tuple[str, str], Tuple[List[Type], Type, str]] = {}
        for sd in program.structs:
            seen_names: Set[str] = set()
            for md in sd.methods:
                if md.name in seen_names:
                    raise SemanticError(
                        f"Method '{md.name}' is already declared on "
                        f"struct '{sd.name}'"
                    )
                seen_names.add(md.name)
                mangled_name = f"{sd.name}.{md.name}"
                param_types = [type_from_name(p.type, self.structs, self.type_aliases) for p in md.params]
                return_type = Type.VOID if md.return_type is None else type_from_name(md.return_type, self.structs, self.type_aliases)
                methods[(sd.name, md.name)] = (param_types, return_type, mangled_name)
                receiver_param = Param(name=md.receiver_name, type=sd.name)
                program.functions.append(Function(
                    name=mangled_name,
                    return_type=md.return_type,
                    params=[receiver_param] + md.params,
                    body=md.body,
                ))
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
                    f"type alias name"
                )
            if ad.name in structs:
                raise SemanticError(
                    f"Type alias '{ad.name}' collides with a struct of "
                    f"the same name -- struct and type-alias names "
                    f"share one namespace and can never be the same"
                )
            if ad.name in seen:
                raise SemanticError(f"Type alias '{ad.name}' is already declared")
            seen[ad.name] = ad

        resolved: Dict[str, Type] = {}
        resolving: Set[str] = set()

        def resolve(name: str) -> Type:
            if name in resolved:
                return resolved[name]
            if name in resolving:
                raise SemanticError(
                    f"Type alias '{name}' is defined in terms of "
                    f"itself (a cycle)"
                )
            resolving.add(name)
            result = resolve_target(seen[name].target_type)
            resolving.discard(name)
            resolved[name] = result
            return result

        def resolve_target(target) -> Type:
            if isinstance(target, ArrayTypeExpr):
                return Type(TypeKind.ARRAY, element_type=resolve_target(target.element_type), size=target.size)
            if isinstance(target, SliceTypeExpr):
                return Type(TypeKind.SLICE, element_type=resolve_target(target.element_type))
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
                f"another type alias"
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
                    f"struct name"
                )
            if sd.name in registry:
                raise SemanticError(f"Struct '{sd.name}' is already declared")
            registry[sd.name] = None
        return registry

    def _resolve_struct_fields(self, struct_defs: List[StructDef], registry: Dict[str, StructInfo]) -> Dict[str, StructInfo]:
        """Resolves every struct's field types, then checks for
        cycles. `registry` already has every struct's NAME reserved (a
        None placeholder), which is what lets a forward reference work
        (`struct A: B b` before `struct B: ...` is declared) -- type_
        from_name's struct-name check only needs NAME membership.

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
                        f"Field '{f.name}' is already declared in struct '{sd.name}'"
                    )
                fields[f.name] = type_from_name(f.field_type, registry, self.type_aliases)
            registry[sd.name] = StructInfo(name=sd.name, fields=fields)

        for sd in struct_defs:
            self._check_struct_contains(sd.name, registry, path=[])

        return registry

    def _check_struct_contains(self, name: str, registry: Dict[str, StructInfo], path: List[str]) -> None:
        """DFS over the struct-containment graph -- X has an edge to Y
        if X has a field of type Y, directly or through any depth of
        array wrapping (an array embeds its element inline, N times
        over, so an array of a struct that contains X is exactly as
        size-infinite as X containing itself). A SLICE field
        deliberately doesn't count: its backing storage is a separate,
        runtime-sized allocation, not embedded inline -- `struct A:
        []A elements` is a genuinely supported pattern (a tree or
        linked structure), not merely tolerated.

        `path` is the visited chain, for a readable error message --
        struct counts are small enough that this doesn't need
        memoization against already-explored, cycle-free structs."""
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
        """If `field_type` is a struct, or an array (at any depth) of
        one, returns that struct's name -- see _check_struct_contains
        for why arrays count and slices don't. None for a scalar field,
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
            self._declare(p.name, type_from_name(p.type, self.structs, self.type_aliases))
        return_type = Type.VOID if fn.return_type is None else type_from_name(fn.return_type, self.structs, self.type_aliases)
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
        `target_type` is expected -- ordinary equality, or the one
        exception this language allows: Type.NONE is compatible with
        ANY slice type (its zero/nil value). Deliberately narrow --
        not int/bool/str/array, even though str is also a pointer
        under the hood.

        Shared by every site with a clear "this is the expected type"
        side (a VarDecl initializer, Assign, IndexAssign, argument,
        return value), so `none` becomes valid at all of them
        uniformly. Equality (`==`/`!=`) has no such fixed side and is
        checked separately, directly in check_binary."""
        if value_type == target_type:
            return True
        return value_type == Type.NONE and target_type.kind == TypeKind.SLICE

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
                        f"{target_type} ({lo} to {hi})"
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
        declared_type = type_from_name(stmt.var_type, self.structs, self.type_aliases)
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
                f"element of type {element_type}"
            )

    def analyze_field_assign(self, stmt: FieldAssign) -> None:
        """`base.name = value` -- mirrors analyze_index_assign one
        level over, for the identical reasons."""
        field_type = self._check_struct_and_field(stmt.base, stmt.name)
        value_type = self._check_value_flowing_into_allowing_struct_literal(stmt.value, field_type)
        if not self._types_compatible(value_type, field_type):
            raise SemanticError(
                f"Cannot assign a value of type {value_type} to field "
                f"'{stmt.name}' of type {field_type}"
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
                f"only arrays and slices support indexing"
            )
        index_type = self.check_expr(index_expr)
        if index_type != Type.INT:
            raise SemanticError(f"Index must be int, got {index_type}")
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
                    f"this bare 'return' returns nothing"
                )
            return
        value_type = self._check_value_flowing_into_allowing_struct_literal(stmt.value, return_type)
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

        # then/else get independent scopes (module docstring), so a
        # name in one is never visible in the other. An elif's else_
        # body is a single nested If (parser.py's If docstring);
        # analyze_if just recurses into it like any other statement.
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
        else:
            raise SemanticError(f"No semantic rule for expression: {expr!r}")
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
            declared_type = type_from_name(expr.type_expr, self.structs, self.type_aliases)
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
                        f"{expected_element_type} (to match the "
                        f"declared element type), but element {i} "
                        f"is {element_type}"
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
        """Shared by check_field (`base.name`) and analyze_field_
        assign (`base.name = value`), mirroring _check_indexable_and_
        index one level over: check base_expr is struct-typed, look up
        field_name in its registered field list, and return the
        field's type -- or raise a clear error for whichever went
        wrong."""
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
                f"declaration order: {field_names}), got {len(expr.args)}"
            )
        for i, (arg, (field_name, field_type)) in enumerate(zip(expr.args, field_items), start=1):
            arg_type = self._check_value_flowing_into_allowing_struct_literal(arg, field_type)
            if not self._types_compatible(arg_type, field_type):
                raise SemanticError(
                    f"Argument {i} to struct literal '{expr.name}' "
                    f"(field '{field_name}') should be {field_type}, "
                    f"got {arg_type}"
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
                    f"'{field_name}' -- valid fields are: {valid_names}"
                )
            if field_name in seen:
                raise SemanticError(
                    f"Field '{field_name}' specified more than once in "
                    f"struct literal for '{expr.name}'"
                )
            seen.add(field_name)
            value_type = self._check_value_flowing_into_allowing_struct_literal(value, field_types[field_name])
            expected_type = field_types[field_name]
            if not self._types_compatible(value_type, expected_type):
                raise SemanticError(
                    f"Field '{field_name}' of struct literal '{expr.name}' "
                    f"should be {expected_type}, got {value_type}"
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
        if receiver_type.kind != TypeKind.STRUCT:
            raise SemanticError(
                f"Cannot call method '{expr.name}' on a value of type "
                f"{receiver_type} -- methods are only defined on structs"
            )
        key = (receiver_type.struct_name, expr.name)
        if key not in self.methods:
            raise SemanticError(
                f"Struct '{receiver_type.struct_name}' has no method "
                f"'{expr.name}'"
            )
        param_types, return_type, mangled_name = self.methods[key]
        if len(expr.args) != len(param_types):
            raise SemanticError(
                f"Method '{expr.name}' on '{receiver_type.struct_name}' "
                f"expects {len(param_types)} argument(s), got "
                f"{len(expr.args)}"
            )
        for i, (arg, expected_type) in enumerate(zip(expr.args, param_types), start=1):
            actual_type = self._check_value_flowing_into_allowing_struct_literal(arg, expected_type)
            if not self._types_compatible(actual_type, expected_type):
                raise SemanticError(
                    f"Argument {i} to method '{expr.name}' on "
                    f"'{receiver_type.struct_name}' should be "
                    f"{expected_type}, got {actual_type}"
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
                f"element, an IndexAssign's own element, or a "
                f"FieldAssign's own field -- not as a bare statement, "
                f"or most other kinds of expressions (a "
                f"Binary operand, a Field-access base, ...); assign it "
                f"to a variable first if you need it in one of those "
                f"positions"
            )
        if expr.kwargs is not None:
            # Named arguments parse into the same shape a named struct
            # literal does -- this is the one place that distinction
            # is made, now that expr.name is known not to be a struct.
            # Named construction is scoped to struct literals only.
            raise SemanticError(
                f"'{expr.name}(...)' uses named arguments, which are "
                f"only supported for struct literals, not function calls"
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
                    f"{expected_type}, got {actual_type}"
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
            if operand_type not in _INTEGER_TYPES:
                raise SemanticError(
                    f"'{expr.op.symbol()}' requires an int, int8, "
                    f"uint8, or int64 operand, got {operand_type}"
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
                    f"`not (x == 0)` instead of `not x`)"
                )
            return Type.BOOL
        raise SemanticError(f"No semantic rule for unary operator: {expr.op}")

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
        target_type = type_from_name(expr.target_type, self.structs, self.type_aliases)
        source_type = self.check_expr(expr.expr)
        if target_type not in _INTEGER_TYPES or source_type not in _INTEGER_TYPES:
            raise SemanticError(
                f"Cannot cast {source_type} to {target_type} -- casting "
                f"is only supported between int, int8, uint8, and int64 "
                f"right now"
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
        return True  # INT, BOOL, STR

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
                f"got {left_type} and {right_type}"
            )

        if op in _INT_ONLY_BINARY_OPS:
            # Stays whichever integer type both operands were -- int8
            # + int8 is int8, never promoted to int (unlike C's own
            # integer-promotion rules).
            return self._require_same_integer_type(left_type, right_type, op)

        if op in _ORDERING_OPS:
            self._require_same_integer_type(left_type, right_type, op)
            return Type.BOOL

        if op in _EQUALITY_OPS:
            # A slice compared to `none` (either order) is checked
            # first, since it's meaningful and allowed -- one of three
            # exceptions to the slice/void/none rejection below.
            none_vs_slice = (
                (left_type == Type.NONE and right_type.kind == TypeKind.SLICE) or
                (right_type == Type.NONE and left_type.kind == TypeKind.SLICE)
            )
            if none_vs_slice:
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
                        f"length and element type"
                    )
                if not self._is_comparable_type(left_type):
                    raise SemanticError(
                        f"'{op.symbol()}' does not support {left_type} "
                        f"operands -- array equality isn't defined yet "
                        f"when the elements are (or contain) a slice, "
                        f"which has no '==' defined for it yet"
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
                        f"same type"
                    )
                if not self._is_comparable_type(left_type):
                    raise SemanticError(
                        f"'{op.symbol()}' does not support {left_type} "
                        f"operands -- struct equality isn't defined yet "
                        f"when a field (directly, or nested inside "
                        f"another struct or an array field) is a slice, "
                        f"which has no '==' defined for it yet"
                    )
                return Type.BOOL

            # A bare slice-vs-slice comparison is rejected outright:
            # codegen has no slice comparison logic, and it isn't even
            # well-defined yet (compare elements, like array equality
            # now does, or the pointer/length/cap triple?) -- a real
            # feature to consider later, not implemented yet.
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
            if left_type.kind in (TypeKind.SLICE, TypeKind.VOID, TypeKind.NONE) or right_type.kind in (TypeKind.SLICE, TypeKind.VOID, TypeKind.NONE):
                raise SemanticError(
                    f"'{op.symbol()}' does not support slice, void, or "
                    f"none operands, except comparing a slice to none"
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

    def _require_same_integer_type(self, left_type: Type, right_type: Type, op) -> Type:
        """Requires left_type and right_type to be the exact SAME
        integer type -- never a mix, even between two different-but-
        both-integer types (`int8 + uint8` is rejected like `bool +
        int` already is). Returns that shared type as the operator's
        own result -- used directly by check_binary's _INT_ONLY_
        BINARY_OPS and ordering-operator cases."""
        if left_type not in _INTEGER_TYPES or left_type != right_type:
            raise SemanticError(
                f"'{op.symbol()}' requires two operands of the same "
                f"integer type (int, int8, uint8, or int64), got "
                f"{left_type} and {right_type}"
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
