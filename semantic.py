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
Two types exist: `int` and `bool`. This is a genuinely *strong* static
type system in the traditional PL sense: there is no implicit
conversion between them in either direction. A bool is not a 0-or-1
int that happens to print differently -- it's a distinct type, and
using one where the other is expected is a type error, full stop. In
particular (and this is the part most likely to surprise someone
coming from C): `not`, `and`, and `or` all require real `bool`
operands. `not 0` is a type error, not "not true". Write
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
from enum import auto, Enum
from typing import Dict, List

from lexer import lex
from parser import (
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
    Node,
    Param,
    Parser,
    Program,
    Return,
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

class Type(Enum):
    INT = auto()
    BOOL = auto()
    STR = auto()

    def __str__(self) -> str:
        return self.name.lower()


_TYPE_NAMES = {
    'int': Type.INT,
    'bool': Type.BOOL,
    'str': Type.STR,
}


def type_from_name(name: str) -> Type:
    """Converts a type keyword's text (as stored in VarDecl.var_type /
    Function.return_type, straight from the token) into a Type. Only
    ever fails for a program that isn't syntactically valid in the
    first place -- parse_type() already restricts these strings to
    'int'/'bool' -- so this is a defensive check, not a user-facing
    validation path."""
    try:
        return _TYPE_NAMES[name]
    except KeyError:
        raise SemanticError(f"Unknown type '{name}'")


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
        if isinstance(expr, Constant):
            return self.check_constant(expr)
        if isinstance(expr, BoolLiteral):
            return Type.BOOL
        if isinstance(expr, StringLiteral):
            return Type.STR
        if isinstance(expr, Variable):
            return self.check_variable(expr)
        if isinstance(expr, Call):
            return self.check_call(expr)
        if isinstance(expr, Unary):
            return self.check_unary(expr)
        if isinstance(expr, Binary):
            return self.check_binary(expr)
        raise SemanticError(f"No semantic rule for expression: {expr!r}")

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
        """`print` takes exactly one argument, of *any* type -- unlike
        an ordinary function it isn't tied to one fixed parameter type,
        since int/bool/str are all printable and there's no reason to
        force a caller to pick a differently-named builtin per type.
        Always "returns" int (see the module docstring's BUILTINS
        section for why 0, specifically) -- Hornet has no void type, and
        this keeps `print(x)` usable as an ordinary expression statement
        via the same ExprStmt path every other call already goes
        through, with nothing print-specific needed there."""
        if len(expr.args) != 1:
            raise SemanticError(
                f"'print' expects exactly 1 argument, got {len(expr.args)}"
            )
        self.check_expr(expr.args[0])  # any type is fine; just validate it
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
