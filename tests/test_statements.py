"""semantic/statements.py's StatementChecker without the analyzer: given the declarations, an
expression checker, and the Context they share, it checks a function's body."""

import pytest

from scopes import build_module_set
from semantic.constants import ConstEvaluator
from semantic.context import Context
from semantic.declarations import DeclarationResolver, Declarations
from semantic.errors import SemanticError
from semantic.expressions import ExpressionChecker
from semantic.facts import Facts
from semantic.flow import Scopes
from semantic.statements import StatementChecker
from semantic.types import TypeResolver
from symbols import SymbolTable
from tests.test_compiler import _parse
from typesys import Type, TypeKind

DECLS = (
    "type Color enum:\n"
    "    Red\n"
    "    Green\n"
    "type Point struct:\n"
    "    int x\n"
    "type Shape is Point | int | none\n"
    "def never stop():\n"
    "    panic('x')\n"
)
POINT = Type(TypeKind.STRUCT, struct_name='Point')


def _check(body: str, signature: str = "def int f(Shape shape, Color c, int n)"):
    """Check the function `signature` with `body`; (the facts, the function, the checker)."""
    source = DECLS + signature + ":\n" + "".join(f"    {line}\n" for line in body.split("\n"))
    program, facts, decls, symbols = build_module_set(_parse(source), {}), Facts(), Declarations(), SymbolTable()
    context = Context(scope=program.files[0][1], scopes=Scopes(symbols, decls, facts))
    types = TypeResolver(decls, program, facts.array_sizes)
    constants = ConstEvaluator(facts, decls.enums, lambda decl: pytest.fail("no constant here"))
    DeclarationResolver(program, decls, types, facts, constants, lambda expr, scope: pytest.fail("no size")).resolve()
    expressions = ExpressionChecker(context, decls, facts, constants, types, program, symbols)
    checker = StatementChecker(context, decls, facts, symbols, types, expressions, program)
    fn = next(fn for fn in decls.all_functions if fn.name == 'f')
    checker.analyze_function(fn)
    return facts, fn, checker


@pytest.mark.parametrize("body", [
    "int total = n + 1\nreturn total",
    "if shape is Point:\n    return shape.x\nreturn 0",                          # narrowed in the branch
    "if shape is none:\n    return 0\nif shape is int:\n    return shape\nreturn shape.x",   # ... and by what is left
    "match shape as s:\n    is Point:\n        return s.x\n    is int:\n        return s\n"
    "    is none:\n        return 0",
    "match c:\n    is Red:\n        return 1\n    is Green:\n        return 2",
    "int total = 0\nwhile total < n:\n    total += 1\n    if total == 3:\n        break\nreturn total",
    "int total = 0\nfor int i = 0; i < n; i += 1:\n    total += i\nreturn total",
    "[]int xs = [1, 2]\nint total = 0\nfor i, x in xs:\n    total += i * x\nreturn total",
    "if n > 0:\n    int n2 = n\n    return n2\nelse:\n    stop()",              # a never call ends its path
    "while true:\n    n += 1",                                                    # never finishes: no return needed
    "if shape is int:\n    shape += 1\n    shape *= 2\n    return shape\nreturn 0",   # `+=` keeps the narrowing
])
def test_a_body_that_checks(body):
    facts, fn, _ = _check(body)
    assert facts.returns[fn.nid] == Type.INT


def test_what_it_records_and_leaves_behind():
    facts, fn, checker = _check("int total = n + 1\nif shape is Point as p:\n    total += p.x\nreturn total")
    declared = {symbol.name: symbol.type for symbol in facts.symbols.values()}
    assert declared['total'] == Type.INT and declared['shape'].kind == TypeKind.SUM
    assert [binding.name for binding in facts.bindings.values()] == ['p']       # the `as NAME` binding, for the tree
    assert checker.loop_depth == 0 and checker.context.scopes.lookup('total')[0] == Type.INT
    assert checker.context.scopes.lookup('p') is None                           # ... which ended with its `if`


@pytest.mark.parametrize("body,message", [
    ("int total = n\nint total = 2\nreturn total", "Variable 'total' is already declared in this scope"),
    ("return 'text'", "str"),
    ("if n > 0:\n    return 1", "does not return a value on all code paths"),
    ("break\nreturn 0", "break"),
    ("continue\nreturn 0", "continue"),
    ("return shape.x", "x"),                                                  # not narrowed: a Shape has no field
    ("if shape is Point:\n    shape = 5\n    return shape.x\nreturn 0", "x"),   # an assignment ends the narrowing
    ("match shape as s:\n    is Point:\n        return 1\n    is int:\n        return 2", "none"),   # not exhaustive
    ("match c:\n    is Red:\n        return 1", "Green"),
    ("while n:\n    n -= 1\nreturn 0", "bool"),
    ("for x in n:\n    n += x\nreturn 0", "for"),
    ("undeclared = 5\nreturn 0", "undeclared"),
    ("n = 'text'\nreturn 0", "str"),
])
def test_what_is_wrong_with_a_body(body, message):
    with pytest.raises(SemanticError, match=message):
        _check(body)


def test_a_never_function_must_not_finish():
    with pytest.raises(SemanticError, match="is declared never, but can finish"):
        _check("if n > 0:\n    stop()", signature="def never f(Shape shape, Color c, int n)")
    _check("stop()", signature="def never f(Shape shape, Color c, int n)")
