"""semantic/expressions.py's ExpressionChecker on its own: given where checking is (a Context),
what the program declares, and the names in scope, it types an expression and records what it
learned, with no statement checker anywhere."""

import pytest

from scopes import build_module_set
from semantic.constants import ConstEvaluator
from semantic.context import Context
from semantic.declarations import DeclarationResolver, Declarations
from semantic.errors import SemanticError
from semantic.expressions import ExpressionChecker
from semantic.facts import Facts
from semantic.flow import Scopes
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
    "    str name\n"
    "type Shape is Point | int | none\n"
    "def int twice(int n):\n"
    "    return n * 2\n"
)
POINT = Type(TypeKind.STRUCT, struct_name='Point')
COLOR = Type(TypeKind.ENUM, enum_name='Color')
SHAPE = Type(TypeKind.SUM, sum_type_name='Shape')
INTS = Type(TypeKind.SLICE, element_type=Type.INT)
PAIR = Type(TypeKind.ARRAY, element_type=Type.INT, size=2)
VARIABLES = {'n': Type.INT, 'small': Type.INT8, 's': Type.STR, 'flag': Type.BOOL, 'p': POINT, 'c': COLOR,
             'shape': SHAPE, 'xs': INTS, 'pair': PAIR, 'ptr': Type(TypeKind.POINTER, element_type=POINT),
             'ages': Type(TypeKind.DICT, key_type=Type.STR, element_type=Type.INT)}


def _checker(expression: str):
    """(an ExpressionChecker with VARIABLES in scope, the expression's node, the facts)."""
    entry = _parse(DECLS + f"def int main():\n    {expression}\n    return 0\n")
    program, facts, decls, symbols = build_module_set(entry, {}), Facts(), Declarations(), SymbolTable()
    context = Context(scope=program.files[0][1], scopes=Scopes(symbols, decls, facts))
    types = TypeResolver(decls, program, facts.array_sizes)
    constants = ConstEvaluator(facts, decls.enums, lambda decl: pytest.fail("no constant here"))
    DeclarationResolver(program, decls, types, facts, constants, lambda expr, scope: pytest.fail("no size")).resolve()
    for name, type_ in VARIABLES.items():
        context.scopes.declare(name, type_, None, symbols.new(name, 'local', type_).id)
    checker = ExpressionChecker(context, decls, facts, constants, types, program, symbols)
    return checker, next(fn for fn in program.functions if fn.name == 'main').body[0].expr, facts


@pytest.mark.parametrize("expression,expected", [
    ("n + 1", Type.INT),
    ("small + 1", Type.INT8),                       # the literal takes the other operand's type
    ("s + 'x'", Type.STR),
    ("s < 'x'", Type.BOOL),
    ("n < 3 and not flag", Type.BOOL),
    ("-n", Type.INT),
    ("p.x", Type.INT),
    ("ptr.name", Type.STR),                          # through the pointer
    ("xs[0]", Type.INT),
    ("xs[1:n]", INTS),
    ("s[0]", Type.UINT8),
    ("ages['a']", Type.INT),
    ("'a' in ages", Type.BOOL),
    ("n in [1, 2, 3]", Type.BOOL),                   # the literal's elements take n's type
    ("pair == [1, 2]", Type.BOOL),
    ("[2]int[n, 3]", PAIR),
    ("[]str['a', s]", Type(TypeKind.SLICE, element_type=Type.STR)),
    ("len(dict[str]int{'a': n})", Type.INT),
    ("&p", VARIABLES['ptr']),
    ("*ptr", POINT),
    ("int8(n)", Type.INT8),
    ("str(c)", Type.STR),
    ("Color.Red", COLOR),
    ("c is Red", Type.BOOL),
    ("shape is Point", Type.BOOL),
    ("shape == none", Type.BOOL),
    ("twice(n) + len(xs)", Type.INT),                # calls, through its CallChecker
    ("Point(n, s).x", Type.INT),
])
def test_the_type_of_an_expression(expression, expected):
    checker, node, facts = _checker(expression)
    assert checker.check_expr_allowing_struct_literal(node) == expected
    assert facts.types[node.nid] == expected        # ... recorded, for the typed tree


def test_what_it_records_about_names_and_narrowing():
    checker, node, facts = _checker("shape is Point")
    checker.check_expr(node)
    assert facts.narrowed[node.nid] == POINT        # what flow.py's Scopes.when reads
    when_true, when_false = checker.context.scopes.when(node)
    shape_id = checker.context.scopes.lookup('shape')[1]
    assert when_true == {shape_id: ('shape', frozenset([POINT]))}
    assert when_false == {shape_id: ('shape', frozenset([Type.INT, Type.NONE]))}
    # A Shape has no field to read...
    checker, node, _ = _checker("shape.x")
    with pytest.raises(SemanticError):
        checker.check_expr(node)
    # ... but once narrowed where checking is, it is a Point, and the same expression has a type.
    checker.context.scopes.apply({checker.context.scopes.lookup('shape')[1]: ('shape', frozenset([POINT]))})
    assert checker.check_expr(node) == Type.INT


@pytest.mark.parametrize("expression,message", [
    ("n + s", "requires two operands of the same integer type"),
    ("n + small", "requires two operands of the same integer type"),
    ("missing + 1", "Reference to undeclared variable 'missing'"),
    ("Color + 1", "'Color' is an enum, not a value"),
    ("p.nothing", "nothing"),
    ("n.x", "x"),
    ("xs['a']", "Index must be int"),
    ("n[0]", "index"),
    ("len([1, 2])", "This array literal has nothing to take its type from"),
    ("[2]int[1, s]", "Array literal declares element type int, but element 2 is str"),
    ("*n", "n"),
    ("shape is Missing", "Missing"),
    ("n is int", r"'n' \(declared int\) is not a sum type"),
    ("flag + flag", "requires two operands of the same integer type"),
    ("not n", "not"),
    ("n == s", "Cannot compare"),
    ("small + 200", "200"),
])
def test_what_is_wrong_with_an_expression(expression, message):
    checker, node, _ = _checker(expression)
    with pytest.raises(SemanticError, match=message):
        checker.check_expr_allowing_struct_literal(node)
