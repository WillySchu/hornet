"""semantic/expressions.py's ExpressionChecker on its own: given where checking is (a Context),
what the program declares, and the names in scope, it types an expression and records what it
learned, with no statement checker anywhere."""

import pytest

from scopes import build_module_set
from semantic.context import Context
from semantic.declarations import DeclarationResolver, Declarations
from semantic.errors import SemanticError
from semantic.expressions import ExpressionChecker
from semantic.facts import Facts
from semantic.flow import Scopes
from semantic.type_resolution import TypeResolver
from symbols import SymbolTable
from tests.test_compiler import _parse
from typesys import Type, TypeKind

DECLS = (
    "const int N = 3\n"
    "const bool FLAG = N > 2\n"
    "const str NAME = 'n' + 'ame'\n"
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


def _checker(expression: str, decls_source: str = DECLS):
    """(an ExpressionChecker with VARIABLES in scope, the expression's node, the facts)."""
    entry = _parse(decls_source + f"def int main():\n    {expression}\n    return 0\n")
    program, facts, decls, symbols = build_module_set(entry, {}), Facts(), Declarations(), SymbolTable()
    context = Context(scope=program.files[0][1], scopes=Scopes(symbols, decls, facts))
    types = TypeResolver(decls, program, facts.array_sizes)
    checker = ExpressionChecker(context, decls, facts, types, program, symbols)   # (it builds its ConstEvaluator)
    DeclarationResolver(program, decls, types, facts, checker.constants, checker.check_array_size).resolve()
    for name, type_ in VARIABLES.items():
        if decls_source is DECLS or name in ('n', 'small', 's', 'flag'):   # (the rest have DECLS's types)
            context.scopes.declare(name, type_, None, symbols.new(name, 'local', type_).id)
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


# -- constant expressions, which the declaration phase asks about

@pytest.mark.parametrize("expression,value", [
    ("N + 1", 4),
    ("N * N - 2", 7),
    ("len(Color) + N", 5),
    ("8", 8),
])
def test_an_array_size_is_a_positive_constant_integer(expression, value):
    checker, node, _ = _checker(expression)
    scope, scopes = checker.context.scope, checker.context.scopes
    assert checker.check_array_size(node, scope) == value
    # Where checking was is where it is again: the same file's scope, and the locals back in sight.
    assert checker.context.scope is scope and checker.context.scopes is scopes and scopes.is_local('n')


@pytest.mark.parametrize("expression,message", [
    ("n + 1", "Array size must be a constant expression, but 'n' isn't a constant"),      # a local is no constant
    ("twice(N)", "Array size must be a constant expression, not a call"),
    ("N - 3", "Array size must be positive, got 0"),
    ("FLAG", "Array size must be an integer, got bool"),
    ("NAME", "Array size must be an integer, got str"),
])
def test_what_is_wrong_with_an_array_size(expression, message):
    checker, node, _ = _checker(expression)
    scope, scopes = checker.context.scope, checker.context.scopes
    with pytest.raises(SemanticError, match=message):
        checker.check_array_size(node, scope)
    assert checker.context.scope is scope and checker.context.scopes is scopes       # ... restored all the same


def test_a_constant_s_declaration_is_checked_and_its_value_worked_out():
    checker, _, _ = _checker("N")
    assert checker.constants.values == {'N': (Type.INT, 3), 'FLAG': (Type.BOOL, True), 'NAME': (Type.STR, 'name')}
    assert checker.check_const_declaration(checker.constants.decls['FLAG']) == Type.BOOL


@pytest.mark.parametrize("declaration,message", [
    ("type P struct:\n    int x\nconst P ORIGIN = P(0)\n", "constants must be an integer type, bool, str, or an enum"),
    ("const int N = 'three'\n", "Constant 'N' is declared int but its value has type str"),
    ("const int8 SMALL = 200\n", "200"),
    ("const int A = B\nconst int B = A\n", "is defined in terms of itself"),
    ("const int N = M + 1\n", "Reference to undeclared variable 'M'"),
])
def test_what_is_wrong_with_a_constant_s_declaration(declaration, message):
    with pytest.raises(SemanticError, match=message):
        _checker("1", decls_source=declaration)

