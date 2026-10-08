"""semantic/calls.py's CallChecker with a stand-in for the expression checker it is part of: what
it decides about a call needs only the declarations and the types of the arguments."""

import pytest

from parser import BoolLiteral, Call, Constant, StringLiteral, Variable
from scopes import build_module_set
from semantic.calls import CallChecker
from semantic.constants import ConstEvaluator
from semantic.declarations import DeclarationResolver, Declarations
from semantic.errors import SemanticError
from semantic.facts import Facts
from semantic.types import TypeResolver
from tests.test_compiler import _parse
from typesys import Type, TypeKind

DECLS = (
    "type Color enum:\n"
    "    Red\n"
    "    Green\n"
    "type Point struct:\n"
    "    int x\n"
    "    str name\n"
    "    def int scaled(self, int by):\n"
    "        return self.x * by\n"
    "def int add(int a, int b):\n"
    "    return a + b\n"
    "def never stop(str why):\n"
    "    panic(why)\n"
)
POINT = Type(TypeKind.STRUCT, struct_name='Point')
COLOR = Type(TypeKind.ENUM, enum_name='Color')
INTS = Type(TypeKind.SLICE, element_type=Type.INT)
VARIABLES = {'n': Type.INT, 's': Type.STR, 'p': POINT, 'xs': INTS, 'flag': Type.BOOL}


class _StandIn:
    """The least an expression checker can be: literals and the variables above have their types."""

    def __init__(self, scope, decls):
        self.scope, self.decls, self.calls = scope, decls, None

    def check_expr(self, expr):
        if isinstance(expr, Call):
            return self.calls.check_call(expr)
        if isinstance(expr, Variable):
            return VARIABLES[expr.name]
        return {Constant: Type.INT, StringLiteral: Type.STR, BoolLiteral: Type.BOOL}[type(expr)]

    def check_expr_allowing_struct_literal(self, expr):
        if self.calls.struct_literal(expr) is not None:
            return self.calls.check_struct_literal(expr)
        return self.check_expr(expr)

    def check_value_flowing_into(self, expr, target_type):
        return self.check_expr(expr)

    def check_value_flowing_into_allowing_struct_literal(self, expr, target_type):
        return self.check_expr_allowing_struct_literal(expr)

    def types_compatible(self, value_type, target_type):
        return value_type == target_type

    def hidden(self, struct, name):
        return False

    def enum_named_by(self, expr):
        return expr.name if isinstance(expr, Variable) and expr.name in self.decls.enums else None

    def as_folded_int_literal(self, expr):
        return expr.value if isinstance(expr, Constant) else None

    def without_zero_value(self, t, seen=None):
        return None


def _check(expression: str):
    """(the type CallChecker gives `expression`, the facts it recorded)."""
    entry = _parse(DECLS + f"def int main():\n    {expression}\n    return 0\n")
    program, facts, decls = build_module_set(entry, {}), Facts(), Declarations()
    constants = ConstEvaluator(facts, decls.enums, lambda decl: pytest.fail("no constant here"))
    types = TypeResolver(decls, program, facts.array_sizes)
    DeclarationResolver(program, decls, types, facts, constants, lambda expr, scope: pytest.fail("no size")).resolve()
    stand_in = _StandIn(program.files[0][1], decls)
    stand_in.calls = CallChecker(stand_in, decls, facts, constants, program)
    call = next(fn for fn in program.functions if fn.name == 'main').body[0].expr
    return stand_in.check_expr_allowing_struct_literal(call), facts


@pytest.mark.parametrize("expression,expected", [
    ("add(n, 2)", Type.INT),
    ("add(add(1, 2), n)", Type.INT),
    ("stop(s)", Type.NEVER),
    ("p.scaled(3)", Type.INT),
    ("Point(1, 'a')", POINT),
    ("Point(name='a', x=1)", POINT),
    ("Point(n, s).scaled(2)", Type.INT),          # a struct literal as a receiver
    ("len(xs)", Type.INT),
    ("len(s)", Type.INT),
    ("len(Color)", Type.INT),
    ("append(xs, n)", INTS),
    ("print(p)", Type.VOID),
    ("bytes(s)", Type(TypeKind.SLICE, element_type=Type.UINT8)),
    ("format('{} and {}', n, s)", Type.STR),
    ("Color(1)", COLOR),
    ("Color(n)", COLOR),
])
def test_what_a_call_is_and_gives(expression, expected):
    assert _check(expression)[0] == expected


def test_what_it_records():
    assert _check("len(Color)")[1].enum_lens != {}                        # the number of members, for later stages
    assert list(_check("format('{} and {}', n, s)")[1].formats.values()) == ['{} and {}']
    (name, args), = _check("p.scaled(3)")[1].calls.values()               # a method call, as the function it is
    assert name == 'Point.scaled' and len(args) == 2


@pytest.mark.parametrize("expression,message", [
    ("add(n)", r"Function 'add' expects 2 argument\(s\), got 1"),
    ("add(n, s)", "Argument 2 to 'add' should be int, got str"),
    ("missing(n)", "Call to undeclared function 'missing'"),
    ("add(a=1, b=2)", "uses named arguments, which are only supported for struct literals"),
    ("p.nothing()", "has no method 'nothing'"),
    ("p.scaled(s)", "int"),
    ("Point(1)", "Point"),
    ("Point(x=1, title='a')", "title"),
    ("len(n)", "len"),
    ("len(xs, xs)", "len"),
    ("append(xs, s)", "append"),
    ("append(n, n)", "append"),
    ("bytes(n)", "bytes"),
    ("format('{} {}', n)", "format"),
    ("format(s, n)", "format"),
    ("Color(2)", "2 is not a member of Color"),
    ("Color(s)", "converts an integer to the enum Color, got str"),
    ("Color(1, 2)", "converts one integer to the enum Color, got 2 arguments"),
    ("print(n, n)", "print"),
])
def test_what_is_wrong_with_a_call(expression, message):
    with pytest.raises(SemanticError, match=message):
        _check(expression)
