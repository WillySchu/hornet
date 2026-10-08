"""semantic/declarations.py's DeclarationResolver on its own: it fills a Declarations from a program
with no body checker in sight, asking only for the value of an array-size expression (and, through
the constant evaluator, the check of a constant's declaration)."""

import pytest

from scopes import build_module_set
from semantic.constants import ConstEvaluator
from semantic.declarations import DeclarationResolver, Declarations
from semantic.errors import SemanticError
from semantic.facts import Facts
from semantic.type_resolution import TypeResolver
from tests.test_compiler import _parse
from typesys import Type, TypeKind


def _resolve(source: str, array_size=None, check_declaration=None):
    """The Declarations of `source`, and the array-size expressions that were asked about."""
    program, facts, decls, asked = build_module_set(_parse(source), {}), Facts(), Declarations(), []

    def size(expr, scope):
        asked.append((expr, scope))
        return array_size
    check = check_declaration or (lambda decl: pytest.fail("no constant here"))
    constants = ConstEvaluator(facts, decls.enums, check)
    types = TypeResolver(decls, program, facts.array_sizes)
    DeclarationResolver(program, decls, types, facts, constants, size).resolve()
    return decls, asked, program


SOURCE = (
    "extern int abs(int n)\n"
    "type Color enum:\n"
    "    Red\n"
    "    Green\n"
    "    def str label(self):\n"
    "        return 'c'\n"
    "type Meters = int\n"
    "type Point struct:\n"
    "    Meters x\n"
    "    Color c\n"
    "    def move(*self, int by):\n"
    "        self.x += by\n"
    "type Shape is Point | none\n"
    "def Shape make(int x):\n"
    "    return Point(x, Color.Red)\n"
    "def never stop():\n"
    "    panic('x')\n"
    "def int main():\n"
    "    return 0\n"
)


def test_what_a_program_declares():
    decls, asked, _ = _resolve(SOURCE)
    point, color = Type(TypeKind.STRUCT, struct_name='Point'), Type(TypeKind.ENUM, enum_name='Color')
    assert decls.enums['Color'].members == ['Red', 'Green']
    assert decls.type_aliases == {'Meters': Type.INT}
    assert decls.structs['Point'].fields == {'x': Type.INT, 'c': color}
    assert decls.sum_types['Shape'].variants == [point, Type.NONE]
    assert decls.methods[('Point', 'move')] == ([Type.INT], Type.VOID, 'Point.move')
    assert decls.methods[('Color', 'label')] == ([], Type.STR, 'Color.label')
    assert decls.pointer_receivers == {('Point', 'move')}
    assert decls.functions['make'] == ([Type.INT], Type(TypeKind.SUM, sum_type_name='Shape'))
    assert decls.functions['stop'] == ([], Type.NEVER) and decls.functions['main'] == ([], Type.INT)
    assert decls.functions['abs'] == ([Type.INT], Type.INT) and decls.extern_names == {'abs'}
    # Every function to check: the program's own, then each method as one, its receiver first.
    assert [fn.name for fn in decls.all_functions] == ['make', 'stop', 'main', 'Point.move', 'Color.label']
    assert decls.functions['Point.move'] == ([Type(TypeKind.POINTER, element_type=point), Type.INT], Type.VOID)
    assert asked == []


def test_an_array_size_is_asked_of_the_checker_with_the_scope_it_is_written_in():
    decls, asked, program = _resolve(
        "const int N = 3\n"
        "type Grid struct:\n"
        "    [N + 1]int cells\n"
        "def int main():\n"
        "    return 0\n",
        array_size=4, check_declaration=lambda decl: Type.INT)
    assert decls.structs['Grid'].fields == {'cells': Type(TypeKind.ARRAY, element_type=Type.INT, size=4)}
    (expr, scope), = asked
    assert type(expr).__name__ == 'Binary' and scope is program.files[0][1]


@pytest.mark.parametrize("source,message", [
    ("type P struct:\n    int x\ndef int P():\n    return 1\n", "Function 'P' collides with a struct of the same name"),
    ("type P struct:\n    Missing m\n", "Unknown type 'Missing'"),
    ("type P struct:\n    P inner\n", "contains itself"),
    ("type E enum:\n    A\n    A\n", "Member 'A' is already declared in enum 'E'"),
    ("type E enum:\n    A\ntype E struct:\n    int x\n", "collides with"),
    ("def int f():\n    return 1\ndef int f():\n    return 2\n", "Function 'f' is already declared"),
    ("extern int f(str s)\n", "str"),
])
def test_what_is_wrong_with_a_declaration_is_found_here(source, message):
    with pytest.raises(SemanticError, match=message):
        _resolve(source + "def int main():\n    return 0\n")
