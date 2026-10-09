"""A sum type named as a variant of another gives that sum its own variants: `type LineResult is
StrResult | none` has the variants `str`, `Error`, and `none`. Every sum's variants are so a flat
list of types that aren't sums, each once, and the rest of the compiler sees nothing new."""

import subprocess

import pytest

from build import build_executable, executable_name
from compile import generate_asm
from scopes import build_module_set
from semantic.constants import ConstEvaluator
from semantic.declarations import DeclarationResolver, Declarations
from semantic.facts import Facts
from semantic.type_resolution import TypeResolver
from target import Target, default_target
from tests.test_compiler import GCC_SKIP, _parse, analyze, assert_program_semantic_error, assert_program_stdout
from typesys import Type, TypeKind

MAIN = "def int main():\n    return 0\n"
ERROR = Type(TypeKind.STRUCT, struct_name='Error')
DECLS = (
    "type Error struct:\n"
    "    str message\n"
    "type StrResult is str | Error\n"
    "type IntResult is int | Error\n"
)


def _variants(source: str, name: str) -> list:
    program, facts, decls = build_module_set(_parse(source + MAIN), {}), Facts(), Declarations()
    constants = ConstEvaluator(facts, decls.enums, lambda decl: pytest.fail("no constant here"))
    types = TypeResolver(decls, program, facts.array_sizes)
    DeclarationResolver(program, decls, types, facts, constants, lambda expr, scope: pytest.fail("no size")).resolve()
    return decls.sum_types[name].variants


def test_a_named_sum_s_variants_take_its_place():
    assert _variants(DECLS + "type LineResult is StrResult | none\n", 'LineResult') == [Type.STR, ERROR, Type.NONE]
    assert _variants(DECLS + "type Line is none | StrResult\n", 'Line') == [Type.NONE, Type.STR, ERROR]  # in place
    # What arrives twice through different sums is one variant, where it first appeared.
    assert _variants(DECLS + "type Either is IntResult | StrResult\n", 'Either') == [Type.INT, ERROR, Type.STR]
    # ... as is what is written and also arrives through a sum.
    assert _variants(DECLS + "type Padded is StrResult | str | bool\n", 'Padded') == [Type.STR, ERROR, Type.BOOL]
    assert _variants(DECLS + "type Padded is str | StrResult | bool\n", 'Padded') == [Type.STR, ERROR, Type.BOOL]


def test_sums_of_sums_of_sums_in_any_order_of_declaration():
    source = (
        "type Wide is Either | Line | bool\n"            # named before either is declared
        "type Line is StrResult | none\n"
        "type Either is IntResult | StrResult\n" + DECLS)
    assert _variants(source, 'Wide') == [Type.INT, ERROR, Type.STR, Type.NONE, Type.BOOL]
    assert _variants(source, 'Line') == [Type.STR, ERROR, Type.NONE]


def test_only_a_sum_named_bare_is_replaced():
    slices = _variants(DECLS + "type Many is []StrResult | *IntResult | int\n", 'Many')
    assert [v.kind for v in slices] == [TypeKind.SLICE, TypeKind.POINTER, TypeKind.INT]   # variants in their own right
    assert slices[0].element_type == Type(TypeKind.SUM, sum_type_name='StrResult')


def test_it_is_the_sum_it_would_be_written_out():
    def assembly(declaration: str) -> str:
        return generate_asm(analyze(_parse(
            DECLS + declaration +
            "def LineResult next(int n):\n"
            "    if n == 0:\n        return none\n"
            "    if n == 1:\n        return Error('bad')\n"
            "    return 'line'\n"
            "def int main():\n"
            "    LineResult l = next(2)\n"
            "    if l is str:\n        print(len(l))\n"
            "    print(next(0))\n"
            "    return 0\n")), target=Target('x86_64', 'linux'))
    assert assembly("type LineResult is StrResult | none\n") == assembly("type LineResult is str | Error | none\n")


@GCC_SKIP
def test_such_a_sum_at_work():
    assert_program_stdout(
        DECLS +
        "type LineResult is StrResult | none\n"
        "type Either is IntResult | StrResult\n"
        "type Wide is Either | LineResult | bool\n"
        "def LineResult next(int n):\n"
        "    if n == 0:\n        return none\n"
        "    if n == 1:\n        return Error('bad')\n"
        "    return 'line'\n"
        "def str describe(Wide w):\n"
        "    match w:\n"                                            # exhaustive over the five it has
        "        is int:\n            return 'int '\n"
        "        is Error:\n            return 'error '\n"
        "        is str:\n            return 'str '\n"
        "        is none:\n            return 'none '\n"
        "        is bool:\n            return 'bool'\n"
        "def int main():\n"
        "    print(next(0))\n    print(next(1))\n    print(next(2))\n"
        "    LineResult l = next(2)\n"
        "    if l is str:\n        print(len(l))\n"
        "    LineResult nothing\n"                                 # its zero value: the none it was given
        "    print(nothing)\n"
        "    Either e = 5\n    e = 'five'\n    e = Error('neither')\n    print(e)\n"
        "    print(describe(1) + describe('s') + describe(Error('x')) + describe(none) + describe(true))\n"
        "    return 0\n",
        "none\nError(message: 'bad')\n'line'\n4\nnone\nError(message: 'neither')\nint str error none bool\n",
    )


@GCC_SKIP
def test_a_sum_from_another_module(tmp_path):
    (tmp_path / "lib.ht").write_text("type Small is str | none\n")
    (tmp_path / "main.ht").write_text(
        "import 'lib'\n"
        "from 'lib' import Small\n"
        "\n"
        "type A is lib.Small | int\n"            # by its qualified name
        "type B is Small | bool\n"               # ... and by the name imported
        "\n"
        "def int main():\n"
        "    A a = 'text'\n"
        "    B b = none\n"
        "    print(a)\n"
        "    print(b)\n"
        "    b = true\n"
        "    print(b)\n"
        "    return 0\n")
    exe = tmp_path / executable_name("main", default_target())
    build_executable(str(tmp_path / "main.ht"), str(exe))
    assert subprocess.run([str(exe)], capture_output=True, text=True).stdout == "'text'\nnone\ntrue\n"


@pytest.mark.parametrize("declarations,message", [
    ("type A is B | int\ntype B is A | str\n",
     r"Sum type 'A' includes itself as a variant \(through 'B'\)"),
    ("type A is B | int\ntype B is C | str\ntype C is A | bool\n",
     r"Sum type 'A' includes itself as a variant \(through 'B', then 'C'\)"),
    ("type A is A | int\n", "Sum type 'A' includes itself as a variant -- "),
    ("type B is str | none\ntype A is B | B\n", "Sum type 'A' lists 'B' as a variant more than once"),
    ("type A is int | int\n", "Sum type 'A' lists 'int' as a variant more than once"),
    ("type A is Missing | int\n", "'Missing' isn't a declared struct, sum type, `none`, or"),
])
def test_what_is_wrong_with_one(declarations, message):
    assert_program_semantic_error(declarations + MAIN, match=message)


@pytest.mark.parametrize("body,message", [
    # Not yet: a whole sum flowing into a wider one, and a test for a sum.
    ("    StrResult s = 'x'\n    LineResult l = s\n",
     "Cannot initialize 'l' .declared LineResult. with a value of type StrResult"),
    ("    LineResult l = 'x'\n    if l is StrResult:\n        print(1)\n",
     "'StrResult' is not one of LineResult's own declared variants"),
])
def test_what_is_still_to_come(body, message):
    assert_program_semantic_error(
        DECLS + "type LineResult is StrResult | none\ndef int main():\n" + body + "    return 0\n", match=message)
