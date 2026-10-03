"""Enums: `type Name enum:` declares a type whose values are its members, written `Name.Member`.
They compare with `==`, key dicts, print as written, convert to an integer (a member's position),
and are tested with `is Member` in `if` and exhaustive `match`."""

import pytest

from build import build_executable
from ir.ir import IRConst, Temp
from ir.program_builder import build_ir_program
from lexer import Lexer
from parser import ParseError, Parser
from target import default_target
from tests.targets import run_binary
from tests.test_compiler import (
    GCC_SKIP, _parse, analyze, assert_program_semantic_error, assert_program_stdout,
)
from typed_ast import dump
from typesys import Type, TypeKind

DECLS = (
    "type Color enum:\n"
    "    Red\n"
    "    Green\n"
    "    Blue\n"
    "type Shade enum:\n"
    "    Dark\n"
    "type Pixel struct:\n"
    "    Color c\n"
    "    int n\n"
    "type Maybe is Color | none\n"
)


def test_an_enum_declaration_lists_one_member_per_line():
    program = Parser(Lexer("type Color enum:\n    Red\n\n    Green\n").tokenize()).parse_program()
    assert [(e.name, [m.name for m in e.members]) for e in program.enums] == [("Color", ["Red", "Green"])]
    for source in ("type E enum:\n    A B\n", "type E enum:\n    A, B\n", "type E enum\n", "type E enum:\n    1\n",
                   "type E enum:\n"):
        with pytest.raises(ParseError):
            Parser(Lexer(source).tokenize()).parse_program()


@GCC_SKIP
def test_enum_values():
    assert_program_stdout(
        DECLS +
        "def str name(Color c):\n"
        "    match c:\n"
        "        is Red:\n"
        "            return 'red'\n"
        "        is Color.Green:\n"
        "            return 'green'\n"
        "        is Blue:\n"
        "            return 'blue'\n"
        "def Color next(Color c):\n"
        "    if c is Red:\n"
        "        return Color.Green\n"
        "    elif c == Color.Green:\n"
        "        return Color.Blue\n"
        "    return Color.Red\n"
        "def int rank(Maybe m):\n"
        "    if m is none:\n"
        "        return -1\n"
        "    if m is Blue:\n"  # m is a Color here
        "        return 30\n"
        "    match m:\n"
        "        is Red:\n"
        "            return 10\n"
        "        else:\n"
        "            return 20\n"
        "def int main():\n"
        "    Color c = Color.Green\n"
        "    Color zero\n"
        "    print(c)\n"
        "    print(zero)\n"                            # the first member
        "    print(name(c) + name(next(c)) + name(next(next(c))))\n"
        "    print(c == Color.Green)\n"
        "    print(c != next(c))\n"
        "    print(int(Color.Blue) + int(c) + 10)\n"
        "    print(byte(c))\n"
        "    print(Pixel(Color.Blue, 3))\n"
        "    print(Pixel(Color.Blue, 3) == Pixel(Color.Blue, 3))\n"
        "    []Color cs = [Color.Red, Color.Blue]\n"
        "    print(cs)\n"
        "    print(c in cs)\n"
        "    print(Color.Blue in cs)\n"
        "    dict[Color]str names = dict[Color]str{Color.Red: 'r', Color.Green: 'g'}\n"
        "    names[Color.Blue] = 'b'\n"
        "    print(names[c])\n"
        "    print(len(names))\n"
        "    print(Color.Red in names)\n"
        "    del(names, Color.Red)\n"
        "    print(Color.Red in names)\n"
        "    for k in cs:\n"
        "        if k is Blue:\n"
        "            print(k)\n"
        "    match next(c) as n:\n"
        "        is Red:\n"
        "            print('red')\n"
        "        else:\n"
        "            print(n)\n"
        "    print(rank(none) + rank(Color.Red) + rank(Color.Green) + rank(Color.Blue))\n"
        "    Maybe m = c\n"
        "    print(m)\n"
        "    return 0\n",
        "Color.Green\nColor.Red\ngreenbluered\ntrue\ntrue\n13\n1\nPixel(c: Color.Blue, n: 3)\ntrue\n"
        "[]Color[Color.Red, Color.Blue]\nfalse\ntrue\ng\n3\ntrue\nfalse\nColor.Blue\nColor.Blue\n59\nColor.Green\n",
    )


MAIN = "def int main():\n    Color c = Color.Red\n"


@pytest.mark.parametrize("before,statement,match", [
    ("", "c = Color.Purple", r"Enum 'Color' has no member 'Purple' \(its members: Red, Green, Blue\)"),
    ("", "if c is Purple:\n        return 1", "'Purple' is not a member of Color"),
    ("", "if c is Shade.Dark:\n        return 1", "'Shade.Dark' is not a member of Color"),
    ("", "match c:\n        is Red:\n            return 1\n        is Green:\n            return 2",
     "doesn't cover every member -- missing: Blue"),
    ("", "match c:\n        is Red:\n            return 1\n        is Color.Red:\n            return 2\n"
         "        else:\n            return 3", "'Red' is tested more than once"),
    # A distinct type: no integers in, no ordering or arithmetic, no other enum.
    ("", "c = 1", "Cannot assign a value of type int"),
    ("", "bool b = c == 1", "Cannot compare Color to int"),
    ("", "bool b = c == Shade.Dark", "Cannot compare Color to Shade"),
    ("", "bool b = c < Color.Blue", "requires two operands of the same integer type"),
    ("", "Color d = c + c", "requires two operands of the same integer type"),
    ("", "c = Color(1)", "would convert a value to the enum Color, which isn't supported yet"),
    ("", "str s = str(c)", r"str\(...\) takes a byte or \[\]byte, got Color"),
    # The enum's name is a type, and a member is not one.
    ("", "print(Color)", "'Color' is an enum, not a value"),
    ("", "Color.Red r = c", "'Color.Red' is an enum's member, not a type"),
    ("", "print(Color.Red.x)", "Cannot access field 'x' on non-struct type Color"),
    ("", "dict[Color]int d = dict[Color]int{Color.Red: 1, Color.Red: 2}", "lists the key 'Color.Red' more than once"),
    ("const Color DEFAULT = Color.Red\n", "return 0", "constants must be an integer type, bool, or str"),
    ("type E enum:\n    A\n    A\n", "return 0", "Member 'A' is already declared in enum 'E'"),
    ("type Pixel enum:\n    A\n", "return 0", "Enum 'Pixel' collides with a struct of the same name"),
    ("type print enum:\n    A\n", "return 0", "'print' is a builtin and can't be used as an enum name"),
])
def test_what_is_an_error(before, statement, match):
    assert_program_semantic_error(DECLS + before + MAIN + f"    {statement}\n    return 0\n", match=match)


def test_a_variable_named_like_an_enum_is_the_variable():
    analyze(_parse(DECLS + "def int main():\n    Pixel Color = Pixel(Color.Blue, 1)\n    return Color.n\n"))


def test_the_typed_tree_and_the_ir():
    program = analyze(_parse(
        DECLS + "def int f(Color c):\n    match c:\n        is Red:\n            return 1\n        else:\n"
        "            return int(c)\ndef int main():\n    return f(Color.Blue)\n"))
    tree = dump(program)
    assert "EnumMember name=Blue index=2 : Color" in tree and "Match" not in tree  # a chain of `if`s
    # The IR knows an enum value only as an int32.
    for fn in build_ir_program(program).functions:
        assert fn.return_type.kind != TypeKind.ENUM
        for instr in fn.body:
            for value in vars(instr).values():
                for item in value if isinstance(value, list) else [value]:
                    if isinstance(item, (Temp, IRConst)):
                        assert item.type.kind != TypeKind.ENUM
                    assert not (isinstance(item, Type) and item.kind == TypeKind.ENUM)


@GCC_SKIP
def test_enums_across_modules(tmp_path):
    (tmp_path / "palette.ht").write_text(
        "type Color enum:\n    Red\n    Green\n    Blue\n\n"
        "type _Hidden enum:\n    A\n\n"
        "def Color favourite():\n    return Color.Blue\n")
    (tmp_path / "main.ht").write_text(
        "import 'palette'\nfrom 'palette' import Color\n\n"
        "def int main():\n"
        "    palette.Color d = palette.Color.Red\n"
        "    print(d)\n"
        "    print(palette.favourite() == Color.Blue)\n"
        "    dict[palette.Color]int stock\n"
        "    stock[Color.Red] = 3\n"
        "    stock[palette.Color.Red] += 1\n"
        "    print(stock)\n"
        "    match palette.favourite() as f:\n"
        "        is Color.Blue:\n"
        "            return int(f)\n"
        "        else:\n"
        "            return 0\n")
    build_executable(str(tmp_path / "main.ht"), str(tmp_path / "out"))
    result = run_binary(default_target(), [tmp_path / "out"], capture_output=True, text=True)
    assert (result.returncode, result.stdout) == (2, "Color.Red\ntrue\ndict[Color]int{Color.Red: 4}\n")
    (tmp_path / "hidden.ht").write_text("import 'palette'\n\ndef int main():\n    print(palette._Hidden.A)\n    return 0\n")
    with pytest.raises(Exception, match="not visible outside the module"):
        build_executable(str(tmp_path / "hidden.ht"), str(tmp_path / "out2"))
