"""`format(template, values...)`: a str, the template with each `{}` replaced by the next value as
print shows it. The template is known when compiling, so it is checked against the values."""

import pytest

from tests.test_compiler import (
    GCC_SKIP, _parse, analyze, assert_program_semantic_error, assert_program_stdout, compile_and_run)
from typed_ast import dump, format_pieces

DECLS = (
    "type Color enum:\n"
    "    Red\n"
    "    Green\n"
    "type Point struct:\n"
    "    int x\n"
    "    str name\n"
    "type Shape is Point | int | none\n"
    "const str TEMPLATE = '<{}>'\n"
    "const str JOINED = '{}' + ' and {}'\n"
)

# Expressions of each kind of type.
VALUES = [
    "n", "-n * 1000000000000", "int32(n) - 50", "int8(n) - 50", "byte(n) + \"a\"", "n > 3", "not n > 3",
    "'text'", "''", "'with \\'quotes\\' and {braces}'", "Color.Green", "Point(n, 'p')",
    "[3]int[1, 2, 3]", "[]str['a', 'b']", "dict[str]int{'k': n}", "shape", "nothing", "[]Point[Point(1, 'a')]",
    "[2][2]bool[[true, false], [false, true]]",
]


@GCC_SKIP
def test_a_value_appears_as_print_shows_it():
    body = "".join(f"    print({value})\n    print(format('{{}}', {value}))\n" for value in VALUES)
    result = compile_and_run(
        DECLS + "def int main():\n    int n = 7\n    Shape shape = Point(n, 'in a sum')\n    Shape nothing = none\n"
        + body + "    return 0\n")
    lines = result.stdout.split("\n")[:-1]
    assert result.returncode == 0 and len(lines) == 2 * len(VALUES)
    assert lines[0::2] == lines[1::2]
    assert lines[14] == "text" and lines[20] == "Color.Green" and lines[22] == "Point(x: 7, name: 'p')"


@GCC_SKIP
def test_text_and_values_together():
    assert_program_stdout(
        DECLS +
        "def str describe(Point p):\n"
        "    return format('{} at {}', p.name, p.x)\n"
        "def int main():\n"
        "    int n = 3\n"
        "    str who = 'world'\n"
        "    print(format('hello {}', who))\n"
        "    print(format('{} + {} = {}', n, 1, n + 1))\n"
        "    print(format('{}{}{}', n, who, n))\n"                       # nothing between them
        "    print(format('{{}} is a brace pair, {{{}}} wraps {}', n, who))\n"
        "    print(format('100%, a \\'quote\\', a tab\\there'))\n"       # no placeholders: the text itself
        "    print(format(''))\n"
        "    print(format(TEMPLATE, n) + format(JOINED, who, true))\n"   # a constant template
        "    print(describe(Point(4, 'origin')))\n"
        "    print(len(format('{}', who)) + len(format('{}{}', who, who)))\n"
        "    print(format('{}', format('[{}]', n)) == '[3]')\n"          # nested, and compared
        "    dict[str]int seen\n"
        "    seen[format('key{}', n)] = n\n"
        "    print('key3' in seen)\n"
        "    print(format('{}!', who)[1:4])\n"
        "    return 0\n",
        "hello world\n3 + 1 = 4\n3world3\n{} is a brace pair, {3} wraps world\n"
        "100%, a 'quote', a tab\there\n\n<3>world and true\norigin at 4\n15\ntrue\ntrue\norl\n",
    )


@GCC_SKIP
def test_values_are_evaluated_once_from_left_to_right_and_each_result_is_its_own():
    assert_program_stdout(
        "def int noisy(int n):\n"
        "    print(n)\n"
        "    return n * 10\n"
        "def int main():\n"
        "    print(format('{} {} {}', noisy(1), noisy(2), noisy(3)))\n"
        "    []str made\n"
        "    for int i = 0; i < 3; i += 1:\n"
        "        made = append(made, format('item {}', i))\n"
        "    print(made)\n"
        "    str long = 'x'\n"
        "    for int i = 0; i < 10; i += 1:\n"
        "        long = format('{}{}', long, long)\n"       # far past the buffer's first size
        "    print(len(long))\n"
        "    print(long[1000:1003])\n"
        "    return 0\n",
        "1\n2\n3\n10 20 30\n[]str['item 0', 'item 1', 'item 2']\n1024\nxxx\n",
    )


@pytest.mark.parametrize("call,match", [
    ("format()", "'format' expects a template, then a value for each '{}' in it"),
    ("format(5)", "'format' expects a str template first, got int"),
    ("format(template, 1)", "needs its template as a string literal or a constant"),
    ("format('{} and {}', 1)", r"has 2 '{}' placeholders, but 1 value was given"),
    ("format('{}', 1, 2)", r"has 1 '{}' placeholder, but 2 values were given"),
    ("format('none', 1)", r"has 0 '{}' placeholders, but 1 value was given"),
    ("format('{0}', 1)", "a placeholder is written '{}', with nothing inside"),
    ("format('{name}', 1)", "a placeholder is written '{}', with nothing inside"),
    ("format('{:>5}', 1)", "a placeholder is written '{}', with nothing inside"),
    ("format('open {', 1)", "a placeholder is written '{}', with nothing inside"),
    ("format('close }', 1)", "a '}' with no '{' before it"),
    ("format('{}', nothing())", "cannot show the result of a function that has no declared return type"),
    ("format('{}', missing)", "undeclared variable 'missing'"),
])
def test_what_the_compiler_rejects(call, match):
    assert_program_semantic_error(
        "def nothing():\n    return\n"
        f"def int main():\n    str template = '{{}}'\n    print({call})\n    return 0\n", match=match)


def test_the_template_and_the_typed_tree():
    assert format_pieces("a{}b{{c}}{}") == ["a", "b{c}", ""] and format_pieces("") == [""]
    tree = dump(analyze(_parse("def str f(int n):\n    return format('n = {}\\n', n)\n"
                              "def str g():\n    return format('{{plain}}')\ndef int main():\n    return 0\n")))
    assert "Format template='n = {}\\n' : str\n" in tree
    assert "StrLit value='{plain}' : str\n" in tree  # nothing to fill in
