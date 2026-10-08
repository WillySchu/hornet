"""An array literal without a type (`[1, 2, 3]`) takes one from where it is used: the declared type
it flows into, or the other operand of `in`, `==`, or `!=`. Anywhere else its type must be written
(`[3]int[1, 2, 3]`), and the error says so."""

import re

import pytest

from tests.test_compiler import GCC_SKIP, assert_program_semantic_error, assert_program_stdout

DECLS = (
    "type Color enum:\n"
    "    Red\n"
    "    Green\n"
    "type Point struct:\n"
    "    int x\n"
    "    [2]int pair\n"
    "type Either is [2]int | []int | str\n"
)


@GCC_SKIP
def test_where_a_declared_type_gives_the_literal_its_own():
    assert_program_stdout(
        DECLS +
        "def int total([3]int xs):\n"
        "    return xs[0] + xs[1] + xs[2]\n"
        "def [2]int8 small():\n"
        "    return [1, 2]\n"                                  # a return type
        "def int main():\n"
        "    [3]int a = [1, 2, 3]\n"                            # a variable's type
        "    []int s = [4, 5]\n"                                # ... a slice's too
        "    a = [7, 8, 9]\n"                                   # an assignment's target
        "    [2][2]int grid = [[1, 2], [3, 4]]\n"               # the inner ones, from the outer
        "    Point p = Point(1, [5, 6])\n"                      # a field, by position
        "    Point q = Point(x=2, pair=[7, 8])\n"               # ... and by name
        "    dict[str][2]int named = dict[str][2]int{'k': [1, 2]}\n"
        "    [][2]int rows = [][2]int[[1, 2]]\n"
        "    rows = append(rows, [3, 4])\n"                     # an element
        "    grid[1] = [9, 9]\n"
        "    print(total([1, 2, 3]) + total(a))\n"              # a parameter
        "    print(small())\n"
        "    print(format('{} {} {} {} {}', s, grid, p.pair, q.pair, named['k']))\n"
        "    print(rows)\n"
        "    return 0\n",
        "30\n[2]int8[1, 2]\n[]int[4, 5] [2][2]int[[2]int[1, 2], [2]int[9, 9]] [2]int[5, 6] [2]int[7, 8] [2]int[1, 2]\n"
        "[][2]int[[2]int[1, 2], [2]int[3, 4]]\n",
    )


@GCC_SKIP
def test_where_the_other_operand_gives_it():
    assert_program_stdout(
        DECLS +
        "def int main():\n"
        "    int n = 2\n"
        "    int8 small = 3\n"
        "    Color c = Color.Green\n"
        "    str word = 'or'\n"
        "    [2]int pair = [1, 2]\n"
        "    print(n in [1, 2, 3])\n"                           # the elements take the left side's type
        "    print(small in [1, 3])\n"
        "    print(c in [Color.Red])\n"
        "    print(word in ['and', 'or', 'not'])\n"
        "    print(\"q\" in [\"a\", \"b\"])\n"
        "    print(7 in [n, n + 5])\n"
        "    print(pair == [1, 2])\n"                           # ... or the other array's
        "    print([1, 2] == pair)\n"
        "    print(pair != [1, 3])\n"
        "    print([2]int[1, 2] == [1, 2])\n"
        "    return 0\n",
        "true\ntrue\nfalse\ntrue\nfalse\ntrue\ntrue\ntrue\ntrue\ntrue\n",
    )


@GCC_SKIP
def test_in_when_one_side_is_only_literals():
    assert_program_stdout(
        DECLS +
        "def int main():\n"
        "    int8 a = 1\n"
        "    int8 b = 2\n"
        "    byte c = \"x\"\n"
        "    [][2]int rows = [][2]int[[1, 2], [3, 4]]\n"
        "    [2][2]int grid = [[1, 2], [3, 4]]\n"
        "    print(1 in [a, b])\n"                    # an integer literal has no type either: the elements say
        "    print(3 in [a, b])\n"
        "    print(120 in [c])\n"
        "    print(5 in [1, a + 4, 3])\n"             # ... the first of them that isn't a literal itself
        "    print(2 in [1, 2, 3])\n"                 # ... or int, if none does
        "    print([3, 4] in rows)\n"                 # a literal on the left is one of the right side's elements
        "    print([9, 9] in rows)\n"
        "    print([1, 2] in grid)\n"
        "    return 0\n",
        "true\nfalse\ntrue\ntrue\ntrue\ntrue\nfalse\ntrue\n",
    )


WRITE = "This array literal has nothing to take its type from -- write the type before it, as in "


@pytest.mark.parametrize("statement,written", [
    ("print([1, 2, 3])", "[3]int"),
    ("print(format('{}', [1, 2]))", "[2]int"),
    ("for x in [1, 2, 3]:\n        print(x)", "[3]int"),
    ("print(len(['a', 'b']))", "[2]str"),
    ("print([10, 20, 30][n])", "[3]int"),
    ("print(len([1, 2, 3][0:n]))", "[3]int"),
    ("print([1, 2] == [1, 2])", "[2]int"),                      # neither side says
    ("print([1] in [[1], [2]])", "[1]int"),
    ("print([n][0] * n)", "[1]int"),                            # an indexed literal, multiplied
    ("Either e = [1, 2]", "[2]int"),                            # an array, or a slice? a sum doesn't say
    ("take([1, 2])", "[2]int"),
    ("print([Point(1, pair)])", "[1]Point"),
    ("print([Color.Red, Color.Green])", "[2]Color"),
    ("print([[1, 2], [3, 4]])", "[2]T"),
    ("print([1, 2] == n)", "[2]int"),                           # compared with what isn't an array
])
def test_where_nothing_does_the_type_must_be_written(statement, written):
    assert_program_semantic_error(
        DECLS +
        "def take(Either e):\n    return\n"
        "def int main():\n"
        "    int n = 1\n"
        "    [2]int pair = [1, 2]\n"
        f"    {statement}\n"
        "    return 0\n",
        match=re.escape(WRITE + f"`{written}[...]`"))


@GCC_SKIP
def test_with_its_type_written_each_of_those_works():
    assert_program_stdout(
        DECLS +
        "def int main():\n"
        "    int n = 1\n"
        "    print([3]int[1, 2, 3])\n"
        "    for x in [2]int[7, 8]:\n"
        "        print(x)\n"
        "    print(len([2]str['a', 'b']) + [3]int[10, 20, 30][n])\n"
        "    print([2]int[1, 2] == [2]int[1, 2])\n"
        "    Either e = [2]int[1, 2]\n"
        "    print(e)\n"
        "    return 0\n",
        "[3]int[1, 2, 3]\n7\n8\n22\ntrue\n[2]int[1, 2]\n",
    )


def test_an_empty_literal_still_needs_elements_or_a_slice_to_be():
    assert_program_semantic_error("def int main():\n    print([])\n    return 0\n",
                                  match="Array literals must have at least one element")
    assert_program_stdout("def int main():\n    []int none_yet = []\n    print(len(none_yet))\n    return 0\n", "0\n")
