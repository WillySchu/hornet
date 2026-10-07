"""`v += 1` on a variable an `is` check has narrowed: it reads the variant v holds, and gives v the
same variant back, so the narrowing still stands. (`v = value` replaces v whole and ends it.)"""

import pytest

from tests.test_compiler import (
    GCC_SKIP, _parse, analyze, assert_program_panics, assert_program_semantic_error, assert_program_stdout)
from typed_ast import dump

DECLS = (
    "type Point struct:\n"
    "    int x\n"
    "type V is int | str | Point | none\n"
)


@GCC_SKIP
def test_each_kind_of_variant():
    assert_program_stdout(
        DECLS +
        "def int bump(V start):\n"
        "    V v = start\n"
        "    if v is int:\n"
        "        v += 1\n"
        "        v *= 10\n"                    # still an int: each one keeps the narrowing
        "        v -= 3\n"
        "        v <<= 1\n"
        "        return v\n"
        "    return -1\n"
        "def int main():\n"
        "    V n = 1\n"
        "    if n is int:\n"
        "        n += 1\n"
        "    print(n)\n"
        "    V s = 'ab'\n"
        "    if s is str:\n"
        "        s += 'cd'\n"
        "        s += s\n"
        "        print(s)\n"
        "    V p = Point(5)\n"
        "    if p is Point:\n"
        "        p.x += 2\n"                   # a field of the narrowed variable
        "        p.x *= 3\n"
        "    print(p)\n"
        "    print(bump(4))\n"
        "    print(bump('x'))\n"
        "    return 0\n",
        "2\nabcdabcd\nPoint(x: 21)\n94\n-1\n",
    )


@GCC_SKIP
def test_in_loops_matches_and_narrowing_by_elimination():
    assert_program_stdout(
        DECLS + "type Maybe is int | none\n"
        "def int main():\n"
        "    V n = 0\n"
        "    int turns = 0\n"
        "    while n is int and n < 5:\n"       # the loop's own condition narrows its body
        "        n += 2\n"
        "        turns += 1\n"
        "    print(n)\n"
        "    print(turns)\n"
        "    V m = 10\n"
        "    match m:\n"
        "        is int:\n"
        "            m += 5\n"
        "            print(m + 1)\n"
        "        else:\n"
        "            print(0)\n"
        "    print(m)\n"
        "    Maybe maybe = 3\n"
        "    if maybe is none:\n"
        "        return 1\n"
        "    maybe += 4\n"                      # an int here: the other branch left
        "    print(maybe)\n"
        "    return 0\n",
        "6\n3\n16\n15\n7\n",
    )


def test_the_target_is_the_variant_the_variable_holds():
    tree = dump(analyze(_parse(
        DECLS + "def int main():\n    V v = 1\n    if v is int:\n        v += 1\n    return 0\n")))
    assert ("      CompoundAssign op=ADD\n"
            "        target:\n"
            "          Payload : int\n"
            "            sum:\n"
            "              Local symbol=v#0 : V\n"
            "        value:\n"
            "          IntLit value=1 : int\n") in tree


@GCC_SKIP
def test_a_variant_changed_through_a_pointer_is_caught_here_too():
    assert_program_panics(
        DECLS +
        "def int main():\n"
        "    V v = 7\n"
        "    *V alias = &v\n"
        "    if v is int:\n"
        "        v += 1\n"
        "        print(v)\n"
        "        *alias = 'a str now'\n"
        "        v += 1\n"
        "    return 0\n",
        "'v' changed variant while narrowed",
        expected_stdout="8\n",
    )


@pytest.mark.parametrize("statement,match", [
    ("v += 1", "requires two operands of the same integer type"),              # not narrowed: a V
    ("if v is str:\n        v += 1", "requires two operands of the same integer type"),
    ("if v is Point:\n        v += 1", "requires two operands of the same integer type"),
    ("if v is int:\n        v += 'x'", "requires two operands of the same integer type"),
])
def test_what_is_still_rejected(statement, match):
    assert_program_semantic_error(DECLS + f"def int main():\n    V v = 1\n    {statement}\n    return 0\n", match=match)
