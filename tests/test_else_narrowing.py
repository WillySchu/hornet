"""What an `is` check rules out: in the `else` branch, and after an `if` whose body always leaves,
the variable isn't the tested variant. With one variant left it has that variant's type."""

import pytest

from tests.test_compiler import (
    GCC_SKIP, _parse, analyze, assert_program_semantic_error, assert_program_stdout,
)

DECLS = (
    "type Circle struct:\n"
    "    int radius\n"
    "type Square struct:\n"
    "    int side\n"
    "type Tri struct:\n"
    "    int base\n"
    "type Shape is Circle | Square\n"
    "type Shape3 is Circle | Square | Tri\n"
    "type Maybe is int | none\n"
)


def _checks(body: str) -> None:
    analyze(_parse(DECLS + body))


def _rejected(body: str, match: str) -> None:
    assert_program_semantic_error(DECLS + body, match=match)


@GCC_SKIP
def test_else_and_guard_narrow_to_the_remaining_variant():
    assert_program_stdout(
        DECLS +
        "def int area(Shape s):\n"
        "    if s is Circle:\n"
        "        return 3 * s.radius * s.radius\n"
        "    else:\n"
        "        return s.side * s.side\n"
        "def int side(Shape s):\n"
        "    if s is Circle:\n"
        "        return -1\n"
        "    return s.side\n"
        "def int size(Shape3 s):\n"
        "    if s is Circle:\n"
        "        return s.radius\n"
        "    elif s is Square:\n"
        "        return s.side\n"
        "    else:\n"
        "        return s.base\n"
        "def int base(Shape3 s):\n"
        "    if s is Circle:\n"
        "        return -1\n"
        "    elif s is Square:\n"
        "        return -2\n"
        "    return s.base\n"
        "def int base_by_two_guards(Shape3 s):\n"
        "    if s is Circle:\n"
        "        return -1\n"
        "    if s is Square:\n"
        "        return -2\n"
        "    return s.base\n"
        "def int base_by_match(Shape3 s):\n"
        "    match s:\n"
        "        is Circle:\n"
        "            return -1\n"
        "        is Square:\n"
        "            return -2\n"
        "        else:\n"
        "            return s.base\n"
        "def Shape squared(Shape s):\n"
        "    if s is Circle:\n"
        "        return Square(s.radius)\n"
        "    s.side += 1\n"
        "    return s\n"
        "def int main():\n"
        "    print(area(Circle(2)) + area(Square(3)))\n"
        "    print(side(Circle(2)) + side(Square(3)))\n"
        "    print(size(Circle(1)) + size(Square(20)) + size(Tri(300)))\n"
        "    print(base(Circle(1)) + base(Square(1)) + base(Tri(10)))\n"
        "    print(base_by_two_guards(Circle(1)) + base_by_two_guards(Square(1)) + base_by_two_guards(Tri(10)))\n"
        "    print(base_by_match(Circle(1)) + base_by_match(Square(1)) + base_by_match(Tri(10)))\n"
        "    print(squared(Circle(4)))\n"
        "    print(squared(Square(4)))\n"
        "    return 0\n",
        "21\n2\n321\n7\n7\n7\nSquare(side: 4)\nSquare(side: 5)\n",
    )


@GCC_SKIP
def test_guards_that_break_or_continue():
    assert_program_stdout(
        DECLS +
        "def int sides([]Shape shapes):\n"
        "    int total = 0\n"
        "    for s in shapes:\n"
        "        if s is Circle:\n"
        "            continue\n"
        "        total += s.side\n"
        "    return total\n"
        "def int first_radius([]Shape shapes):\n"
        "    int i = 0\n"
        "    Shape s = Square(0)\n"
        "    while true:\n"
        "        s = shapes[i]\n"  # before the guard, so not yet narrowed
        "        i += 1\n"
        "        if s is Square:\n"
        "            continue\n"
        "        return s.radius\n"
        "def int until_circle([]Shape shapes):\n"
        "    int total = 0\n"
        "    for s in shapes:\n"
        "        if s is Circle:\n"
        "            break\n"
        "        total += s.side\n"
        "    return total\n"
        "def int stepped():\n"
        "    int hits = 0\n"
        "    for Shape s = Circle(1); hits < 3; s = Square(hits):\n"  # the step assigns: the body's narrowing is over
        "        if s is Circle:\n"
        "            hits += 1\n"
        "            continue\n"
        "        hits += s.side + 1\n"
        "    return hits\n"
        "def int main():\n"
        "    []Shape shapes = [Square(1), Square(2), Circle(30), Square(4)]\n"
        "    print(sides(shapes))\n"
        "    print(first_radius(shapes))\n"
        "    print(until_circle(shapes))\n"
        "    print(stepped())\n"
        "    return 0\n",
        "7\n30\n3\n3\n",
    )


@GCC_SKIP
def test_bindings_none_and_checks_of_a_narrowed_variable():
    assert_program_stdout(
        DECLS +
        "def int first([]Shape shapes):\n"
        "    if shapes[0] is Circle as c:\n"
        "        return c.radius\n"
        "    else:\n"
        "        return c.side\n"
        "def int or_zero(Maybe m):\n"
        "    if m is none:\n"
        "        return 0\n"
        "    return m + 1\n"
        "def int describe(Maybe m):\n"
        "    if m is int:\n"
        "        return m\n"
        "    else:\n"
        "        print(m)\n"  # `none` has no value of its own: m is still a Maybe
        "        if m == none:\n"
        "            return -1\n"
        "    return -2\n"
        "def int again(Shape s):\n"
        "    if s is Circle:\n"
        "        return 1\n"
        "    if s is Circle:\n"  # never true here; still allowed
        "        return 2\n"
        "    match s:\n"
        "        is Circle:\n"
        "            return 3\n"
        "        is Square:\n"
        "            return s.side\n"
        "def int main():\n"
        "    print(first([Circle(5)]) + first([Square(6)]))\n"
        "    print(or_zero(none) + or_zero(41))\n"
        "    print(describe(7) + describe(none))\n"
        "    print(again(Circle(9)) + again(Square(9)))\n"
        "    return 0\n",
        "11\n42\nnone\n6\n10\n",
    )


@pytest.mark.parametrize("body,match", [
    # Several variants remain: still the sum.
    ("def int f(Shape3 s):\n    if s is Circle:\n        return 1\n    return s.side\n",
     "Cannot access field 'side' on non-struct type Shape3"),
    ("def int f(Shape3 s):\n    if s is Circle:\n        return 1\n    else:\n        return s.side\n",
     "Cannot access field 'side' on non-struct type Shape3"),
    # The body may fall through: nothing is known afterwards.
    ("def int f(Shape s):\n    if s is Circle:\n        print(1)\n    return s.side\n",
     "Cannot access field 'side' on non-struct type Shape"),
    # Both branches may fall through: nothing is known afterwards.
    ("def int f(Shape s):\n    if s is Circle:\n        print(1)\n    else:\n        print(2)\n    return s.side\n",
     "Cannot access field 'side' on non-struct type Shape"),
    # The rest of the block, and no further.
    ("def int f(Shape s, bool b):\n    if b:\n        if s is Circle:\n            return 1\n        print(s.side)\n"
     "    return s.side\n",
     "Cannot access field 'side' on non-struct type Shape"),
    # An `as NAME` binding ends with its `if`.
    ("def int f([]Shape shapes):\n    if shapes[0] is Circle as c:\n        return 1\n    return c.side\n",
     "Reference to undeclared variable 'c'"),
    # A narrowed variable can't be reassigned, whether or not one variant is left.
    ("def int f(Shape s):\n    if s is Circle:\n        return 1\n    s = Circle(2)\n    return 0\n",
     "Cannot reassign 's'"),
    ("def int f(Shape s):\n    if s is Circle:\n        return 1\n    else:\n        s = Circle(2)\n    return 0\n",
     "Cannot reassign 's'"),
    ("def int f(Shape3 s):\n    if s is Circle:\n        return 1\n    s = Circle(2)\n    return 0\n",
     "Cannot reassign 's'"),
    ("def int f([]Shape shapes):\n    Shape s = shapes[0]\n    int i = 0\n    while true:\n        if s is Circle:\n"
     "            break\n        i += 1\n        s = shapes[i]\n    return i\n",
     "Cannot reassign 's'"),
    # A guard doesn't make room for a second declaration in the same scope.
    ("def int f():\n    Shape s = Circle(1)\n    if s is Circle:\n        return 1\n    Shape s = Square(2)\n    return 0\n",
     "Variable 's' is already declared in this scope"),
])
def test_what_stays_an_error(body, match):
    _rejected(body, match)


def test_a_new_variable_of_the_same_name_is_not_narrowed():
    _checks(
        "def int f(Shape s):\n"
        "    if true:\n"
        "        if s is Circle:\n"
        "            return 1\n"
        "        Shape s = Circle(7)\n"  # shadows the parameter in this block
        "        s = Square(2)\n"
        "        if s is Square:\n"
        "            return s.side\n"
        "    return 0\n"
    )
    _checks(
        "def int g(Shape s):\n"
        "    if s is Circle:\n"
        "        for int i = 0; i < 1; i += 1:\n"
        "            Shape s = Square(1)\n"
        "            s = Circle(2)\n"
        "    return 0\n"
    )
