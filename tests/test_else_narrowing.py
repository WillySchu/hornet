"""What an `is` check rules out: in the `else` branch, and after an `if` whose body always leaves,
the variable isn't the tested variant. With one variant left it has that variant's type."""

import pytest

from tests.test_compiler import (
    GCC_SKIP, _parse, analyze, assert_program_panics, assert_program_semantic_error, assert_program_stdout,
)

MAIN = "def int main():\n    return 0\n"

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
        "    match s:\n"         # (an arm for Circle is not: see the match tests below)
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
    # Assigning to a narrowed variable ends the narrowing.
    ("def int f(Shape s):\n    if s is Circle:\n        return 1\n    s = Square(2)\n    return s.side\n",
     "Cannot access field 'side' on non-struct type Shape"),
    ("def int f(Shape s):\n"
     "    if s is Circle:\n"
     "        return 1\n"
     "    else:\n"
     "        s = Square(2)\n"
     "    return s.side\n",
     "Cannot access field 'side' on non-struct type Shape"),
    # A guard doesn't make room for a second declaration in the same scope.
    ("def int f():\n"
     "    Shape s = Circle(1)\n"
     "    if s is Circle:\n"
     "        return 1\n"
     "    Shape s = Square(2)\n"
     "    return 0\n",
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


# -- a `match` on a variable that checks have narrowed: its arms are for what it can still hold

MATCHED = (
    "type Circle struct:\n"
    "    int radius\n"
    "type Square struct:\n"
    "    int side\n"
    "type Figure is Circle | Square | none\n"
    "type Solid is Circle | Square\n"
)


@GCC_SKIP
def test_a_match_needs_arms_only_for_what_is_left():
    assert_program_stdout(
        MATCHED +
        "def int after_a_guard(Figure f):\n"
        "    if f is none:\n        return 0\n"
        "    match f:\n"                                           # no arm for none: it was ruled out
        "        is Circle:\n            return f.radius\n"
        "        is Square:\n            return f.side\n"
        "def int one_left(Figure f):\n"
        "    if f is Circle:\n"
        "        match f:\n"
        "            is Circle:\n                return f.radius\n"
        "    return 0\n"
        "def int by_a_sum(Figure f):\n"
        "    if f is Solid:\n"                                     # a test for a sum narrows to its variants
        "        match f:\n"
        "            is Circle:\n                return f.radius\n"
        "            is Square:\n                return f.side\n"
        "    return 0\n"
        "def int part_of_an_arm(Figure f):\n"
        "    if f is Circle:\n        return 1\n"
        "    match f:\n"
        "        is Solid:\n            return f.side\n"           # Solid, less the Circle ruled out: a Square
        "        is none:\n            return 0\n"
        "def int with_else(Figure f):\n"
        "    if f is none:\n        return 0\n"
        "    match f:\n"
        "        is Circle:\n            return f.radius\n"
        "        else:\n            return f.side\n"
        "def int main():\n"
        "    []Figure all = []Figure[Circle(2), Square(3), none]\n"
        "    for f in all:\n"
        "        print(format('{} {} {} {} {}', after_a_guard(f), one_left(f), by_a_sum(f), part_of_an_arm(f),\n"
        "                     with_else(f)))\n"
        "    return 0\n",
        "2 2 2 1 2\n3 0 3 3 3\n0 0 0 0 0\n",
    )


@pytest.mark.parametrize("body,message", [
    # An arm for what was ruled out can't be reached.
    ("    if f is none:\n        return 0\n    match f:\n        is Circle:\n            return 1\n"
     "        is Square:\n            return 2\n        is none:\n            return 3\n",
     "'none' is never reached in this match on 'f': checks before the match have ruled it out, so 'f' can only be "
     "Circle or Square here"),
    ("    if f is Solid:\n        return 0\n    match f:\n        is Solid:\n            return 1\n"
     "        is none:\n            return 2\n",
     "'Solid' is never reached in this match on 'f': checks before the match have ruled it out, so 'f' can only be "
     "none here"),
    # What is left still needs its arms.
    ("    if f is none:\n        return 0\n    match f:\n        is Circle:\n            return 1\n",
     r"This match on 'f' \(declared Figure\) doesn't cover every variant it can hold here -- missing: Square "),
    # Assigned since: it may hold any again.
    ("    if f is none:\n        return 0\n    f = none\n    match f:\n        is Circle:\n            return 1\n"
     "        is Square:\n            return 2\n",
     r"This match on 'f' \(declared Figure\) doesn't cover every variant -- missing: none "),
    # Never narrowed: as it always was.
    ("    match f:\n        is Circle:\n            return 1\n        is Square:\n            return 2\n",
     r"doesn't cover every variant -- missing: none \(add an arm for each, or an 'else:' to cover the rest\)"),
])
def test_what_such_a_match_refuses(body, message):
    assert_program_semantic_error(MATCHED + "def int g(Figure f):\n" + body + "    return 9\n" + MAIN, match=message)


def test_a_subject_that_is_no_variable_is_never_narrowed():
    # `match EXPR as NAME` binds a fresh variable: every variant needs its arm, whatever was known of EXPR's parts.
    assert_program_semantic_error(
        MATCHED + "type Holder struct:\n    Figure f\n"
        "def int g(Holder h):\n    if h.f is none:\n        return 0\n"
        "    match h.f as f:\n        is Circle:\n            return 1\n"
        "        is Square:\n            return 2\n" + MAIN,
        match="doesn't cover every variant -- missing: none")


@GCC_SKIP
def test_a_variant_ruled_out_and_put_back_through_a_pointer_is_caught():
    source = (
        MATCHED +
        "def int g(bool change):\n"
        "    Figure f = Circle(2)\n"
        "    *Figure p = &f\n"
        "    if f is none:\n        return 0\n"
        "    if change:\n        *p = none\n"                       # what the match was told it can't be
        "    match f:\n"
        "        is Circle:\n            return f.radius\n"
        "        is Square:\n            return f.side\n")
    assert_program_stdout(source + "def int main():\n    print(g(false))\n    return 0\n", "2\n")
    assert_program_panics(source + "def int main():\n    print(g(true))\n    return 0\n",
                          "'f' changed variant while narrowed")


def test_a_variable_that_can_hold_nothing_is_told_of_all_its_variants():
    # Inside two checks that can't both hold, what the variable "can hold" is nothing: the hint lists none of that.
    assert_program_semantic_error(
        MATCHED + "def int g(Figure f):\n    if f is Circle:\n        if f is Square:\n            return f.side\n"
        "    return 0\n" + MAIN,
        match="Figure is a sum type: test which variant 'f' holds first, as in `if f is Square:`")
