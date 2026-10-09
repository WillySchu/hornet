"""`==` and `!=` on sum types: equal is holding the same variant, with equal payloads. The two sides
are compared as one sum, the wider: the other may be a narrower sum, or a value of one of its
variants. A sum has `==` when each of its variants does."""

import pytest

from tests.test_compiler import GCC_SKIP, assert_program_semantic_error, assert_program_stdout

DECLS = (
    "type Error struct:\n"
    "    str message\n"
    "type Point struct:\n"
    "    int x\n"
    "    int y\n"
    "type StrResult is str | Error\n"
    "type IntResult is int | Error\n"
    "type LineResult is StrResult | none\n"
    "type Shape is Point | int | [2]int | *Point | none\n"
    "type Holder struct:\n"
    "    Shape shape\n"
    "    str label\n"
    "type Small is int8 | str\n"
    "type Ints is int8 | int32\n"
    "type Listed is []int | none\n"
    "type Keyed is dict[str]int | int\n"
)


def _main(body: str) -> str:
    return DECLS + "def int main():\n" + "".join(f"    {line}\n" for line in body.split("\n")) + "    return 0\n"


@GCC_SKIP
def test_what_is_equal():
    checks = [
        ("StrResult a = 'same'\nStrResult b = 'same'\nStrResult c = Error('same')", None),
        ("a == b", True),                          # the same variant, equal payloads
        ("a == c", False),                         # different variants
        ("a != c", True),
        ("c == Error('same')", True),              # a sum and a value of one of its variants
        ("'same' == a", True),                     # ... on either side
        ("a != 'other'", True),
        ("LineResult l = 'same'", None),
        ("l == a", True),                          # a sum and a wider one
        ("a == l", True),
        ("LineResult gone", None),
        ("gone == a", False),
        ("gone == none", True),
        ("Point p = Point(1, 2)\nShape s = Point(1, 2)", None),
        ("s == p", True),
        ("s == Point(1, 3)", False),
        ("s == 5", False),                         # an integer literal: the sum's int
        ("s = 5\nShape t = 5", None),
        ("s == t", True),                          # though s still has a Point's bytes past its int
        ("s = [2]int[3, 4]\nt = [2]int[3, 4]", None),
        ("s == t", True),
        ("s = &p\nt = &p", None),
        ("s == t", True),                          # pointers: the same address
        ("Point q = Point(1, 2)\nt = &q", None),
        ("s == t", False),                         # ... not what is at it
        ("Holder h = Holder(Point(1, 2), 'h')", None),
        ("h == Holder(p, 'h')", True),             # a struct with a sum field
        ("h == Holder(none, 'h')", False),
        ("[2]Shape pair = [5, none]", None),
        ("pair == [2]Shape[5, none]", True),       # an array of sums
        ("pair != [2]Shape[5, 6]", True),
        ("[]Shape many = []Shape[1, none, p]", None),
        ("p in many", True),                       # ... and `in`
        ("none in many", True),
        ("7 in many", False),
        ("Small m = int8(7)", None),
        ("m == 7", True),                          # no int among its variants: its one integer variant
        ("m == 'seven'", False),
        ("if a is str:\n    print(a == b)\n    print(a == l)", [True, True]),    # a narrowed side is its variant
    ]
    body, expected = [], []
    for source, result in checks:
        if result is None:
            body.append(source)
        elif isinstance(result, list):
            body.append(source)
            expected += result
        else:
            body.append(f"print({source})")
            expected.append(result)
    assert_program_stdout(_main("\n".join(body)), "".join(f"{str(r).lower()}\n" for r in expected))


@GCC_SKIP
def test_each_side_is_evaluated_once_and_in_order():
    assert_program_stdout(
        DECLS +
        "def StrResult say(str what):\n"
        "    print(what)\n"
        "    return what\n"
        "def int main():\n"
        "    print(say('left') == say('right'))\n"
        "    print(say('same') != 'same')\n"
        "    return 0\n",
        "left\nright\nfalse\nsame\nfalse\n",
    )


@GCC_SKIP
def test_only_none_narrows():
    assert_program_stdout(_main("Shape s = 5\nif s != none:\n    print(s == 5)\nShape t\nprint(t == none)"),
                          "true\ntrue\n")
    # `s == 5` being true says s is an int, but it stays a Shape: the value is known already.
    assert_program_semantic_error(_main("Shape s = 5\nif s == 5:\n    int n = s"),
                                  match="Cannot initialize 'n' .declared int. with a value of type Shape")


@pytest.mark.parametrize("body,message", [
    ("StrResult a = 'x'\nIntResult b = 1\nprint(a == b)",
     "Cannot compare StrResult to IntResult with '==' -- neither has all the other's variants -- a StrResult can "
     "hold str, which IntResult has no variant for"),
    ("StrResult a = 'x'\nprint(a == 5)",
     r"Cannot compare StrResult to int with '==' -- int is not one of StrResult's variants \(str, Error\)"),
    ("StrResult a = 'x'\nprint(Point(1, 2) != a)",
     "Cannot compare Point to StrResult with '!=' -- Point is not one of StrResult's variants"),
    ("Ints n = int8(1)\nprint(n == 1)",
     r"Ints has more than one integer variant \(int8, int32\) and none is int -- say which this is, as in "
     r"`int8\(1\)`"),
    ("Small m = int8(7)\nprint(m == 300)", "300 is out of range for int8"),
    ("Listed a\nListed b\nprint(a == b)",
     r"'==' does not support Listed operands -- equality isn't defined yet for a variant that is \(or contains\) "
     r"a slice \(\[\]int\)"),
    ("Keyed a = 1\nprint(a == 1)", r"a variant that is \(or contains\) a dict \(dict\[str\]int\)"),
    ("[]Listed many\nListed one\nprint(one in many)", "'in' does not support an element type of Listed"),
])
def test_what_cannot_be_compared(body, message):
    assert_program_semantic_error(_main(body), match=message)


def test_a_struct_holding_a_sum_that_has_no_equality():
    assert_program_semantic_error(
        DECLS + "type Box struct:\n    Listed items\n" +
        "def int main():\n    Box a\n    Box b\n    print(a == b)\n    return 0\n",
        match=r"struct equality isn't defined yet when a field \(directly, or nested inside another struct, an "
              r"array field, or a sum's variant\) is a slice or a dict")
