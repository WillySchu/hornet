"""An integer literal (or its negation) beside an operand of a narrower integer type takes that
type, range-checked: `fd < 0`, `1 + a`, `a += 1`, `1 in xs`, `a in [1, 2]`."""

import pytest

from tests.test_compiler import GCC_SKIP, _parse, analyze, assert_program_semantic_error, assert_program_stdout
from typed_ast import dump

SETUP = (
    "const int BIG = 10\n"
    "def int main():\n"
    "    int8 a = 127\n"
    "    int32 fd = -1\n"
    "    byte b = \"a\"\n"
    "    int n = 7\n"
    "    []int8 xs = [1, 2]\n"
    "    dict[int8]int d\n"
    "    d[3] = 30\n"
)


@GCC_SKIP
def test_literal_operands_take_the_other_operands_type():
    lines = [
        ("print(a + 1)", "-128"),  # int8 arithmetic: wraps
        ("print(1 + a)", "-128"),
        ("print(fd < 0)", "true"),
        ("print(fd == -1)", "true"),
        ("print(0 > fd)", "true"),
        ("print(b == 97)", "true"),
        ("print(b + 1)", "98"),
        ("print(b * 2)", "194"),
        ("print(a >> 1)", "63"),
        ("print(a & 15)", "15"),
        ("print(b << 1)", "194"),
        ("print(-fd + 1)", "2"),
        ("print(fd * 2 + 10)", "8"),
        ("print(1 in xs)", "true"),
        ("print(5 in xs)", "false"),
        ("print(a in [127, 2])", "true"),
        ("print(a in [-1, 2])", "false"),
        ("print(3 in d)", "true"),
        ("a += 1", None),
        ("print(a)", "-128"),
        ("a /= 2", None),
        ("print(a)", "-64"),
        ("b -= 97", None),
        ("print(b)", "0"),
        ("fd <<= 3", None),
        ("print(fd)", "-8"),
    ]
    assert_program_stdout(
        SETUP + "".join(f"    {statement}\n" for statement, _ in lines) + "    return 0\n",
        "".join(f"{output}\n" for _, output in lines if output is not None),
    )


@pytest.mark.parametrize("statement,match", [
    # The literal must fit the other operand's type.
    ("int8 r = a + 200", r"200 is out of range for int8 \(-128 to 127\)"),
    ("bool r = b == 300", r"300 is out of range for uint8 \(0 to 255\)"),
    ("bool r = b > -1", r"-1 is out of range for uint8 \(0 to 255\)"),
    ("a += 200", r"200 is out of range for int8"),
    ("bool r = 300 in xs", r"300 is out of range for int8"),
    ("bool r = a in [1, 200]", r"200 is out of range for int8"),
    # Only a literal adapts: not a constant, a byte literal, an expression of literals, or a variable.
    ("bool r = fd < BIG", "requires two operands of the same integer type .*got int32 and int"),
    ("bool r = n == \"a\"", "Cannot compare int to uint8"),
    ("int8 r = a + (1 + 2)", "requires two operands of the same integer type .*got int8 and int"),
    ("int8 r = 1 + 2", "Cannot initialize 'r' \\(declared int8\\) with a value of type int"),
    ("bool r = a == b", "Cannot compare int8 to uint8"),
    ("bool r = n in xs", "declares element type int8, but 'in's own left operand is int"),
])
def test_what_stays_an_error(statement, match):
    assert_program_semantic_error(SETUP + f"    {statement}\n    return 0\n", match=match)


def test_the_literal_is_typed_in_the_tree():
    tree = dump(analyze(_parse(SETUP + "    bool r = fd < -5\n    a += 1\n    return 0\n")))
    assert "IntLit value=5 : int32" in tree and "IntLit value=1 : int8" in tree


# -- an integer literal where a sum is wanted

SUMS = (
    "type Small is int8 | str\n"
    "type Bytes is byte | none\n"
    "type Both is int | int8 | str\n"
    "type Ints is int8 | int32\n"
    "type Wordy is str | none\n"
    "type Holder struct:\n"
    "    Small held\n"
)


@GCC_SKIP
def test_a_literal_flowing_into_a_sum_is_its_int_or_its_one_integer_variant():
    assert_program_stdout(
        SUMS +
        "def Small pick(int n):\n"
        "    if n == 0:\n        return 7\n"                     # a return value
        "    return 'seven'\n"
        "def str show(Small s):\n"
        "    if s is int8:\n        return format('int8 {}', s)\n"
        "    return 'a str'\n"
        "def int main():\n"
        "    Small m = 7\n"                                       # an initializer: there is no int, so its int8
        "    print(show(m))\n"
        "    m = -128\n"                                          # an assignment, and a negative one
        "    print(show(m))\n"
        "    print(show(100) + ', ' + show(pick(0)) + ', ' + show(pick(1)))\n"   # an argument
        "    Holder h = Holder(5)\n"                              # a field
        "    print(show(h.held))\n"
        "    []Small many = []Small[1, 'two', 3]\n"               # elements
        "    many = append(many, 4)\n"
        "    print(many)\n"
        "    print(3 in many)\n"
        "    print(m == -128)\n"                                  # ... as it is in a comparison
        "    Bytes b = 255\n"
        "    print(b)\n"
        "    Both both = 300\n"                                   # with an int among them, a literal is that
        "    if both is int:\n        print(both)\n"
        "    return 0\n",
        "int8 7\nint8 -128\nint8 100, int8 7, a str\nint8 5\n[]Small[1, 'two', 3, 4]\ntrue\ntrue\n255\n300\n",
    )


@pytest.mark.parametrize("statement,message", [
    ("Small m = 200", r"200 is out of range for int8 \(-128 to 127\)"),
    ("Bytes b = -1", r"-1 is out of range for uint8 \(0 to 255\)"),
    ("Ints n = 1", r"Ints has more than one integer variant \(int8, int32\) and none is int -- say which this is, "
                   r"as in `int8\(1\)`"),
    ("Wordy w = 1", "Cannot initialize 'w' .declared Wordy. with a value of type int"),    # no integer variant
    ("int n = 1\n    Small m = n", "Cannot initialize 'm' .declared Small. with a value of type int"),  # not a literal
])
def test_a_literal_that_no_variant_takes(statement, message):
    assert_program_semantic_error(SUMS + f"def int main():\n    {statement}\n    return 0\n", match=message)
