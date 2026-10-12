"""A statement's value must be used: an expression statement is refused unless it gives nothing (a call of
a function with no result, or one that never returns). `_ = value` evaluates a value and throws it away;
`_` is not a name, except as a `for ... in` binding, which it leaves unbound."""

import pytest

from tests.test_compiler import GCC_SKIP, assert_program_panics, assert_program_semantic_error, assert_program_stdout

DECLS = (
    "type P struct:\n    int x\n\n    def int get(p):\n        return p.x\n"
    "def int f():\n    print('f ran')\n    return 1\n"
    "def note():\n    print('noted')\n"
)

UNUSED = "gives a value \\({}\\) that nothing uses -- use it, or throw it away with `_ = ...`"


@pytest.mark.parametrize("statement,message", [
    ("f()", "This call to 'f' " + UNUSED.format("int")),
    ("p.get()", "This call to method 'get' " + UNUSED.format("int")),
    ("P(1)", "This struct literal " + UNUSED.format("P")),
    ("append(xs, 2)", "This call to 'append' " + UNUSED.format("\\[\\]int")),
    ("xs[0] + 1", "This expression " + UNUSED.format("int")),
    ("_ = note()", "This call to 'note' gives no value to throw away"),
    ("_ = panic('x')", "This call to 'panic' gives no value to throw away"),
    ("_ += 1", "`_ \\+= ...` has nothing to work on"),
    ("int _ = 1", "'_' can't be declared: it is kept for throwing a value away, as in `_ = f\\(\\)`"),
    ("if xs is []int as _:\n        return 1", "'_' can't be declared"),
    ("print(_)", "Reference to undeclared variable '_'"),
    ("for _ in xs:\n        print(_)", "Reference to undeclared variable '_'"),
])
def test_what_is_refused(statement, message):
    assert_program_semantic_error(
        DECLS + "def int main():\n    []int xs = [1]\n    P p = P(4)\n    " + statement + "\n    return 0\n", message)


def test_a_parameter_cannot_be_called_underscore():
    assert_program_semantic_error(DECLS + "def g(int _):\n    return\ndef int main():\n    return 0\n",
                                  "'_' can't be declared")


@GCC_SKIP
def test_throwing_values_away():
    assert_program_stdout(
        DECLS +
        "def int main():\n"
        "    _ = f()\n"                                       # evaluated, then thrown away
        "    []int xs = [3, 4]\n"
        "    _ = append(xs, 5)\n"
        "    _ = P(2).get()\n"
        "    note()\n"                                        # nothing to throw away: as before
        "    defer f()\n"                                     # a deferred call's value goes unused
        "    int n = 0\n"
        "    for _ in xs:\n        n += 1\n"
        "    for i, _ in xs:\n        n += i\n"
        "    for _, x in xs:\n        n += x\n"
        "    print(n)\n"
        "    return 0\n",
        "f ran\nnoted\n10\nf ran\n")


@GCC_SKIP
def test_a_thrown_away_value_keeps_its_checks():
    assert_program_panics(
        "def int main():\n    []int xs = [1]\n    int i = len(xs) + 2\n    _ = xs[i]\n    return 0\n",
        "index out of bounds: index 3, length 1")
