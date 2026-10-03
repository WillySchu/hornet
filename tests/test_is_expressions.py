"""`is` as an ordinary boolean expression (`x is T`, `x is not T`, and `x == none` for a sum), and
the narrowing that follows it through `and`, `or`, `not`, `else`, and guards. Each condition says
something when true and something when false (SemanticAnalyzer._when); only variables are narrowed."""

import pytest

from tests.test_compiler import GCC_SKIP, assert_program_semantic_error, assert_program_stdout
from lexer import Lexer
from parser import ParseError, Parser

DECLS = """type A struct:
    int x

type B struct:
    int y

type U is A | B | none

type H struct:
    U u

type Color enum:
    Red
    Green

"""

PROGRAM = DECLS + """def U make(int n):
    if n == 0:
        return none
    if n == 1:
        return A(10)
    return B(20)

def int value(U u, bool ok):
    if u is A and ok:
        return u.x                     # narrowed through `and`
    if u is B and u.y > 5:             # the right side sees the left's narrowing
        return u.y
    return -1

def int either(U u):
    if u is A or u is B:
        if u is A:
            return u.x
        return u.y                     # A or B, and not A
    return 0

def int guard(U u, bool ok):
    if not (u is A and ok):
        return -1
    return u.x                         # the guard left unless u is an A

def int guard_none(U u):
    if u == none or u is B:
        return 0
    return u.x                         # neither none nor B

def int not_none(U u):
    if u != none and not u is B:
        return u.x
    return 7

def int one_leaves(U u):
    if u is not A:
        return 3
    else:
        print('an A')
    return u.x                         # only the else gets here

def int main():
    U a = make(1)
    U b = make(2)
    U n = make(0)
    print(value(a, true) + value(a, false) + value(b, true))
    print(either(a) + either(b) + either(n))
    print(guard(a, true) + guard(a, false) + guard(b, true))
    print(guard_none(a) + guard_none(b) + guard_none(n))
    print(not_none(a) + not_none(b) + not_none(n))
    print(one_leaves(a) + one_leaves(b))
    bool is_a = a is A                 # a value
    bool neither = not (b is A) and b is not none
    print(is_a)
    print(neither)
    H h = H(b)
    print(h.u is B)                    # a field: a test, nothing to narrow
    print(make(1) is A)
    print(make(0) is none)
    int count = 0
    while n is none and count < 2:
        count += 1
    print(count)
    Color c = Color.Green
    print(c is Green and not c is Red)
    print(c is not Green)
    return 0
"""


@GCC_SKIP
def test_is_in_expressions_and_what_it_narrows():
    assert_program_stdout(
        PROGRAM,
        "29\n30\n8\n10\n24\nan A\n13\ntrue\ntrue\ntrue\ntrue\ntrue\n2\ntrue\nfalse\n",
    )


F = "def int f(U u, bool ok, *U p):\n"


@pytest.mark.parametrize("body,match", [
    # Nothing is known about u in these places.
    ("    if u is A or ok:\n        return u.x\n    return 0\n", "Cannot access field 'x' on non-struct type U"),
    ("    if u is A and ok:\n        return 1\n    return u.x\n", "Cannot access field 'x' on non-struct type U"),
    ("    if u is A and ok:\n        return 1\n    else:\n        return u.y\n", "Cannot access field 'y'"),
    ("    if not u is A:\n        return u.x\n    return 0\n", "Cannot access field 'x' on non-struct type U"),
    ("    if u is A or u is B:\n        return u.x\n    return 0\n", "Cannot access field 'x' on non-struct type U"),
    ("    bool a = u is A\n    if a:\n        return u.x\n    return 0\n", "Cannot access field 'x'"),
    ("    while u is A:\n        return u.x\n    return 0\n", "Cannot access field 'x' on non-struct type U"),
    ("    if ok or u is A:\n        return 0\n    return u.y\n", "Cannot access field 'y'"),  # none remains too
    # Only a variable is narrowed; a tested expression is not.
    ("    if *p is A:\n        return (*p).x\n    return 0\n", "Cannot access field 'x' on non-struct type U"),
    # A narrowed variable still can't be reassigned.
    ("    if u is A and ok:\n        u = B(1)\n    return 0\n", "Cannot reassign 'u' while it's narrowed"),
    ("    if u is A:\n        return 0\n    u = B(1)\n    return 0\n", "Cannot reassign 'u' while it's narrowed"),
    # What `is` takes.
    ("    return 5 is A\n", "'is' tests a sum type's variant or an enum's member, but this value is int"),
    ("    if ok is A:\n        return 1\n    return 0\n", "'ok' \\(declared bool\\) is not a sum type"),
    ("    if u is int:\n        return 1\n    return 0\n", "'int' is not one of U's own declared variants"),
    ("    if *p is Red:\n        return 1\n    return 0\n", "Unknown type 'Red'"),
])
def test_what_is_not_narrowed_and_what_is_an_error(body, match):
    assert_program_semantic_error(DECLS + F + body + "def int main():\n    return 0\n", match=match)


@pytest.mark.parametrize("condition", ["u is A as a and ok", "u is not A as a", "ok and u is A as a"])
def test_as_binds_only_a_whole_condition(condition):
    source = DECLS + F + f"    if {condition}:\n        return 1\n    return 0\n"
    with pytest.raises(ParseError, match="'as NAME' binds the subject of an 'is' check that is the whole condition"):
        Parser(Lexer(source).tokenize()).parse_program()
