"""`is` as an ordinary boolean expression (`x is T`, `x is not T`, and `x == none` for a sum), and
the narrowing that follows it through `and`, `or`, `not`, `else`, and guards. Each condition says
something when true and something when false (flow.py's Scopes.when); only variables are narrowed."""

import re

import pytest

from tests.test_compiler import (
    GCC_SKIP, SemanticError, _parse, analyze, assert_program_semantic_error, assert_program_stdout)
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
    ("    if ok or u is A:\n        return 0\n    return u.y\n", "Cannot access field 'y'"),  # none remains too
    # Only a variable is narrowed; a tested expression is not.
    ("    if *p is A:\n        return (*p).x\n    return 0\n", "Cannot access field 'x' on non-struct type U"),
    # Assigning to a narrowed variable ends the narrowing.
    ("    if u is A and ok:\n        u = A(1)\n        return u.x\n    return 0\n", "Cannot access field 'x'"),
    ("    if u is A:\n        if ok:\n            u = B(1)\n        return u.x\n    return 0\n",
     "Cannot access field 'x'"),
    ("    if u is A:\n        u = B(1)\n    else:\n        return 0\n    return u.x\n", "Cannot access field 'x'"),
    # A loop's body may already have run: what it assigns is not known on the way in.
    ("    if u is A:\n        while ok:\n            int x = u.x\n            u = B(x)\n    return 0\n",
     "Cannot access field 'x'"),
    ("    if u is A:\n"
     "        for int i = 0; i < 2; i += 1:\n"
     "            int x = u.x\n"
     "            u = B(x)\n"
     "    return 0\n",
     "Cannot access field 'x'"),
    ("    if u is A:\n        for b in 'ab':\n            int x = u.x\n            u = B(x)\n    return 0\n",
     "Cannot access field 'x'"),
    # A loop left by `break` says nothing about its condition.
    ("    while u is not A:\n        if ok:\n            break\n        u = A(1)\n    return u.x\n",
     "Cannot access field 'x'"),
    # What `is` takes.
    ("    return 5 is A\n", "'is' tests a sum type's variant or an enum's member, but this value is int"),
    ("    if ok is A:\n        return 1\n    return 0\n", "'ok' \\(declared bool\\) is not a sum type"),
    ("    if u is int:\n        return 1\n    return 0\n", "'int' is not one of U's own declared variants"),
    ("    if *p is Red:\n        return 1\n    return 0\n", "Unknown type 'Red'"),
])
def test_what_is_not_narrowed_and_what_is_an_error(body, match):
    assert_program_semantic_error(DECLS + F + body + "def int main():\n    return 0\n", match=match)


BINDINGS = """type Num struct:
    int v

type Neg struct:
    *Expr operand

type Add struct:
    *Expr left
    *Expr right

type Expr is Num | Neg | Add | none

type Holder struct:
    Expr e
    int limit


def Expr make(int n):
    if n == 0:
        return none
    return Num(n)

# -(-x) is x; -(number) folds.
def str describe(Expr e):
    if e is Neg as neg and *neg.operand is Neg as inner:
        return 'double negation'
    if e is Neg as neg and *neg.operand is Num as n and n.v > 0:
        return 'negative literal'
    if e is Add as a and *a.left is Num as l and *a.right is Num as r:
        if l.v + r.v == 0:
            return 'zero sum'
        return 'constant sum'
    return 'other'

def int first_big(Holder h):
    if h.limit > 0 and h.e is Num as n and n.v > h.limit:
        return n.v
    else:
        return -1

def int main():
    Expr one = Num(1)
    Expr neg = Neg(&Num(5))
    Expr dneg = Neg(&Neg(&Num(2)))
    Expr sum = Add(&Num(2), &Num(-2))
    Expr sum2 = Add(&Num(2), &Num(3))
    Expr mixed = Add(&Num(2), &Neg(&Num(3)))
    print(describe(one))
    print(describe(neg))
    print(describe(dneg))
    print(describe(sum))
    print(describe(sum2))
    print(describe(mixed))
    print(first_big(Holder(Num(9), 5)) + first_big(Holder(Num(3), 5)) + first_big(Holder(Num(9), 0)))
    int n = 7
    if make(n) is Num as m and m.v == n and make(0) is none:
        print(m.v)
    return 0
"""


@GCC_SKIP
def test_as_bindings_joined_by_and():
    assert_program_stdout(
        BINDINGS,
        "other\nnegative literal\ndouble negation\nzero sum\nconstant sum\nother\n7\n7\n",
    )


@GCC_SKIP
def test_a_binding_is_made_only_when_the_checks_before_it_hold():
    assert_program_stdout(
        DECLS +
        "def U noisy(int n):\n"
        "    print(n)\n"
        "    return A(n)\n"
        "def int main():\n"
        "    bool no = false\n"
        "    if no and noisy(1) is A as a:\n"       # noisy(1) is never called
        "        print(a.x)\n"
        "    for int i = 0; i < 3; i += 1:\n"        # a new binding each time round
        "        if i != 1 and noisy(i * 10) is A as a and a.x > 5:\n"
        "            print(a.x + 1)\n"
        "    return 0\n",
        "0\n20\n21\n",
    )


@pytest.mark.parametrize("body,match", [
    # A binding is for an `if`/`elif` condition: alone, or one of the checks its `and`s join.
    ("    if ok or u is A as a:\n        return 1\n    return 0\n", "'as NAME' can bind only in an `if` or `elif`"),
    ("    if not (u is A as a):\n        return 1\n    return 0\n", "'as NAME' can bind only in an `if` or `elif`"),
    ("    if ok and (ok or u is A as a):\n        return 1\n    return 0\n", "'as NAME' can bind only"),
    ("    while u is A as a:\n        return 1\n    return 0\n", "'as NAME' can bind only in an `if` or `elif`"),
    ("    bool b = u is A as a\n    return 0\n", "'as NAME' can bind only in an `if` or `elif`"),
    # The name lasts for the rest of the condition and the body; not the `else`, nor afterwards.
    ("    if ok and u is A as a:\n        return a.x\n    else:\n        return a.x\n", "undeclared variable 'a'"),
    ("    if ok and u is A as a:\n        return a.x\n    return a.x\n", "undeclared variable 'a'"),
    ("    if a.x > 0 and u is A as a:\n        return 1\n    return 0\n", "undeclared variable 'a'"),
    ("    if u is A as a and u is B as a:\n        return 1\n    return 0\n", "'a' is already declared"),
    # It is a copy: narrowing it says nothing about the subject.
    ("    if u is A as a and ok:\n        return u.x\n    return 0\n", "Cannot access field 'x' on non-struct type U"),
])
def test_where_as_may_bind(body, match):
    assert_program_semantic_error(DECLS + F + body + "def int main():\n    return 0\n", match=match)


def test_as_cannot_follow_is_not():
    source = DECLS + F + "    if u is not A as a:\n        return 1\n    return 0\n"
    with pytest.raises(ParseError, match="'as NAME' can't follow 'is not'"):
        Parser(Lexer(source).tokenize()).parse_program()


LOOPS = """type Cons struct:
    int value
    *List next

type List is Cons | none

type A struct:
    int x

type B struct:
    int y

type U is A | B | none

def List build(int n):
    List list = none
    for int i = n; i > 0; i -= 1:
        List rest = list               # each element points to its own copy of the rest
        list = Cons(i, &rest)
    return list

def int sum(List list):
    int total = 0
    List node = list
    while node is Cons:
        total += node.value            # narrowed by the condition
        node = *node.next              # the assignment ends the narrowing
    return total

def int length(List list):
    int n = 0
    List node = list
    while node != none:
        n += 1
        if node is Cons:
            node = *node.next
    return n

def int last(List list):
    List node = list
    int value = -1
    while node is Cons:
        value = node.value
        node = *node.next
    if node is none:                    # after the loop: its condition was false
        return value
    return -2

def U step(U u):
    if u is A:
        return B(u.x + 1)
    if u is B:
        return none
    return A(1)

def int reassigned(U u):
    if u is A:
        int before = u.x
        u = step(u)                    # u is a U again
        if u is B:
            return before * 10 + u.y
        return -1
    return 0

def int in_branch(U u, bool change):
    if u is A:
        if change:
            u = B(7)
        if u is A:                     # must be tested again: it may have been reassigned
            return u.x
        if u is B:
            return u.y
    return 0

def int leaving_branch(U u, bool stop):
    if u is A:
        if stop:
            u = B(7)
            return -1
        return u.x                     # the reassigning branch never gets here
    return 0

def int main():
    List list = build(4)
    print(sum(list))
    print(length(list))
    print(last(list))
    print(last(none))
    print(reassigned(A(3)))
    print(in_branch(A(5), false) + in_branch(A(5), true))
    print(leaving_branch(A(9), false) + leaving_branch(A(9), true))
    return 0
"""


@GCC_SKIP
def test_while_narrows_and_assignment_ends_narrowing():
    assert_program_stdout(LOOPS, "10\n4\n4\n-1\n34\n12\n8\n")


# -- reaching into a sum without narrowing it: the error says how to

REACH = (
    "type Circle struct:\n"
    "    int radius\n"
    "\n"
    "    def int area(c):\n"
    "        return c.radius * c.radius\n"
    "type Square struct:\n"
    "    int side\n"
    "type Shape is Circle | Square | none\n"
    "type Holder struct:\n"
    "    Shape shape\n"
    "    *Shape far\n"
    "    []Shape many\n"
    "def Shape make():\n"
    "    return Circle(1)\n"
    "def int main():\n"
    "    Shape s = Circle(1)\n"
    "    Holder h = Holder(s, &s, []Shape[s])\n"
    "    *Holder p = &h\n"
    "    int i = 0\n"
)
ONLY_A_VARIABLE = "Shape is a sum type, and 'is' narrows only a variable: bind this value to reach its variant, as in "


@pytest.mark.parametrize("use,message", [
    # A field or an element, tested or not: it has to be bound.
    ("if h.shape is Circle:\n        print(h.shape.radius)",
     "Cannot access field 'radius' on non-struct type Shape -- " + ONLY_A_VARIABLE
     + "`if h.shape is Circle as NAME:`"),
    ("print(h.many[i].side)", ONLY_A_VARIABLE + r"`if h.many\[i\] is Square as NAME:`"),
    ("print(h.many[0].side)", ONLY_A_VARIABLE + r"`if h.many\[0\] is Square as NAME:`"),
    ("print(p.shape.radius)", ONLY_A_VARIABLE + "`if p.shape is Circle as NAME:`"),
    ("print(h.far.radius)", r"on non-struct type \*Shape -- " + ONLY_A_VARIABLE + r"`if \*h.far is Circle as NAME:`"),
    ("print((*h.far).radius)", ONLY_A_VARIABLE + r"`if \*h.far is Circle as NAME:`"),
    ("print(make().radius)", ONLY_A_VARIABLE + r"`if \.\.\. is Circle as NAME:`"),      # (not written out)
    # A variable only has to be tested, for a variant it can still hold that has what was asked for.
    ("print(s.radius)",
     "Shape is a sum type: test which variant 's' holds first, as in `if s is Circle:`"),
    ("if s is Circle:\n        return 1\n    print(s.side)", "as in `if s is Square:`"),
    # Methods are told the same.
    ("print(h.shape.area())",
     "Cannot call method 'area' on a value of type Shape -- " + ONLY_A_VARIABLE + "`if h.shape is Circle as NAME:`"),
    ("print(s.area())", "test which variant 's' holds first, as in `if s is Circle:`"),
    # What no variant has.
    ("print(h.shape.width)",
     r"Shape is a sum type, and none of its variants \(Circle, Square, none\) has a field 'width'"),
    ("print(s.volume())", r"none of its variants \(Circle, Square, none\) has a method 'volume'"),
])
def test_reaching_into_a_sum_says_how_to_get_at_the_variant(use, message):
    assert_program_semantic_error(REACH + f"    {use}\n    return 0\n", match=message)


@pytest.mark.parametrize("use,message", [
    ("print(i.radius)", "Cannot access field 'radius' on non-struct type int at line"),
    ("print(i.area())", "on a value of type int -- methods are only defined on structs and enums at line"),
])
def test_what_is_no_sum_is_told_nothing_more(use, message):
    assert_program_semantic_error(REACH + f"    {use}\n    return 0\n", match=message)


@pytest.mark.parametrize("use,reached", [
    ("print(h.shape.radius)", "NAME.radius"),
    ("print(h.many[i].side)", "NAME.side"),
    ("print(h.far.radius)", "NAME.radius"),
    ("print(p.shape.area())", "NAME.area()"),
    ("print(s.radius)", "s.radius"),
    ("print(s.area())", "s.area()"),
])
def test_what_the_error_suggests_is_right(use, reached):
    # Take the line the error offers, and use it: the program then compiles.
    with pytest.raises(SemanticError) as refused:
        analyze(_parse(REACH + f"    {use}\n    return 0\n"))
    suggestion = re.search(r"as in `(if .*:)`", refused.value.message).group(1)
    analyze(_parse(REACH + f"    {suggestion}\n        print({reached})\n    return 0\n"))
