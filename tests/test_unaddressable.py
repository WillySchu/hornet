"""A dict entry and a narrowed sum's payload have no address: a dict moves its entries when it grows, and
giving a sum another variant puts that variant where the payload was. So `&`, a pointer-receiver method
call, and slicing an array are refused anywhere in one (through fields and array elements, not pointers).
What is done in place without an address left behind still works: assignment, compound assignment (the
entry found again by key if the value may have grown the dict), and `for ... in` (over a copy)."""

import pytest

from tests.test_compiler import GCC_SKIP, assert_program_semantic_error, assert_program_stdout

DECLS = (
    "type P struct:\n    int x\n"
    "type C struct:\n"
    "    int n\n    str label\n    [3]int arr\n    *P at\n"
    "\n"
    "    def inc(*c):\n        c.n += 1\n"
    "    def int get(c):\n        return c.n\n"
    "type B struct:\n    str s\n"
    "type V is C | B | []int | *P | none\n"
    "type T is [2]int | str\n"
)

ENTRY = "Cannot {} a dict entry, or anything in one: the dict moves its entries when it grows"
PAYLOAD = "Cannot {} the payload of narrowed '{}', or anything in it"


@pytest.mark.parametrize("statement,message", [
    ("*C p = &d[0]", ENTRY.format("take the address of")),
    ("*int p = &d[0].n", ENTRY.format("take the address of")),
    ("*int p = &d[0].arr[1]", ENTRY.format("take the address of")),
    ("*int p = &a[0][1]", ENTRY.format("take the address of")),
    ("d[0].inc()", ENTRY.format("call method 'inc' \\(its receiver is a pointer\\) on")),
    ("[]int s = d[0].arr[:]", ENTRY.format("slice an array in")),
    ("[]int s = a[0][1:]", ENTRY.format("slice an array in")),
    ("if v is C:\n        *C p = &v", PAYLOAD.format("take the address of", "v")),
    ("if v is C:\n        *int p = &v.n", PAYLOAD.format("take the address of", "v")),
    ("if v is C:\n        *int p = &v.arr[2]", PAYLOAD.format("take the address of", "v")),
    ("if v is C:\n        v.inc()", PAYLOAD.format("call method 'inc' \\(its receiver is a pointer\\) on", "v")),
    ("if v is C:\n        []int s = v.arr[:]", PAYLOAD.format("slice an array in", "v")),
    ("if t is [2]int:\n        []int s = t[:]", PAYLOAD.format("slice an array in", "t")),
    ("if v is C as w:\n        *int p = &w.n", PAYLOAD.format("take the address of", "w")),
    ("match v as w:\n        is C:\n            w.inc()\n        else:\n            return 1",
     PAYLOAD.format("call method 'inc' \\(its receiver is a pointer\\) on", "w")),
])
def test_what_is_refused(statement, message):
    assert_program_semantic_error(
        DECLS + "def int main():\n    dict[int]C d\n    dict[int][3]int a\n    V v = none\n    T t = 'x'\n"
                "    " + statement + "\n    return 0\n", message)


def test_the_payload_refusal_says_how_to_do_without():
    assert_program_semantic_error(
        DECLS + "def int main():\n    V v = none\n    if v is C:\n        v.inc()\n    return 0\n",
        "Copy it out \\(`C copy = v`\\), change the copy, and assign it back")


@GCC_SKIP
def test_what_still_works():
    # An address into what a pointer points to, a slice's elements (which are elsewhere), the whole sum, a
    # value receiver, and the copies the refusals suggest.
    assert_program_stdout(
        DECLS +
        "def int main():\n"
        "    P p = P(1)\n"
        "    dict[int]C d\n"
        "    d[0] = C(1, 'one', [1, 2, 3], &p)\n"
        "    *int x = &d[0].at.x\n    *x = 5\n    print(p.x)\n"
        "    d[0].n = 10\n    d[0].n += 1\n    d[0].arr[2] *= 7\n    print(d[0].get())\n    print(d[0].arr)\n"
        "    C c = d[0]\n    c.inc()\n    d[0] = c\n    print(d[0].n)\n"
        "    V v = []int[4, 5]\n"
        "    *V whole = &v\n"
        "    if v is []int:\n        *int e = &v[1]\n        *e = 50\n        print(v)\n"
        "    v = C(1, 'c', [0, 0, 0], none)\n"
        "    if v is C:\n        C copy = v\n        copy.inc()\n        v = copy\n"
        "    if v is C:\n        print(v.get())\n"
        "    return 0\n",
        "5\n11\n[3]int[1, 2, 21]\n12\n[]int[4, 50]\n2\n")


@GCC_SKIP
def test_compound_assignment_into_an_entry_that_moves():
    # Evaluating the value inserts enough to grow the table: the entry is found again for the write.
    assert_program_stdout(
        DECLS +
        "def int fill(dict[int]C d):\n"
        "    for int i = 1; i < 100; i += 1:\n        d[i] = C(i, 'x', [i, i, i], none)\n"
        "    return 5\n"
        "def str more(dict[int]C d):\n    _ = fill(d)\n    return '-more'\n"
        "def int main():\n"
        "    dict[int]C d\n"
        "    d[0] = C(1, 'one', [1, 2, 3], none)\n"
        "    d[0].n += fill(d)\n"
        "    d[0].arr[1] *= fill(d)\n"
        "    d[0].label += more(d)\n"
        "    print(d[0].n)\n    print(d[0].arr)\n    print(d[0].label)\n"
        "    return 0\n",
        "6\n[3]int[1, 10, 3]\none-more\n")


@GCC_SKIP
def test_iterating_an_array_that_may_move_goes_over_a_copy():
    assert_program_stdout(
        DECLS +
        "type Ps is [2]*P | str\n"
        "def int main():\n"
        "    P a = P(7)\n"
        "    Ps t = [2]*P[&a, &a]\n"
        "    if t is [2]*P:\n"
        "        for q in t:\n"
        "            t = 'a str now'\n"              # the array iterated is the one the loop began with
        "            print(q.x)\n"
        "    dict[int][3]int g\n"
        "    g[0] = [1, 2, 3]\n"
        "    for x in g[0]:\n"
        "        for int i = 1; i < 100; i += 1:\n"
        "            g[i] = [i, i, i]\n"
        "        print(x)\n"
        "    return 0\n",
        "7\n7\n1\n2\n3\n")
