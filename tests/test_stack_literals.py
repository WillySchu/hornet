"""Where a literal's storage is: the elements of a slice literal, and the struct of an `&S(...)`, are
kept in the frame unless what points to them may outlive the function, or the loop iteration, that
made them (escape_analysis.py decides; then they are on the heap, as they always used to be)."""

import pytest

from ir.ir import IRCall
from ir.program_builder import build_ir_program
from tests.test_compiler import GCC_SKIP, _parse, analyze, assert_program_stdout

DECLS = (
    "type Point struct:\n"
    "    int x\n"
    "    int y\n"
    "\n"
    "    def int sum(self):\n"
    "        return self.x + self.y\n"
    "type Box struct:\n"
    "    []int items\n"
    "    *Point at\n"
    "\n"
    "    def keep(*self, []int xs):\n"
    "        self.items = xs\n"
    "type Ints is []int | none\n"
    "extern int strlen(*byte s)\n"
    "intrinsic str _from_raw_parts(*byte p, int n)\n"
    "def int total([]int xs):\n"                      # reads its argument, and keeps nothing of it
    "    int n = 0\n"
    "    for x in xs:\n"
    "        n += x\n"
    "    return n\n"
    "def int norm(*Point p):\n"
    "    return p.x + p.y\n"
    "def []int same([]int xs):\n"                     # these keep it: by returning it, directly or not,
    "    return xs\n"
    "def []int through([]int xs):\n"
    "    return same(xs)\n"
    "def []int after(int n, []int xs):\n"
    "    if n == 0:\n"
    "        return xs\n"
    "    return after(n - 1, xs)\n"
    "def put(*[]int out, []int xs):\n"                # ... or by storing it where the caller can see
    "    *out = xs\n"
    "def *Point same_point(*Point p):\n"
    "    return p\n"
    "def []int items_of(Box b):\n"                   # ... or a part of what it is given by value
    "    return b.items\n"
    "def *Point first([]*Point ps):\n"               # (keeps what the slice points to, not the slice)
    "    return ps[0]\n"
    # Uses the stack below its caller, so that what a returned function left there is overwritten.
    "def int clobber(int a):\n"
    "    [64]int scratch\n"
    "    for int i = 0; i < 64; i += 1:\n"
    "        scratch[i] = a + i\n"
    "    return scratch[0] + scratch[63]\n"
)


def _allocations(function: str) -> int:
    """How many times `function`, as its body, asks for memory of its own."""
    program = build_ir_program(analyze(_parse(DECLS + function + "def int main():\n    return 0\n")),
                               keep_unreachable=True)
    fn = next(f for f in program.functions if f.name.rstrip('$') == 'f')
    return sum(1 for instr in fn.body if isinstance(instr, IRCall) and instr.name.startswith('hornet_alloc'))


STAYS = [
    "def int f(int a):\n    return total([a, 2, 3])\n",                               # an argument, only read
    "def int f(int a):\n    []int xs = [a, 2, 3]\n    return xs[0] + len(xs)\n",        # a local that never leaves
    "def int f(int a):\n    []int xs = [a, 2, 3]\n    return total(xs[1:])\n",          # part of it, only read
    "def int f(int a):\n    for x in []int[a, 2]:\n        a += x\n    return a\n",
    "def int f(int a):\n    int n = 0\n    for int i = 0; i < 3; i += 1:\n"            # made each time round, and
    "        []int xs = [a, i]\n        n += total(xs)\n    return n\n",               # used only that time
    "def int f(int a):\n    int n = 0\n    while total([a, n]) < 9:\n        n += 1\n    return n\n",
    "def int f(int a):\n    [][]int rows = [[a], [a, a]]\n    return len(rows[1])\n",  # literals in a literal
    "def int f(int a):\n    defer total([a, 2])\n    return a\n",
    "def int f(int a):\n    return norm(&Point(a, 4))\n",                               # `&S(...)`, only read
    "def int f(int a):\n    *Point p = &Point(a, 4)\n    p.x += 1\n    return p.x\n",
    "def int f(int a):\n    Box b = Box([a, 2], &Point(a, a))\n    return total(b.items) + b.at.x\n",
    "def bool f(int a):\n    return a in [1, 2, 3]\n",
    "def int f(int a):\n    []*Point ps = [&Point(a, 1), &Point(a, 2)]\n    return ps[1].y\n",
    "def int f(int a):\n    Ints s = []int[a, 2]\n    if s is []int as xs:\n"       # in a sum
    "        return total(xs)\n    return 0\n",
    "def int f(int a):\n    int n = 0\n    for []int xs = [a, 1]; n < 3; n += 1:\n"  # a loop's init clause
    "        n += xs[1]\n    return n\n",                                             # runs once
]

LEAVES = [
    "def []int f(int a):\n    return [a, 2, 3]\n",                                      # returned
    "def []int f(int a):\n    []int xs = [a, 2, 3]\n    return xs[1:]\n",               # ... a part of it
    "def *int f(int a):\n    []int xs = [a, 2, 3]\n    return &xs[0]\n",                # ... an element's address
    "def Box f(int a):\n    return Box([a, 2], none)\n",                                # ... inside a struct
    "def [][]int f(int a):\n    [][]int rows = [[a], [a, a]]\n    return rows\n",      # ... inside another
    "def []int f(int a):\n    return same([a, 2])\n",                                   # kept by what it is passed to
    "def []int f(int a):\n    return through([a, 2])\n",                                # ... at one remove
    "def []int f(int a):\n    return after(3, [a, 2])\n",                               # ... after calling itself
    "def f(*[]int out, int a):\n    *out = [a, 2]\n",                                   # stored where the caller sees
    "def f(*[]int out, int a):\n    put(out, [a, 2])\n",                                # ... by what it is passed to
    "def f(*Box b, int a):\n    b.items = [a, 2]\n",
    "def f(*Box b, int a):\n    b.keep([a, 2])\n",
    "def f(dict[int][]int d, int a):\n    d[a] = [a, 2]\n",                             # (a dict's table is shared)
    "def int f(int a):\n    []byte s = [104, 105, 0]\n    return strlen(&s[0])\n",    # passed to C: unknown
    "def [][]int f(int a):\n    [][]int rows\n    for int i = 0; i < 3; i += 1:\n"     # made each time round, and
    "        rows = append(rows, [a, i])\n    return rows\n",                           # every one kept
    "def int f(int a):\n    []int last\n    for int i = 0; i < 3; i += 1:\n"           # outlives its time round,
    "        last = [a, i]\n    return total(last)\n",                                  # though not the function
    "def *Point f(int a):\n    return &Point(a, 4)\n",
    "def *Point f(int a):\n    return same_point(&Point(a, 4))\n",
    "def f(*Box b, int a):\n    b.at = &Point(a, 4)\n",
    "def int f(int a):\n    *Point last = none\n    for int i = 0; i < 3; i += 1:\n"
    "        last = &Point(a, i)\n    return last.y\n",
    "def int f(int a):\n    return total([" + ", ".join(["a"] * 2100) + "])\n",         # too large for the stack
    "def []int f(int a):\n    []int xs = [a, 2, 3]\n    return append(xs[:1], 9)\n",  # appended to in place
    "def Ints f(int a):\n    return []int[a, 2]\n",                                    # returned in a sum
    "def []int f(int a):\n    Ints s = []int[a, 2]\n    match s as xs:\n"            # ... and out of one
    "        is []int:\n            return xs\n        is none:\n            return []int[]\n",
    "def []int f(int a):\n    Ints s = []int[a, 2]\n    if s is []int:\n        return s\n    return []int[]\n",
    "def []int f(int a):\n    for xs in [][]int[[a, 1]]:\n        return xs\n    return []int[]\n",
    "def []int f(int a):\n    return items_of(Box([a, 2], none))\n",                   # kept from a struct's copy
    "def *Point f(int a):\n    return first([&Point(a, 1)])\n",
    "def str f(int a):\n    []byte s = [104, 105]\n    return _from_raw_parts(&s[0], 2)\n",
    "def int f(int a):\n    [][]int rows = [[0], [0]]\n    for int i = 0; i < 2; i += 1:\n"  # held by a literal
    "        rows[i] = [a, i]\n    return rows[0][1] + rows[1][1]\n",                 # from outside the loop
    "def int f(int a):\n    int n = 0\n    for []int xs = [0, 1]; n < 5; n += 1:\n"  # held by the loop's own
    "        xs = [xs[1], xs[0] + xs[1]]\n    return n\n",                            # variable, which the next
    "def int f(int a):\n    int n = 0\n"                                              # time round still has
    "    for []int xs = [0, 1]; n < 5; xs = [xs[1], xs[0] + xs[1]]:\n        n += 1\n    return n\n",
]


@pytest.mark.parametrize("function", STAYS, ids=range(len(STAYS)))
def test_a_literal_nothing_keeps_is_in_the_frame(function):
    assert _allocations(function) == 0


@pytest.mark.parametrize("function", LEAVES, ids=range(len(LEAVES)))
def test_a_literal_that_may_be_kept_is_on_the_heap(function):
    assert _allocations(function) >= 1


@pytest.mark.parametrize("function,expected", [
    ("def *Point f(int a):\n    return first([&Point(a, 1)])\n", 1),                   # the Point, not the slice
    ("def []int f(int a):\n    [][]int rows = [[a, 1], [a, 2]]\n    return rows[0]\n", 2),  # the rows, not `rows`
    ("def int f(int a):\n    *Point last = &Point(a, 0)\n    for int i = 0; i < 3; i += 1:\n"
     "        last = &Point(a, i)\n        a += norm(&Point(i, i))\n    return last.y + a\n", 1),
])
def test_only_what_may_be_kept_is_on_the_heap(function, expected):
    assert _allocations(function) == expected


@GCC_SKIP
def test_what_was_kept_is_still_there():
    # Each of these keeps a literal past the function that made it; the stack that function used is then
    # written over. A literal wrongly left in the frame would no longer hold what it was given.
    assert_program_stdout(
        DECLS +
        "def []int returned(int a):\n    return [a, 2, 3]\n"
        "def []int part(int a):\n    []int xs = [a, 2, 3]\n    return xs[1:]\n"
        "def *int element(int a):\n    []int xs = [a, 2, 3]\n    return &xs[0]\n"
        "def Box boxed(int a):\n    return Box([a, 2], &Point(a, 4))\n"
        "def []int handed_on(int a):\n    return through([a, 2])\n"
        "def stored(*[]int out, int a):\n    put(out, [a, 2])\n"
        "def fill(*Box b, int a):\n    b.keep([a, 5])\n    b.at = &Point(a, 6)\n"
        "def remember(dict[int][]int d, int a):\n    d[a] = [a, 7]\n"
        "def [][]int rounds(int a):\n"
        "    [][]int rows\n"
        "    for int i = 0; i < 3; i += 1:\n"
        "        rows = append(rows, [a, i])\n"
        "    return rows\n"
        "def []*Point points(int a):\n"
        "    []*Point all\n"
        "    for int i = 0; i < 3; i += 1:\n"
        "        all = append(all, &Point(a, i))\n"
        "    return all\n"
        "def int last_round(int a):\n"
        "    []int last\n"
        "    *Point at = none\n"
        "    for int i = 0; i < 3; i += 1:\n"
        "        last = [a, i]\n"
        "        at = &Point(a, i)\n"
        "        _ = clobber(i)\n"
        "    return total(last) + at.sum()\n"
        "def int main():\n"
        "    []int a = returned(1)\n"
        "    []int b = part(1)\n"
        "    *int c = element(7)\n"
        "    Box d = boxed(1)\n"
        "    []int e = handed_on(1)\n"
        "    []int f\n"
        "    stored(&f, 1)\n"
        "    Box g = Box([0], none)\n"
        "    fill(&g, 1)\n"
        "    dict[int][]int h\n"
        "    remember(h, 1)\n"
        "    [][]int i = rounds(1)\n"
        "    []*Point j = points(1)\n"
        "    print(clobber(100))\n"
        "    print(a)\n    print(b)\n    print(*c)\n    print(d.items)\n    print(*d.at)\n    print(e)\n    print(f)\n"
        "    print(g.items)\n    print(*g.at)\n    print(h[1])\n    print(i)\n"
        "    print(*j[0])\n    print(*j[2])\n"
        "    print(last_round(1))\n"
        "    return 0\n",
        "263\n[]int[1, 2, 3]\n[]int[2, 3]\n7\n[]int[1, 2]\nPoint(x: 1, y: 4)\n[]int[1, 2]\n[]int[1, 2]\n"
        "[]int[1, 5]\nPoint(x: 1, y: 6)\n[]int[1, 7]\n[][]int[[]int[1, 0], []int[1, 1], []int[1, 2]]\n"
        "Point(x: 1, y: 0)\nPoint(x: 1, y: 2)\n6\n",
    )


@GCC_SKIP
def test_literals_in_the_frame_behave_as_they_did():
    assert_program_stdout(
        DECLS +
        "def int deep(int n):\n"                        # each call's literal is its own
        "    []int mine = [n, n * 2]\n"
        "    *Point here = &Point(n, n)\n"
        "    if n > 0:\n"
        "        _ = deep(n - 1)\n"
        "    return mine[0] + mine[1] + here.sum()\n"
        "def int main():\n"
        "    print(deep(40))\n"
        "    []int xs = [1, 2, 3]\n"
        "    []int more = append(xs, 4)\n"              # growing one copies it out of the frame
        "    xs[0] = 9\n"
        "    print(xs)\n    print(more)\n"
        "    int n = 0\n"
        "    for int i = 0; i < 1000; i += 1:\n"        # one place in the frame, written each time round
        "        []int pair = [i, i + 1]\n"
        "        *Point p = &Point(i, 1)\n"
        "        n += total(pair) + norm(p)\n"
        "    print(n)\n"
        "    [][]int rows = [[1], [2, 3]]\n"
        "    rows[0] = append(rows[0], 5)\n"
        "    print(rows)\n"
        "    print(total([4, 5, 6]) + norm(&Point(3, 4)) + Point(1, 2).sum())\n"
        "    return 0\n",
        "200\n[]int[9, 2, 3]\n[]int[1, 2, 3, 4]\n1500500\n[][]int[[]int[1, 5], []int[2, 3]]\n25\n",
    )
