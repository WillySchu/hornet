"""`for ... in` over any expression that gives an array, slice, dict, or string: a call, a method
call, what is reached through one (a field, an element, part of a slice), a dereference. It is
evaluated once, before the first time round."""

import pytest

from ir.ir import IRCall
from ir.program_builder import build_ir_program
from tests.test_compiler import (
    GCC_SKIP, _parse, analyze, assert_program_panics, assert_program_semantic_error, assert_program_stdout)

DECLS = (
    "type Box struct:\n"
    "    []int items\n"
    "    [3]int fixed\n"
    "    dict[str]int ages\n"
    "\n"
    "    def []int doubled(self):\n"
    "        []int out\n"
    "        for x in self.items:\n"
    "            out = append(out, x * 2)\n"
    "        return out\n"
    "\n"
    "def int loud(str what):\n"                    # (says when it is called)
    "    print(what)\n"
    "    return 0\n"
    "def []int nums():\n"
    "    loud('nums')\n"
    "    return []int[1, 2, 3]\n"
    "def [3]int trio():\n"
    "    return [7, 8, 9]\n"
    "def str word():\n"
    "    return 'hi'\n"
    "def dict[str]int ages():\n"
    "    return dict[str]int{'a': 1, 'b': 2}\n"
    "def Box box():\n"
    "    loud('box')\n"
    "    return Box([]int[4, 5], [1, 2, 3], dict[str]int{'k': 9})\n"
    "def *Box boxed():\n"
    "    return &Box([]int[6], [0, 0, 1], dict[str]int{})\n"
)


def _main(body: str) -> str:
    return DECLS + "def int main():\n" + "".join(f"    {line}\n" for line in body.split("\n")) + "    return 0\n"


LOOPS = [
    # (the loop, with what comes before it; what it prints)
    ("for x in nums():\n    print(x)", "nums 1 2 3"),                               # the call is made once
    ("for i, x in nums():\n    print(i + x)", "nums 1 3 5"),
    ("for x in trio():\n    print(x)", "7 8 9"),
    ("for b in word():\n    print(b)", "104 105"),
    ("for k, v in ages():\n    print(v)", "1 2"),
    ("Box b = Box([]int[4, 5], [1, 2, 3], dict[str]int{})\nfor x in b.doubled():\n    print(x)", "8 10"),
    # What is reached through a call: it too is called once.
    ("for x in box().items:\n    print(x)", "box 4 5"),
    ("for x in box().items[1:]:\n    print(x)", "box 5"),
    ("for x in box().fixed:\n    print(x)", "box 1 2 3"),
    ("for k in box().ages:\n    print(k)", "box k"),
    ("for x in nums()[1:]:\n    print(x)", "nums 2 3"),
    ("for x in boxed().items:\n    print(x)", "6"),
    ("[][]int rows = [][]int[[]int[1], []int[2, 3]]\nfor x in rows[loud('index') + 1]:\n    print(x)", "index 2 3"),
    # A dereference, and values made for the loop.
    ("[3]int a = [1, 2, 3]\n*[3]int p = &a\nfor x in *p:\n    print(x)", "1 2 3"),
    ("[]int s = []int[1, 2]\nfor x in append(s, 3):\n    print(x)", "1 2 3"),
    ("for b in 'a' + 'b':\n    print(b)", "97 98"),
    ("for b in bytes('hi'):\n    print(b)", "104 105"),
    # The rest of a loop is as ever.
    ("for x in nums():\n    defer print(0)\n    if x == 2:\n        break", "nums 0 0"),
    ("for x in nums():\n    for y in trio():\n        if y == 8:\n            continue\n        print(x * y)",
     "nums 7 9 14 18 21 27"),
    ("[]*int kept\nfor x in nums():\n    kept = append(kept, &x)\nprint(*kept[0] + *kept[2])", "nums 4"),
]


@GCC_SKIP
def test_what_can_be_iterated():
    body = "\nprint('--')\n".join(loop for loop, _ in LOOPS)
    expected = "\n--\n".join(printed.replace(" ", "\n") for _, printed in LOOPS) + "\n"
    assert_program_stdout(_main(body), expected)


@pytest.mark.parametrize("loop", [loop for loop, printed in LOOPS if printed.startswith(("nums", "box", "index"))])
def test_the_iterable_s_call_is_made_once(loop):
    program = build_ir_program(analyze(_parse(_main(loop))))
    main = next(fn for fn in program.functions if fn.name == 'main')
    called = [instr.name.rstrip('$') for instr in main.body if isinstance(instr, IRCall)]
    assert sum(called.count(name) for name in ('nums', 'box', 'loud')) == 1


@GCC_SKIP
def test_a_call_s_result_is_what_it_was_when_the_loop_began():
    # A variable that is appended to while it is iterated panics; a call's result has no variable to
    # watch, and the loop goes through what the call gave.
    source = (
        DECLS +
        "type Holder struct:\n"
        "    []int items\n"
        "\n"
        "    def []int all(self):\n"
        "        return self.items\n"
        "def int main():\n"
        "    Holder h = Holder([]int[1, 2])\n"
        "    for x in ITERABLE:\n"
        "        h.items = append(h.items, x * 10)\n"
        "    print(h.items)\n"
        "    return 0\n")
    assert_program_stdout(source.replace("ITERABLE", "h.all()"), "[]int[1, 2, 10, 20]\n")
    assert_program_panics(source.replace("ITERABLE", "h.items"),
                          "for ... in: slice was reallocated (e.g. by append) during iteration")


@GCC_SKIP
def test_a_dict_from_a_call_is_still_watched():
    # Copies of a dict are one table, so growing it through another name is seen.
    assert_program_panics(
        DECLS +
        "def dict[int]int same(dict[int]int d):\n"
        "    return d\n"
        "def int main():\n"
        "    dict[int]int d = dict[int]int{1: 1}\n"
        "    for k in same(d):\n"
        "        for int i = 10; i < 200; i += 1:\n"
        "            d[i] = i\n"
        "    return 0\n",
        "for ... in: dict's own buckets were reallocated (e.g. by an insert that triggered growth) during iteration")


@pytest.mark.parametrize("loop,message", [
    ("int n = 3\nfor x in n + 1:\n    print(x)",
     "requires an array, slice, dict, or str as its own iterable .*, got int"),
    ("for x in loud('x'):\n    print(x)", "as its own iterable .*, got int"),
    ("for x in box():\n    print(x)", "as its own iterable .*, got Box"),
    ("for i, k, v in ages():\n    print(k)", "."),                                    # (however it is refused)
    ("for x in missing():\n    print(x)", "Call to undeclared function 'missing'"),
])
def test_what_is_still_refused(loop, message):
    if message == ".":
        with pytest.raises(Exception):
            analyze(_parse(_main(loop)))
        return
    assert_program_semantic_error(_main(loop), match=message)
