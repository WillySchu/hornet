"""`append(s, a, b, ...)` and `append(s, vs...)`: several values, or a slice's elements (a str's bytes for a
[]byte), after `s`'s own; one check for room, and what is added is read in full before any of it is written."""

import pytest

from ir.ir import IRCall
from ir.program_builder import build_ir_program
from tests.test_compiler import GCC_SKIP, _parse, analyze, assert_program_semantic_error, assert_program_stdout
from typed_ast import dump


@pytest.mark.parametrize("call,message", [
    ("append(xs)", "'append' needs something to add after the slice"),
    ("append()", "'append' expects a slice and what to add to it, got nothing"),
    ("append(xs...)", "'append' takes the slice to add to as itself, not spread"),
    ("append(xs, 1, ys...)", "takes a spread slice \\(`xs...`\\) as all it adds, got 1 value\\(s\\) before it"),
    ("append(xs, ss...)", "'append' can't spread a value of type \\[\\]str into a \\[\\]int -- what is spread is a "
                          "\\[\\]int at"),
    ("append(xs, 'ab'...)", "can't spread a value of type str into a \\[\\]int"),
    ("append(bs, xs...)", "can't spread a value of type \\[\\]int into a \\[\\]uint8 -- .* \\(or a str's bytes\\)"),
    ("append(xs, 1, 'a')", "'append' cannot append a value of type str to a \\[\\]int"),
])
def test_what_is_refused(call, message):
    assert_program_semantic_error(
        "def int main():\n    []int xs\n    []int ys\n    []str ss\n    []byte bs\n    xs = " + call + "\n"
        "    return 0\n", message)


def test_what_it_becomes():
    tree = dump(analyze(_parse(
        "def int main():\n    []int xs\n    []byte bs\n"
        "    xs = append(xs, 1)\n    xs = append(xs, 2, 3)\n    xs = append(xs, xs...)\n    bs = append(bs, 'hi'...)\n"
        "    return 0\n")))
    assert "Append : []int\n" in tree                                  # one value: as before
    assert ("AppendAll : []int\n"                                       # several: their slice literal
            "        slice:\n          Local symbol=xs#0 : []int\n"
            "        values:\n          SliceLiteral : []int\n") in tree
    assert "values:\n          Local symbol=xs#0 : []int\n" in tree
    assert "values:\n          StrLit value='hi' : str\n" in tree


def _calls(function: str) -> list:
    program = build_ir_program(analyze(_parse(
        "type Point struct:\n    int x\n" + function + "def int main():\n    return 0\n")), keep_unreachable=True)
    fn = next(f for f in program.functions if f.name.rstrip('$') == 'f')
    return [i.name for i in fn.body if isinstance(i, IRCall) and i.name.startswith('hornet_alloc')]


def test_what_is_added_is_copied_and_only_what_it_points_to_is_kept():
    assert _calls("def []int f([]int xs, int a):\n    return append(xs, a, 2, 3)\n") == []
    assert _calls("def []int f([]int xs, int a):\n    []int ys = [a, 2]\n    return append(xs, ys...)\n") == []
    assert len(_calls("def []*Point f([]*Point ps, int a):\n    return append(ps, &Point(a), &Point(2))\n")) == 2
    assert len(_calls("def [][]int f([][]int rows, int a):\n    return append(rows, [a], [a, a])\n")) == 2


@GCC_SKIP
def test_appending_many():
    assert_program_stdout(
        "type P struct:\n    int x\n    str s\n"
        "def [][]int rows(int a):\n    [][]int out\n    return append(out, [a], [a, a])\n"
        "def int clobber(int a):\n    [64]int scratch\n    for int i = 0; i < 64; i += 1:\n"
        "        scratch[i] = a + i\n    return scratch[0] + scratch[63]\n"
        "def int main():\n"
        "    []int xs = [1, 2]\n"
        "    xs = append(xs, 3, 4, 5)\n    print(xs)\n"
        "    []int ys = [6, 7]\n"
        "    xs = append(xs, ys...)\n    print(xs)\n"
        "    []int none_yet\n"
        "    none_yet = append(none_yet, none_yet...)\n    print(len(none_yet))\n"
        "    []int a = [0, 1, 2, 3]\n"
        "    print(append(a[:1], a[2], a[1]))\n"         # each value read before any is written over
        "    []int c = [0, 1, 2, 3]\n"
        "    print(append(c[:1], c[0:2]...))\n"         # a spread overlapping where it goes
        "    []int f = [1, 2, 3]\n"
        "    f = append(f, f...)\n    print(f)\n"
        "    []byte buf\n"
        "    buf = append(buf, 'hello'...)\n    buf = append(buf, \" \", \"w\")\n    buf = append(buf, 'orld'...)\n"
        "    print(str(buf))\n"
        "    []P ps\n"
        "    ps = append(ps, P(1, 'a'), P(2, 'b'))\n    ps = append(ps, ps...)\n    print(ps)\n"
        "    [][]int kept = rows(9)\n    _ = clobber(100)\n    print(kept)\n"
        "    []int big\n"
        "    for int i = 0; i < 300; i += 1:\n        big = append(big, i, i)\n"
        "    big = append(big, big[:250]...)\n"
        "    print(len(big))\n    print(big[599] + big[849])\n"
        "    return 0\n",
        "[]int[1, 2, 3, 4, 5]\n[]int[1, 2, 3, 4, 5, 6, 7]\n0\n[]int[0, 2, 1]\n[]int[0, 0, 1]\n"
        "[]int[1, 2, 3, 1, 2, 3]\nhello world\n"
        "[]P[P(x: 1, s: 'a'), P(x: 2, s: 'b'), P(x: 1, s: 'a'), P(x: 2, s: 'b')]\n"
        "[][]int[[]int[9], []int[9, 9]]\n850\n423\n")
