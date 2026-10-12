"""Variadic parameters, `T... name`: the last parameter of a function or method, a `[]T` inside it. A call
gives it the arguments after the others, packed into a slice, or one slice spread as them, `xs...`."""

import pytest

from ir.ir import IRCall
from ir.program_builder import build_ir_program
from lexer import Lexer, TokenType
from parser import ParseError, Spread, SliceTypeExpr
from tests.test_compiler import GCC_SKIP, _parse, analyze, assert_program_semantic_error, assert_program_stdout
from typed_ast import dump

DECLS = (
    "type Kind enum:\n    Int\n    Str\n    Name\n"
    "type Value is int | str\n"
    "type Parser struct:\n"
    "    Kind at\n"
    "\n"
    "    def bool check(self, Kind... kinds):\n"
    "        return self.at in kinds\n"
    "def int total(int... xs):\n"
    "    int n = 0\n"
    "    for x in xs:\n"
    "        n += x\n"
    "    return n\n"
    "def int lead(str s, int... xs):\n"
    "    return len(s) + len(xs)\n"
    "def int pair(int a, int b):\n"
    "    return a + b\n"
    "def []int same(int... xs):\n"
    "    return xs\n"
    "extern int puts(*byte s)\n"
)


def test_how_it_lexes_and_parses():
    assert [t.type for t in Lexer("xs... .. ....").tokenize()][:5] == [
        TokenType.IDENTIFIER, TokenType.ELLIPSIS, TokenType.DOT, TokenType.DOT, TokenType.ELLIPSIS]
    program = _parse("def f(str s, int... xs):\n    f('a', 1, 2)\n    f('a', ys[1:]...,)\n")
    param = program.functions[0].params[1]
    assert param.variadic and isinstance(param.type, SliceTypeExpr) and param.type.element_type == 'int'
    assert not program.functions[0].params[0].variadic
    packed, spread = (s.expr for s in program.functions[0].body)
    assert len(packed.args) == 3
    assert isinstance(spread.args[1], Spread) and (spread.args[1].line, spread.args[1].col) == (3, 12)
    assert 'variadic' not in program.functions[0].params[0].pretty()


@pytest.mark.parametrize("source,message", [
    ("def f(int... a, int b):\n    return\n", "A variadic parameter comes last -- another follows 'a'"),
    ("type S struct:\n    int x\n    def m(self, int... a, int b):\n        return\n", "comes last"),
    ("def f(...int a):\n    return\n", "with '...' after its element type, as in `int... xs`"),
    ("extern int printf(*byte f, int... xs)\n", "An extern function can't have a variadic parameter \\('xs'\\)"),
    ("intrinsic int _raw_len(str... s)\n", "An intrinsic can't have a variadic parameter"),
    ("def f():\n    g(xs..., 1)\n", "A spread argument \\(`xs...`\\) comes last in a call"),
    ("def f():\n    g(a=xs...)\n", "A named argument can't be spread"),
])
def test_what_does_not_parse(source, message):
    with pytest.raises(ParseError, match=message):
        _parse(source)


@pytest.mark.parametrize("call,message", [
    ("total(1, 'a')", "Argument 2 to 'total' should be int, got str"),
    ("lead()", "Function 'lead' expects at least 1 argument\\(s\\), got 0"),
    ("lead(xs...)", "Function 'lead' expects 1 argument\\(s\\) before the spread slice, got 0 -- a spread slice"),
    ("total(1, xs...)", "expects 0 argument\\(s\\) before the spread slice, got 1"),
    ("lead('a', ss...)", "Argument 2 to 'lead' should be \\[\\]int, got \\[\\]str"),
    ("lead('a', 1...)", "Argument 2 to 'lead' should be \\[\\]int, got int"),
    ("pair(xs...)", "Function 'pair' has no variadic parameter, so it can't be given a slice spread"),
    ("puts(xs...)", "Function 'puts' has no variadic parameter"),
    ("p.check(1)", "Argument 1 to method 'check' on 'Parser' should be Kind, got int"),
    ("p.check(Kind.Int, ks...)", "Method 'check' on 'Parser' expects 0 argument\\(s\\) before the spread slice"),
    ("print(xs...)", "A slice spread with '...' can only be given to a variadic parameter"),
    ("len(xs...)", "can only be given to a variadic parameter"),
    ("Parser q = Parser(ks...)", "can only be given to a variadic parameter"),
    ("Kind k = Kind(xs...)", "can only be given to a variadic parameter"),
])
def test_what_is_refused(call, message):
    assert_program_semantic_error(
        DECLS + "def int main():\n    []int xs = [1]\n    []str ss = ['a']\n    []Kind ks = [Kind.Int]\n"
                "    Parser p = Parser(Kind.Int)\n    " + call + "\n    return 0\n", message)


def test_what_a_call_becomes():
    tree = dump(analyze(_parse(DECLS + "def int main():\n    []int ys = [1]\n"
                                       "    return total(1, 2) + total() + lead('a', ys...)\n")))
    # Packed into a slice literal, none at all an empty slice, a spread slice passed as it is.
    assert ("Call name=total kind=function : int\n"
            "                args:\n"
            "                  SliceLiteral : []int\n"
            "                    elements:\n"
            "                      IntLit value=1 : int\n"
            "                      IntLit value=2 : int\n") in tree
    assert "Call name=total kind=function : int\n                args:\n                  EmptySlice : []int\n" in tree
    assert "args:\n              StrLit value='a' : str\n              Local symbol=ys#" in tree


def _allocations(main_body: str) -> int:
    program = build_ir_program(analyze(_parse(DECLS + "def int main():\n" + main_body)), keep_unreachable=True)
    fn = next(f for f in program.functions if f.name.rstrip('$') == 'main')
    return sum(1 for i in fn.body if isinstance(i, IRCall) and i.name in ('hornet_alloc', 'hornet_alloc_zeroed'))


def test_packing_allocates_only_what_the_callee_keeps():
    assert _allocations("    Parser p = Parser(Kind.Int)\n    p.check(Kind.Int, Kind.Str)\n"
                        "    return total(1, 2, 3) + lead('a', 1)\n") == 0
    assert _allocations("    []int kept = same(1, 2)\n    return len(kept)\n") == 1


@GCC_SKIP
def test_variadic_calls():
    assert_program_stdout(
        DECLS +
        "def str show(str label, Value... vs):\n"
        "    str out = label + ':'\n"
        "    for v in vs:\n"
        "        match v as w:\n"
        "            is int:\n"
        "                out = out + ' ' + format('{}', w)\n"
        "            is str:\n"
        "                out = out + ' ' + w\n"
        "    return out\n"
        "def int main():\n"
        "    Parser p = Parser(Kind.Str)\n"
        "    print(p.check(Kind.Int, Kind.Str))\n"
        "    print(p.check(Kind.Name))\n"
        "    print(p.check())\n"
        "    []Kind wanted = [Kind.Name, Kind.Str]\n"
        "    print(p.check(wanted...))\n"
        "    print(p.check(wanted[:1]...))\n"
        "    print(total())\n"
        "    print(total(1, 2, 3))\n"
        "    print(lead('ab', 7))\n"
        "    print(show('values', 1, 'two', 3))\n"               # each argument widened into the sum
        "    []int ys = [4, 5]\n"
        "    []int shared = same(ys...)\n"                        # a spread slice is the caller's own
        "    shared[0] = 40\n"
        "    print(ys)\n"
        "    []int first = same(1, 2)\n"
        "    []int second = same(3)\n"
        "    print(first)\n"
        "    print(second)\n"
        "    for int i = 0; i < 3; i += 1:\n"
        "        print(total(i, i * 10))\n"
        "    defer print(total(ys...))\n"
        "    return 0\n",
        "true\nfalse\nfalse\ntrue\nfalse\n0\n6\n3\nvalues: 1 two 3\n[]int[40, 5]\n[]int[1, 2]\n[]int[3]\n"
        "0\n11\n22\n45\n")


@GCC_SKIP
def test_another_modules_variadics(tmp_path):
    from tests.test_module_identity import _run, _tree
    entry = _tree(
        tmp_path,
        util=("type Op enum:\n    Add\n    Mul\n\n"
              "    def int apply(self, int... xs):\n"
              "        int n = int(self)\n"
              "        for x in xs:\n"
              "            if self == Op.Add:\n                n += x\n            else:\n                n *= x\n"
              "        return n\n"
              "def int total(int... xs):\n    return Op.Add.apply(xs...)\n"),
        main=("import 'util'\nfrom 'util' import Op, total\n"
              "def int main():\n"
              "    []int xs = [1, 2, 3]\n"
              "    print(util.total(4, 5))\n    print(total(xs...))\n"
              "    print(Op.Mul.apply(2, 3, 4))\n    print(util.Op.Add.apply())\n"
              "    return 0\n"))
    assert _run(entry) == "9\n6\n24\n0\n"



@GCC_SKIP
def test_what_a_packed_argument_may_be():
    # As anywhere a value of the element type is wanted: struct literals, `&S(...)` and `none`, untyped
    # literals, and integer literals into a sum.
    assert_program_stdout(
        "type Point struct:\n    int x\n    int y\n"
        "type Small is int8 | str\n"
        "def int area(Point... ps):\n"
        "    int n = 0\n    for p in ps:\n        n += p.x * p.y\n    return n\n"
        "def int set(*Point... ps):\n"
        "    int n = 0\n    for p in ps:\n        if p != none:\n            n += 1\n    return n\n"
        "def int lengths([]int... rs):\n"
        "    int n = 0\n    for r in rs:\n        n += len(r)\n    return n\n"
        "def str smalls(Small... ss):\n    return format('{}', ss)\n"
        "def int main():\n"
        "    print(area(Point(1, 2), Point(3, 4)))\n"
        "    print(set(&Point(1, 1), none, &Point(2, 2)))\n"
        "    print(lengths([1, 2], [3], []int[]))\n"
        "    print(smalls(1, 'a', -3))\n"
        "    return 0\n",
        "14\n2\n3\n[]Small[1, 'a', -3]\n")
