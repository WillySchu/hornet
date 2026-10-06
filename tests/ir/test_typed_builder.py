"""ir/typed_builder.py: IR built from the typed tree."""

import pytest

from ir.program_builder import build_ir_program
from ir.typed_builder import link_name
from tests.test_compiler import GCC_SKIP, _parse, analyze, assert_program_stdout
from typesys import Type

SCALARS = """\
def int fib(int n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)
def bool noisy(int k):
    print(k)
    return k > 1
def *int leak():
    int x = 41
    return &x
def int main(int argc, *byte argv):
    int x = 5
    *int p = &x
    *p += 2
    int8 small = int8(127) + int8(1)
    bool b = x > 3 and not (x == 9)
    print(b)
    print(noisy(0) and noisy(2))
    print(noisy(3) or noisy(4))
    for int i = 0; i < 5; i += 1:
        if i == 1:
            continue
        if i == 3:
            break
        print(i)
    int j = 0
    while j < 3:
        j += 1
    *int q = none
    print(q == none)
    print(*leak() + j)
    print(int(small))
    print(fib(15) + x + argc)
    return 0
"""


def _built_functions(source: str) -> list:
    program = _parse(source)
    program = analyze(program)
    return [fn.name for fn in build_ir_program(program).functions]


def test_scalar_functions_are_built():
    assert _built_functions(SCALARS) == [link_name(n) for n in ('fib', 'noisy', 'leak', 'main')]


def test_a_shape_with_no_rule_is_a_compiler_error(monkeypatch):
    from ir.errors import IRError
    from ir.typed_builder import NotYetPorted, TypedFunctionBuilder

    def refuse(self, fn):
        raise NotYetPorted("for this test")
    monkeypatch.setattr(TypedFunctionBuilder, 'build', refuse)
    with pytest.raises(IRError, match="No IR for main: for this test"):
        _built_functions("def int main():\n    return 0\n")


@GCC_SKIP
def test_scalar_programs_run_correctly():
    assert_program_stdout(SCALARS, "true\n0\nfalse\n3\ntrue\n0\n2\ntrue\n44\n-128\n618\n")


COMPOSITES = """\
type P struct:
    int x
    int y
type Num struct:
    int v
type Bin struct:
    *Expr left
    *Expr right
type Expr is Num | Bin | none
def P swap(P p):
    return P(p.y, p.x)
def []int evens(int n):
    []int s = []int[0, 2, 4, 6, 8, 10]
    return s[0:n]
def str greet(str name):
    return 'hi ' + name + str(byte(33))
def int eval(*Expr p):
    match *p as e:
        is Num:
            return e.v
        is Bin:
            return eval(e.left) + eval(e.right)
        is none:
            return 0
    return 0
def int main():
    P p = P(1, 2)
    p = P(p.y, p.x)
    print(p)
    print(swap(p))
    [3]P ps
    ps[1].x = 7
    print(ps)
    []int s = evens(3)
    print(s)
    print(len(s) + s[2])
    str g = greet('bob')
    print(g)
    print(g[0:2] == 'hi')
    print(len(g))
    Expr e = Bin(&Num(1), &Bin(&Num(2), &Num(3)))
    print(eval(&e))
    Expr nothing
    print(nothing == none)
    if e is Bin:
        print(eval(e.left))
    return 0
"""


def test_composite_functions_are_built():
    assert _built_functions(COMPOSITES) == [link_name(n) for n in ('swap', 'evens', 'greet', 'eval', 'main')]


@GCC_SKIP
def test_composite_programs_run_correctly():
    # `p = P(p.y, p.x)` reads its own target: the old builder printed P(x: 2, y: 2).
    assert_program_stdout(COMPOSITES, "P(x: 2, y: 1)\n"
                                      "P(x: 1, y: 2)\n"
                                      "[3]P[P(x: 0, y: 0), P(x: 7, y: 0), P(x: 0, y: 0)]\n"
                                      "[]int[0, 2, 4]\n7\nhi bob!\ntrue\n7\n6\ntrue\n1\n")


CONTAINERS = """\
type P struct:
    int x
    str name
def dict[str]int count([]str words):
    dict[str]int d
    for w in words:
        if w in d:
            d[w] += 1
        else:
            d[w] = 1
    return d
def int main():
    dict[str]int d = count(['a', 'b', 'a'])
    print(d['a'] * 10 + d['b'])
    del(d, 'b')
    print(len(d))
    dict[int]P people = dict[int]P{1: P(1, 'one')}
    people[2] = P(2, 'two')
    int total = 0
    for k, v in people:
        total += k * v.x
    print(total)
    []int s
    for int i = 0; i < 100; i += 1:
        s = append(s, i)
    print(len(s) + s[99])
    []*int ptrs
    for int i = 0; i < 3; i += 1:
        ptrs = append(ptrs, &i)
    print(*ptrs[0] + *ptrs[2])
    []byte bs = bytes('hey')
    print(str(bs))
    print(3 in s and not (500 in s))
    print(P(1, 'a') == P(1, 'a'))
    print([2]str['a', 'b'] != [2]str['a', 'c'])
    int n = 0
    for i, c in 'abc':
        n += i * int(c)
    print(n)
    return 0
"""


def test_container_functions_are_built():
    assert _built_functions(CONTAINERS) == [link_name(n) for n in ('count', 'main')]


@GCC_SKIP
def test_container_programs_run_correctly():
    assert_program_stdout(CONTAINERS, "21\n1\n5\n199\n2\nhey\ntrue\ntrue\ntrue\n296\n")


@GCC_SKIP
def test_iterating_over_a_slice_that_grows_panics():
    from tests.test_compiler import assert_program_panics
    assert_program_panics("def int main():\n"
                          "    []int s = [1, 2]\n"
                          "    for x in s:\n"
                          "        s = append(s, x)\n"
                          "    return 0\n",
                          "for ... in: slice was reallocated (e.g. by append) during iteration")


@GCC_SKIP
def test_scalar_zero_values():
    source = ("def int main():\n"
              "    int x\n"
              "    bool b\n"
              "    *int p\n"
              "    print(x)\n"
              "    print(b)\n"
              "    print(p == none)\n"
              "    return 0\n")
    assert _built_functions(source) == ['main']
    assert_program_stdout(source, "0\nfalse\ntrue\n")


CONDITIONS = """\
def bool noisy(int n, bool v):
    print(n)
    return v
def int main():
    int a = 1
    if (a < 2 and noisy(1, false)) or not (a == 1 or noisy(2, true)):
        print(10)
    elif not noisy(3, false) and (true or noisy(4, true)):
        print(20)
    bool x = a > 0 and not (a > 5)
    bool y = false or noisy(5, false)
    int i = 0
    while not (i >= 3 or false):
        i += 1
    print(x)
    print(y)
    print(i)
    return 0
"""


@GCC_SKIP
def test_conditions_short_circuit_and_negate():
    assert_program_stdout(CONDITIONS, "1\n3\n20\n5\ntrue\nfalse\n3\n")


def test_a_condition_made_of_and_or_not_branches_without_computing_bools():
    """Comparisons inside `and`/`or`/`not` each feed a branch; no bool is computed and then tested."""
    from ir.ir import IRMove
    program = _parse("def int f(int a, int b):\n    if (a < b and b < 10) or not (a == 3):\n"
                     "        return 1\n    return 0\n")
    fn = next(f for f in build_ir_program(analyze(program)).functions if f.name == link_name('f'))
    assert not [i for i in fn.body if isinstance(i, IRMove) and i.dst.type == Type.BOOL]
