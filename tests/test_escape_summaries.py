"""Interprocedural escape summaries: sound for every way a parameter can leak, precise otherwise."""

import pytest

from desugar import desugar_methods
from escape_analysis import analyze_array_escapes, compute_escape_summaries
from elaborate import elaborate
from typed_ast import Declare
from semantic import analyze
from tests.test_compiler import GCC_SKIP, _parse, assert_program_stdout


def _escapes(source: str, fn_name: str = 'leak'):
    program = _parse(source)
    desugar_methods(program)
    analyze(program)
    typed = elaborate(program)
    summaries = compute_escape_summaries(typed.functions, program.struct_registry)
    fn = next(f for f in typed.functions if f.name == fn_name)
    escaping = analyze_array_escapes(fn, program.struct_registry, summaries)
    return [s.symbol.name for s in fn.body if isinstance(s, Declare) and s.symbol.id in escaping], summaries


# Each leaks `x` (or `n`) out of leak() through a callee; the local must be heap-allocated.
LEAKS = {
    'e1': (  # returned by the callee
        'def *int ident(*int p):\n'
        '    return p\n'
        'def *int leak():\n'
        '    int x = 42\n'
        '    return ident(&x)\n'
        , 'ident', [True]),
    'e2': (  # stored into another parameter's memory
        'type Out struct:\n'
        '    *int p\n'
        'def put(*Out o, *int p):\n'
        '    o.p = p\n'
        'def *int leak():\n'
        '    int x = 42\n'
        '    Out r = Out(none)\n'
        '    put(&r, &x)\n'
        '    return r.p\n'
        , 'put', [False, True]),
    'e3': (  # stored two calls deep
        'type Out struct:\n'
        '    *int p\n'
        'def pass_on(*Out o, *int p):\n'
        '    store(o, p)\n'
        'def store(*Out o, *int p):\n'
        '    o.p = p\n'
        'def *int leak():\n'
        '    int x = 42\n'
        '    Out r = Out(none)\n'
        '    pass_on(&r, &x)\n'
        '    return r.p\n'
        , 'store', [False, True]),
    'e4': (  # returned through recursion
        'def *int rec(*int p, int n):\n'
        '    if n == 0:\n'
        '        return p\n'
        '    return rec(p, n - 1)\n'
        'def *int leak():\n'
        '    int x = 42\n'
        '    return rec(&x, 5)\n'
        , 'rec', [True, False]),
    'e5': (  # loaded back out of a struct the callee only reads
        'type Box struct:\n'
        '    *int p\n'
        'def *int get(*Box b):\n'
        '    return b.p\n'
        'def *int leak():\n'
        '    int x = 42\n'
        '    Box b = Box(&x)\n'
        '    return get(&b)\n'
        , 'get', [False]),
    'e6': (  # loaded out of a slice argument
        'def *int first([]*int s):\n'
        '    return s[0]\n'
        'def *int leak():\n'
        '    int x = 42\n'
        '    [1]*int arr = [&x]\n'
        '    return first(arr[:])\n'
        , 'first', [False]),
    'e7': (  # appended by a pointer-receiver method
        'type Reg struct:\n'
        '    []*int items\n'
        '    def add(*r, *int p):\n'
        '        r.items = append(r.items, p)\n'
        'def *int leak():\n'
        '    int x = 42\n'
        '    Reg reg = Reg([])\n'
        '    reg.add(&x)\n'
        '    return reg.items[0]\n'
        , 'Reg.add', [False, True]),
    'e8': (  # a pointer receiver returning itself
        'type Node struct:\n'
        '    int v\n'
        '    def *Node self_ptr(*n):\n'
        '        return n\n'
        'def *Node leak():\n'
        '    Node n = Node(42)\n'
        '    return n.self_ptr()\n'
        , 'Node.self_ptr', [True]),
    'e9': (  # copied between two arguments' memory
        'type Out struct:\n'
        '    *int p\n'
        'def swap_in(*Out a, *Out b):\n'
        '    a.p = b.p\n'
        'def *int leak():\n'
        '    int x = 42\n'
        '    Out src = Out(&x)\n'
        '    Out dst = Out(none)\n'
        '    swap_in(&dst, &src)\n'
        '    return dst.p\n'
        , 'swap_in', [False, False]),
}


@pytest.mark.parametrize('name', sorted(LEAKS))
def test_leaked_local_is_heap_allocated(name):
    source, callee, summary = LEAKS[name]
    escaping, summaries = _escapes(source)
    assert escaping in (['x'], ['n'])
    assert summaries[callee] == summary


def test_non_escaping_parameters_keep_locals_on_the_stack():
    source = (
        "type Counter struct:\n"
        "    int n\n"
        "    def inc(*c):\n"
        "        c.n += 1\n"
        "def int total([]int s):\n"
        "    int t = 0\n"
        "    for v in s:\n"
        "        t += v\n"
        "    return t\n"
        "def int leak():\n"
        "    Counter a = Counter(0)\n"
        "    a.inc()\n"
        "    [3]int arr = [1, 2, 3]\n"
        "    return total(arr[:]) + a.n\n"
    )
    escaping, summaries = _escapes(source)
    assert escaping == []
    assert summaries['Counter.inc'] == [False] and summaries['total'] == [False]


def test_extern_calls_stay_conservative():
    source = (
        "extern *byte memcpy(*byte dst, *byte src, int n)\n"
        "def int leak():\n"
        "    byte b = \"a\"\n"
        "    memcpy(&b, &b, 1)\n"
        "    return 0\n"
    )
    escaping, _ = _escapes(source)
    assert escaping == ['b']


_NOISE = (
    "def int noise(int a):\n"
    "    [64]int z\n"
    "    for int i = 0; i < 64; i += 1:\n"
    "        z[i] = a * 1000 + i\n"
    "    return z[3] + z[60]\n"
)


@GCC_SKIP
@pytest.mark.parametrize('name', sorted(LEAKS))
def test_leaked_local_survives_its_frame(name):
    source, _, _ = LEAKS[name]
    ret, deref = ('*Node', 'r.v') if name == 'e8' else ('*int', '*r')
    assert_program_stdout(
        source + _NOISE + f"def int main():\n    {ret} r = leak()\n    noise(7)\n    print({deref})\n    return 0\n",
        "42\n")
