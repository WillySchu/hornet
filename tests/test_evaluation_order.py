"""Operands and arguments are evaluated left to right, and each one's value is fixed when it is
evaluated: a later one that changes an earlier one's variable, through a pointer, comes too late."""

from ir.ir import IRCopy
from ir.program_builder import build_ir_program
from ir.typed_builder import link_name
from tests.test_compiler import GCC_SKIP, _parse, analyze, assert_program_stdout

DECLS = (
    "type P struct:\n"
    "    int x\n"
    "    def int get(self, int ignored):\n"
    "        return self.x\n"
    "    def int reset(*self):\n"
    "        self.x = 99\n"
    "        return 0\n"
    "type Q struct:\n"
    "    P p\n"
    "    int n\n"
    "type U is P | int\n"
    "def int set(*P p):\n"
    "    p.x = 99\n"
    "    return 0\n"
    "def P changed(*P p):\n"
    "    p.x = 99\n"
    "    return P(1)\n"
    "def int set_q(*Q q):\n"
    "    q.p.x = 99\n"
    "    return 0\n"
    "def int set_arr(*[2]int a):\n"
    "    *a = [99, 99]\n"
    "    return 0\n"
    "def int set_dict(*dict[str]int d):\n"
    "    *d = dict[str]int{'k': 99}\n"
    "    return 0\n"
    "def int set_sum(*U u):\n"
    "    *u = 99\n"
    "    return 0\n"
    "def int fill([]int xs):\n"
    "    xs[0] = 99\n"
    "    return 0\n"
    "def int set_entry(dict[str]P d):\n"
    "    d['a'] = P(99)\n"
    "    return 0\n"
    "def int bump(*int n):\n"
    "    *n = 99\n"
    "    return 0\n"
    "def int take_p(P p, int ignored):\n"
    "    return p.x\n"
    "def int take_arr([2]int a, int ignored):\n"
    "    return a[0]\n"
    "def int take_dict(dict[str]int d, int ignored):\n"
    "    return d['k']\n"
    "def int take_sum(U u, int ignored):\n"
    "    if u is int:\n"
    "        return u\n"
    "    return u.x\n"
    "def int take_int(int n, int ignored):\n"
    "    return n\n"
)

# (setup, expression): the expression's later operand changes what its first one names.
CASES = [
    ("P s = P(1)", "take_p(s, set(&s))"),                                   # a struct argument
    ("P s = P(1)", "s.get(set(&s))"),                                       # a value receiver
    ("P s = P(1)", "take_p(s, s.reset())"),                                 # changed by a pointer-receiver method
    ("Q q = Q(P(1), 0)", "take_p(q.p, set_q(&q))"),                         # a field
    ("Q q = Q(P(1), 0)", "take_p(q.p, set(&q.p))"),                         # ... changed through its own address
    ("[2]int a = [1, 1]", "take_arr(a, set_arr(&a))"),                      # an array
    ("[2]int a = [1, 1]", "take_arr(a, fill(a[:]))"),                       # ... changed through a slice of it
    ("dict[str]int d = dict[str]int{'k': 1}", "take_dict(d, set_dict(&d))"),  # a dict
    ("U u = P(1)", "take_sum(u, set_sum(&u))"),                             # a sum
    ("dict[str]P e = dict[str]P{'a': P(1)}", "take_p(e['a'], set_entry(e))"),  # a dict entry
    ("int n = 1", "take_int(n, bump(&n))"),                                 # a scalar whose address is taken
    ("int n = 1", "n + bump(&n)"),
    ("int n = 1", "n * 2 - bump(&n) - 1"),
]


@GCC_SKIP
def test_a_later_operand_cannot_change_an_earlier_ones_value():
    body = "".join(f"    if true:\n        {setup}\n        print({expression})\n" for setup, expression in CASES)
    assert_program_stdout(
        DECLS + "def int main():\n" + body +
        "    P s = P(1)\n"
        "    print(s == changed(&s))\n"          # compared as it was before the call
        "    print(changed(&s) == s)\n"          # ... and here the call comes first
        "    int n = 1\n"
        "    print(bump(&n) + n)\n"
        "    return 0\n",
        "1\n" * len(CASES) + "true\nfalse\n99\n",
    )


def _copies_in_main(call: str) -> int:
    program = build_ir_program(analyze(_parse(
        DECLS + f"def int main():\n    P s = P(1)\n    int n = 2\n    return {call}\n")))
    main = next(fn for fn in program.functions if fn.name == link_name('main'))
    return sum(isinstance(instr, IRCopy) for instr in main.body)


def test_an_argument_is_copied_only_when_a_later_one_could_change_it():
    assert _copies_in_main("take_p(s, n)") == 0  # the callee's copy is the only one
    assert _copies_in_main("take_p(s, take_int(n, n))") == 0  # no pointer or slice reaches s: no call can
    assert _copies_in_main("take_p(s, set(&s))") == 1
    assert _copies_in_main("take_p(changed(&s), set(&s))") == 0  # a call's result is already its own
