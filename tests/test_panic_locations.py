"""A runtime check that fails reports where: `file:line:col: panic: message`, the position being the
start of the expression (or statement) that was checked and the file named without its directory."""

import signal

import pytest

from build import build_executable
from ir.ir import IRBoundsCheck, IRConst
from ir.panics import located
from ir.program_builder import build_ir_program
from target import default_target
from tests.targets import run_binary
from tests.test_compiler import GCC_SKIP, _parse, analyze, compile_and_run
from typesys import Type

DECLS = (
    "type P struct:\n"      # lines 1-7
    "    int x\n"
    "    def bump(*p):\n"
    "        p.x += 1\n"
    "type A struct:\n"
    "    str s\n"
    "type U is A | P\n"
)

# (statements of main, from line 9; `line:col: message` of the panic)
CASES = [
    ("    [3]int a = [1, 2, 3]\n    int i = 5\n    return a[i]\n", "11:12: index out of bounds: index 5, length 3"),
    ("    []int xs = [1, 2, 3]\n    int i = 5\n    return 100 + xs[i]\n",
     "11:18: index out of bounds: index 5, length 3"),
    ("    str s = 'abc'\n    int i = 5\n    return int(s[i])\n", "11:16: index out of bounds: index 5, length 3"),
    # Each check reports its own expression.
    ("    [3]int a = [1, 2, 3]\n    [3]int b = [1, 2, 3]\n    int i = 5\n    return a[0] + b[i]\n",
     "12:19: index out of bounds: index 5, length 3"),
    ("    [3]int a = [1, 2, 3]\n    int i = 5\n    return (a[0] +\n            a[i])\n",
     "12:13: index out of bounds: index 5, length 3"),
    ("    []int xs = [1, 2, 3]\n    int n = 9\n    []int t = xs[1:n]\n    return len(t)\n",
     "11:15: slice bounds out of range: end 9, capacity 3"),
    ("    *int p = none\n    return *p\n", "10:12: dereference of none"),
    ("    *P p = none\n    return p.x\n", "10:12: dereference of none"),
    ("    *P p = none\n    p.bump()\n    return 0\n", "4:9: dereference of none"),  # inside the method
    ("    int z = 0\n    return 10 / z\n", "10:12: integer division by zero"),
    ("    int z = 0\n    return 10 % z\n", "10:12: integer division by zero"),
    ("    int m = -9223372036854775808\n    int n = -1\n    return m / n\n", "11:12: integer overflow in division"),
    ("    int x = 10\n    int z = 0\n    x /= z\n    return x\n", "11:5: integer division by zero"),
    ("    dict[str]int d\n    return d['k']\n", "10:12: dict lookup: key not found"),
    ("    dict[str]int d\n    d['k'] += 1\n    return 0\n", "10:5: dict lookup: key not found"),
    ("    dict[str]int d\n    del(d, 'k')\n    return 0\n", "10:5: dict delete: key not found"),
    ("    []int xs = [1, 2]\n    for x in xs:\n        xs = append(xs, x)\n    return 0\n",
     "10:5: for ... in: slice was reallocated (e.g. by append) during iteration"),
    ("    dict[int]int d\n    d[-1] = 1\n    d[-2] = 2\n    for k in d:\n        for int i = 0; i < 9; i += 1:\n"
     "            d[i] = i\n    return 0\n",
     "12:5: for ... in: dict's own buckets were reallocated (e.g. by an insert that triggered growth) "
     "during iteration"),
    ("    U u = A('a')\n    *U p = &u\n    if u is A:\n        *p = P(1)\n        return len(u.s)\n    return 0\n",
     "13:20: 'u' changed variant while narrowed"),
]


@GCC_SKIP
@pytest.mark.parametrize("body,expected", CASES, ids=[c[1] for c in CASES])
def test_each_check_reports_its_position(body, expected):
    result = compile_and_run(DECLS + "def int main():\n" + body)
    position, message = expected.split(": ", 1)
    assert (result.returncode, result.stderr) == (-signal.SIGABRT, f"program.lang:{position}: panic: {message}\n")


@GCC_SKIP
def test_a_module_is_named_by_its_file(tmp_path):
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "picker.ht").write_text("def int pick([]int xs, int i):\n    return xs[i]\n")
    (tmp_path / "main.ht").write_text(
        "from 'lib/picker' import pick\n\ndef int main():\n    []int xs = [1, 2]\n    return pick(xs, 2)\n")
    build_executable(str(tmp_path / "main.ht"), str(tmp_path / "main"))
    result = run_binary(default_target(), [tmp_path / "main"], capture_output=True, text=True)
    assert (result.returncode, result.stderr) == (
        -signal.SIGABRT, "picker.ht:2:12: panic: index out of bounds: index 2, length 2\n")


def test_positions_reach_the_ir_and_are_optional():
    program = build_ir_program(analyze(_parse("def int main():\n    [3]int a\n    int i = 5\n    return a[i]\n")))
    checks = [instr for fn in program.functions for instr in fn.body if isinstance(instr, IRBoundsCheck)]
    assert [check.where for check in checks] == ["program.lang:4:12"]
    # A position isn't part of what an instruction is, and a check built without one still panics.
    assert checks[0] == IRBoundsCheck(index=checks[0].index, length=IRConst(3, Type.INT))
    assert located("index out of bounds", "program.lang:4:12") == \
        "program.lang:4:12: panic: index out of bounds"
    assert located("index out of bounds", None) == "index out of bounds"
