"""Safepoint checks (optimize/safepoints.py, runtime/safepoint.c), placed with HORNET_SAFEPOINTS set: one
on each loop and at the entry of each recursive function; and, with every check calling into the
runtime (HORNET_SAFEPOINT_STRESS), programs behave as without them, on every target."""

import signal

from ir.ir import IRSafepoint
from ir.program_builder import build_ir_program
from optimize.optimizer import optimize
from tests.targets import build_and_run, on_every_target
from tests.test_compiler import GCC_SKIP, _parse, analyze

SOURCE = (
    "def int leaf(int n):\n"                       # no loop, no recursion: no check
    "    return n * 2 + 1\n"
    "def int calls(int n):\n"                      # calls, but isn't in a cycle of calls: no check
    "    return leaf(n) + leaf(n + 1)\n"
    "def int down(int n):\n"                       # recursive: a check at its entry
    "    if n == 0:\n"
    "        return 0\n"
    "    return up(n - 1) + 1\n"
    "def int up(int n):\n"                         # in the same cycle
    "    return down(n)\n"
    "def int loops(int n):\n"                      # two loops, one inside the other: two checks
    "    int total = 0\n"
    "    for int i = 0; i < n; i += 1:\n"
    "        int j = 0\n"
    "        while j < i:\n"
    "            total += j * i\n"
    "            j += 1\n"
    "    return total\n"
    "def int main():\n"
    "    int a = 1\n"
    "    int b = 2\n"
    "    int c = 3\n"
    "    int d = 4\n"
    "    int e = 5\n"
    "    int f = 6\n"
    "    int sum = 0\n"
    "    for int i = 0; i < 3000; i += 1:\n"        # many values live across the check
    "        a += b\n        b += c ^ i\n        c += d\n        d += e & i\n        e += f\n        f += a | i\n"
    "        sum += calls(i) + (a ^ b ^ c ^ d ^ e ^ f) % 1000\n"
    "    print(sum)\n"
    "    print(down(2000))\n"
    "    print(loops(300))\n"
    "    return 0\n"
)


def _checks(monkeypatch) -> dict:
    monkeypatch.setenv('HORNET_SAFEPOINTS', '1')
    program = optimize(build_ir_program(analyze(_parse(SOURCE))))
    return {fn.name: sum(isinstance(i, IRSafepoint) for i in fn.body) for fn in program.functions}


def test_where_checks_go(monkeypatch):
    checks = _checks(monkeypatch)
    assert {name.rstrip('$'): n for name, n in checks.items()} == {
        'leaf': 0, 'calls': 0, 'down': 1, 'up': 1, 'loops': 2, 'main': 1}


def test_none_without_the_setting(monkeypatch):
    monkeypatch.delenv('HORNET_SAFEPOINTS', raising=False)
    program = optimize(build_ir_program(analyze(_parse(SOURCE))))
    assert not any(isinstance(i, IRSafepoint) for fn in program.functions for i in fn.body)


@GCC_SKIP
def test_every_check_calling_in_changes_nothing(monkeypatch):
    plain = on_every_target(lambda target: build_and_run(SOURCE, target))
    monkeypatch.setenv('HORNET_SAFEPOINTS', '1')
    monkeypatch.setenv('HORNET_SAFEPOINT_STRESS', '7')  # (and a collection at every 7th)
    checked = on_every_target(lambda target: build_and_run(SOURCE, target))
    assert (checked.returncode, checked.stdout, checked.stderr) == (plain.returncode, plain.stdout, plain.stderr)
    assert plain.returncode == 0


@GCC_SKIP
def test_a_stack_overflow_in_a_safepoint_still_names_the_function(monkeypatch):
    # Every call calling in (and collecting at every 1000th), deep in a recursion: the stack may run
    # out under the stub, and the handler unwinds through it to the program's function.
    monkeypatch.setenv('HORNET_SAFEPOINTS', '1')
    monkeypatch.setenv('HORNET_SAFEPOINT_STRESS', '1000')
    source = ("def int depth(int n):\n    return depth(n + 1) + 1\n"
              "def int main():\n    print('before')\n    return depth(0)\n")

    def run(target):
        result = build_and_run(source, target, timeout=60)
        assert (result.returncode, result.stdout) == (-signal.SIGABRT, "before\n"), target
        assert result.stderr == "panic: stack overflow in 'depth'\n", target
        return result
    on_every_target(run)
