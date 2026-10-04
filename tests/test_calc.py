"""examples/calc, the integer calculator: its sample input on every target, and random
expressions checked against a Python model of the same arithmetic (64-bit wraparound, division
that truncates, `and`/`or` that stop early, errors with their columns)."""

import random
import subprocess
from pathlib import Path

import pytest

from build import build_executable, run_prefix
from tests.targets import E2E_TARGETS
from tests.test_compiler import GCC_SKIP

CALC = Path(__file__).resolve().parent.parent / 'examples' / 'calc'

pytestmark = GCC_SKIP


@pytest.fixture(scope='session', params=E2E_TARGETS, ids=str)
def calc(request, tmp_path_factory):
    """Command running the calculator built for each E2E target."""
    exe = tmp_path_factory.mktemp(f'calc-{request.param}') / 'calc'
    build_executable(str(CALC / 'main.ht'), str(exe), target=request.param)
    return run_prefix(request.param) + [str(exe)]


def _run(calc, *args, stdin=None):
    return subprocess.run([*calc, *args], input=stdin, capture_output=True, text=True, timeout=60)


def test_the_sample_input(calc):
    result = _run(calc, str(CALC / 'demo.calc'))
    assert result.stdout.split() == [
        '7', '9', '3', '-6', '-2', '3', '-3', '1',                 # arithmetic
        '82', '1', '0', '1', '1',                                   # variables, comparisons, and/or/not
        '-9223372036854775808', '-9223372036854775808', '-9223372036854775808',  # wraparound
        '0', '1']                                                   # `and` and `or` stopping early
    name = str(CALC / 'demo.calc')
    assert result.stderr == ''.join(f"{name}:{where}: error: {message}\n" for where, message in [
        ('24:3', "division by zero"), ('25:1', "undefined variable 'z'"), ('26:4', "unexpected end of line"),
        ('27:7', "expected ')'"), ('28:3', "unexpected '2'"), ('29:3', "unexpected character '$'"),
        ('30:1', "number too large")])
    assert result.returncode == 1


def test_standard_input_and_a_missing_file(calc):
    result = _run(calc, stdin="n = 20\r\nn * 2 + 2  # comment\r\n\r\n")  # lines may end in \\r\\n
    assert (result.returncode, result.stdout, result.stderr) == (0, "42\n", "")
    missing = _run(calc, "no-such-file.calc")
    assert missing.returncode == 1 and missing.stderr.startswith("calc: could not open 'no-such-file.calc': ")


# ---- a model of the calculator

LEVEL = {'or': 1, 'and': 2, '==': 4, '!=': 4, '<': 4, '<=': 4, '>': 4, '>=': 4, '+': 5, '-': 5, '*': 6, '/': 6, '%': 6}
NOT_LEVEL, NEGATE_LEVEL, ATOM_LEVEL = 3, 7, 8
VARIABLES = {'a': 7, 'b': -3, 'c': 0}


class Failed(Exception):
    def __init__(self, message, col):
        self.message, self.col = message, col


def _wrap(value: int) -> int:
    return (value + 2 ** 63) % 2 ** 64 - 2 ** 63


def _random_tree(rng, depth):
    choice = rng.random()
    if depth == 0 or choice < 0.25:
        if rng.random() < 0.3:
            return ['name', rng.choice(['a', 'b', 'c', 'undefined'] if rng.random() < 0.1 else ['a', 'b', 'c'])]
        return ['num', rng.choice([0, 1, 2, 3, 10, 2 ** 31, 2 ** 62, 2 ** 63 - 1, rng.randrange(1000)])]
    if choice < 0.35:
        return [rng.choice(['neg', 'not']), _random_tree(rng, depth - 1)]
    return ['binary', rng.choice(list(LEVEL)), _random_tree(rng, depth - 1), _random_tree(rng, depth - 1)]


def _level(tree) -> int:
    return {'num': ATOM_LEVEL, 'name': ATOM_LEVEL, 'neg': NEGATE_LEVEL, 'not': NOT_LEVEL}.get(tree[0]) or LEVEL[tree[1]]


def _render(tree, out: list, at_least: int) -> None:
    """Append the tree's text to `out` with only the parentheses its place needs (an operand must
    bind at least as tightly as `at_least`), noting the column of each name and binary operator."""
    def text():
        return ''.join(out)
    parenthesized = _level(tree) < at_least
    if parenthesized:
        out.append('(')
    if tree[0] == 'num':
        out.append(str(tree[1]))
    elif tree[0] == 'name':
        tree.append(len(text()) + 1)
        out.append(tree[1])
    elif tree[0] == 'neg':
        out.append('-')
        _render(tree[1], out, NEGATE_LEVEL)
    elif tree[0] == 'not':
        out.append('not ')
        _render(tree[1], out, NOT_LEVEL)
    else:
        _render(tree[2], out, LEVEL[tree[1]])  # operators of one level group from the left
        out.append(' ')
        tree.append(len(text()) + 1)
        out.append(tree[1] + ' ')
        _render(tree[3], out, LEVEL[tree[1]] + 1)
    if parenthesized:
        out.append(')')


def _value(tree) -> int:
    kind = tree[0]
    if kind == 'num':
        return tree[1]
    if kind == 'name':
        if tree[1] not in VARIABLES:
            raise Failed(f"undefined variable '{tree[1]}'", tree[2])
        return VARIABLES[tree[1]]
    if kind == 'neg':
        return _wrap(-_value(tree[1]))
    if kind == 'not':
        return int(_value(tree[1]) == 0)
    _, op, left_tree, right_tree, col = tree
    left = _value(left_tree)
    if op == 'and' and left == 0:
        return 0
    if op == 'or' and left != 0:
        return 1
    right = _value(right_tree)
    if op in ('and', 'or'):
        return int(right != 0)
    if op in ('/', '%'):
        if right == 0:
            raise Failed("division by zero", col)
        quotient = abs(left) // abs(right) * (1 if (left < 0) == (right < 0) else -1)  # toward zero
        return _wrap(quotient) if op == '/' else left - right * quotient
    return _wrap({'+': left + right, '-': left - right, '*': left * right, '==': left == right, '!=': left != right,
                  '<': left < right, '<=': left <= right, '>': left > right, '>=': left >= right}[op])


def test_random_expressions_agree_with_the_model(calc, tmp_path):
    rng = random.Random(20261004)
    lines = [f"{name} = {value}" if value >= 0 else f"{name} = -{-value}" for name, value in VARIABLES.items()]
    stdout, stderr = [], []
    path = tmp_path / "random.calc"
    for _ in range(400):
        tree = _random_tree(rng, depth=rng.randrange(1, 6))
        parts: list = []
        _render(tree, parts, 0)
        lines.append(''.join(parts))
        try:
            stdout.append(str(_value(tree)))
        except Failed as failed:
            stderr.append(f"{path}:{len(lines)}:{failed.col}: error: {failed.message}")
    path.write_text('\n'.join(lines) + '\n')
    result = _run(calc, str(path))
    assert len(stderr) > 20 and len(stdout) > 200  # the model itself exercises both
    assert result.stdout.split('\n')[:-1] == stdout
    assert result.stderr.split('\n')[:-1] == stderr
    assert result.returncode == 1
