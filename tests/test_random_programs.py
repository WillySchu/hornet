"""Seeded random programs checked end to end against a Python model of Hornet's integer semantics.

Many live variables, values live across calls (memory operands), array loads and stores,
branches, and 64/32-bit arithmetic. int8/uint8 appear only through casts. Operand aliasing
and illegal-operand cases are covered more directly by tests/backend/x86_64/test_direct_selection.py.
"""

import random

import pytest

from tests.test_compiler import GCC_SKIP, assert_program_stdout

N_INT = 18
N_I32 = 4
NARROW = {'b0': 'int8', 'b1': 'int8', 'u0': 'uint8', 'u1': 'uint8'}
ARR = 4
LOOP = 3


def _wrap(v: int, bits: int) -> int:
    v &= (1 << bits) - 1
    return v - (1 << bits) if v >> (bits - 1) else v


def _div(a: int, b: int) -> int:
    q = abs(a) // abs(b)
    return q if (a < 0) == (b < 0) else -q


def _mod(a: int, b: int) -> int:
    return a - _div(a, b) * b


class Gen:
    """Builds (source, expected_stdout) for one seed."""

    def __init__(self, seed: int):
        self.r = random.Random(seed)
        self.calls = seed % 2 == 0  # values live across calls vs. pure register pressure

    # Expressions are (hornet_text, python_eval) pairs; python_eval takes the state dict.

    def const(self, bits: int):
        r = self.r
        choice = r.random()
        if choice < 0.6:
            v = r.randint(-20, 300)
        elif choice < 0.85:
            v = r.randint(-2 ** 31, 2 ** 31 - 1) if bits == 64 else r.randint(-2 ** 20, 2 ** 20)
        else:
            v = r.randint(-2 ** 62, 2 ** 62) if bits == 64 else r.randint(-2 ** 31, 2 ** 31 - 1)
        text = f"({v})" if v < 0 else str(v)
        if bits == 32:
            text = f"int32({text})"
        return text, (lambda s, v=v: v)

    def var(self, bits: int):
        name = f"v{self.r.randrange(N_INT)}" if bits == 64 else f"w{self.r.randrange(N_I32)}"
        return name, (lambda s, n=name: s[n])

    def expr(self, bits: int, depth: int = 0, prefer=None):
        r = self.r
        if prefer is not None and r.random() < 0.5 and depth == 0:
            target = (prefer, lambda s, n=prefer: s[n])
            other = self.expr(bits, depth + 1)
            return self.binop(bits, target, other) if r.random() < 0.5 else self.binop(bits, other, target)
        roll = r.random()
        if depth >= 2 or roll < 0.3:
            return self.var(bits) if r.random() < 0.7 else self.const(bits)
        if roll < 0.75:
            return self.binop(bits, self.expr(bits, depth + 1), self.expr(bits, depth + 1))
        if roll < 0.82:
            op = r.choice(['-', '~'])
            t, f = self.expr(bits, depth + 1)
            fn = (lambda s: -f(s)) if op == '-' else (lambda s: ~f(s))
            return f"({op}{t})", (lambda s: _wrap(fn(s), bits))
        if roll < 0.9:
            t, f = self.expr(bits, depth + 1)
            narrow = r.choice(['int8', 'uint8'])
            model = (lambda s: _wrap(f(s), 8)) if narrow == 'int8' else (lambda s: f(s) & 255)
            back = 'int' if bits == 64 else 'int32'
            return f"{back}({narrow}({t}))", model
        if bits == 64 and r.random() < 0.5:
            t, f = self.expr(32, depth + 1)
            return f"int({t})", f
        if bits == 64 and self.calls:
            a, fa = self.expr(64, depth + 1)
            b, fb = self.expr(64, depth + 1)
            return f"mix({a}, {b})", (lambda s: _wrap(_wrap(fa(s) * 3, 64) ^ fb(s), 64))
        if bits == 64:
            return self.var(64)
        t, f = self.expr(64, depth + 1)
        return f"int32({t})", (lambda s: _wrap(f(s), 32))

    def binop(self, bits: int, left, right):
        r = self.r
        (lt, lf), (rt, rf) = left, right
        op = r.choice(['+', '-', '*', '&', '|', '^', '<<', '>>', '/', '%', 'arr'] if bits == 64
                      else ['+', '-', '*', '&', '|', '^', '<<', '>>', '/', '%'])
        mask = 63 if bits == 64 else 31
        c255 = '255' if bits == 64 else 'int32(255)'
        c1 = '1' if bits == 64 else 'int32(1)'
        cm = str(mask) if bits == 64 else f'int32({mask})'
        if op == 'arr':
            return f"arr[{rt} & 3]", (lambda s: s['arr'][rf(s) & 3])
        if op in ('<<', '>>'):
            text = f"({lt} {op} ({rt} & {cm}))"
            if op == '<<':
                return text, (lambda s: _wrap(lf(s) << (rf(s) & mask), bits))
            return text, (lambda s: lf(s) >> (rf(s) & mask))
        if op in ('/', '%'):
            fn = _div if op == '/' else _mod
            if r.random() < 0.4:  # literal divisor: multiply-high lowering
                k = r.choice([2, 3, 7, 10, 16, 255, 1000003, -5, -8, 2 ** 40 + 3] if bits == 64
                             else [2, 3, 7, 10, 16, 255, 1000003, -5, -8, 65537])
                kt = f"({k})" if k < 0 else str(k)
                if bits == 32:
                    kt = f"int32({kt})"
                return f"({lt} {op} {kt})", (lambda s, k=k: _wrap(fn(lf(s), k), bits))
            text = f"({lt} {op} (({rt} & {c255}) | {c1}))"
            return text, (lambda s: _wrap(fn(lf(s), (rf(s) & 255) | 1), bits))
        py = {'+': lambda a, b: a + b, '-': lambda a, b: a - b, '*': lambda a, b: a * b,
              '&': lambda a, b: a & b, '|': lambda a, b: a | b, '^': lambda a, b: a ^ b}[op]
        return f"({lt} {op} {rt})", (lambda s: _wrap(py(lf(s), rf(s)), bits))

    def nexpr(self, ty: str, depth: int = 0):
        """int8/uint8 expression; the model wraps every result to 8 bits."""
        r = self.r
        signed = ty == 'int8'
        w = (lambda v: _wrap(v, 8)) if signed else (lambda v: v & 255)
        roll = r.random()
        if depth >= 2 or roll < 0.35:
            if r.random() < 0.6:
                name = r.choice([n for n, t in NARROW.items() if t == ty])
                return name, (lambda s, n=name: s[n])
            v = w(r.randint(-300, 300))
            return f"{ty}({v})" if v >= 0 else f"{ty}(({v}))", (lambda s, v=v: v)
        if roll < 0.5:
            t, f = self.expr(64, 2)
            return f"{ty}({t})", (lambda s: w(f(s)))
        if roll < 0.6:
            t, f = self.nexpr(ty, depth + 1)
            if signed:
                return f"(-{t})", (lambda s: w(-f(s)))
            return f"(~{t})", (lambda s: w(~f(s)))
        (lt, lf), (rt, rf) = self.nexpr(ty, depth + 1), self.nexpr(ty, depth + 1)
        op = r.choice(['+', '+', '-', '*', '*', '&', '|', '^', '/', '%', '<<', '>>'])
        seven, one = f"{ty}(7)", f"{ty}(1)"
        if op in ('/', '%'):
            fn = _div if op == '/' else _mod
            return f"({lt} {op} (({rt} & {seven}) | {one}))", (lambda s: w(fn(lf(s), (rf(s) & 7) | 1)))
        if op in ('<<', '>>'):
            if op == '<<':
                return f"({lt} << ({rt} & {seven}))", (lambda s: w(lf(s) << (rf(s) & 7)))
            return f"({lt} >> ({rt} & {seven}))", (lambda s: lf(s) >> (rf(s) & 7))
        py = {'+': lambda a, b: a + b, '-': lambda a, b: a - b, '*': lambda a, b: a * b,
              '&': lambda a, b: a & b, '|': lambda a, b: a | b, '^': lambda a, b: a ^ b}[op]
        return f"({lt} {op} {rt})", (lambda s: w(py(lf(s), rf(s))))

    def cond(self):
        r = self.r
        if r.random() < 0.25:
            ty = r.choice(['int8', 'uint8'])
            (a, fa), (b, fb) = self.nexpr(ty, 1), self.nexpr(ty, 1)
            op = r.choice(['<', '>', '=='])
            py = {'<': lambda x, y: x < y, '>': lambda x, y: x > y, '==': lambda x, y: x == y}[op]
            return f"{a} {op} {b}", (lambda s: py(fa(s), fb(s)))
        bits = r.choice([64, 32])
        if r.random() < 0.5:
            (a, fa), (b, fb) = self.var(bits), self.var(bits)
        else:
            (a, fa), (b, fb) = self.expr(bits, 1), self.expr(bits, 1)
        op = r.choice(['<', '<=', '>', '>=', '==', '!='])
        py = {'<': lambda x, y: x < y, '<=': lambda x, y: x <= y, '>': lambda x, y: x > y,
              '>=': lambda x, y: x >= y, '==': lambda x, y: x == y, '!=': lambda x, y: x != y}[op]
        return f"{a} {op} {b}", (lambda s: py(fa(s), fb(s)))

    def assign(self, indent: str):
        r = self.r
        roll = r.random()
        if roll < 0.1:
            (it, itf), (et, ef) = self.expr(64, 1), self.expr(64)
            def run(s):
                idx, val = itf(s) & 3, ef(s)
                s['arr'][idx] = val
            return [f"{indent}arr[{it} & 3] = {et}"], run
        if roll < 0.2:
            ct, cf = self.cond()
            def run(s):
                s['flag'] = cf(s)
            return [f"{indent}flag = {ct}"], run
        if roll < 0.32:
            target = r.choice(list(NARROW))
            t, f = self.nexpr(NARROW[target])
            def run(s):
                s[target] = f(s)
            return [f"{indent}{target} = {t}"], run
        bits = 64 if roll < 0.82 else 32
        target = f"v{r.randrange(N_INT)}" if bits == 64 else f"w{r.randrange(N_I32)}"
        t, f = self.expr(bits, prefer=target)
        def run(s):
            s[target] = f(s)
        return [f"{indent}{target} = {t}"], run

    def statement(self, indent: str):
        r = self.r
        if r.random() < 0.2:
            use_flag = r.random() < 0.4
            if use_flag:
                ct, cf = 'flag', (lambda s: s['flag'])
            else:
                ct, cf = self.cond()
            then_lines, then_run = self.assign(indent + '    ')
            else_lines, else_run = self.assign(indent + '    ')
            lines = [f"{indent}if {ct}:"] + then_lines + [f"{indent}else:"] + else_lines
            def run(s):
                (then_run if cf(s) else else_run)(s)
            return lines, run
        return self.assign(indent)

    def program(self):
        r = self.r
        state = {'arr': [0] * ARR, 'flag': False}
        lines = ["def int mix(int a, int b):", "    return (a * 3) ^ b", "", "def int main():"]
        for i in range(N_INT):
            t, f = self.const(64)
            lines.append(f"    int v{i} = {t}")
            state[f"v{i}"] = f(state)
        for i in range(N_I32):
            t, f = self.const(32)
            lines.append(f"    int32 w{i} = {t}")
            state[f"w{i}"] = f(state)
        for name, ty in NARROW.items():
            v = r.randint(-128, 127) if ty == 'int8' else r.randint(0, 255)
            state[name] = v
            lines.append(f"    {ty} {name} = {ty}({v})" if v >= 0 else f"    {ty} {name} = {ty}(({v}))")
        lines.append(f"    [{ARR}]int arr")
        lines.append("    bool flag = false")
        body = [self.statement('        ') for _ in range(r.randint(10, 18))]
        lines.append("    int i = 0")
        lines.append(f"    while i < {LOOP}:")
        for stmt_lines, _ in body:
            lines.extend(stmt_lines)
        lines.append("        i = i + 1")
        for _ in range(LOOP):
            for _, run in body:
                run(state)
        out = []
        for i in range(N_INT):
            lines.append(f"    print(v{i})")
            out.append(str(state[f"v{i}"]))
        for i in range(N_I32):
            lines.append(f"    print(w{i})")
            out.append(str(state[f"w{i}"]))
        for name in NARROW:
            lines.append(f"    print(int({name}))")
            out.append(str(state[name]))
        lines.append("    print(arr)")
        out.append(f"[{ARR}]int[" + ', '.join(str(x) for x in state['arr']) + "]")
        lines.append("    print(flag)")
        out.append('true' if state['flag'] else 'false')
        lines.append("    return 0")
        return '\n'.join(lines) + '\n', '\n'.join(out) + '\n'


@GCC_SKIP
@pytest.mark.parametrize('seed', range(40))
def test_random_program_matches_model(seed):
    source, expected = Gen(seed).program()
    assert_program_stdout(source, expected)
