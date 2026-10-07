"""ir/division_checks.py."""

from ir.ir import IRBinOp, IRCall, IRConst
from ir.program_builder import build_ir_program
from ir.typed_builder import link_name
from ops import BinaryOp
from tests.test_compiler import _parse, analyze


def _main_body(expr: str, type_: str = 'int'):
    source = f"def {type_} f({type_} n, {type_} d):\n    return {expr}\ndef int main():\n    return 0\n"
    program = _parse(source)
    program = analyze(program)
    built = build_ir_program(program, keep_unreachable=True)  # (main doesn't call f)
    fn = next(f for f in built.functions if f.name == link_name('f'))
    return fn.body


def _divisions(body):
    return [i for i in body if isinstance(i, IRBinOp) and i.op in (BinaryOp.DIVIDE, BinaryOp.MODULO)]


def _panics(body):
    return [i for i in body if isinstance(i, IRCall) and i.name == 'hornet_panic']


def test_variable_divisor_gets_zero_and_overflow_checks():
    body = _main_body("n / d")
    assert len(_panics(body)) == 2


def test_narrow_types_get_only_the_zero_check():
    assert len(_panics(_main_body("n % d", 'int8'))) == 1


def test_divisor_computed_from_constants_becomes_a_constant_with_no_checks():
    for expr, value in [("n / -3", -3), ("n % int32(-9)", -9)]:
        body = _main_body(expr, 'int32' if 'int32' in expr else 'int')
        assert [d.right for d in _divisions(body)] == [IRConst(value, _divisions(body)[0].right.type)]
        assert not _panics(body)


def test_constant_minus_one_gets_only_the_overflow_check():
    body = _main_body("n / -1")
    assert len(_panics(body)) == 1
