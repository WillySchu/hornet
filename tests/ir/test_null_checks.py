"""ir/null_checks.py."""

from ir.ir import IRBinOp, IRBranch, IRCall, IRConst, IRFunction, IRLabel, IRLoad, IRMove, IRNullCheck, IRReturn, Temp
from ir.null_checks import expand_null_checks
from ir.id_allocator import IdAllocator
from ir.ir import IRProgram
from ops import BinaryOp
from typesys import Type, TypeKind

PTR = Type(TypeKind.POINTER, element_type=Type.INT)


def _expand(body):
    program = IRProgram(functions=[], ids=IdAllocator())
    fn = IRFunction(name='f')
    fn.body = body
    expand_null_checks(fn, program)
    return fn.body, program


def _checks(body):
    return [i for i in body if isinstance(i, IRBinOp) and i.op == BinaryOp.EQUAL and i.right == IRConst(0, PTR)]


def test_each_check_becomes_a_branch_to_one_panic_block():
    p, q, v = Temp(1, PTR), Temp(2, PTR), Temp(3, Type.INT)
    body, program = _expand([IRNullCheck(p), IRLoad(v, p), IRNullCheck(q), IRLoad(v, q), IRReturn(None)])
    assert [c.left for c in _checks(body)] == [p, q]
    panics = [i for i in body if isinstance(i, IRCall) and i.name == 'hornet_panic']
    assert len(panics) == 1 and ('dereference of none' in dict(program.string_literals).values())


def test_a_pointer_already_checked_in_the_block_is_not_checked_again():
    p, v = Temp(1, PTR), Temp(3, Type.INT)
    body, _ = _expand([IRNullCheck(p), IRLoad(v, p), IRNullCheck(p), IRLoad(v, p), IRReturn(None)])
    assert len(_checks(body)) == 1


def test_a_rewritten_pointer_or_a_new_block_is_checked_again():
    p, q, v, c = Temp(1, PTR), Temp(2, PTR), Temp(3, Type.INT), Temp(4, Type.BOOL)
    body, _ = _expand([
        IRNullCheck(p), IRMove(p, q), IRNullCheck(p),
        IRBranch(c, 'L', 'L'), IRLabel('L'), IRNullCheck(p), IRLoad(v, p), IRReturn(None),
    ])
    assert len(_checks(body)) == 3


def test_none_itself_always_panics():
    body, _ = _expand([IRNullCheck(IRConst(0, PTR)), IRReturn(None)])
    assert len(_checks(body)) == 1
