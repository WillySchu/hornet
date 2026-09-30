"""Tests for copy coalescing, copy propagation, dead code elimination, and branch simplification."""

from ir.ir import (
    IRBinOp, IRBranch, IRCall, IRConst, IRFunction, IRJump, IRLabel, IRLocalAddress, IRMove, IRReadArgument, IRReturn, IRStore, Temp,
)
from ir.verify import verify_function
from ops import BinaryOp
from optimize.branch_simplification import simplify_branches
from optimize.copy_coalescing import coalesce_copies
from optimize.copy_propagation import propagate_copies
from optimize.dead_code import remove_dead_code
from optimize.optimizer import optimize_function, pinned_temps
from typesys import Type


def t(n: int, ty: Type = Type.INT) -> Temp:
    return Temp(id=n, type=ty)


def c(v: int, ty: Type = Type.INT) -> IRConst:
    return IRConst(v, ty)


def fn(*body) -> IRFunction:
    return IRFunction(name='f', body=list(body))


def add(dst, a, b):
    return IRBinOp(dst=dst, op=BinaryOp.ADD, left=a, right=b)


# -- coalescing ----------------------------------------------------------------

def test_coalesce_single_use_copy():
    f = fn(add(t(1), t(0), c(1)), IRMove(dst=t(2), src=t(1)), IRReturn(value=t(2)))
    coalesce_copies(f, set())
    assert f.body == [add(t(2), t(0), c(1)), IRReturn(value=t(2))]


def test_no_coalesce_when_temp_read_again():
    body = [add(t(1), t(0), c(1)), IRMove(dst=t(2), src=t(1)), IRReturn(value=t(1))]
    f = fn(*body)
    coalesce_copies(f, set())
    assert f.body == body


def test_no_coalesce_when_not_adjacent():
    body = [add(t(1), t(0), c(1)), IRMove(dst=t(3), src=c(0)), IRMove(dst=t(2), src=t(1)), IRReturn(value=t(2))]
    f = fn(*body)
    coalesce_copies(f, set())
    assert f.body == body


def test_no_coalesce_across_types_or_into_pinned():
    body = [add(t(1), t(0), c(1)), IRMove(dst=t(2, Type.INT32), src=t(1)), IRReturn(value=t(2, Type.INT32))]
    f = fn(*body)
    coalesce_copies(f, set())
    assert f.body == body
    body = [add(t(1), t(0), c(1)), IRMove(dst=t(2), src=t(1)), IRReturn(value=t(2))]
    f = fn(*body)
    coalesce_copies(f, {2})
    assert f.body == body


def test_coalesce_call_result():
    f = fn(IRCall(dst=t(1), name='g', args=[]), IRMove(dst=t(2), src=t(1)), IRReturn(value=t(2)))
    coalesce_copies(f, set())
    assert f.body == [IRCall(dst=t(2), name='g', args=[]), IRReturn(value=t(2))]


# -- propagation ---------------------------------------------------------------

def test_propagates_copies_and_constants_within_a_block():
    f = fn(IRMove(dst=t(1), src=t(0)), IRMove(dst=t(2), src=c(5)), add(t(3), t(1), t(2)), IRReturn(value=t(3)))
    propagate_copies(f, set())
    assert f.body[2] == add(t(3), t(0), c(5))


def test_redefinition_stops_propagation():
    f = fn(IRMove(dst=t(1), src=t(0)), IRMove(dst=t(0), src=c(9)), add(t(3), t(1), t(1)), IRReturn(value=t(3)))
    propagate_copies(f, set())
    assert f.body[2] == add(t(3), t(1), t(1))  # t0 changed, so t1 no longer equals it
    f = fn(IRMove(dst=t(1), src=c(4)), IRMove(dst=t(1), src=t(0)), IRReturn(value=t(1)))
    propagate_copies(f, set())
    assert f.body[2] == IRReturn(value=t(0))


def test_no_propagation_across_blocks():
    f = fn(IRMove(dst=t(1), src=c(4)), IRJump('.L'), IRLabel('.L'), IRReturn(value=t(1)))
    propagate_copies(f, set())
    assert f.body[3] == IRReturn(value=t(1))


def test_pinned_temps_are_not_propagated():
    body = [IRMove(dst=t(1), src=c(4)), IRCall(dst=None, name='g', args=[]), IRReturn(value=t(1))]
    f = fn(*body)
    propagate_copies(f, {1})
    assert f.body == body
    body = [IRMove(dst=t(2), src=t(1)), IRReturn(value=t(2))]
    f = fn(*body)
    propagate_copies(f, {1})
    assert f.body == body


# -- dead code -----------------------------------------------------------------

def test_removes_unused_pure_instructions():
    f = fn(add(t(1), t(0), c(1)), IRMove(dst=t(2), src=t(1)), IRReturn(value=t(0)))
    remove_dead_code(f, set())
    assert f.body == [IRReturn(value=t(0))]


def test_keeps_division_calls_stores_and_pinned_writes():
    body = [
        IRBinOp(dst=t(1), op=BinaryOp.DIVIDE, left=t(0), right=t(0)),
        IRBinOp(dst=t(2), op=BinaryOp.MODULO, left=t(0), right=t(0)),
        IRCall(dst=t(3), name='g', args=[]),
        IRStore(address=t(0), value=t(0), value_type=Type.INT),
        IRMove(dst=t(4), src=c(1)),
        IRReturn(value=None),
    ]
    f = fn(*body)
    remove_dead_code(f, {4})
    assert f.body == body


def test_keeps_loop_carried_value():
    body = [
        IRMove(dst=t(1), src=c(0)),
        IRJump('.top'),
        IRLabel('.top'),
        IRBranch(cond=t(9, Type.BOOL), true_label='.body', false_label='.end'),
        IRLabel('.body'),
        add(t(1), t(1), c(1)),
        IRJump('.top'),
        IRLabel('.end'),
        IRReturn(value=t(1)),
    ]
    f = fn(*body)
    remove_dead_code(f, set())
    assert f.body == body


# -- branches --------------------------------------------------------------------

def test_constant_branch_becomes_jump_and_dead_arm_is_removed():
    f = fn(
        IRBranch(cond=c(0, Type.BOOL), true_label='.then', false_label='.else'),
        IRLabel('.then'), IRReturn(value=c(1)),
        IRLabel('.else'), IRReturn(value=c(2)),
    )
    simplify_branches(f)
    assert f.body == [IRJump('.else'), IRLabel('.else'), IRReturn(value=c(2))]


# -- driver --------------------------------------------------------------------------

def test_pinned_temps_are_those_homed_in_address_taken_slots():
    f = fn(IRLocalAddress(dst=t(5), slot=3), IRReturn(value=None))
    assert pinned_temps(f, {1: 3, 2: 4}) == {1}


def test_optimize_function_result_still_verifies():
    f = fn(
        IRMove(dst=t(1), src=c(2)),
        IRBinOp(dst=t(2), op=BinaryOp.LESS_THAN, left=t(1), right=c(3)),
        IRBranch(cond=t(2), true_label='.a', false_label='.b'),
        IRLabel('.a'), add(t(3), t(0), t(1)), IRMove(dst=t(4), src=t(3)), IRReturn(value=t(4)),
        IRLabel('.b'), IRReturn(value=c(0)),
    )
    f.body.insert(0, IRReadArgument(dst=t(0), index=0))
    optimize_function(f, {})
    verify_function(f)
    assert f.body[-2:] == [add(t(3), t(0), t(1)), IRReturn(value=t(3))]
    assert not any(isinstance(i, (IRBranch, IRMove)) and getattr(i, 'dst', None) == t(4) for i in f.body)
    assert not any(isinstance(i, IRBranch) for i in f.body)
