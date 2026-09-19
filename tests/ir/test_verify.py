"""Tests for ir/verify.py -- one test per check listed in its own
module docstring, plus a few edge cases the docstring calls out
explicitly (def-after-use, IRSliceGrow's two dsts)."""

import pytest

from ir.ir import (
    IRBinOp,
    IRBranch,
    IRCall,
    IRFunction,
    IRJump,
    IRLabel,
    IRLocalAddress,
    IRMove,
    IRProgram,
    IRReadArgument,
    IRReturn,
    IRSliceGrow,
    Temp,
)
from ir.verify import IRVerificationError, verify_function, verify_program
from parser import BinaryOp
from semantic import Type


def t(n: int) -> Temp:
    return Temp(id=n, type=Type.INT)


def fn(name: str = 'f', **kwargs) -> IRFunction:
    return IRFunction(name=name, **kwargs)


# -- non-empty body -----------------------------------------------------

def test_empty_body_raises():
    with pytest.raises(IRVerificationError, match="f: empty body"):
        verify_function(fn(body=[]))


# -- duplicate labels -----------------------------------------------------

def test_duplicate_label_raises():
    body = [
        IRLabel('.l'),
        IRLabel('.l'),
        IRReturn(value=None),
    ]
    with pytest.raises(IRVerificationError, match="f: duplicate label '.l'"):
        verify_function(fn(body=body))


# -- every block ends in exactly one terminator --------------------------

def test_falls_off_end_without_terminator_raises():
    body = [IRMove(dst=t(0), src=t(0))]
    with pytest.raises(IRVerificationError, match="falls through off the end of body"):
        verify_function(fn(body=body))


def test_falls_through_into_label_without_terminator_raises():
    body = [
        IRMove(dst=t(0), src=t(0)),
        IRLabel('.next'),
        IRReturn(value=None),
    ]
    with pytest.raises(IRVerificationError, match="falls through into label '.next'"):
        verify_function(fn(body=body))


def test_two_consecutive_labels_without_terminator_raises():
    # The first label is itself a non-terminator op immediately
    # followed by another label -- same "falls through into a label"
    # case as any other op, not a special case of its own.
    body = [IRLabel('.a'), IRLabel('.b'), IRReturn(value=None)]
    with pytest.raises(IRVerificationError, match="falls through into label '.b'"):
        verify_function(fn(body=body))


def test_valid_straight_line_passes():
    body = [
        IRMove(dst=t(0), src=t(0)),
        IRReturn(value=t(0)),
    ]
    verify_function(fn(body=body))  # no raise


# -- jump/branch targets name a real label ---------------------------------

def test_jump_to_undefined_label_raises():
    body = [IRJump('.nowhere')]
    with pytest.raises(IRVerificationError, match="IRJump targets undefined label '.nowhere'"):
        verify_function(fn(body=body))


def test_branch_true_label_undefined_raises():
    body = [
        IRBranch(cond=t(0), true_label='.nowhere', false_label='.else'),
        IRLabel('.else'),
        IRReturn(value=None),
    ]
    with pytest.raises(IRVerificationError, match="IRBranch targets undefined label '.nowhere'"):
        verify_function(fn(body=body))


def test_branch_false_label_undefined_raises():
    body = [
        IRBranch(cond=t(0), true_label='.then', false_label='.nowhere'),
        IRLabel('.then'),
        IRReturn(value=None),
    ]
    with pytest.raises(IRVerificationError, match="IRBranch targets undefined label '.nowhere'"):
        verify_function(fn(body=body))


def test_valid_branch_and_jump_passes():
    body = [
        IRMove(dst=t(0), src=t(0)),
        IRBranch(cond=t(0), true_label='.then', false_label='.else'),
        IRLabel('.then'),
        IRJump('.end'),
        IRLabel('.else'),
        IRJump('.end'),
        IRLabel('.end'),
        IRReturn(value=None),
    ]
    verify_function(fn(body=body))  # no raise


# -- IRLocalAddress/hidden_return_ptr_slot name a real slot ---------------

def test_local_address_undefined_slot_raises():
    body = [
        IRLocalAddress(dst=t(0), slot=3),
        IRReturn(value=None),
    ]
    with pytest.raises(IRVerificationError, match="references unknown slot 3"):
        verify_function(fn(body=body, slot_widths={}))


def test_local_address_defined_slot_passes():
    body = [
        IRLocalAddress(dst=t(0), slot=3),
        IRReturn(value=None),
    ]
    verify_function(fn(body=body, slot_widths={3: 8}))  # no raise


def test_hidden_return_ptr_slot_undefined_raises():
    body = [IRReturn(value=None)]
    with pytest.raises(IRVerificationError, match="hidden_return_ptr_slot references unknown slot 5"):
        verify_function(fn(body=body, slot_widths={}, hidden_return_ptr_slot=5))


# -- every read Temp is defined somewhere in the function ------------------

def test_use_before_any_def_raises():
    body = [IRReturn(value=t(0))]  # t(0) is never written by anything
    with pytest.raises(IRVerificationError, match="Temp 0 used by .* but never defined"):
        verify_function(fn(body=body))


def test_def_after_use_still_passes():
    """Documented as deliberate: this checks a Temp is defined
    SOMEWHERE in the function, not that the def dominates the use."""
    body = [
        IRJump('.later'),
        IRLabel('.later'),
        IRMove(dst=t(0), src=t(0)),  # t(0) "read" here counts as fine too
        IRReturn(value=t(0)),
    ]
    verify_function(fn(body=body))  # no raise


def test_read_argument_counts_as_definition():
    body = [
        IRReadArgument(dst=t(0), index=0),
        IRReturn(value=t(0)),
    ]
    verify_function(fn(body=body))  # no raise


def test_slice_grow_defines_both_dst_temps():
    body = [
        IRSliceGrow(dst_ptr=t(0), dst_cap=t(1), ptr=t(2), length=t(2), cap=t(2), element_width=4),
        IRReadArgument(dst=t(2), index=0),  # define t(2), used above -- order doesn't matter (see above)
        IRBinOp(dst=t(3), op=BinaryOp.ADD, left=t(0), right=t(1)),
        IRReturn(value=t(3)),
    ]
    verify_function(fn(body=body))  # no raise


def test_call_argument_must_be_defined_raises():
    body = [
        IRCall(dst=None, name='some_func', args=[t(0)]),  # t(0) never defined
        IRReturn(value=None),
    ]
    with pytest.raises(IRVerificationError, match="Temp 0 used by .* but never defined"):
        verify_function(fn(body=body))


# -- verify_program: verify_function, for every function -------------------

def test_verify_program_all_valid_passes():
    good_a = fn(name='a', body=[IRReturn(value=None)])
    good_b = fn(name='b', body=[IRReturn(value=None)])
    verify_program(IRProgram(functions=[good_a, good_b]))  # no raise


def test_verify_program_names_the_offending_function():
    good = fn(name='good', body=[IRReturn(value=None)])
    broken = fn(name='broken', body=[])
    with pytest.raises(IRVerificationError, match="broken: empty body"):
        verify_program(IRProgram(functions=[good, broken]))
