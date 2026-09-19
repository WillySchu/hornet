"""Shared, small pieces of calling-convention accounting used by both
real-IR call construction (see _ir_call/_ir_composite_call in
scalars.py, and _ir_call_arguments) and gen_function_ir's own
prologue/epilogue: total_arg_slots and CALLEE_SAVED_SCRATCH_
REGISTERS."""

from ir.utils import type_of
from parser import Node, NoneLiteral
from semantic import TypeKind

# Every function's prologue/epilogue saves and restores these
# unconditionally regardless of whether that specific function uses
# them itself, since the callee-saved contract must hold for any call.
CALLEE_SAVED_SCRATCH_REGISTERS = ['rbx', 'r12', 'r13', 'r14']


def total_arg_slots(args: list[Node]) -> int:
    """Total argument-register slots `args` will need: 3 per
    slice-typed (or `none`) argument, 1 for everything else.
    `none` is checked via isinstance rather than its resolved
    type, since Type.NONE != SLICE; semantic.py guarantees `none`
    only ever appears where a slice is expected, so this is safe.
    Used by every call-codegen entry point's "too many arguments"
    check."""
    return sum(
        3 if type_of(a).kind == TypeKind.SLICE or isinstance(a, NoneLiteral) else 1
        for a in args
    )
