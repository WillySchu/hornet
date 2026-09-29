"""Calling-convention accounting shared by IR construction and lowering."""

from ir.utils import type_of
from parser import Node, NoneLiteral
from semantic import TypeKind

# Saved unconditionally in every prologue.
CALLEE_SAVED_SCRATCH_REGISTERS = ['rbx', 'r12', 'r13', 'r14']


def total_arg_slots(args: list[Node]) -> int:
    """Argument slots `args` need: 3 per slice or `none`, else 1."""
    return sum(
        3 if type_of(a).kind == TypeKind.SLICE or isinstance(a, NoneLiteral) else 1
        for a in args
    )
