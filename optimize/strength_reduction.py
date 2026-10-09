"""Strength reduction: a multiplication by a power of two becomes the shift that leaves the same
bits, `x * 8` -> `x << 3`. The two wrap alike at every width."""

from dataclasses import replace

from ir.ir import IRBinOp, IRConst, IRFunction
from ops import BinaryOp


def reduce_strength(ir_fn: IRFunction) -> None:
    """Reduce ir_fn.body's multiplications in place."""
    for i, instr in enumerate(ir_fn.body):
        if not (isinstance(instr, IRBinOp) and instr.op == BinaryOp.MULTIPLY):
            continue
        value, by = (instr.right, instr.left) if isinstance(instr.left, IRConst) else (instr.left, instr.right)
        # (Two constants are constant folding's; 1 and 0 are identity reduction's.)
        if not isinstance(by, IRConst) or isinstance(value, IRConst):
            continue
        if by.value > 1 and by.value & (by.value - 1) == 0:
            count = IRConst(by.value.bit_length() - 1, by.type)
            ir_fn.body[i] = replace(instr, op=BinaryOp.SHIFT_LEFT, left=value, right=count)
