"""Branches on constants become jumps; unreachable blocks are then removed."""

from ir.cfg import remove_unreachable
from ir.ir import IRBranch, IRConst, IRFunction, IRJump


def simplify_branches(ir_fn: IRFunction) -> None:
    """Simplify constant branches in ir_fn.body and drop unreachable blocks."""
    body = []
    for instr in ir_fn.body:
        if isinstance(instr, IRBranch) and isinstance(instr.cond, IRConst):
            instr = IRJump(instr.true_label if instr.cond.value else instr.false_label)
        elif isinstance(instr, IRBranch) and instr.true_label == instr.false_label:
            instr = IRJump(instr.true_label)
        body.append(instr)
    ir_fn.body = remove_unreachable(body)
