"""Constant folding: IRBinOp/IRUnOp/IRCast with all-constant operands -> IRMove. Results match runtime semantics (wraparound, truncating division)."""

from folding import fold_binary_op, fold_cast, fold_unary_op  # noqa: F401 (re-exported)
from ir.ir import IRBinOp, IRCast, IRConst, IRFunction, IRMove, IRUnOp


def fold_constants(ir_fn: IRFunction) -> None:
    """Fold all-constant ops in ir_fn.body in place."""
    for i, instr in enumerate(ir_fn.body):
        if isinstance(instr, IRBinOp) and isinstance(instr.left, IRConst) and isinstance(instr.right, IRConst):
            folded = fold_binary_op(instr.op, instr.left.value, instr.right.value, instr.dst.type)
            if folded is not None:
                ir_fn.body[i] = IRMove(dst=instr.dst, src=IRConst(folded, instr.dst.type))
        elif isinstance(instr, IRUnOp) and isinstance(instr.operand, IRConst):
            folded = fold_unary_op(instr.op, instr.operand.value, instr.dst.type)
            ir_fn.body[i] = IRMove(dst=instr.dst, src=IRConst(folded, instr.dst.type))
        elif isinstance(instr, IRCast) and isinstance(instr.src, IRConst):
            folded = fold_cast(instr.dst.type, instr.src.value, instr.src.type)
            ir_fn.body[i] = IRMove(dst=instr.dst, src=IRConst(folded, instr.dst.type))
