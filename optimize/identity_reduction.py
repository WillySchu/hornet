"""Identity reduction: IRBinOp with an identity or absorbing constant operand (x + 0, x * 1, x * 0, ...) -> IRMove."""

from ir.ir import IRBinOp, IRConst, IRFunction, IRMove, IRValue
from ops import BinaryOp
from typesys import Type


def reduce_binary_op(op: BinaryOp, left: IRValue, right: IRValue, result_type: Type) -> "IRValue | None":
    """The operand or constant `left OP right` always equals, else None."""
    left_value = left.value if isinstance(left, IRConst) else None
    right_value = right.value if isinstance(right, IRConst) else None

    if op == BinaryOp.ADD:
        if right_value == 0:
            return left
        if left_value == 0:
            return right
    elif op == BinaryOp.SUBTRACT:
        if right_value == 0:
            return left
    elif op == BinaryOp.MULTIPLY:
        if right_value == 0 or left_value == 0:
            return IRConst(0, result_type)
        if right_value == 1:
            return left
        if left_value == 1:
            return right
    elif op == BinaryOp.DIVIDE:
        if right_value == 1:
            return left
    elif op == BinaryOp.MODULO:
        if right_value == 1:
            return IRConst(0, result_type)
    elif op in (BinaryOp.SHIFT_LEFT, BinaryOp.SHIFT_RIGHT):
        if right_value == 0:
            return left
        if left_value == 0:
            return IRConst(0, result_type)
    elif op == BinaryOp.BITWISE_AND:
        if right_value == 0 or left_value == 0:
            return IRConst(0, result_type)
    elif op == BinaryOp.BITWISE_OR:
        if right_value == 0:
            return left
        if left_value == 0:
            return right
    elif op == BinaryOp.BITWISE_XOR:
        if right_value == 0:
            return left
        if left_value == 0:
            return right
    return None


def reduce_identities(ir_fn: IRFunction) -> None:
    """Reduce identities in ir_fn.body in place."""
    for i, instr in enumerate(ir_fn.body):
        if not isinstance(instr, IRBinOp):
            continue
        reduced = reduce_binary_op(instr.op, instr.left, instr.right, instr.dst.type)
        if reduced is not None:
            ir_fn.body[i] = IRMove(dst=instr.dst, src=reduced)
