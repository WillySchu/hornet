"""Constant folding: IRBinOp/IRUnOp/IRCast with all-constant operands -> IRMove. Results match runtime semantics (wraparound, truncating division)."""

from ir.ir import IRBinOp, IRCast, IRConst, IRFunction, IRMove, IRUnOp
from ops import BinaryOp, UnaryOp
from typesys import Type


def _wrap(value: int, t: Type) -> int:
    """Wrap `value` to `t`'s runtime width and signedness."""
    if t == Type.BOOL:
        return value
    if t == Type.INT8:
        value &= 0xFF
        return value - 0x100 if value >= 0x80 else value
    if t == Type.UINT8:
        return value & 0xFF
    if t == Type.INT64:
        value &= 0xFFFFFFFFFFFFFFFF
        return value - 0x10000000000000000 if value >= 0x8000000000000000 else value
    # INT
    value &= 0xFFFFFFFF
    return value - 0x100000000 if value >= 0x80000000 else value


def _truncated_div(a: int, b: int) -> int:
    """Round-toward-zero division, like idiv."""
    quotient = abs(a) // abs(b)
    return -quotient if (a < 0) != (b < 0) else quotient


def _truncated_mod(a: int, b: int) -> int:
    """Remainder matching _truncated_div."""
    return a - _truncated_div(a, b) * b


def fold_binary_op(op: BinaryOp, left: int, right: int, result_type: Type) -> "int | None":
    """Folded value, or None (division by zero is left to trap at runtime)."""
    if op == BinaryOp.ADD:
        return _wrap(left + right, result_type)
    if op == BinaryOp.SUBTRACT:
        return _wrap(left - right, result_type)
    if op == BinaryOp.MULTIPLY:
        return _wrap(left * right, result_type)
    if op == BinaryOp.DIVIDE:
        return _wrap(_truncated_div(left, right), result_type) if right != 0 else None
    if op == BinaryOp.MODULO:
        return _wrap(_truncated_mod(left, right), result_type) if right != 0 else None
    if op == BinaryOp.SHIFT_LEFT:
        mask = 63 if result_type == Type.INT64 else 31
        return _wrap(left << (right & mask), result_type)
    if op == BinaryOp.SHIFT_RIGHT:
        mask = 63 if result_type == Type.INT64 else 31
        return _wrap(left >> (right & mask), result_type)
    if op == BinaryOp.LESS_THAN:
        return int(left < right)
    if op == BinaryOp.GREATER_THAN:
        return int(left > right)
    if op == BinaryOp.LESS_THAN_OR_EQUAL:
        return int(left <= right)
    if op == BinaryOp.GREATER_THAN_OR_EQUAL:
        return int(left >= right)
    if op == BinaryOp.EQUAL:
        return int(left == right)
    if op == BinaryOp.NOT_EQUAL:
        return int(left != right)
    if op == BinaryOp.BITWISE_AND:
        return _wrap(left & right, result_type)
    if op == BinaryOp.BITWISE_XOR:
        return _wrap(left ^ right, result_type)
    if op == BinaryOp.BITWISE_OR:
        return _wrap(left | right, result_type)
    return None


def fold_unary_op(op: UnaryOp, operand: int, result_type: Type) -> int:
    """Folded value of a unary op."""
    if op == UnaryOp.NEGATE:
        return _wrap(-operand, result_type)
    if op == UnaryOp.COMPLEMENT:
        return _wrap(~operand, result_type)
    return _wrap(int(operand == 0), result_type)


def fold_cast(target_type: Type, src: int, source_type: Type) -> int:
    """Folded cast; must match gen_cast_narrowing_into."""
    if target_type == Type.INT8:
        return _wrap(src, Type.INT8)
    if target_type == Type.UINT8:
        return _wrap(src, Type.UINT8)
    if target_type == Type.INT64:
        if source_type == Type.INT64:
            return src
        return _wrap(_wrap(src, Type.INT), Type.INT64)
    return _wrap(src, Type.INT)


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
