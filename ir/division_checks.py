"""Integer division and modulo panic on a zero divisor, and on MIN / -1 for int and int32 (where
the quotient doesn't fit), identically on every target. Runs on each function's IR after it is built."""

from dataclasses import replace

from folding import fold_cast, fold_unary_op
from ir.cfg import writes
from ir.ir import IRBinOp, IRBranch, IRCast, IRConst, IRLabel, IRMove, IRUnOp, Temp
from ir.panics import PanicBlocks, located
from ops import BinaryOp
from typesys import Type

ZERO_MESSAGE = "integer division by zero"
OVERFLOW_MESSAGE = "integer overflow in division"
_MINIMUM = {Type.INT: -(2 ** 63), Type.INT32: -(2 ** 31)}


def _track_constants(instr, constants: dict) -> None:
    def known(v):
        if isinstance(v, IRConst):
            return v.value
        return constants.get(v.id) if isinstance(v, Temp) else None

    value = None
    if isinstance(instr, IRMove):
        value = known(instr.src)
    elif isinstance(instr, IRUnOp) and known(instr.operand) is not None:
        value = fold_unary_op(instr.op, known(instr.operand), instr.dst.type)
    elif isinstance(instr, IRCast) and known(instr.src) is not None:
        value = fold_cast(instr.dst.type, known(instr.src), instr.src.type)
    for t in writes(instr):
        constants.pop(t.id, None)
    if value is not None:
        constants[instr.dst.id] = value


def insert_division_checks(ir_fn, ir_program) -> None:
    ids = ir_program.ids
    panics = PanicBlocks(ir_program, "division_panic")

    out = []
    constants: dict = {}  # temp id -> value, for temps set to a constant earlier in the same block
    for instr in ir_fn.body:
        if isinstance(instr, IRLabel):
            constants.clear()
        if not (isinstance(instr, IRBinOp) and instr.op in (BinaryOp.DIVIDE, BinaryOp.MODULO)):
            _track_constants(instr, constants)
            out.append(instr)
            continue
        if isinstance(instr.right, Temp) and instr.right.id in constants:
            # A divisor computed from constants (such as `-3`) becomes a constant, so it needs no
            # check and backends can divide by it with multiplication.
            instr = replace(instr, right=IRConst(constants[instr.right.id], instr.right.type))
        _track_constants(instr, constants)
        divisor, t = instr.right, instr.left.type
        known = divisor.value if isinstance(divisor, IRConst) else None
        if known is None or known == 0:
            is_zero, ok = ids.new_temp(Type.BOOL), ids.new_label("divisor_nonzero")
            out += [IRBinOp(dst=is_zero, op=BinaryOp.EQUAL, left=divisor, right=IRConst(0, divisor.type)),
                    IRBranch(cond=is_zero, true_label=panics.label(located(ZERO_MESSAGE, instr.where)),
                             false_label=ok),
                    IRLabel(ok)]
        if t in _MINIMUM and (known is None or known == -1):
            is_minus_one, is_min = ids.new_temp(Type.BOOL), ids.new_temp(Type.BOOL)
            check_min, ok = ids.new_label("divisor_minus_one"), ids.new_label("division_fits")
            out += [IRBinOp(dst=is_minus_one, op=BinaryOp.EQUAL, left=divisor, right=IRConst(-1, divisor.type)),
                    IRBranch(cond=is_minus_one, true_label=check_min, false_label=ok),
                    IRLabel(check_min),
                    IRBinOp(dst=is_min, op=BinaryOp.EQUAL, left=instr.left, right=IRConst(_MINIMUM[t], t)),
                    IRBranch(cond=is_min, true_label=panics.label(located(OVERFLOW_MESSAGE, instr.where)),
                             false_label=ok),
                    IRLabel(ok)]
        out.append(instr)
    ir_fn.body = out + panics.blocks()
