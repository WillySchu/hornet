"""Integer division and modulo panic on a zero divisor, and on MIN / -1 for int and int32 (where
the quotient doesn't fit), identically on every target. Runs on each function's IR after it is built."""

from ir.ir import IRBinOp, IRBranch, IRCall, IRConst, IRJump, IRLabel, IRStaticDataAddress
from ops import BinaryOp
from typesys import Type

ZERO_MESSAGE = "integer division by zero"
OVERFLOW_MESSAGE = "integer overflow in division"
_MINIMUM = {Type.INT: -(2 ** 63), Type.INT32: -(2 ** 31)}


def _message_label(ir_program, message: str) -> str:
    labels = ir_program.__dict__.setdefault('_panic_message_labels', {})
    if message not in labels:
        labels[message] = ir_program.ids.new_label("panic_msg")
        ir_program.string_literals.append((labels[message], message))
    return labels[message]


def insert_division_checks(ir_fn, ir_program) -> None:
    ids = ir_program.ids
    panics: dict = {}  # message -> label of this function's panic block

    def panic_label(message: str) -> str:
        if message not in panics:
            panics[message] = ids.new_label("division_panic")
        return panics[message]

    out = []
    for instr in ir_fn.body:
        if not (isinstance(instr, IRBinOp) and instr.op in (BinaryOp.DIVIDE, BinaryOp.MODULO)):
            out.append(instr)
            continue
        divisor, t = instr.right, instr.left.type
        known = divisor.value if isinstance(divisor, IRConst) else None
        if known is None or known == 0:
            is_zero, ok = ids.new_temp(Type.BOOL), ids.new_label("divisor_nonzero")
            out += [IRBinOp(dst=is_zero, op=BinaryOp.EQUAL, left=divisor, right=IRConst(0, divisor.type)),
                    IRBranch(cond=is_zero, true_label=panic_label(ZERO_MESSAGE), false_label=ok),
                    IRLabel(ok)]
        if t in _MINIMUM and (known is None or known == -1):
            is_minus_one, is_min = ids.new_temp(Type.BOOL), ids.new_temp(Type.BOOL)
            check_min, ok = ids.new_label("divisor_minus_one"), ids.new_label("division_fits")
            out += [IRBinOp(dst=is_minus_one, op=BinaryOp.EQUAL, left=divisor, right=IRConst(-1, divisor.type)),
                    IRBranch(cond=is_minus_one, true_label=check_min, false_label=ok),
                    IRLabel(check_min),
                    IRBinOp(dst=is_min, op=BinaryOp.EQUAL, left=instr.left, right=IRConst(_MINIMUM[t], t)),
                    IRBranch(cond=is_min, true_label=panic_label(OVERFLOW_MESSAGE), false_label=ok),
                    IRLabel(ok)]
        out.append(instr)
    for message, label in panics.items():
        msg = ids.new_temp(Type.INT64)
        out += [IRLabel(label),
                IRStaticDataAddress(dst=msg, label=_message_label(ir_program, message)),
                IRCall(dst=None, name='hornet_panic', args=[msg]),
                IRJump(label)]  # not reached: hornet_panic aborts
    ir_fn.body = out
