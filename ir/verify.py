"""Structural IR invariants, checked after building and after optimization:
non-empty body; unique labels; every block ends in one terminator; jump targets exist;
slots exist; every read Temp is written somewhere in the function.
Not checked: operand type consistency, call target existence.
"""

from ir.ir import (
    IRBinOp,
    IRBoundsCheck,
    IRBranch,
    IRCall,
    IRCast,
    IRCopy,
    IRFunction,
    IRJump,
    IRLabel,
    IRLoad,
    IRLocalAddress,
    IRMove,
    IRProgram,
    IRReadArgument,
    IRReturn,
    IRSliceBoundsCheck,
    IRSliceGrow,
    IRStaticDataAddress,
    IRStore,
    IRUnOp,
    Temp,
)

_TERMINATORS = (IRJump, IRBranch, IRReturn)


class IRVerificationError(Exception):
    """IR invariant violated; a compiler bug."""


def _op_defs(op) -> list:
    """Temp ids written by `op`."""
    if isinstance(op, (
            IRMove, IRBinOp, IRUnOp, IRCast, IRReadArgument, IRLoad, IRLocalAddress, IRStaticDataAddress)):
        return [op.dst.id]
    if isinstance(op, IRCall):
        return [op.dst.id] if op.dst is not None else []
    if isinstance(op, IRSliceGrow):
        return [op.dst_ptr.id, op.dst_cap.id]
    return []


def _op_uses(op) -> list:
    """Temp ids read by `op`."""
    if isinstance(op, IRMove):
        values = [op.src]
    elif isinstance(op, IRBinOp):
        values = [op.left, op.right]
    elif isinstance(op, IRUnOp):
        values = [op.operand]
    elif isinstance(op, IRCast):
        values = [op.src]
    elif isinstance(op, IRCall):
        values = list(op.args)
    elif isinstance(op, IRReturn):
        values = [op.value] if op.value is not None else []
    elif isinstance(op, IRLoad):
        values = [op.address]
    elif isinstance(op, IRStore):
        values = [op.address, op.value]
    elif isinstance(op, IRCopy):
        values = [op.dst_address, op.src_address]
    elif isinstance(op, IRBoundsCheck):
        values = [op.index, op.length]
    elif isinstance(op, IRSliceBoundsCheck):
        values = [op.value, op.bound]
    elif isinstance(op, IRSliceGrow):
        values = [op.ptr, op.length, op.cap]
    elif isinstance(op, IRBranch):
        values = [op.cond]
    else:
        values = []
    return [v.id for v in values if isinstance(v, Temp)]


def verify_function(ir_fn: IRFunction) -> None:
    """Check `ir_fn`; see module docstring."""
    if not ir_fn.body:
        raise IRVerificationError(f"{ir_fn.name}: empty body -- every function must end in a terminator")

    label_names = set()
    for op in ir_fn.body:
        if isinstance(op, IRLabel):
            if op.name in label_names:
                raise IRVerificationError(f"{ir_fn.name}: duplicate label {op.name!r}")
            label_names.add(op.name)

    body_len = len(ir_fn.body)
    for i, op in enumerate(ir_fn.body):
        if isinstance(op, _TERMINATORS):
            continue
        is_last = i == body_len - 1
        next_op = None if is_last else ir_fn.body[i + 1]
        if is_last or isinstance(next_op, IRLabel):
            where = "off the end of body" if is_last else f"into label {next_op.name!r}"
            raise IRVerificationError(f"{ir_fn.name}: block falls through {where} without a terminator (after {op!r})")

    for op in ir_fn.body:
        if isinstance(op, IRJump) and op.label not in label_names:
            raise IRVerificationError(f"{ir_fn.name}: IRJump targets undefined label {op.label!r}")
        if isinstance(op, IRBranch):
            for label in (op.true_label, op.false_label):
                if label not in label_names:
                    raise IRVerificationError(f"{ir_fn.name}: IRBranch targets undefined label {label!r}")

    for op in ir_fn.body:
        if isinstance(op, IRLocalAddress) and op.slot not in ir_fn.slot_widths:
            raise IRVerificationError(f"{ir_fn.name}: IRLocalAddress references unknown slot {op.slot}")
    if ir_fn.hidden_return_ptr_slot is not None and ir_fn.hidden_return_ptr_slot not in ir_fn.slot_widths:
        raise IRVerificationError(
            f"{ir_fn.name}: hidden_return_ptr_slot references unknown slot {ir_fn.hidden_return_ptr_slot}"
        )

    defined_ids = set()
    for op in ir_fn.body:
        defined_ids.update(_op_defs(op))
    for op in ir_fn.body:
        for temp_id in _op_uses(op):
            if temp_id not in defined_ids:
                raise IRVerificationError(
                    f"{ir_fn.name}: Temp {temp_id} used by {op!r} but never defined anywhere in this function"
                )


def verify_program(ir_program: IRProgram) -> None:
    for ir_fn in ir_program.functions:
        verify_function(ir_fn)
