"""Structural IR invariants, checked after building and after optimization:
non-empty body; unique labels; every block ends in one terminator; jump targets exist;
slots exist; every Temp fits in a register; every read Temp is written somewhere in the function.
Not checked: operand type consistency, call target existence.
"""

from diagnostics import InternalCompilerError

from ir.cfg import TERMINATORS, reads, writes
from ir.ir import IRBranch, IRFunction, IRJump, IRLabel, IRLocalAddress, IRProgram
from typesys import TypeKind


class IRVerificationError(InternalCompilerError):
    """IR invariant violated; a compiler bug."""


# Temps hold one machine word: integers, bools, pointers, and str (a descriptor's address).
_REGISTER_KINDS = {TypeKind.INT, TypeKind.INT32, TypeKind.INT8, TypeKind.UINT8, TypeKind.BOOL, TypeKind.POINTER,
                   TypeKind.STR}


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
        if isinstance(op, TERMINATORS):
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

    for op in ir_fn.body:
        for temp in (*reads(op), *writes(op)):
            if temp.type.kind not in _REGISTER_KINDS:
                raise IRVerificationError(
                    f"{ir_fn.name}: Temp {temp.id} has type {temp.type}, which doesn't fit in a register "
                    f"(composite values live in slots and are handled by address): {op!r}"
                )

    defined_ids = {t.id for t in ir_fn.params}
    for op in ir_fn.body:
        defined_ids.update(t.id for t in writes(op))
    for op in ir_fn.body:
        for temp_id in sorted(t.id for t in reads(op)):
            if temp_id not in defined_ids:
                raise IRVerificationError(
                    f"{ir_fn.name}: Temp {temp_id} used by {op!r} but never defined anywhere in this function"
                )


def verify_program(ir_program: IRProgram) -> None:
    for ir_fn in ir_program.functions:
        verify_function(ir_fn)
