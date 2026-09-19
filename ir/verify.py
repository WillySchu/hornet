"""Checks a handful of structural invariants this IR is supposed to
hold by construction (see ir.ir's own module docstring), mechanically
rather than by inspection -- catching a bug in an IR-to-IR transform
(an optimization pass) as an immediate, specific IRVerificationError
naming the exact function and op responsible, rather than a failure at
the assembler, the linker, or a wrong answer at runtime.

Checks, in the order verify_function runs them:
- body is non-empty.
- No two IRLabels in the same function share a name.
- Every block (the run of ops between one IRLabel, or the start of
  body, and the next) ends in exactly one terminator (IRJump, IRBranch,
  IRReturn) -- no implicit fallthrough into a label or off the end of
  body.
- Every IRJump/IRBranch target names a real IRLabel in the same
  function.
- Every IRLocalAddress.slot, and hidden_return_ptr_slot when set,
  names a real entry in slot_widths.
- Every Temp read by some op (through any IRValue-typed field) was
  written by some op earlier in the SAME function (an IRReadArgument,
  or any op with a dst, counts as a write) -- "defined somewhere at
  all," not a dominance-based def-before-use analysis; enough to catch
  a transform deleting or renaming a Temp's own producer while
  something else still reads it.

Deliberately NOT checked: type consistency between an op and its own
operands (IRBinOp.left/right agreeing on width, say) -- easy to get
subtly wrong and worse than not checking at all if it produces false
positives; and whether IRCall.name refers to a real function -- user
Hornet functions and runtime/libc symbols (malloc, strcmp,
hornet_print, ...) share this field with no registry of valid names to
check against, and a bad name is already caught at link time
regardless."""

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
    """Raised by verify_function/verify_program on the first structural
    problem found. Distinct from CodegenError: this signals a bug in
    the IR itself, never a mistake in how the compiler was invoked."""


def _op_defs(op) -> list:
    """The Temp id(s) `op` writes to -- 0, 1 (the common case), or 2
    (IRSliceGrow alone)."""
    if isinstance(op, (
            IRMove, IRBinOp, IRUnOp, IRCast, IRReadArgument, IRLoad, IRLocalAddress, IRStaticDataAddress)):
        return [op.dst.id]
    if isinstance(op, IRCall):
        return [op.dst.id] if op.dst is not None else []
    if isinstance(op, IRSliceGrow):
        return [op.dst_ptr.id, op.dst_cap.id]
    return []


def _op_uses(op) -> list:
    """The Temp id(s) `op` reads -- every IRValue-typed field it has,
    with any IRConst operand filtered back out (only a Temp has an id
    at all)."""
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
    """See this module's own docstring for the checks and their order.
    Each error names ir_fn.name and, where there's one specific op
    responsible, that op's own repr."""
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
    """verify_function, for every function in ir_program -- see its
    own docstring for what's actually checked."""
    for ir_fn in ir_program.functions:
        verify_function(ir_fn)
