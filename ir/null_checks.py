"""Dereferencing none panics ("dereference of none") instead of faulting. The IR builder marks each
dereference with IRNullCheck; this pass drops checks of a pointer already checked earlier in the
same block (the common `p.a + p.b`), then expands the rest into a compare and branch."""

from ir.cfg import TERMINATORS, writes
from ir.ir import IRBinOp, IRBranch, IRConst, IRLabel, IRNullCheck, Temp
from ir.panics import PanicBlocks
from ops import BinaryOp
from typesys import Type

MESSAGE = "dereference of none"


def expand_null_checks(ir_fn, ir_program) -> None:
    ids = ir_program.ids
    panics = PanicBlocks(ir_program, "none_panic")
    out = []
    checked: set = set()  # temp ids known non-none in the current block
    for instr in ir_fn.body:
        if isinstance(instr, IRLabel) or isinstance(instr, TERMINATORS):
            checked.clear()
        if not isinstance(instr, IRNullCheck):
            for t in writes(instr):
                checked.discard(t.id)
            out.append(instr)
            continue
        p = instr.pointer
        if isinstance(p, Temp) and p.id in checked:
            continue
        is_none, ok = ids.new_temp(Type.BOOL), ids.new_label("not_none")
        out += [IRBinOp(dst=is_none, op=BinaryOp.EQUAL, left=p, right=IRConst(0, p.type)),
                IRBranch(cond=is_none, true_label=panics.label(MESSAGE), false_label=ok),
                IRLabel(ok)]
        if isinstance(p, Temp):
            checked.add(p.id)
    ir_fn.body = out + panics.blocks()
