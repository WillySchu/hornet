"""Address folding: `t = base + c` (c a constant) used as a load, store, or copy address becomes that
access's offset, `[base + c]`, so the backends address it directly. Within straight-line code only
(temps aren't SSA): a label, terminator, or redefinition of `t` or `base` ends what's known. Folded
additions left without uses are removed by dead-code elimination.
"""

from dataclasses import replace

from ir.cfg import TERMINATORS, writes
from ir.ir import IRBinOp, IRConst, IRCopy, IRFunction, IRLabel, IRLoad, IRStore, Temp
from ops import BinaryOp


def fold_addresses(ir_fn: IRFunction, pinned: set) -> None:
    known: dict = {}  # temp id -> (base temp, constant): what it holds right now
    out = []
    for instr in ir_fn.body:
        if isinstance(instr, IRLabel):
            known.clear()
        instr = _fold(instr, known)
        for w in writes(instr):
            known.pop(w.id, None)
            for tid in [tid for tid, (base, _) in known.items() if base.id == w.id]:
                del known[tid]
        if isinstance(instr, IRBinOp) and instr.dst.id not in pinned:
            sum_ = _base_plus_constant(instr, known, pinned)
            if sum_ is not None:
                known[instr.dst.id] = sum_
        if isinstance(instr, TERMINATORS):
            known.clear()
        out.append(instr)
    ir_fn.body = out


def _base_plus_constant(instr: IRBinOp, known: dict, pinned: set):
    """(base, constant) if `instr` computes a temp plus (or minus) a constant."""
    if instr.op not in (BinaryOp.ADD, BinaryOp.SUBTRACT):
        return None
    left, right = instr.left, instr.right
    if instr.op == BinaryOp.ADD and isinstance(left, IRConst):
        left, right = right, left
    if not isinstance(left, Temp) or not isinstance(right, IRConst) or left.id in pinned or left.id == instr.dst.id:
        return None
    c = right.value if instr.op == BinaryOp.ADD else -right.value
    base, inner = known.get(left.id, (left, 0))
    return base, inner + c


def _fold(instr, known: dict):
    def folded(address, offset):
        if isinstance(address, Temp) and address.id in known:
            base, c = known[address.id]
            return base, offset + c
        return address, offset

    if isinstance(instr, (IRLoad, IRStore)):
        address, offset = folded(instr.address, instr.offset)
        return replace(instr, address=address, offset=offset) if address is not instr.address else instr
    if isinstance(instr, IRCopy):
        dst, dst_offset = folded(instr.dst_address, instr.dst_offset)
        src, src_offset = folded(instr.src_address, instr.src_offset)
        if dst is instr.dst_address and src is instr.src_address:
            return instr
        return replace(instr, dst_address=dst, dst_offset=dst_offset, src_address=src, src_offset=src_offset)
    return instr
