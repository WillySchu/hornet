"""Dead code elimination: drop pure instructions whose result is never read.

Division and modulo stay, though their panics now come from the checks ir/division_checks.py
puts before them; calls, loads, stores, and checks always stay; writes to pinned temps always stay. Run after unreachable blocks are removed.
"""

from ir.cfg import PURE, build_blocks, liveness, reads, writes
from ir.ir import IRBinOp, IRFunction
from ops import BinaryOp


def _removable(instr, live: set, pinned: set) -> bool:
    if not isinstance(instr, PURE):
        return False
    if isinstance(instr, IRBinOp) and instr.op in (BinaryOp.DIVIDE, BinaryOp.MODULO):
        return False
    ws = writes(instr)
    return bool(ws) and all(w.id not in live and w.id not in pinned for w in ws)


def remove_dead_code(ir_fn: IRFunction, pinned: set) -> None:
    """Remove dead pure instructions from ir_fn.body in place."""
    blocks = build_blocks(ir_fn.body)
    _, live_out = liveness(blocks)
    out = []
    for block, block_live in zip(blocks, live_out):
        live = {t.id for t in block_live}
        kept = []
        for instr in reversed(block.instructions):
            if _removable(instr, live, pinned):
                continue
            live -= {w.id for w in writes(instr)}
            live |= {r.id for r in reads(instr)}
            kept.append(instr)
        out.extend(reversed(kept))
    ir_fn.body = out
