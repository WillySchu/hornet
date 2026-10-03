"""Dead code elimination: drop instructions that only compute a result nothing reads.

Pure instructions and loads go. That includes a division: its panics come from the checks
ir/division_checks.py puts before it, which stay, as the none and bounds checks before a load do
(so this pass expects IR that has been through build_ir_program). Calls, stores, copies, and checks
always stay; writes to pinned temps always stay. Run after unreachable blocks are removed.
"""

from ir.cfg import PURE, build_blocks, liveness, reads, writes
from ir.ir import IRFunction, IRLoad

# Instructions with no effect but writing `dst`. A load isn't in PURE: its result depends on memory.
_ONLY_WRITES_DST = PURE + (IRLoad,)


def _removable(instr, live: set, pinned: set) -> bool:
    if not isinstance(instr, _ONLY_WRITES_DST):
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
