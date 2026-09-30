"""Forward copy and constant propagation within each block.

After `x = y` or `x = 5`, later reads of x in the block use y or 5 until x or y is written.
Pinned temps (address taken) are never propagated from or into.
"""

from ir.cfg import build_blocks, replace_reads, writes
from ir.ir import IRConst, IRFunction, IRMove, Temp


def propagate_copies(ir_fn: IRFunction, pinned: set) -> None:
    """Propagate copies in ir_fn.body in place."""
    body = ir_fn.body
    for block in build_blocks(body):
        env: dict = {}
        for offset in range(len(block.instructions)):
            idx = block.start + offset
            instr = replace_reads(body[idx], env)
            body[idx] = instr
            for w in writes(instr):
                env.pop(w.id, None)
                for k in [k for k, v in env.items() if isinstance(v, Temp) and v.id == w.id]:
                    del env[k]
            if isinstance(instr, IRMove) and instr.dst.id not in pinned and instr.src != instr.dst:
                src = instr.src
                if isinstance(src, IRConst) and src.type == instr.dst.type:
                    env[instr.dst.id] = src
                elif isinstance(src, Temp) and src.id not in pinned and src.type == instr.dst.type:
                    env[instr.dst.id] = src
