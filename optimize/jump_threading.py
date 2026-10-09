"""Jump threading and block merging: both go by the shape of the control flow alone.

Threading: a jump or branch to a block that does nothing but jump on goes straight to where that
leads, and the blocks this leaves unentered are dropped. Merging: a block entered only by another
block's jump is joined to the end of that block, wherever it stood.
"""

from dataclasses import replace

from ir.cfg import build_blocks, remove_unreachable
from ir.ir import IRBranch, IRFunction, IRJump


def thread_jumps(ir_fn: IRFunction) -> None:
    """Retarget ir_fn.body's jumps and branches past jump-only blocks, in place."""
    onward = {}  # a jump-only block's label -> the label it jumps to
    for block in build_blocks(ir_fn.body):
        if block.label is not None and len(block.instructions) == 2 and isinstance(block.instructions[1], IRJump):
            onward[block.label] = block.instructions[1].label
    if not onward:
        return

    def end(label: str) -> str:
        """Where a jump to `label` ends up. A chain that loops (an empty infinite loop) is left as it is."""
        seen, at = {label}, label
        while at in onward:
            at = onward[at]
            if at in seen:
                return label
            seen.add(at)
        return at

    body = []
    for instr in ir_fn.body:
        if isinstance(instr, IRJump) and end(instr.label) != instr.label:
            instr = IRJump(end(instr.label))
        elif isinstance(instr, IRBranch):
            on_true, on_false = end(instr.true_label), end(instr.false_label)
            if on_true == on_false:
                instr = IRJump(on_true)
            elif (on_true, on_false) != (instr.true_label, instr.false_label):
                instr = replace(instr, true_label=on_true, false_label=on_false)
        body.append(instr)
    ir_fn.body = remove_unreachable(body)


def merge_blocks(ir_fn: IRFunction) -> None:
    """Join each block entered only by another block's jump onto that block, in place."""
    blocks = build_blocks(ir_fn.body)
    joined = {}  # a block's index -> the index of the block whose jump is its only way in
    for i, block in enumerate(blocks):
        if i > 0 and block.label is not None and len(block.predecessors) == 1:  # (the entry stays first)
            before = block.predecessors[0]
            if before != i and isinstance(blocks[before].instructions[-1], IRJump):
                joined[i] = before
    if not joined:
        return
    body = []
    for i, block in enumerate(blocks):
        if i in joined:
            continue  # it goes where the block it is joined to goes
        body += block.instructions
        at = i
        while isinstance(body[-1], IRJump) and joined.get(blocks[at].successors[0]) == at:
            at = blocks[at].successors[0]
            body.pop()
            body += blocks[at].instructions[1:]  # (without its label)
    ir_fn.body = body
