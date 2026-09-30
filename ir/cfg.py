"""Control-flow analysis over a function's flat IR list.

Blocks are computed on demand; IRFunction.body stays a flat list.
A block starts at a label or after a terminator and ends at the next one.
"""

from dataclasses import dataclass, field, replace
from typing import Optional

from ir.ir import (
    IRBinOp,
    IRBoundsCheck,
    IRBranch,
    IRCall,
    IRCast,
    IRCopy,
    IRJump,
    IRLabel,
    IRLoad,
    IRLocalAddress,
    IRMove,
    IRReadArgument,
    IRReturn,
    IRSliceBoundsCheck,
    IRStaticDataAddress,
    IRStore,
    IRUnOp,
    Temp,
)

TERMINATORS = (IRJump, IRBranch, IRReturn)

# Fields each instruction reads ('args' is a list). Must agree with reads().
READ_FIELDS = {
    IRMove: ('src',), IRCast: ('src',), IRBinOp: ('left', 'right'), IRUnOp: ('operand',),
    IRCall: ('args',), IRReturn: ('value',), IRBranch: ('cond',), IRLoad: ('address',),
    IRStore: ('address', 'value'), IRCopy: ('dst_address', 'src_address'),
    IRBoundsCheck: ('index', 'length'), IRSliceBoundsCheck: ('value', 'bound'),
}

# Instructions whose only effect is writing `dst`.
PURE = (IRMove, IRBinOp, IRUnOp, IRCast, IRLocalAddress, IRStaticDataAddress, IRReadArgument)


def replace_reads(instr, mapping: dict):
    """`instr` with every read Temp whose id is in `mapping` replaced; the same object if none."""
    names = READ_FIELDS.get(type(instr), ())
    changes = {}
    for name in names:
        value = getattr(instr, name)
        if name == 'args':
            new = [mapping.get(a.id, a) if isinstance(a, Temp) else a for a in value]
            if new != value:
                changes[name] = new
        elif isinstance(value, Temp) and value.id in mapping:
            changes[name] = mapping[value.id]
    return replace(instr, **changes) if changes else instr


def reads(instr) -> set:
    """Temps `instr` reads."""
    if isinstance(instr, IRMove):
        return {instr.src} if isinstance(instr.src, Temp) else set()
    if isinstance(instr, IRCast):
        return {instr.src} if isinstance(instr.src, Temp) else set()
    if isinstance(instr, IRBinOp):
        return {v for v in (instr.left, instr.right) if isinstance(v, Temp)}
    if isinstance(instr, IRUnOp):
        return {instr.operand} if isinstance(instr.operand, Temp) else set()
    if isinstance(instr, IRCall):
        return {a for a in instr.args if isinstance(a, Temp)}
    if isinstance(instr, IRReturn):
        return {instr.value} if isinstance(instr.value, Temp) else set()
    if isinstance(instr, IRBranch):
        return {instr.cond} if isinstance(instr.cond, Temp) else set()
    if isinstance(instr, IRLoad):
        return {instr.address} if isinstance(instr.address, Temp) else set()
    if isinstance(instr, IRStore):
        return {v for v in (instr.address, instr.value) if isinstance(v, Temp)}
    if isinstance(instr, IRCopy):
        return {v for v in (instr.dst_address, instr.src_address) if isinstance(v, Temp)}
    if isinstance(instr, IRBoundsCheck):
        return {v for v in (instr.index, instr.length) if isinstance(v, Temp)}
    if isinstance(instr, IRSliceBoundsCheck):
        return {v for v in (instr.value, instr.bound) if isinstance(v, Temp)}
    return set()


def writes(instr) -> set:
    """Temps `instr` writes."""
    if isinstance(instr, (IRMove, IRBinOp, IRUnOp, IRLoad, IRLocalAddress, IRStaticDataAddress, IRCast, IRReadArgument)):
        return {instr.dst}
    if isinstance(instr, IRCall):
        return {instr.dst} if instr.dst is not None else set()
    return set()


@dataclass
class Block:
    """Maximal straight-line run of IR. start is its index in the flat list."""
    label: Optional[str]
    start: int
    instructions: list
    successors: list = field(default_factory=list)  # block indices
    predecessors: list = field(default_factory=list)  # block indices


def build_blocks(body: list) -> list[Block]:
    """Split `body` into blocks with successor and predecessor edges."""
    if not body:
        return []

    leader_indices = {0}
    for i, instr in enumerate(body):
        if isinstance(instr, IRLabel):
            leader_indices.add(i)
        if isinstance(instr, TERMINATORS) and i + 1 < len(body):
            leader_indices.add(i + 1)
    leader_indices = sorted(leader_indices)

    blocks: list[Block] = []
    label_to_block: dict[str, int] = {}
    for idx, start in enumerate(leader_indices):
        end = leader_indices[idx + 1] if idx + 1 < len(leader_indices) else len(body)
        instructions = body[start:end]
        label = instructions[0].name if instructions and isinstance(instructions[0], IRLabel) else None
        blocks.append(Block(label=label, start=start, instructions=instructions))
        if label is not None:
            label_to_block[label] = idx

    for idx, block in enumerate(blocks):
        last = block.instructions[-1] if block.instructions else None
        if isinstance(last, IRJump):
            block.successors = [label_to_block[last.label]]
        elif isinstance(last, IRBranch):
            block.successors = [label_to_block[last.true_label], label_to_block[last.false_label]]
        elif isinstance(last, IRReturn):
            block.successors = []
        else:
            block.successors = [idx + 1] if idx + 1 < len(blocks) else []
    for idx, block in enumerate(blocks):
        for succ in block.successors:
            if idx not in blocks[succ].predecessors:
                blocks[succ].predecessors.append(idx)

    return blocks


def flatten(blocks: list[Block]) -> list:
    """Blocks back to a flat instruction list. flatten(build_blocks(b)) == b."""
    return [instr for block in blocks for instr in block.instructions]


def _block_use_def(block: Block) -> tuple[set, set]:
    """(use, def): read-before-written and written."""
    use, defined = set(), set()
    for instr in block.instructions:
        for t in reads(instr):
            if t not in defined:
                use.add(t)
        defined |= writes(instr)
    return use, defined


def liveness(blocks: list[Block]) -> tuple[list, list]:
    """Backward liveness to a fixed point: (live_in, live_out) per block."""
    use_def = [_block_use_def(b) for b in blocks]
    live_in = [set() for _ in blocks]
    live_out = [set() for _ in blocks]
    changed = True
    while changed:
        changed = False
        for i in reversed(range(len(blocks))):
            new_out = set()
            for succ in blocks[i].successors:
                new_out |= live_in[succ]
            use, defined = use_def[i]
            new_in = use | (new_out - defined)
            if new_in != live_in[i] or new_out != live_out[i]:
                changed = True
            live_in[i], live_out[i] = new_in, new_out
    return live_in, live_out


def reverse_postorder(blocks: list[Block]) -> list[int]:
    """Block indices reachable from block 0, in reverse postorder."""
    if not blocks:
        return []
    order, seen = [], {0}
    stack = [(0, iter(blocks[0].successors))]
    while stack:
        idx, succs = stack[-1]
        nxt = next(succs, None)
        if nxt is None:
            stack.pop()
            order.append(idx)
        elif nxt not in seen:
            seen.add(nxt)
            stack.append((nxt, iter(blocks[nxt].successors)))
    return order[::-1]


def remove_unreachable(body: list) -> list:
    """`body` without blocks unreachable from the entry."""
    blocks = build_blocks(body)
    reachable = set(reverse_postorder(blocks))
    return flatten([b for i, b in enumerate(blocks) if i in reachable])


def uses(body: list) -> dict:
    """temp.id -> indices of the instructions in `body` that read it."""
    out: dict = {}
    for i, instr in enumerate(body):
        for t in reads(instr):
            out.setdefault(t.id, []).append(i)
    return out
