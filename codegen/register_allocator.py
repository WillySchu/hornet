"""Linear-scan register allocation over the IR. Temps live across an IRCall or IRSliceGrow are not allocated."""

from dataclasses import dataclass, field
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
    IRSliceGrow,
    IRStaticDataAddress,
    IRStore,
    IRUnOp,
    Temp,
)

# Allocatable pool: no SysV argument role, no implicit instruction role, not %rax scratch.
# r10/r11/r15 are caller-saved; rbx/r12-r14 are callee-saved and always saved in the prologue.
# rbx/r12/r13 are also IRSliceGrow's fixed scratch; see its lowering for write ordering.
ALLOCATABLE_REGISTERS = ['r10d', 'r11d', 'r15d', 'ebx', 'r12d', 'r13d', 'r14d']


@dataclass
class BasicBlock:
    """Maximal straight-line run of IR."""
    label: Optional[str]
    start: int
    instructions: list
    successors: list = field(default_factory=list)  # indices into the block list


def build_cfg(ir: list) -> list[BasicBlock]:
    """Split IR into basic blocks with successor edges."""
    if not ir:
        return []

    leader_indices = {0}
    for i, instr in enumerate(ir):
        if isinstance(instr, IRLabel):
            leader_indices.add(i)
        if isinstance(instr, (IRJump, IRBranch, IRReturn)) and i + 1 < len(ir):
            leader_indices.add(i + 1)
    leader_indices = sorted(leader_indices)

    blocks: list[BasicBlock] = []
    label_to_block: dict[str, int] = {}
    for idx, start in enumerate(leader_indices):
        end = leader_indices[idx + 1] if idx + 1 < len(leader_indices) else len(ir)
        instructions = ir[start:end]
        label = instructions[0].name if instructions and isinstance(instructions[0], IRLabel) else None
        blocks.append(BasicBlock(label=label, start=start, instructions=instructions))
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

    return blocks


def _reads(instr) -> set:
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
    if isinstance(instr, IRSliceGrow):
        return {v for v in (instr.ptr, instr.length, instr.cap) if isinstance(v, Temp)}
    return set()


def _writes(instr) -> set:
    """Temps `instr` writes."""
    if isinstance(instr, (IRMove, IRBinOp, IRUnOp, IRLoad, IRLocalAddress, IRStaticDataAddress, IRCast, IRReadArgument)):
        return {instr.dst}
    if isinstance(instr, IRCall):
        return {instr.dst} if instr.dst is not None else set()
    if isinstance(instr, IRSliceGrow):
        return {instr.dst_ptr, instr.dst_cap}
    return set()


def _block_use_def(block: BasicBlock) -> tuple[set, set]:
    """(use, def): read-before-written and written."""
    use, defined = set(), set()
    for instr in block.instructions:
        for t in _reads(instr):
            if t not in defined:
                use.add(t)
        defined |= _writes(instr)
    return use, defined


def compute_liveness(blocks: list[BasicBlock]) -> tuple[list, list]:
    """Backward liveness to a fixed point."""
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


@dataclass
class LiveInterval:
    """One Temp's live range as a single [start, end] span."""
    temp: Temp
    start: int
    end: int


def compute_live_intervals(blocks: list[BasicBlock], live_in: list, live_out: list) -> dict:
    """One interval per Temp from liveness and per-block reads/writes."""
    bounds: dict = {}  # temp.id -> [Temp, start, end]

    def extend(temp: Temp, index: int) -> None:
        if temp.id not in bounds:
            bounds[temp.id] = [temp, index, index]
        else:
            entry = bounds[temp.id]
            entry[1] = min(entry[1], index)
            entry[2] = max(entry[2], index)

    for i, block in enumerate(blocks):
        if not block.instructions:
            continue
        block_end = block.start + len(block.instructions) - 1
        for temp in live_in[i]:
            extend(temp, block.start)
        for temp in live_out[i]:
            extend(temp, block_end)
        for local_index, instr in enumerate(block.instructions):
            global_index = block.start + local_index
            for temp in _reads(instr) | _writes(instr):
                extend(temp, global_index)

    return {tid: LiveInterval(temp=entry[0], start=entry[1], end=entry[2]) for tid, entry in bounds.items()}


def eligible_intervals(ir: list, intervals: dict, temp_home_slots: Optional[dict] = None) -> dict:
    """Drop intervals live across an unsafe position (IRCall, IRSliceGrow)."""
    unsafe_positions = [i for i, instr in enumerate(ir) if isinstance(instr, (IRCall, IRSliceGrow))]
    addressed_slots = {instr.slot for instr in ir if isinstance(instr, IRLocalAddress)} if temp_home_slots is not None else None
    result = {}
    for tid, interval in intervals.items():
        if any(_is_hazard(interval, pos, ir) for pos in unsafe_positions):
            continue
        if addressed_slots is not None and temp_home_slots.get(tid) in addressed_slots:
            continue
        result[tid] = interval
    return result


def _is_hazard(interval: LiveInterval, pos: int, ir: list) -> bool:
    """Whether unsafe position `pos` threatens `interval`."""
    if pos <= interval.start or pos > interval.end:
        return False
    if pos < interval.end:
        return True
    instr = ir[pos]
    return not (isinstance(instr, IRCall) and interval.temp in instr.args)


def linear_scan(intervals: dict, available_registers: list[str] = ALLOCATABLE_REGISTERS) -> dict:
    """Poletto & Sarkar linear scan; spill the interval ending last."""
    sorted_intervals = sorted(intervals.values(), key=lambda iv: iv.start)
    active: list[tuple[LiveInterval, str]] = []  # sorted by end
    free_registers = list(available_registers)
    assignment: dict = {}

    for interval in sorted_intervals:
        still_active = []
        for active_interval, reg in active:
            if active_interval.end < interval.start:
                free_registers.append(reg)
            else:
                still_active.append((active_interval, reg))
        active = still_active
        active.sort(key=lambda pair: pair[0].end)

        if free_registers:
            reg = free_registers.pop()
            assignment[interval.temp.id] = reg
            active.append((interval, reg))
            active.sort(key=lambda pair: pair[0].end)
        elif active and active[-1][0].end > interval.end:
            spilled_interval, reg = active[-1]
            del assignment[spilled_interval.temp.id]
            assignment[interval.temp.id] = reg
            active[-1] = (interval, reg)
            active.sort(key=lambda pair: pair[0].end)

    return assignment


def allocate_registers(ir: list, temp_home_slots: Optional[dict] = None) -> dict:
    """temp.id -> register for one function's IR."""
    blocks = build_cfg(ir)
    live_in, live_out = compute_liveness(blocks)
    intervals = compute_live_intervals(blocks, live_in, live_out)
    eligible = eligible_intervals(ir, intervals, temp_home_slots)
    return linear_scan(eligible)
