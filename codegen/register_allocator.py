"""Linear-scan register allocation over the IR. Temps live across an IRCall or IRSliceGrow are not allocated."""

from dataclasses import dataclass
from typing import Optional

from ir.cfg import Block, build_blocks, liveness, reads, writes
from ir.ir import IRCall, IRLocalAddress, IRSliceGrow, Temp

# Allocatable pool: no SysV argument role, no implicit instruction role, not %rax scratch.
# r10/r11/r15 are caller-saved; rbx/r12-r14 are callee-saved and always saved in the prologue.
# rbx/r12/r13 are also IRSliceGrow's fixed scratch; see its lowering for write ordering.
ALLOCATABLE_REGISTERS = ['r10d', 'r11d', 'r15d', 'ebx', 'r12d', 'r13d', 'r14d']


@dataclass
class LiveInterval:
    """One Temp's live range as a single [start, end] span."""
    temp: Temp
    start: int
    end: int


def compute_live_intervals(blocks: list[Block], live_in: list, live_out: list) -> dict:
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
            for temp in reads(instr) | writes(instr):
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
    blocks = build_blocks(ir)
    live_in, live_out = liveness(blocks)
    intervals = compute_live_intervals(blocks, live_in, live_out)
    eligible = eligible_intervals(ir, intervals, temp_home_slots)
    return linear_scan(eligible)
