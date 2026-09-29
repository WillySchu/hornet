"""Linear-scan register allocation over the IR. Temps live across a call get callee-saved registers only."""

from dataclasses import dataclass
from typing import Optional

from ir.cfg import Block, build_blocks, liveness, reads, writes
from ir.ir import IRCall, IRLocalAddress, Temp

# Allocatable registers: no SysV argument role, no implicit instruction role, not scratch
# (%rax, %rcx, %rdx, %r8, %r9). Caller-saved come first so values not live across a call
# leave the callee-saved ones free. Callee-saved registers are saved by the prologue when used.
CALLER_SAVED_POOL = ['r10d', 'r11d']
CALLEE_SAVED_POOL = ['ebx', 'r12d', 'r13d', 'r14d', 'r15d']
ALLOCATABLE_REGISTERS = CALLER_SAVED_POOL + CALLEE_SAVED_POOL


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
    """Intervals that may get a register: all but named temps whose slot address is taken."""
    if temp_home_slots is None:
        return dict(intervals)
    addressed_slots = {instr.slot for instr in ir if isinstance(instr, IRLocalAddress)}
    return {tid: iv for tid, iv in intervals.items() if temp_home_slots.get(tid) not in addressed_slots}


def call_crossing(ir: list, intervals: dict) -> set:
    """Temp ids whose value must survive some IRCall."""
    calls = [i for i, instr in enumerate(ir) if isinstance(instr, IRCall)]
    return {tid for tid, iv in intervals.items() if any(_is_hazard(iv, pos, ir) for pos in calls)}


def _is_hazard(interval: LiveInterval, pos: int, ir: list) -> bool:
    """Whether the call at `pos` happens while `interval`'s value is still needed."""
    if pos <= interval.start or pos > interval.end:
        return False
    if pos < interval.end:
        return True
    return interval.temp not in ir[pos].args


def linear_scan(intervals: dict, available_registers: list[str] = ALLOCATABLE_REGISTERS,
                crossing: frozenset = frozenset(), callee_saved: list[str] = CALLEE_SAVED_POOL) -> dict:
    """Poletto & Sarkar linear scan. Intervals in `crossing` may only use `callee_saved`
    registers; others take the first free register in `available_registers` order.
    When none is allowed and free, spill whichever interval ends last."""
    sorted_intervals = sorted(intervals.values(), key=lambda iv: iv.start)
    active: list[tuple[LiveInterval, str]] = []
    free = set(available_registers)
    assignment: dict = {}

    for interval in sorted_intervals:
        still_active = []
        for active_interval, reg in active:
            if active_interval.end < interval.start:
                free.add(reg)
            else:
                still_active.append((active_interval, reg))
        active = still_active

        allowed = [r for r in available_registers if interval.temp.id not in crossing or r in callee_saved]
        reg = next((r for r in allowed if r in free), None)
        if reg is not None:
            free.discard(reg)
            assignment[interval.temp.id] = reg
            active.append((interval, reg))
            continue
        victims = [pair for pair in active if pair[1] in allowed]
        if victims:
            victim, reg = max(victims, key=lambda pair: pair[0].end)
            if victim.end > interval.end:
                del assignment[victim.temp.id]
                active.remove((victim, reg))
                assignment[interval.temp.id] = reg
                active.append((interval, reg))

    return assignment


def allocate_registers(ir: list, temp_home_slots: Optional[dict] = None) -> dict:
    """temp.id -> register for one function's IR."""
    blocks = build_blocks(ir)
    live_in, live_out = liveness(blocks)
    intervals = compute_live_intervals(blocks, live_in, live_out)
    eligible = eligible_intervals(ir, intervals, temp_home_slots)
    return linear_scan(eligible, crossing=frozenset(call_crossing(ir, eligible)))
