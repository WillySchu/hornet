"""Register allocation over the IR (see ir.py) -- v1, linear scan.

Only two kinds of Temp are excluded from allocation, both for
correctness, not performance: a named-variable Temp
(Temp.is_named_local -- see its own docstring) whose own variable was
ever touched by old-style code -- since that code reads or writes its
memory slot directly, bypassing the Temp entirely -- unless legacy
access tracking (see CodeGenerator._escaped_offsets) has established
it never was, in which case it's exempt from THIS exclusion
specifically (see eligible_intervals' own docstring for what
safe_named_locals does and doesn't change); and any Temp that needs
to SURVIVE THROUGH an IRRaw or IRCall it doesn't own (see
eligible_intervals' own docstring for why being defined BY one is a
different, safe case), because neither the caller-saved registers
(which a call definitely clobbers) nor the existing callee-saved ones
(already used internally, for unrelated purposes, by old-style
string/append code) can be trusted to carry a value across an opaque
block untouched. Everything else here -- basic blocks, liveness,
linear scan itself -- is standard and unsurprising; the interesting
decisions are those two exclusions and the register pool choice (see
ALLOCATABLE_REGISTERS below), not the algorithm.
"""

from dataclasses import dataclass, field
from typing import Optional

from codegen.ir import (
    IRBinOp,
    IRBranch,
    IRCall,
    IRCopy,
    IRJump,
    IRLabel,
    IRLoad,
    IRMove,
    IRRaw,
    IRReturn,
    IRStore,
    IRUnOp,
    Temp,
)

# %r10d, %r11d, %r15d (the ordinary 32-bit-named form, matching every
# other register this codebase passes around by default -- widened via
# as_qword_register when a Temp's type needs it, exactly like %eax/
# %ecx already are): caller-saved, general-purpose, with no SysV
# argument role (unlike %rdi/%rsi/%rdx/%rcx/%r8/%r9) and no implicit
# instruction-level role (unlike %rcx's shift-count, %rdx's div/mul
# high half), and not this compiler's own universal scratch
# convention (%rax, used throughout gen_expr_into and every IRRaw).
# Existing old-style code already uses all three as short-lived,
# single-method scratch (83, 49, and 2 call sites respectively,
# checked directly rather than assumed) -- safe to also hand out here
# because an allocated Temp's live range never spans an IRRaw/IRCall
# (see this module's own docstring), so the allocator's own use and
# any old-style method's internal use of the same register are always
# sequential, never concurrent.
ALLOCATABLE_REGISTERS = ['r10d', 'r11d', 'r15d']


@dataclass
class BasicBlock:
    """One maximal straight-line run of IR instructions: starts at a
    label or at whatever immediately follows a terminator
    (IRJump/IRBranch/IRReturn), and runs up to (and including) its own
    terminator, if it has one. `start` is this block's own first
    instruction's index in the whole-function ir list this was built
    from -- the same numbering live_intervals uses, so a block's own
    instructions are always range(start, start + len(instructions))."""
    label: Optional[str]
    start: int
    instructions: list
    successors: list = field(default_factory=list)  # indices into the owning block list


def build_cfg(ir: list) -> list[BasicBlock]:
    """Splits a flat, whole-function ir list (see gen_function) into
    basic blocks and computes each one's successor edges.

    A block boundary ("leader") is: the first instruction, every
    IRLabel, and whatever immediately follows a terminator -- the
    last case matters even when a label doesn't happen to follow one
    (e.g. an early `return` inside an if, with ordinary code after the
    if): that code is unreachable from the return itself, but it's
    still a real block, reachable via the other branch, and needs its
    own leader so it isn't silently fused into the block before it.

    Successors: IRJump goes to one block (its target label's own);
    IRBranch goes to two (true_label's and false_label's); IRReturn
    goes to none (it leaves the function); anything else falls through
    to the next block in program order, if there is one -- there
    might not be, for a void function's trailing statement, which
    relies on gen_function's own epilogue appended outside this list
    entirely.
    """
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
    """The Temps `instr` reads as input -- never includes IRRaw's own
    `instructions`, which are already-lowered and never reference a
    Temp at all (see this module's own docstring)."""
    if isinstance(instr, IRMove):
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
    return set()


def _writes(instr) -> set:
    """The Temps `instr` defines. IRRaw's own `dst`, when present,
    counts here even though it's set from outside the wrapped
    instructions -- see IRRaw's own docstring."""
    if isinstance(instr, (IRMove, IRBinOp, IRUnOp, IRLoad)):
        return {instr.dst}
    if isinstance(instr, (IRCall, IRRaw)):
        return {instr.dst} if instr.dst is not None else set()
    return set()


def _block_use_def(block: BasicBlock) -> tuple[set, set]:
    """A block's own use/def sets: `use` is whatever it reads before
    ever writing to it itself (i.e. must already be live coming in);
    `def` is whatever it writes at all, regardless of order relative
    to any of its own reads."""
    use, defined = set(), set()
    for instr in block.instructions:
        for t in _reads(instr):
            if t not in defined:
                use.add(t)
        defined |= _writes(instr)
    return use, defined


def compute_liveness(blocks: list[BasicBlock]) -> tuple[list, list]:
    """The standard backward liveness dataflow, iterated to a fixed
    point: live_out[B] is the union of live_in over B's successors;
    live_in[B] is whatever B uses itself, plus whatever live_out[B]
    needs that B doesn't itself overwrite. Iterating to convergence
    (rather than a single pass) is what makes this correct in the
    presence of a loop's back-edge -- a single forward or backward
    pass would get a loop-carried Temp's own range wrong."""
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
    """One Temp's live range, approximated (as linear scan always
    does) as a single [start, end] span rather than the possibly-
    disjoint set of points it's actually live at -- sound (never
    under-estimates how long a Temp needs protecting), just not
    maximally precise."""
    temp: Temp
    start: int
    end: int


def compute_live_intervals(blocks: list[BasicBlock], live_in: list, live_out: list) -> dict:
    """Builds one LiveInterval per Temp referenced anywhere, keyed by
    temp.id, from per-block live-in/live-out plus each block's own
    internal reads/writes -- the latter is what gives a Temp that's
    purely local to one block (never live-in or live-out of it at
    all) a tight interval, rather than the whole block's own span."""
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


def eligible_intervals(ir: list, intervals: dict, safe_named_locals: frozenset = frozenset()) -> dict:
    """Filters out every interval that can't be safely register-
    allocated -- see this module's own docstring for why named-local
    Temps and any Temp SURVIVING THROUGH an IRRaw/IRCall it doesn't
    own are excluded unconditionally, not just usually.

    `safe_named_locals` (a set of Temp ids, from gen_function -- see
    its own docstring for how it's computed from legacy access
    tracking) lifts the named-local exclusion specifically, not the
    hazard check below it: a named-local Temp whose own variable was
    never touched by old-style code is only exempt from the "its
    memory slot might be read behind its back" reasoning -- it still
    needs to survive an IRRaw/IRCall it doesn't own like anything
    else, so it falls through to exactly the same _is_hazard check
    every other Temp goes through, not an automatic pass.

    Two boundary cases are deliberately safe, not excluded, even
    though they touch an unsafe position -- see _is_hazard for the
    precise reasoning behind each:

      pos == interval.start: this Temp's own def, via IRRaw/IRCall's
      dst -- that same op can't put it at risk, only one running
      strictly after its definition can.

      pos == interval.end, when ir[pos] is an IRCall and this Temp is
      one of ITS OWN args: the read that places it into an argument
      register happens before that same call's own clobbering, not
      across it -- symmetric to the start case, just on the other
      side of the unsafe op.

    Anything strictly between start and end is always a hazard,
    regardless of which op it is."""
    unsafe_positions = [i for i, instr in enumerate(ir) if isinstance(instr, (IRRaw, IRCall))]
    result = {}
    for tid, interval in intervals.items():
        if interval.temp.is_named_local and tid not in safe_named_locals:
            continue
        if any(_is_hazard(interval, pos, ir) for pos in unsafe_positions):
            continue
        result[tid] = interval
    return result


def _is_hazard(interval: LiveInterval, pos: int, ir: list) -> bool:
    """Whether unsafe position `pos` genuinely threatens `interval`'s
    own Temp -- see eligible_intervals' own docstring for the two safe
    boundary cases this rules out (pos == start; pos == end when
    ir[pos] is an IRCall reading this exact Temp as one of its own
    args). Anything strictly between start and end is always a
    hazard, regardless of what `ir[pos]` actually is: it's impossible
    for interval.end to land exactly ON an unrelated IRCall's own
    position purely from block-boundary liveness extension, since any
    genuine downstream need would already have pulled `end` out
    further than that -- so this only ever needs to special-case the
    Temp's own true last position, never an earlier one it merely
    passes through. `unsafe_positions` spans the WHOLE function, not
    just this one interval's own span, so pos > end (entirely after
    this Temp is already dead) needs its own explicit case too --
    falling through to the IRCall-membership check for that case,
    rather than returning early, was a real bug this method shipped
    with initially: conservative rather than unsafe (it could only
    ever produce a spurious exclusion, never a wrong allocation), but
    a real mismatch with the intended design regardless."""
    if pos <= interval.start or pos > interval.end:
        return False
    if pos < interval.end:
        return True
    instr = ir[pos]
    return not (isinstance(instr, IRCall) and interval.temp in instr.args)


def linear_scan(intervals: dict, available_registers: list[str] = ALLOCATABLE_REGISTERS) -> dict:
    """Poletto & Sarkar's linear-scan allocator: walk intervals in
    start order, expiring any active interval that's already ended
    (freeing its register) before deciding the new one's own fate.
    Spilling (when no register is free) always targets whichever
    interval -- among the active ones and the new one itself -- ends
    FURTHEST in the future: that's the one most likely to still be
    blocking a free register by the time anything else needs one, so
    spilling it first minimizes the total number of spills over the
    whole function.

    Returns temp.id -> register name for however many intervals fit;
    anything spilled is simply absent from the result, needing no
    special marking -- it just falls back to its existing, always-
    correct memory-slot behavor, exactly as every Temp already
    behaves today.
    """
    sorted_intervals = sorted(intervals.values(), key=lambda iv: iv.start)
    active: list[tuple[LiveInterval, str]] = []  # kept sorted by end point
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
        # else: no free register, and every active interval ends no
        # later than this one -- the new interval itself is the one
        # that gets spilled; nothing to do, it's just absent from
        # `assignment`.

    return assignment


def allocate_registers(ir: list, safe_named_locals: frozenset = frozenset()) -> dict:
    """The whole pipeline, run over one function's own accumulated IR:
    build the CFG, compute liveness, derive live intervals, filter to
    what's actually eligible, and run linear scan over the result.
    `safe_named_locals` is passed straight through to eligible_
    intervals -- see its own docstring for what it means. Returns
    temp.id -> register name, exactly like linear_scan itself."""
    blocks = build_cfg(ir)
    live_in, live_out = compute_liveness(blocks)
    intervals = compute_live_intervals(blocks, live_in, live_out)
    eligible = eligible_intervals(ir, intervals, safe_named_locals)
    return linear_scan(eligible)
