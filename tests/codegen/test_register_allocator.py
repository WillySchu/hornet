"""Tests for register_allocator.py's CFG construction and liveness
dataflow (build_cfg/compute_liveness) -- the linear-scan algorithm
itself is tested separately, in test_register_allocator.py."""

from semantic import Type
from codegen.ir import Temp, IRConst, IRMove, IRBinOp, IRUnOp, IRLabel, IRJump, IRBranch, IRReturn, IRRaw, IRCall
from codegen.register_allocator import (
    build_cfg,
    compute_liveness,
    compute_live_intervals,
    eligible_intervals,
    linear_scan,
    LiveInterval,
)
from parser import BinaryOp


def t(n: int) -> Temp:
    return Temp(id=n, type=Type.INT)


# -- build_cfg --------------------------------------------------------------

def test_build_cfg_empty():
    assert build_cfg([]) == []


def test_build_cfg_straight_line_is_one_block():
    ir = [
        IRMove(dst=t(0), src=IRConst(1, Type.INT)),
        IRMove(dst=t(1), src=IRConst(2, Type.INT)),
        IRReturn(value=t(1)),
    ]
    blocks = build_cfg(ir)
    assert len(blocks) == 1
    assert blocks[0].start == 0
    assert blocks[0].instructions == ir
    assert blocks[0].successors == []


def test_build_cfg_if_else_shape():
    """Mirrors _ir_if_head/gen_statement_ir's own If shape exactly:
    branch, then-body, jump to end, else-label, else-body, end-label."""
    ir = [
        IRBranch(cond=t(0), true_label='.then', false_label='.else'),
        IRLabel('.then'),
        IRMove(dst=t(1), src=IRConst(1, Type.INT)),
        IRJump('.end'),
        IRLabel('.else'),
        IRMove(dst=t(1), src=IRConst(2, Type.INT)),
        IRLabel('.end'),
        IRReturn(value=t(1)),
    ]
    blocks = build_cfg(ir)
    # Leaders: 0 (first), 1 (.then label), 4 (.else label), 6 (.end
    # label) -- NOT 3->4 as a separate leader pair, since .else IS the
    # very next instruction after the jump at 3, so IRLabel's own
    # leader-detection and the post-terminator rule agree on the same
    # split point rather than doubling up.
    assert [b.start for b in blocks] == [0, 1, 4, 6]
    assert [b.label for b in blocks] == [None, '.then', '.else', '.end']
    # Block 0 (the branch) goes to .then (block 1) or .else (block 2).
    assert blocks[0].successors == [1, 2]
    # Block 1 (.then body + jump) goes straight to .end (block 3).
    assert blocks[1].successors == [3]
    # Block 2 (.else body) falls through to .end (block 3).
    assert blocks[2].successors == [3]
    # Block 3 (.end + return) leaves the function.
    assert blocks[3].successors == []


def test_build_cfg_while_loop_back_edge():
    """Mirrors _ir_while_head/gen_statement_ir's own While shape:
    start label, condition+branch, body label, body, jump back to
    start, end label."""
    ir = [
        IRLabel('.start'),
        IRBranch(cond=t(0), true_label='.body', false_label='.end'),
        IRLabel('.body'),
        IRMove(dst=t(1), src=IRConst(1, Type.INT)),
        IRJump('.start'),
        IRLabel('.end'),
        IRReturn(value=None),
    ]
    blocks = build_cfg(ir)
    assert [b.label for b in blocks] == ['.start', '.body', '.end']
    # The body's own jump goes BACK to .start (block 0) -- the actual
    # back-edge this whole analysis exists to get right.
    body_block = next(b for b in blocks if b.label == '.body')
    start_index = next(i for i, b in enumerate(blocks) if b.label == '.start')
    assert body_block.successors == [start_index]


def test_build_cfg_unreachable_code_after_return_gets_its_own_block():
    """An early return inside an if, with ordinary code after the if
    entirely -- that code is unreachable from the return itself, but
    still needs its own leader, since it's reachable via the other
    branch (the label right after it, from the if's own end)."""
    ir = [
        IRBranch(cond=t(0), true_label='.then', false_label='.end'),
        IRLabel('.then'),
        IRReturn(value=t(1)),
        IRLabel('.end'),  # immediately follows a terminator AND is a label -- one leader, not two
        IRMove(dst=t(2), src=IRConst(1, Type.INT)),
    ]
    blocks = build_cfg(ir)
    assert [b.start for b in blocks] == [0, 1, 3]
    assert blocks[1].successors == []  # the return leaves the function
    assert blocks[2].successors == []  # falls off the end -- gen_function's own epilogue is appended outside this list


# -- compute_liveness ---------------------------------------------------------

def test_liveness_straight_line_def_then_use():
    ir = [
        IRMove(dst=t(0), src=IRConst(1, Type.INT)),
        IRBinOp(dst=t(1), op=BinaryOp.ADD, left=t(0), right=IRConst(1, Type.INT)),
        IRReturn(value=t(1)),
    ]
    blocks = build_cfg(ir)
    live_in, live_out = compute_liveness(blocks)
    # Nothing is live before this block even starts -- t(0) is defined
    # here, not received from anywhere else.
    assert live_in[0] == set()
    assert live_out[0] == set()


def test_liveness_dead_store_never_shows_live():
    """A Temp written but never read anywhere is live nowhere at all
    -- not a real program shape (nothing produces dead IR today), but
    a real check that _block_use_def's own def-before-use ordering
    doesn't accidentally treat every write as a use."""
    ir = [
        IRMove(dst=t(0), src=IRConst(1, Type.INT)),
        IRReturn(value=None),
    ]
    blocks = build_cfg(ir)
    live_in, live_out = compute_liveness(blocks)
    assert t(0) not in live_in[0]
    assert t(0) not in live_out[0]


def test_liveness_if_else_join_point():
    """Mirrors _ir_short_circuit's own real shape: a Temp defined in
    TWO different blocks (once per branch), read once after they join
    -- exactly the case a naive single-pass scan (rather than real
    per-block liveness) would get wrong."""
    ir = [
        IRBranch(cond=t(0), true_label='.then', false_label='.else'),
        IRLabel('.then'),
        IRMove(dst=t(1), src=IRConst(1, Type.INT)),
        IRJump('.end'),
        IRLabel('.else'),
        IRMove(dst=t(1), src=IRConst(0, Type.INT)),
        IRLabel('.end'),
        IRReturn(value=t(1)),
    ]
    blocks = build_cfg(ir)
    live_in, live_out = compute_liveness(blocks)
    then_block = next(i for i, b in enumerate(blocks) if b.label == '.then')
    else_block = next(i for i, b in enumerate(blocks) if b.label == '.else')
    end_block = next(i for i, b in enumerate(blocks) if b.label == '.end')
    # t(1) is defined fresh in both branches -- neither needs it live-in.
    assert t(1) not in live_in[then_block]
    assert t(1) not in live_in[else_block]
    # Both branches need it live-out, to reach the join point.
    assert t(1) in live_out[then_block]
    assert t(1) in live_out[else_block]
    # The join block itself needs it live-in, for the return.
    assert t(1) in live_in[end_block]


def test_liveness_loop_carried_variable_spans_the_back_edge():
    """THE critical case this whole module exists to get right: a
    Temp read AND written inside a loop body (mirroring `i = i + 1`)
    must be live across the loop's own back-edge -- read at the TOP of
    iteration N+1 needs to see the write from the BOTTOM of iteration
    N. A naive forward-only scan would miss this entirely."""
    ir = [
        IRMove(dst=t(0), src=IRConst(0, Type.INT)),  # i = 0, before the loop
        IRLabel('.start'),
        IRBinOp(dst=t(1), op=BinaryOp.LESS_THAN, left=t(0), right=IRConst(10, Type.INT)),
        IRBranch(cond=t(1), true_label='.body', false_label='.end'),
        IRLabel('.body'),
        IRBinOp(dst=t(0), op=BinaryOp.ADD, left=t(0), right=IRConst(1, Type.INT)),  # i = i + 1
        IRJump('.start'),
        IRLabel('.end'),
        IRReturn(value=t(0)),
    ]
    blocks = build_cfg(ir)
    live_in, live_out = compute_liveness(blocks)
    start_block = next(i for i, b in enumerate(blocks) if b.label == '.start')
    body_block = next(i for i, b in enumerate(blocks) if b.label == '.body')
    end_block = next(i for i, b in enumerate(blocks) if b.label == '.end')
    # t(0) must be live-in to .start on every pass -- both the first
    # time (from before the loop) and every subsequent time (from the
    # body's own back-edge).
    assert t(0) in live_in[start_block]
    # The body reads t(0) (as part of `i + 1`) before writing it, so
    # it's live-in there too -- and, because of the back-edge, this
    # can only be correct if it's ALSO live-out of the body, carrying
    # the new value back around to .start for the next iteration.
    assert t(0) in live_in[body_block]
    assert t(0) in live_out[body_block]
    # After the loop, only t(0) (the return value) is needed.
    assert t(0) in live_in[end_block]


def test_liveness_irraw_dst_counts_as_a_definition():
    """IRRaw never reads a Temp (see this module's own docstring), but
    its own `dst`, when present, is a real definition -- confirmed
    here so a later read of it correctly shows nothing live-in before
    the IRRaw runs."""
    ir = [
        IRRaw(instructions=[], dst=t(0)),
        IRReturn(value=t(0)),
    ]
    blocks = build_cfg(ir)
    live_in, live_out = compute_liveness(blocks)
    assert t(0) not in live_in[0]


def test_liveness_ircall_args_and_dst():
    ir = [
        IRCall(dst=t(0), name='foo', args=[t(1)]),
        IRReturn(value=t(0)),
    ]
    blocks = build_cfg(ir)
    live_in, live_out = compute_liveness(blocks)
    # t(1), used as an argument, must already be live coming in --
    # this block doesn't produce it itself.
    assert t(1) in live_in[0]
    # t(0), the call's own result, is defined here, not received.
    assert t(0) not in live_in[0]


# -- compute_live_intervals ---------------------------------------------------

def test_compute_live_intervals_tight_span_for_a_purely_local_temp():
    ir = [
        IRMove(dst=t(0), src=IRConst(1, Type.INT)),  # index 0: irrelevant filler
        IRBinOp(dst=t(1), op=BinaryOp.ADD, left=t(0), right=IRConst(1, Type.INT)),  # index 1: t(1) born here
        IRUnOp(dst=t(2), op=None, operand=t(1)),  # index 2: t(1)'s only use
        IRReturn(value=t(2)),  # index 3
    ]
    blocks = build_cfg(ir)
    live_in, live_out = compute_liveness(blocks)
    intervals = compute_live_intervals(blocks, live_in, live_out)
    assert intervals[1].start == 1
    assert intervals[1].end == 2


def test_compute_live_intervals_loop_carried_spans_the_whole_loop():
    ir = [
        IRMove(dst=t(0), src=IRConst(0, Type.INT)),           # 0
        IRLabel('.start'),                                     # 1
        IRBinOp(dst=t(1), op=BinaryOp.LESS_THAN, left=t(0), right=IRConst(10, Type.INT)),  # 2
        IRBranch(cond=t(1), true_label='.body', false_label='.end'),  # 3
        IRLabel('.body'),                                      # 4
        IRBinOp(dst=t(0), op=BinaryOp.ADD, left=t(0), right=IRConst(1, Type.INT)),  # 5
        IRJump('.start'),                                       # 6
        IRLabel('.end'),                                        # 7
        IRReturn(value=t(0)),                                   # 8
    ]
    blocks = build_cfg(ir)
    live_in, live_out = compute_liveness(blocks)
    intervals = compute_live_intervals(blocks, live_in, live_out)
    # t(0)'s interval must cover from its own first def (index 0) all
    # the way through the loop body's own last use of it (index 5) --
    # NOT stop short partway through, which a naive scan could do.
    assert intervals[0].start == 0
    assert intervals[0].end >= 5


# -- eligible_intervals -------------------------------------------------------

def _interval(temp, start, end):
    return LiveInterval(temp=temp, start=start, end=end)


def test_eligible_intervals_excludes_named_local():
    named = Temp(id=0, type=Type.INT, is_named_local=True)
    intervals = {0: _interval(named, 0, 5)}
    ir = [IRMove(dst=named, src=IRConst(1, Type.INT))] * 6
    assert eligible_intervals(ir, intervals) == {}


def test_eligible_intervals_excludes_span_across_irraw():
    ir = [
        IRMove(dst=t(0), src=IRConst(1, Type.INT)),   # 0: t(0) defined
        IRRaw(instructions=[], dst=t(1)),              # 1: opaque -- could clobber anything
        IRBinOp(dst=t(2), op=BinaryOp.ADD, left=t(0), right=t(1)),  # 2: t(0) finally used
    ]
    intervals = {0: _interval(t(0), 0, 2)}
    assert eligible_intervals(ir, intervals) == {}


def test_eligible_intervals_excludes_span_across_ircall():
    ir = [
        IRMove(dst=t(0), src=IRConst(1, Type.INT)),
        IRCall(dst=t(1), name='foo', args=[]),
        IRBinOp(dst=t(2), op=BinaryOp.ADD, left=t(0), right=t(1)),
    ]
    intervals = {0: _interval(t(0), 0, 2)}
    assert eligible_intervals(ir, intervals) == {}


def test_eligible_intervals_includes_pure_temp_arithmetic():
    ir = [
        IRMove(dst=t(0), src=IRConst(1, Type.INT)),
        IRMove(dst=t(1), src=IRConst(2, Type.INT)),
        IRBinOp(dst=t(2), op=BinaryOp.ADD, left=t(0), right=t(1)),
    ]
    intervals = {0: _interval(t(0), 0, 2), 1: _interval(t(1), 1, 2)}
    result = eligible_intervals(ir, intervals)
    assert set(result.keys()) == {0, 1}


# -- linear_scan ---------------------------------------------------------------

def test_linear_scan_empty():
    assert linear_scan({}) == {}


def test_linear_scan_fewer_intervals_than_registers_all_assigned():
    intervals = {0: _interval(t(0), 0, 2), 1: _interval(t(1), 1, 3)}
    result = linear_scan(intervals, available_registers=['r10', 'r11', 'r15'])
    assert set(result.keys()) == {0, 1}
    assert result[0] != result[1]


def test_linear_scan_non_overlapping_intervals_may_share_a_register():
    # t(0) is dead (its last use at index 1) before t(1) is even born
    # (index 2) -- a single register suffices for both.
    intervals = {0: _interval(t(0), 0, 1), 1: _interval(t(1), 2, 3)}
    result = linear_scan(intervals, available_registers=['r10'])
    assert set(result.keys()) == {0, 1}


def test_linear_scan_overlapping_intervals_need_different_registers():
    intervals = {0: _interval(t(0), 0, 3), 1: _interval(t(1), 1, 2)}
    result = linear_scan(intervals, available_registers=['r10', 'r11'])
    assert result[0] != result[1]


def test_linear_scan_spills_the_interval_with_the_furthest_end():
    """Three intervals, all mutually overlapping, only two registers:
    the one ending LATEST (t(2), ending at 10) should be the one left
    unassigned -- Poletto & Sarkar's own heuristic, minimizing total
    spills by always evicting whichever value is needed longest."""
    intervals = {
        0: _interval(t(0), 0, 5),
        1: _interval(t(1), 1, 4),
        2: _interval(t(2), 2, 10),
    }
    result = linear_scan(intervals, available_registers=['r10', 'r11'])
    assert 2 not in result
    assert 0 in result and 1 in result


def test_linear_scan_never_assigns_the_same_register_to_two_live_intervals():
    """A denser, more realistic mix -- some overlapping, some not --
    checked generically (rather than against one hand-picked
    assignment) since linear scan's own register choice among
    equally-valid options isn't itself meaningful, only that no two
    SIMULTANEOUSLY live intervals ever collide."""
    intervals = {
        0: _interval(t(0), 0, 2),
        1: _interval(t(1), 1, 5),
        2: _interval(t(2), 3, 6),
        3: _interval(t(3), 4, 4),
    }
    result = linear_scan(intervals, available_registers=['r10', 'r11', 'r15'])
    by_reg: dict = {}
    for tid, reg in result.items():
        by_reg.setdefault(reg, []).append(intervals[tid])
    for assigned in by_reg.values():
        for a in assigned:
            for b in assigned:
                if a is not b:
                    assert a.end < b.start or b.end < a.start
