"""backend/common/regalloc.py, driven with the x86-64 register lists."""

from semantic import Type
from ir.ir import (
    IRBinOp,
    IRBoundsCheck,
    IRBranch,
    IRCall,
    IRCopy,
    IRConst,
    IRJump,
    IRLabel,
    IRMove,
    IRReturn,
    IRUnOp,
    Temp,
)
from ir.cfg import build_blocks, liveness
from backend.x86_64.calling_convention import ALLOCATABLE_REGISTERS, CALLEE_SAVED_POOL
from backend.common.regalloc import (
    compute_live_intervals,
    call_crossing,
    eligible_intervals,
    linear_scan,
    LiveInterval,
    allocate_registers,
)
from parser import BinaryOp


def t(n: int) -> Temp:
    return Temp(id=n, type=Type.INT)


# -- compute_live_intervals ---------------------------------------------------

def test_compute_live_intervals_tight_span_for_a_purely_local_temp():
    ir = [
        IRMove(dst=t(0), src=IRConst(1, Type.INT)),  # index 0: irrelevant filler
        IRBinOp(dst=t(1), op=BinaryOp.ADD, left=t(0), right=IRConst(1, Type.INT)),  # index 1: t(1) born here
        IRUnOp(dst=t(2), op=None, operand=t(1)),  # index 2: t(1)'s only use
        IRReturn(value=t(2)),  # index 3
    ]
    blocks = build_blocks(ir)
    live_in, live_out = liveness(blocks)
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
    blocks = build_blocks(ir)
    live_in, live_out = liveness(blocks)
    intervals = compute_live_intervals(blocks, live_in, live_out)
    # t(0)'s interval must cover from its own first def (index 0) all
    # the way through the loop body's own last use of it (index 5) --
    # NOT stop short partway through, which a naive scan could do.
    assert intervals[0].start == 0
    assert intervals[0].end >= 5


# -- eligible_intervals -------------------------------------------------------

def _interval(temp, start, end):
    return LiveInterval(temp=temp, start=start, end=end)


def test_call_crossing_includes_span_across_ircall():
    ir = [
        IRMove(dst=t(0), src=IRConst(1, Type.INT)),
        IRCall(dst=t(1), name='foo', args=[]),
        IRBinOp(dst=t(2), op=BinaryOp.ADD, left=t(0), right=t(1)),
    ]
    intervals = {0: _interval(t(0), 0, 2)}
    assert 0 in call_crossing(ir, intervals)


def test_call_crossing_excludes_temp_surviving_across_an_ircopy():
    """IRCopy isn't a call, so a Temp live across one doesn't cross a call. t(0) has nothing to do
    with the copy, and stays eligible for any register."""
    ir = [
        IRMove(dst=t(0), src=IRConst(1, Type.INT)),
        IRCopy(dst_address=t(1), src_address=t(2), value_type=Type.INT),
        IRBinOp(dst=t(3), op=BinaryOp.ADD, left=t(0), right=IRConst(1, Type.INT)),
    ]
    intervals = {0: _interval(t(0), 0, 2)}
    assert 0 not in call_crossing(ir, intervals)


def test_call_crossing_excludes_temp_surviving_across_an_irboundscheck():
    """Nor is IRBoundsCheck."""
    ir = [
        IRMove(dst=t(0), src=IRConst(1, Type.INT)),
        IRBoundsCheck(index=t(1), length=t(2)),
        IRBinOp(dst=t(3), op=BinaryOp.ADD, left=t(0), right=IRConst(1, Type.INT)),
    ]
    intervals = {0: _interval(t(0), 0, 2)}
    assert 0 not in call_crossing(ir, intervals)


def test_call_crossing_excludes_argument_temp_consumed_by_the_call_it_ends_at():
    """The symmetric end-side counterpart to the def-side fix above:
    a Temp used as one of an IRCall's OWN args, with nothing needed
    after, is safe -- the read that places it into an argument
    register happens before that same call's own clobbering, not
    across it. Regression test for a real, deliberately deferred
    refinement mentioned when call-argument IR semantics first
    shipped, now actually implemented."""
    ir = [
        IRMove(dst=t(0), src=IRConst(1, Type.INT)),  # 0: t(0) defined
        IRCall(dst=t(1), name='foo', args=[t(0)]),    # 1: t(0) consumed here, its own last use
    ]
    intervals = {0: _interval(t(0), 0, 1)}
    assert 0 not in call_crossing(ir, intervals)


def test_call_crossing_includes_temp_surviving_through_an_unrelated_call_before_being_used_as_an_argument():
    """The fix above must not overcorrect: a Temp that needs to
    survive through an EARLIER, unrelated call before finally being
    consumed as a LATER call's own argument is still genuinely unsafe
    -- mirrors `foo(t); bar(t)` where t is computed once and passed to
    two separate calls."""
    ir = [
        IRMove(dst=t(0), src=IRConst(1, Type.INT)),  # 0: t(0) defined
        IRCall(dst=t(1), name='foo', args=[]),         # 1: unrelated call -- t(0) must survive through this
        IRCall(dst=t(2), name='bar', args=[t(0)]),     # 2: t(0)'s own last use, as bar's argument
    ]
    intervals = {0: _interval(t(0), 0, 2)}
    assert 0 in call_crossing(ir, intervals)


def test_call_crossing_includes_argument_temp_also_needed_after_the_call():
    """If a Temp used as an IRCall's own argument is ALSO needed
    again afterward, its interval extends past that call, and the
    call becomes a genuine survive-through hazard again -- the end-
    side fix only ever applies when the argument use IS the Temp's
    own true last position."""
    ir = [
        IRMove(dst=t(0), src=IRConst(1, Type.INT)),  # 0: t(0) defined
        IRCall(dst=t(1), name='foo', args=[t(0)]),     # 1: used as an argument here...
        IRBinOp(dst=t(2), op=BinaryOp.ADD, left=t(0), right=IRConst(1, Type.INT)),  # 2: ...but needed again after
    ]
    intervals = {0: _interval(t(0), 0, 2)}
    assert 0 in call_crossing(ir, intervals)


def test_call_crossing_unaffected_by_an_unrelated_ircall_entirely_after_its_own_lifetime():
    """A real bug the end-side fix shipped with initially: an unsafe
    position AFTER a Temp's own interval has already ended (it's
    already dead by then) must never count against it -- this Temp's
    own interval [0,1] ends well before the SECOND call at position 2,
    which has nothing to do with it at all. Caught empirically (not by
    the unit tests above, which didn't happen to cover a later,
    unrelated call existing at all) by inspecting real compiler
    output where this exact shape appeared."""
    ir = [
        IRMove(dst=t(0), src=IRConst(1, Type.INT)),   # 0: t(0) defined
        IRBinOp(dst=t(1), op=BinaryOp.ADD, left=t(0), right=IRConst(1, Type.INT)),  # 1: t(0)'s own last use
        IRCall(dst=t(2), name='foo', args=[]),          # 2: unrelated, and AFTER t(0) is already dead
    ]
    intervals = {0: _interval(t(0), 0, 1)}
    assert 0 not in call_crossing(ir, intervals)


def test_call_crossing_excludes_pure_temp_arithmetic():
    ir = [
        IRMove(dst=t(0), src=IRConst(1, Type.INT)),
        IRMove(dst=t(1), src=IRConst(2, Type.INT)),
        IRBinOp(dst=t(2), op=BinaryOp.ADD, left=t(0), right=t(1)),
    ]
    intervals = {0: _interval(t(0), 0, 2), 1: _interval(t(1), 1, 2)}
    assert call_crossing(ir, intervals) == set()


def test_call_crossing_excludes_temp_defined_by_its_own_ircall():
    """A Temp's own defining IRCall isn't a hazard to itself; only surviving through another call is
    (see call_crossing). The usual shape: a call's result used by the very next instruction."""
    ir = [
        IRCall(dst=t(0), name='foo', args=[]),  # 0: t(0) defined BY this call
        IRBinOp(dst=t(1), op=BinaryOp.ADD, left=t(0), right=IRConst(1, Type.INT)),  # 1: used right after
    ]
    intervals = {0: _interval(t(0), 0, 1)}
    assert 0 not in call_crossing(ir, intervals)


# -- linear_scan ---------------------------------------------------------------

def test_linear_scan_empty():
    assert linear_scan({}, ALLOCATABLE_REGISTERS) == {}


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


def test_crossing_intervals_get_only_callee_saved_registers():
    intervals = {i: LiveInterval(temp=t(i), start=i, end=10) for i in range(4)}
    result = linear_scan(intervals, ALLOCATABLE_REGISTERS, frozenset({0, 2}), CALLEE_SAVED_POOL)
    assert result[0] in CALLEE_SAVED_POOL and result[2] in CALLEE_SAVED_POOL
    assert result[1] == 'r10d' and result[3] == 'r11d'


def test_crossing_interval_evicts_a_non_crossing_one_holding_a_callee_saved_register():
    n = len(ALLOCATABLE_REGISTERS)  # every register taken when the crossing interval starts
    intervals = {i: LiveInterval(temp=t(i), start=i, end=40 - i) for i in range(n)}
    intervals[n] = LiveInterval(temp=t(n), start=n, end=n + 1)
    result = linear_scan(intervals, ALLOCATABLE_REGISTERS, frozenset({n}), CALLEE_SAVED_POOL)
    assert result[n] in CALLEE_SAVED_POOL
    assert len(result) == n and len(set(result.values())) == n


def test_crossing_intervals_beyond_callee_saved_pool_spill():
    intervals = {i: LiveInterval(temp=t(i), start=i, end=10) for i in range(6)}
    result = linear_scan(intervals, ALLOCATABLE_REGISTERS, frozenset(range(6)), CALLEE_SAVED_POOL)
    assert len(result) == 5
    assert set(result.values()) == set(CALLEE_SAVED_POOL)


def test_eligible_intervals_excludes_named_temp_whose_slot_address_is_taken():
    from ir.ir import IRLocalAddress
    ir = [IRLocalAddress(dst=t(1), slot=7), IRReturn(value=t(0))]
    intervals = {0: LiveInterval(temp=t(0), start=0, end=1), 1: LiveInterval(temp=t(1), start=0, end=0)}
    assert set(eligible_intervals(ir, intervals, {0: 7})) == {1}
    assert set(eligible_intervals(ir, intervals)) == {0, 1}


def test_allocate_registers_keeps_call_crossing_temp_in_callee_saved_register():
    from ir.ir import IRCall
    ir = [
        IRMove(dst=t(0), src=IRConst(1, Type.INT)),
        IRCall(dst=t(1), name='f', args=[]),
        IRBinOp(dst=t(2), op=BinaryOp.ADD, left=t(0), right=t(1)),
        IRReturn(value=t(2)),
    ]
    result = allocate_registers(ir, ALLOCATABLE_REGISTERS, CALLEE_SAVED_POOL)
    assert result[0] in CALLEE_SAVED_POOL


def test_allocate_registers_no_longer_spills_four_simultaneously_live_temps():
    """Four purely-arithmetic Temps, all overlapping, no calls in
    sight -- exactly the shape that had to spill one of them with the
    old 3-register pool. All four now fit."""
    ir = [
        IRMove(dst=t(0), src=IRConst(1, Type.INT)),
        IRMove(dst=t(1), src=IRConst(2, Type.INT)),
        IRMove(dst=t(2), src=IRConst(3, Type.INT)),
        IRMove(dst=t(3), src=IRConst(4, Type.INT)),
        # t(4) reads all four at once, forcing them to be
        # simultaneously live right up to this point.
        IRBinOp(dst=t(4), op=BinaryOp.ADD, left=t(0), right=t(1)),
        IRBinOp(dst=t(4), op=BinaryOp.ADD, left=t(4), right=t(2)),
        IRBinOp(dst=t(4), op=BinaryOp.ADD, left=t(4), right=t(3)),
        IRReturn(value=t(4)),
    ]
    result = allocate_registers(ir, ALLOCATABLE_REGISTERS, CALLEE_SAVED_POOL)
    assert {0, 1, 2, 3} <= result.keys()
    assigned = [result[i] for i in (0, 1, 2, 3)]
    assert len(set(assigned)) == 4  # four genuinely distinct registers, not a spill in disguise


def test_param_live_across_a_call_at_the_first_instruction_gets_a_callee_saved_register():
    """Params arrive before instruction 0, so a call there crosses them."""
    from ir.ir import IRCall
    ir = [
        IRCall(dst=t(1), name='malloc', args=[IRConst(8, Type.INT)]),
        IRBinOp(dst=t(2), op=BinaryOp.ADD, left=t(0), right=t(1)),
        IRReturn(value=t(2)),
    ]
    assert allocate_registers(ir, ALLOCATABLE_REGISTERS, CALLEE_SAVED_POOL, params=[t(0)])[0] in CALLEE_SAVED_POOL
