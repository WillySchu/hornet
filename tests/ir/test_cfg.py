"""Tests for ir/cfg.py."""

from typesys import Type
from ir.ir import (
    IRBinOp,
    IRBoundsCheck,
    IRBranch,
    IRCall,
    IRCopy,
    IRConst,
    IRJump,
    IRLabel,
    IRLoad,
    IRMove,
    IRReturn,
    IRStore,
    Temp,
)
from ir.cfg import build_blocks, flatten, liveness, remove_unreachable, reverse_postorder, uses
from ops import BinaryOp


def t(n: int) -> Temp:
    return Temp(id=n, type=Type.INT)


# -- build_blocks --------------------------------------------------------------

def test_build_blocks_empty():
    assert build_blocks([]) == []


def test_build_blocks_straight_line_is_one_block():
    ir = [
        IRMove(dst=t(0), src=IRConst(1, Type.INT)),
        IRMove(dst=t(1), src=IRConst(2, Type.INT)),
        IRReturn(value=t(1)),
    ]
    blocks = build_blocks(ir)
    assert len(blocks) == 1
    assert blocks[0].start == 0
    assert blocks[0].instructions == ir
    assert blocks[0].successors == []


def test_build_blocks_if_else_shape():
    """The IR of an `if`/`else`: branch, then-body, jump to end, else-label, else-body, end-label."""
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
    blocks = build_blocks(ir)
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


def test_build_blocks_while_loop_back_edge():
    """The IR of a `while`: start label, condition and branch, body label, body, jump back to
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
    blocks = build_blocks(ir)
    assert [b.label for b in blocks] == ['.start', '.body', '.end']
    # The body's own jump goes BACK to .start (block 0) -- the actual
    # back-edge this whole analysis exists to get right.
    body_block = next(b for b in blocks if b.label == '.body')
    start_index = next(i for i, b in enumerate(blocks) if b.label == '.start')
    assert body_block.successors == [start_index]


def test_build_blocks_unreachable_code_after_return_gets_its_own_block():
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
    blocks = build_blocks(ir)
    assert [b.start for b in blocks] == [0, 1, 3]
    assert blocks[1].successors == []  # the return leaves the function
    assert blocks[2].successors == []  # falls off the end


# -- liveness ---------------------------------------------------------

def test_liveness_straight_line_def_then_use():
    ir = [
        IRMove(dst=t(0), src=IRConst(1, Type.INT)),
        IRBinOp(dst=t(1), op=BinaryOp.ADD, left=t(0), right=IRConst(1, Type.INT)),
        IRReturn(value=t(1)),
    ]
    blocks = build_blocks(ir)
    live_in, live_out = liveness(blocks)
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
    blocks = build_blocks(ir)
    live_in, live_out = liveness(blocks)
    assert t(0) not in live_in[0]
    assert t(0) not in live_out[0]


def test_liveness_if_else_join_point():
    """A Temp defined in two blocks (once per branch) and read after they join, as a short-circuit
    `and`/`or` value is: a single pass over the list, without per-block liveness, gets it wrong."""
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
    blocks = build_blocks(ir)
    live_in, live_out = liveness(blocks)
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
    blocks = build_blocks(ir)
    live_in, live_out = liveness(blocks)
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


def test_liveness_ircall_args_and_dst():
    ir = [
        IRCall(dst=t(0), name='foo', args=[t(1)]),
        IRReturn(value=t(0)),
    ]
    blocks = build_blocks(ir)
    live_in, live_out = liveness(blocks)
    # t(1), used as an argument, must already be live coming in --
    # this block doesn't produce it itself.
    assert t(1) in live_in[0]
    # t(0), the call's own result, is defined here, not received.
    assert t(0) not in live_in[0]


def test_liveness_irload_address_and_dst():
    """A real bug this module shipped with initially: IRLoad/IRStore
    had no case in _reads/_writes at all, making them completely
    invisible to liveness -- an address Temp's interval would end
    right where it was DEFINED, not where an IRStore/IRLoad actually
    LAST USED it, silently letting the allocator hand its register to
    something else while it was still needed."""
    ir = [
        IRLoad(dst=t(1), address=t(0)),
        IRReturn(value=t(1)),
    ]
    blocks = build_blocks(ir)
    live_in, live_out = liveness(blocks)
    # t(0), the address being read through, must already be live
    # coming in -- this block doesn't produce it itself.
    assert t(0) in live_in[0]
    # t(1), the loaded value, is defined here, not received.
    assert t(1) not in live_in[0]


def test_liveness_irstore_address_and_value():
    """Same bug, the write side: BOTH address and value must count as
    reads -- an IRStore never defines anything."""
    ir = [
        IRStore(address=t(0), value=t(1), value_type=Type.INT),
        IRReturn(value=None),
    ]
    blocks = build_blocks(ir)
    live_in, live_out = liveness(blocks)
    assert t(0) in live_in[0]
    assert t(1) in live_in[0]


def test_liveness_ircopy_reads_both_addresses_and_writes_nothing():
    """IRCopy reads dst_address and src_address alike -- unlike every
    other op's own `dst` field, neither one is a result Temp here
    (see IRCopy's own docstring for why they're named dst_address/
    src_address specifically, not dst/src): a copy writes to memory,
    never to a Temp, so it must never appear in _writes either."""
    ir = [
        IRCopy(dst_address=t(0), src_address=t(1), value_type=Type.INT),
        IRReturn(value=None),
    ]
    blocks = build_blocks(ir)
    live_in, live_out = liveness(blocks)
    assert t(0) in live_in[0]
    assert t(1) in live_in[0]


def test_liveness_irboundscheck_reads_index_and_length_and_writes_nothing():
    """IRBoundsCheck reads both index and length -- like IRCopy, it
    writes to no Temp at all (it only conditionally jumps elsewhere in
    the function, via a shared, already-existing panic label; see its
    own docstring)."""
    ir = [
        IRBoundsCheck(index=t(0), length=t(1)),
        IRReturn(value=None),
    ]
    blocks = build_blocks(ir)
    live_in, live_out = liveness(blocks)
    assert t(0) in live_in[0]
    assert t(1) in live_in[0]


def test_param_read_before_any_write_is_live_into_the_entry():
    """Params aren't written by any instruction, so they are live coming into the entry block."""
    blocks = build_blocks([IRReturn(value=t(0))])
    live_in, _ = liveness(blocks)
    assert t(0) in live_in[0]


# -- predecessors, flatten, reverse_postorder, remove_unreachable, uses -------

_IF_ELSE = [
    IRBranch(cond=t(0), true_label='.then', false_label='.else'),
    IRLabel('.then'),
    IRMove(dst=t(1), src=IRConst(1, Type.INT)),
    IRJump('.end'),
    IRLabel('.else'),
    IRMove(dst=t(1), src=IRConst(2, Type.INT)),
    IRJump('.end'),
    IRLabel('.end'),
    IRReturn(value=t(1)),
]


def test_predecessors_mirror_successors():
    blocks = build_blocks(_IF_ELSE)
    assert [b.predecessors for b in blocks] == [[], [0], [0], [1, 2]]
    for i, b in enumerate(blocks):
        for s in b.successors:
            assert i in blocks[s].predecessors


def test_loop_back_edge_predecessor():
    ir = [
        IRJump('.start'),
        IRLabel('.start'),
        IRBranch(cond=t(0), true_label='.body', false_label='.end'),
        IRLabel('.body'),
        IRJump('.start'),
        IRLabel('.end'),
        IRReturn(value=None),
    ]
    blocks = build_blocks(ir)
    start = next(i for i, b in enumerate(blocks) if b.label == '.start')
    body = next(i for i, b in enumerate(blocks) if b.label == '.body')
    assert sorted(blocks[start].predecessors) == [0, body]


def test_flatten_round_trips():
    assert flatten(build_blocks(_IF_ELSE)) == _IF_ELSE
    assert flatten(build_blocks([])) == []


def test_reverse_postorder_visits_entry_first_and_join_last():
    order = reverse_postorder(build_blocks(_IF_ELSE))
    assert order[0] == 0
    assert order[-1] == 3
    assert sorted(order) == [0, 1, 2, 3]


def test_remove_unreachable_drops_code_after_return():
    ir = [
        IRReturn(value=None),
        IRMove(dst=t(0), src=IRConst(1, Type.INT)),
        IRReturn(value=t(0)),
        IRLabel('.orphan'),
        IRReturn(value=None),
    ]
    assert remove_unreachable(ir) == [IRReturn(value=None)]


def test_remove_unreachable_keeps_everything_reachable():
    assert remove_unreachable(_IF_ELSE) == _IF_ELSE


def test_uses_maps_temps_to_reading_instructions():
    u = uses(_IF_ELSE)
    assert u[0] == [0]
    assert u[1] == [8]


def test_liveness_needs_several_passes_through_nested_loops():
    """t0 is read only in the outer loop's tail; the inner loop must still carry it."""
    ir = [
        IRMove(dst=t(0), src=IRConst(1, Type.INT)),
        IRJump('.outer'),
        IRLabel('.outer'),
        IRBranch(cond=t(1), true_label='.inner', false_label='.done'),
        IRLabel('.inner'),
        IRBranch(cond=t(2), true_label='.inner_body', false_label='.tail'),
        IRLabel('.inner_body'),
        IRJump('.inner'),
        IRLabel('.tail'),
        IRBinOp(dst=t(3), op=BinaryOp.ADD, left=t(0), right=IRConst(1, Type.INT)),
        IRJump('.outer'),
        IRLabel('.done'),
        IRReturn(value=None),
    ]
    blocks = build_blocks(ir)
    live_in, _ = liveness(blocks)
    for label in ('.outer', '.inner', '.inner_body', '.tail'):
        idx = next(i for i, b in enumerate(blocks) if b.label == label)
        assert t(0) in live_in[idx], label
    done = next(i for i, b in enumerate(blocks) if b.label == '.done')
    assert t(0) not in live_in[done]


def test_read_fields_agree_with_reads():
    from ir.cfg import READ_FIELDS, reads, replace_reads
    from ir.ir import IRCast, IRNullCheck, IRSliceBoundsCheck, IRUnOp
    samples = [
        IRMove(dst=t(0), src=t(1)), IRCast(dst=t(0), src=t(1)), IRBinOp(dst=t(0), op=BinaryOp.ADD, left=t(1), right=t(2)),
        IRCall(dst=t(0), name='g', args=[t(1), t(2)]), IRReturn(value=t(1)), IRBranch(cond=t(1), true_label='a', false_label='b'),
        IRLoad(dst=t(0), address=t(1)), IRStore(address=t(1), value=t(2), value_type=Type.INT),
        IRCopy(dst_address=t(1), src_address=t(2), value_type=Type.INT), IRBoundsCheck(index=t(1), length=t(2)),
        IRSliceBoundsCheck(value=t(1), bound=t(2)), IRUnOp(dst=t(0), op=None, operand=t(1)), IRNullCheck(pointer=t(1)),
    ]
    assert {type(s) for s in samples} == set(READ_FIELDS)
    for s in samples:
        renamed = replace_reads(s, {1: t(11), 2: t(12)})
        assert {x.id for x in reads(renamed)} == {x.id + 10 for x in reads(s)}, s
