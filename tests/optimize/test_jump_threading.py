"""Tests for jump threading and block merging."""

from compile import compile_to_asm
from ir.cfg import build_blocks
from ir.ir import IRBinOp, IRBranch, IRCall, IRConst, IRFunction, IRJump, IRLabel, IRMove, IRReturn, Temp
from ir.program_builder import build_ir_program
from ir.verify import verify_function
from modules import discover_modules
from ops import BinaryOp
from optimize.jump_threading import merge_blocks, thread_jumps
from optimize.optimizer import optimize
from semantic import analyze
from typesys import Type


def t(n: int, ty: Type = Type.INT) -> Temp:
    return Temp(id=n, type=ty)


def c(v: int) -> IRConst:
    return IRConst(v, Type.INT)


def set_(n: int, v: int) -> IRMove:
    return IRMove(dst=t(n), src=c(v))


def branch(true_label: str, false_label: str) -> IRBranch:
    return IRBranch(cond=t(0, Type.BOOL), true_label=true_label, false_label=false_label)


def _after(a_pass, body) -> list:
    f = IRFunction(name='f', body=list(body), params=[t(0, Type.BOOL)])  # (t0: what `branch` goes by)
    a_pass(f)
    verify_function(f)
    return f.body


def threaded(*body) -> list:
    return _after(thread_jumps, body)


def merged(*body) -> list:
    return _after(merge_blocks, body)


RET = IRReturn(value=None)


# -- threading -----------------------------------------------------------------

def test_a_jump_goes_past_a_block_that_only_jumps():
    assert threaded(IRJump('.a'), IRLabel('.a'), IRJump('.b'), IRLabel('.b'), RET) == [IRJump('.b'), IRLabel('.b'), RET]


def test_a_chain_is_followed_to_its_end():
    assert threaded(
        branch('.a', '.end'),
        IRLabel('.a'), IRJump('.b'), IRLabel('.b'), IRJump('.c'), IRLabel('.c'), set_(1, 1), IRJump('.end'),
        IRLabel('.end'), RET,
    ) == [branch('.c', '.end'), IRLabel('.c'), set_(1, 1), IRJump('.end'), IRLabel('.end'), RET]


def test_a_branch_whose_ways_meet_becomes_a_jump():
    assert threaded(
        branch('.a', '.b'), IRLabel('.a'), IRJump('.end'), IRLabel('.b'), IRJump('.end'), IRLabel('.end'), RET,
    ) == [IRJump('.end'), IRLabel('.end'), RET]


def test_a_block_that_does_more_than_jump_is_kept():
    body = [branch('.a', '.end'), IRLabel('.a'), set_(1, 1), IRJump('.end'), IRLabel('.end'), RET]
    assert threaded(*body) == body


def test_an_empty_loop_is_left_as_it_is():
    # `while true: continue`: threading through it would only turn it round, forever.
    body = [IRJump('.a'), IRLabel('.a'), IRJump('.b'), IRLabel('.b'), IRJump('.a')]
    assert threaded(*body) == body
    alone = [IRJump('.a'), IRLabel('.a'), IRJump('.a')]
    assert threaded(*alone) == alone
    # A jump into one from outside it stays too.
    into = [branch('.in', '.end'), IRLabel('.in'), IRJump('.a'), IRLabel('.a'), IRJump('.b'), IRLabel('.b'),
            IRJump('.a'), IRLabel('.end'), RET]
    assert threaded(*into) == into


# -- merging -------------------------------------------------------------------

def test_a_block_entered_by_one_jump_joins_the_block_that_jumps():
    assert merged(set_(1, 1), IRJump('.a'), IRLabel('.a'), set_(2, 2), RET) == [set_(1, 1), set_(2, 2), RET]


def test_a_block_is_moved_to_where_its_way_in_is():
    assert merged(
        set_(1, 1), IRJump('.far'),
        IRLabel('.other'), set_(3, 3), RET,
        IRLabel('.far'), set_(2, 2), branch('.other', '.other2'),
        IRLabel('.other2'), RET,
    ) == [set_(1, 1), set_(2, 2), branch('.other', '.other2'), IRLabel('.other'), set_(3, 3), RET,
          IRLabel('.other2'), RET]


def test_a_chain_of_blocks_joins_into_one():
    assert merged(
        IRJump('.a'), IRLabel('.b'), set_(2, 2), IRJump('.c'), IRLabel('.a'), set_(1, 1), IRJump('.b'),
        IRLabel('.c'), set_(3, 3), RET,
    ) == [set_(1, 1), set_(2, 2), set_(3, 3), RET]


def test_a_block_with_two_ways_in_stays():
    body = [branch('.a', '.end'), IRLabel('.a'), set_(1, 1), IRJump('.end'), IRLabel('.end'), RET]
    assert merged(*body) == body  # .end: from the branch and from .a; .a: from a branch, not a jump


def test_a_loop_keeps_its_head():
    body = [set_(1, 0), IRJump('.head'),
            IRLabel('.head'), IRBinOp(dst=t(5, Type.BOOL), op=BinaryOp.LESS_THAN, left=t(1), right=c(9)),
            IRBranch(cond=t(5, Type.BOOL), true_label='.body', false_label='.end'),
            IRLabel('.body'), IRBinOp(dst=t(1), op=BinaryOp.ADD, left=t(1), right=c(1)), IRJump('.head'),
            IRLabel('.end'), RET]
    assert merged(*body) == body
    alone = [IRJump('.a'), IRLabel('.a'), IRCall(dst=None, name='g', args=[]), IRJump('.a')]
    assert merged(*alone) == alone  # (.a is entered from itself too)


def test_the_entry_stays_first():
    body = [IRLabel('.entry'), set_(1, 1), branch('.again', '.end'), IRLabel('.again'), set_(2, 2), IRJump('.entry'),
            IRLabel('.end'), RET]
    assert merged(*body) == body


# -- in the optimizer ----------------------------------------------------------

PROGRAM = """\
def int count(int limit):
    int total = 0
    for int i = 0; i < limit; i += 1:
        for int j = 0; j < limit; j += 1:
            if j == i:
                continue
            if j > 5 and i > 5 or j == 2:
                total += 1
            else:
                total += 2
        while total > 1000:
            total -= 1000
    return total

def int main():
    print(count(20))
    return 0
"""


def test_optimized_code_has_no_block_either_would_take(tmp_path):
    (tmp_path / "p.ht").write_text(PROGRAM)
    entry, modules = discover_modules(str(tmp_path / "p.ht"))
    program = optimize(build_ir_program(analyze(entry, modules)))
    for function in program.functions:
        blocks = build_blocks(function.body)
        assert len(blocks) > 1 or function.name != 'count$'
        for i, block in enumerate(blocks):
            assert not (block.label and len(block.instructions) == 2 and isinstance(block.instructions[1], IRJump))
            entered_by_a_jump = [p for p in block.predecessors if isinstance(blocks[p].instructions[-1], IRJump)]
            assert i == 0 or not (len(block.predecessors) == 1 and entered_by_a_jump)
    assert compile_to_asm(str(tmp_path / "p.ht"))  # (and it lowers)
