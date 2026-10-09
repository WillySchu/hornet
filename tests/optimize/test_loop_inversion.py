"""Tests for loop inversion."""

from ir.cfg import build_blocks, reverse_postorder, uses
from ir.ir import IRBinOp, IRBranch, IRCall, IRConst, IRFunction, IRJump, IRLabel, IRMove, IRReturn, Temp
from ir.program_builder import build_ir_program
from ir.verify import verify_function
from modules import discover_modules
from ops import BinaryOp
from optimize.loop_inversion import MAX_TEST, invert_loops
from optimize.optimizer import optimize
from semantic import analyze
from tests.test_compiler import GCC_SKIP, assert_program_stdout, compile_and_run
from typesys import Type

RET = IRReturn(value=None)
COMPARISONS = (BinaryOp.LESS_THAN, BinaryOp.GREATER_THAN, BinaryOp.LESS_THAN_OR_EQUAL,
               BinaryOp.GREATER_THAN_OR_EQUAL, BinaryOp.EQUAL, BinaryOp.NOT_EQUAL)


def t(n: int, ty: Type = Type.INT) -> Temp:
    return Temp(id=n, type=ty)


def b(n: int) -> Temp:
    return t(n, Type.BOOL)


def c(v: int) -> IRConst:
    return IRConst(v, Type.INT)


def op(dst, operator, left, right) -> IRBinOp:
    return IRBinOp(dst=dst, op=operator, left=left, right=right)


def less(dst, left, right) -> IRBinOp:
    return op(dst, BinaryOp.LESS_THAN, left, right)


def step(n: int) -> IRBinOp:
    return op(t(n), BinaryOp.ADD, t(n), c(1))


def branch(cond, true_label: str, false_label: str) -> IRBranch:
    return IRBranch(cond=cond, true_label=true_label, false_label=false_label)


class Ids:
    """Fresh temps from 100 up, clear of a test's own."""

    def __init__(self):
        self.next = 100

    def new_temp(self, ty: Type) -> Temp:
        self.next += 1
        return Temp(id=self.next - 1, type=ty)


def inverted(*body, params=(), homes=None):
    """(the body after inversion, whether anything was inverted)."""
    f = IRFunction(name='f', body=list(body), params=list(params), temp_homes=homes or {})
    changed = invert_loops(f, Ids())
    verify_function(f)
    return f.body, changed


def loop(*test, body=(step(1),)) -> list:
    """`t1 = 0`, then a loop of `body` whose test block is `test` and a branch on t2."""
    return [IRMove(dst=t(1), src=c(0)), IRJump('.test'),
            IRLabel('.test'), *test, branch(b(2), '.body', '.end'),
            IRLabel('.body'), *body, IRJump('.test'),
            IRLabel('.end'), RET]


# -- the pass ------------------------------------------------------------------

def test_the_jump_back_becomes_a_copy_of_the_test():
    before = loop(less(b(2), t(1), c(9)))
    after, changed = inverted(*before)
    assert changed
    assert after == before[:7] + [less(b(100), t(1), c(9)), branch(b(100), '.body', '.end'), IRLabel('.end'), RET]
    assert after[:5] == before[:5]  # the test still guards the way in, and the jump to it is no way back


def test_a_temp_read_past_the_test_keeps_its_name():
    next_ = op(t(3), BinaryOp.ADD, t(1), c(1))
    after, _ = inverted(*loop(next_, less(b(2), t(3), c(9)), body=(IRMove(dst=t(1), src=t(3)),)))
    assert after[8:11] == [next_, less(b(100), t(3), c(9)), branch(b(100), '.body', '.end')]


def test_a_temp_the_test_carries_from_turn_to_turn_keeps_its_name():
    count = op(t(3), BinaryOp.ADD, t(3), c(1))  # (read before it is written: each turn needs the last one's)
    after, _ = inverted(*loop(count, less(b(2), t(3), c(9))), params=[t(3)])
    assert after[8:11] == [count, less(b(100), t(3), c(9)), branch(b(100), '.body', '.end')]


def test_a_variable_s_temp_keeps_its_name():
    before = loop(less(b(2), t(1), c(9)))
    after, changed = inverted(*before, homes={2: 0})
    assert changed and after[7:9] == [less(b(2), t(1), c(9)), branch(b(2), '.body', '.end')]


def test_each_way_back_gets_its_own_copy():
    call = IRCall(dst=b(4), name='g', args=[t(1)])
    after, _ = inverted(*loop(call, less(b(2), t(1), c(9)), body=(
        step(1), branch(b(4), '.again', '.rest'), IRLabel('.again'), IRJump('.test'), IRLabel('.rest'), step(1))))
    copies = [i for i in after if isinstance(i, IRBranch) and i.true_label == '.body']
    assert [i.cond for i in copies] == [b(2), b(100), b(101)]
    calls = [i for i in after if isinstance(i, IRCall)]
    assert calls == [call] * 3 and len({id(i.args) for i in calls}) == 3  # (b4 is read in the body: no new name)


def test_what_is_not_a_loop_with_a_test_is_left():
    forward = [IRJump('.test'), IRLabel('.test'), less(b(2), t(1), c(9)), branch(b(2), '.a', '.b'),
               IRLabel('.a'), RET, IRLabel('.b'), RET]
    assert inverted(*forward, params=[t(1)]) == (forward, False)
    endless = [IRJump('.a'), IRLabel('.a'), IRCall(dst=None, name='g', args=[]), IRJump('.a')]
    assert inverted(*endless) == (endless, False)


def test_a_long_test_is_left():
    long = loop(*[op(t(9), BinaryOp.ADD, t(1), c(n)) for n in range(MAX_TEST)], less(b(2), t(1), c(9)))
    assert inverted(*long) == (long, False)
    assert inverted(*loop(*long[3:-7][1:], less(b(2), t(1), c(9))))[1]  # one fewer, and it is inverted


# -- in the optimizer ----------------------------------------------------------

PROGRAM = """\
def int total([]int xs, int limit):
    int sum = 0
    for int i = 0; i < limit; i += 1:
        sum += i
    for int j = 0; j < len(xs); j += 1:
        if xs[j] < 0:
            continue
        sum += xs[j]
    int k = limit
    while k > 0 and sum > 0:
        k -= 2
    return sum + k

def int main():
    print(total([1, -2, 3], 5))
    return 0
"""


def _optimized(tmp_path, name: str) -> IRFunction:
    (tmp_path / "p.ht").write_text(PROGRAM)
    entry, modules = discover_modules(str(tmp_path / "p.ht"))
    program = optimize(build_ir_program(analyze(entry, modules)))
    return next(f for f in program.functions if f.name == name)


def test_no_loop_is_left_with_its_test_at_the_top(tmp_path):
    f = _optimized(tmp_path, 'total$')
    blocks = build_blocks(f.body)
    position = {i: n for n, i in enumerate(reverse_postorder(blocks))}
    back = [(i, s) for i, block in enumerate(blocks) for s in block.successors if position[s] <= position[i]]
    assert len(back) >= 3
    assert all(isinstance(blocks[i].instructions[-1], IRBranch) for i, _ in back)


def test_a_counting_loop_is_its_step_its_test_and_one_branch(tmp_path):
    f = _optimized(tmp_path, 'total$')
    first = next(block for block in build_blocks(f.body)
                 if isinstance(block.instructions[-1], IRBranch) and block.instructions[-1].true_label == block.label)
    step_, test, back = first.instructions[-3:]
    assert step_.op == BinaryOp.ADD and step_.dst == step_.left and step_.right == IRConst(1, Type.INT)
    assert test.op == BinaryOp.LESS_THAN and test.left == step_.dst and back.cond == test.dst
    assert not any(isinstance(i, IRMove) for i in first.instructions)  # (the step was coalesced first)


def test_each_branch_s_comparison_has_the_one_reader(tmp_path):
    f = _optimized(tmp_path, 'total$')
    read_at = uses(f.body)
    compared = [(i, instr) for i, instr in enumerate(f.body[:-1])
                if isinstance(instr, IRBinOp) and instr.op in COMPARISONS and isinstance(f.body[i + 1], IRBranch)]
    assert len(compared) >= 6  # each loop's guard and its copy, at least
    assert all(read_at[instr.dst.id] == [i + 1] for i, instr in compared)


# -- what a program does -------------------------------------------------------

LOOPS = """\
def bool more(*int calls, int limit):
    *calls += 1
    return *calls <= limit

def int main():
    int calls = 0
    int turns = 0
    while more(&calls, 3):
        turns += 1
    print(format('{} turns, {} tests', turns, calls))
    while more(&calls, 0):
        turns += 100
    print(format('{} turns, {} tests', turns, calls))
    int odd = 0
    int n = 0
    while n < 7:
        n += 1
        if n % 2 == 0:
            continue
        odd += n
    print(odd)
    []int xs = [3, 1, 4, 1, 5]
    int i = 0
    while i < len(xs) and xs[i] != 4:
        i += 1
    print(i)
    for int a = 0; a < 0; a += 1:
        print('never')
    int pairs = 0
    for int a = 0; a < 4; a += 1:
        for int z = a; z < 4; z += 1:
            if z == 3:
                break
            pairs += 1
    print(pairs)
    for x in xs:
        if x == 1:
            continue
        pairs += x
    print(pairs)
    return 0
"""

OUT_OF_BOUNDS = """\
def int main():
    []int xs = [1, 2, 3]
    int i = 0
    while xs[i] > 0:
        i += 1
    return 0
"""


@GCC_SKIP
def test_loops_do_what_they_did():
    assert_program_stdout(LOOPS, "3 turns, 4 tests\n3 turns, 5 tests\n16\n2\n6\n18\n")


@GCC_SKIP
def test_a_test_s_panic_on_a_later_turn_is_at_the_test():
    result = compile_and_run(OUT_OF_BOUNDS)  # (the fourth test is a copy's)
    assert result.stderr.endswith(":4:11: panic: index out of bounds: index 3, length 3\n")
