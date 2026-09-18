"""Identity reduction: replaces an IRBinOp where one operand is a
known identity or absorbing constant for its own op (`x + 0`, `x * 1`,
`x * 0`, and the rest -- see reduce_binary_op's own table) with an
IRMove of whatever the result always is, regardless of the OTHER
operand's own runtime value -- the operand that isn't the identity/
absorbing constant itself, or a fixed constant for an absorbing case.
The second IR-to-IR transform this compiler has (see constant_folding.
py's own module docstring for the first, and this package's own
optimizer.py for how the two compose).

Reduces in place, one instruction at a time, into the SAME dst, for
the identical reason constant_folding.py does: this IR isn't SSA, so
an IRMove is safe where deleting the op and rewriting its own Temp
id's later uses wouldn't be, without first proving no other write to
that id exists in between.

Deliberately narrower than "every algebraic identity a compiler could
know," in two different ways:

- Only IRBinOp is handled -- not IRUnOp. A unary identity (double
  negation, double complement) needs to recognize one op's own OUTPUT
  feeding directly into a second op of the same kind -- a genuinely
  different, cross-instruction pattern from "one instruction, one
  identity operand," which is all this pass (and constant_folding.py's
  own single-instruction folds) currently knows how to see.

- Several reductions that hold algebraically are deliberately left
  out because they aren't simplifications, or would change this
  compiler's own runtime behavior:

  - `0 - x` (-> `-x`) and `x * -1` (-> `-x`) are strength reductions,
    not identity ones -- the result is a TRANSFORMED value, not `x`
    itself unchanged, so rewriting either needs a real IRUnOp(NEGATE,
    x) in its place, not an IRMove; left for a future pass to add
    rather than folded into this one's own, narrower job.
  - `0 / x` and `0 % x` are NOT reduced to 0, even though both are 0
    for every x except x == 0 -- exactly the x == 0 case where the
    unreduced code traps (a hardware SIGFPE, see constant_folding.py's
    own docstring for the identical reasoning applied to a literal
    zero divisor there). Reducing away the division would silently
    turn that trap into a normal 0, so this rule is limited to
    scalar-constant DIVISOR positions only (`x / 1`, `x % 1`), never
    a DIVIDEND one.
  - `x & <all-ones>` (-> `x`) and `x | <all-ones>` (-> all-ones) are
    left out: unlike 0/1, what "all-ones" actually looks like as a
    literal value depends on the operand's own runtime width (0xFF
    for uint8, -1 for a signed type at any other width) -- real,
    useful reductions, just deferred rather than rushed alongside the
    zero/one identities every other case here needs no such width
    reasoning for at all."""

from ir.ir import IRBinOp, IRConst, IRFunction, IRMove, IRValue
from parser import BinaryOp
from semantic import Type


def reduce_binary_op(op: BinaryOp, left: IRValue, right: IRValue, result_type: Type) -> "IRValue | None":
    """The value `dst = left OP right` always computes, regardless of
    whatever the non-constant operand's own runtime value turns out to
    be -- either the other operand itself (an identity), a fixed
    IRConst (an absorbing result), or None if this op/operand-shape
    isn't one of the reductions this pass knows -- see this module's
    own docstring for the ones deliberately left out and why."""
    left_value = left.value if isinstance(left, IRConst) else None
    right_value = right.value if isinstance(right, IRConst) else None

    if op == BinaryOp.ADD:
        if right_value == 0:
            return left
        if left_value == 0:
            return right
    elif op == BinaryOp.SUBTRACT:
        if right_value == 0:
            return left
    elif op == BinaryOp.MULTIPLY:
        if right_value == 0 or left_value == 0:
            return IRConst(0, result_type)
        if right_value == 1:
            return left
        if left_value == 1:
            return right
    elif op == BinaryOp.DIVIDE:
        if right_value == 1:
            return left
    elif op == BinaryOp.MODULO:
        if right_value == 1:
            return IRConst(0, result_type)
    elif op in (BinaryOp.SHIFT_LEFT, BinaryOp.SHIFT_RIGHT):
        if right_value == 0:
            return left
        if left_value == 0:
            return IRConst(0, result_type)
    elif op == BinaryOp.BITWISE_AND:
        if right_value == 0 or left_value == 0:
            return IRConst(0, result_type)
    elif op == BinaryOp.BITWISE_OR:
        if right_value == 0:
            return left
        if left_value == 0:
            return right
    elif op == BinaryOp.BITWISE_XOR:
        if right_value == 0:
            return left
        if left_value == 0:
            return right
    return None


def reduce_identities(ir_fn: IRFunction) -> None:
    """Rewrites ir_fn.body in place, replacing every IRBinOp that
    reduce_binary_op recognizes with an IRMove of the always-true
    result -- see this module's own docstring for what's covered and
    what's deliberately not."""
    for i, instr in enumerate(ir_fn.body):
        if not isinstance(instr, IRBinOp):
            continue
        reduced = reduce_binary_op(instr.op, instr.left, instr.right, instr.dst.type)
        if reduced is not None:
            ir_fn.body[i] = IRMove(dst=instr.dst, src=reduced)
