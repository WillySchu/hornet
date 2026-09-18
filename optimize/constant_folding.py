"""Constant folding: replaces an IRBinOp/IRUnOp/IRCast whose operand(s)
are all already IRConst with an IRMove of the computed result --
`5 + 3` becomes the constant 8 at compile time, never a runtime add.
The first real IR-to-IR transform this compiler has (see codegen.py's
own module docstring, and generate_asm's own body for the seam this
runs in) -- see this package's own __init__.py for why optimize()
itself is a plain function, not a class.

Rewrites in place, one instruction at a time, each op replaced by a
strictly simpler one with the SAME dst: this IR isn't SSA (a named-
local Temp's own id can be written more than once across a function's
body -- see Temp's own docstring), so folding can't assume deleting an
op and rewriting every later use of its own Temp id is safe without
first proving no other write to that same id exists in between. An
IRMove into that identical Temp needs no such proof: whatever already
read or will read that Temp's own value keeps doing so, now from a
simpler computation. Never touches control flow, block structure, or
instruction count/ordering -- ir.verify's own invariants (every block
still ends in exactly one terminator, and the rest) hold automatically
before and after, since nothing here adds, removes, or reorders an
instruction, only ever replaces one in place with another.

Three correctness rules specific to this compiler's own type system,
not generic to constant folding in the abstract -- get any of these
wrong and this pass would silently miscompile a program that already
worked correctly before it ran:

- DIVIDE/MODULO by a constant zero are deliberately never folded (see
  fold_binary_op's own guard): dividing by zero is a genuine, intended
  runtime trap today (a hardware SIGFPE, the same "abnormal
  termination" character an out-of-bounds access already has -- see
  arrays_slices_lowering.py's own _gen_bounds_check_panic_block),
  which folding would silently replace with whatever the fold
  happened to compute instead.
- DIVIDE/MODULO that ARE folded use truncating (round-toward-zero)
  division, matching x86's own idiv/idivq -- not Python's own //,
  which floors instead (`-7 // 2` is `-4` in Python, but this
  compiler's own runtime `-7 / 2` is `-3`; see _truncated_div's own
  docstring).
- Every arithmetic/bitwise/shift result is wrapped to fit the actual
  runtime width of its own result type (int8/uint8 are genuinely 1
  byte, int64 genuinely 8, everything else 4 -- see _wrap's own
  docstring) before being folded into the replacement IRConst:
  int8(100) + int8(100) has to fold to -56, the identical two's-
  complement wraparound the real add instruction would have produced,
  not the mathematically \"correct\" 200. A shift's own COUNT is
  additionally masked to the actual shift instruction's own operand
  width (31 for everything except int64, 63 for int64 -- x86's own sal/
  sar mask the count in %cl this same way at the hardware level; see
  gen_binary_op's own SHIFT_LEFT/SHIFT_RIGHT cases) before the shift
  itself is computed, not just the result afterward -- `x << 35` on a
  32-bit value is a shift by 3, not 35, on real hardware."""

from ir.ir import IRBinOp, IRCast, IRConst, IRFunction, IRMove, IRUnOp
from parser import BinaryOp, UnaryOp
from semantic import Type


def _wrap(value: int, t: Type) -> int:
    """Wraps `value` to fit within the range this compiler's own
    runtime representation of type `t` can actually hold -- the same
    widths type_byte_width's own docstring describes (1 byte for
    int8/uint8, 8 for int64, 4 for everything else), reinterpreted as
    a two's-complement signed value for every type except uint8 and
    bool (which are never negative in the first place). A bool result
    is always already exactly 0 or 1, needing no wrapping at all --
    included here anyway so every caller can route through this one
    function unconditionally rather than special-casing bool itself."""
    if t == Type.BOOL:
        return value
    if t == Type.INT8:
        value &= 0xFF
        return value - 0x100 if value >= 0x80 else value
    if t == Type.UINT8:
        return value & 0xFF
    if t == Type.INT64:
        value &= 0xFFFFFFFFFFFFFFFF
        return value - 0x10000000000000000 if value >= 0x8000000000000000 else value
    # INT -- also the width every OTHER type's own 32-bit register view
    # actually holds, so this same branch is what a same-width
    # reinterpretation (see fold_cast) routes through too.
    value &= 0xFFFFFFFF
    return value - 0x100000000 if value >= 0x80000000 else value


def _truncated_div(a: int, b: int) -> int:
    """Truncating (round-toward-zero) integer division, matching
    x86's own idiv/idivq -- unlike Python's own // (floor division),
    which rounds toward negative infinity instead: `-7 // 2` is `-4`
    in Python, but truncated division gives `-3`, the actual value
    this compiler's own runtime `-7 / 2` produces. Computed via abs/
    floor-division on nonnegative operands, then sign-corrected, to
    stay exact for every int64 value -- float division (`int(a / b)`)
    would silently lose precision well within int64's own range,
    since a Python float only carries 53 bits of mantissa."""
    quotient = abs(a) // abs(b)
    return -quotient if (a < 0) != (b < 0) else quotient


def _truncated_mod(a: int, b: int) -> int:
    """The truncating-division remainder: whatever satisfies
    `a == _truncated_div(a, b) * b + result`, matching x86's own
    idiv/idivq (which computes quotient and remainder together) --
    unlike Python's own % (floor-division remainder), the sign of the
    result here always matches `a`'s own sign, not `b`'s."""
    return a - _truncated_div(a, b) * b


def fold_binary_op(op: BinaryOp, left: int, right: int, result_type: Type) -> "int | None":
    """The value `dst = left OP right` computes at compile time, or
    None if it can't (or, for DIVIDE/MODULO by zero, deliberately
    shouldn't) be folded -- see this module's own docstring for why
    each of the three rules applied here exists."""
    if op == BinaryOp.ADD:
        return _wrap(left + right, result_type)
    if op == BinaryOp.SUBTRACT:
        return _wrap(left - right, result_type)
    if op == BinaryOp.MULTIPLY:
        return _wrap(left * right, result_type)
    if op == BinaryOp.DIVIDE:
        return _wrap(_truncated_div(left, right), result_type) if right != 0 else None
    if op == BinaryOp.MODULO:
        return _wrap(_truncated_mod(left, right), result_type) if right != 0 else None
    if op == BinaryOp.SHIFT_LEFT:
        mask = 63 if result_type == Type.INT64 else 31
        return _wrap(left << (right & mask), result_type)
    if op == BinaryOp.SHIFT_RIGHT:
        mask = 63 if result_type == Type.INT64 else 31
        return _wrap(left >> (right & mask), result_type)
    if op == BinaryOp.LESS_THAN:
        return int(left < right)
    if op == BinaryOp.GREATER_THAN:
        return int(left > right)
    if op == BinaryOp.LESS_THAN_OR_EQUAL:
        return int(left <= right)
    if op == BinaryOp.GREATER_THAN_OR_EQUAL:
        return int(left >= right)
    if op == BinaryOp.EQUAL:
        return int(left == right)
    if op == BinaryOp.NOT_EQUAL:
        return int(left != right)
    if op == BinaryOp.BITWISE_AND:
        return _wrap(left & right, result_type)
    if op == BinaryOp.BITWISE_XOR:
        return _wrap(left ^ right, result_type)
    if op == BinaryOp.BITWISE_OR:
        return _wrap(left | right, result_type)
    return None


def fold_unary_op(op: UnaryOp, operand: int, result_type: Type) -> int:
    """The value `dst = OP operand` computes at compile time. NOT's
    own result_type is always bool, needing no wrapping (see _wrap's
    own docstring) -- routed through it anyway, like every other case
    here, rather than special-cased out."""
    if op == UnaryOp.NEGATE:
        return _wrap(-operand, result_type)
    if op == UnaryOp.COMPLEMENT:
        return _wrap(~operand, result_type)
    return _wrap(int(operand == 0), result_type)  # UnaryOp.NOT


def fold_cast(target_type: Type, src: int) -> int:
    """The value an explicit `target_type(expr)` cast computes at
    compile time -- deliberately mirrors gen_cast_narrowing_into's own
    lowering instruction for instruction, rather than independently
    re-deriving "correct" cast semantics: that method's own docstring
    is the actual spec for what a cast means on this compiler's own
    hardware, so this has to match it exactly, quirks included, or a
    folded cast could silently disagree with an unfolded one.

    int8/uint8 targets: MovSX/MovZX's own low-byte reinterpretation --
    _wrap already does exactly this for either type, regardless of
    src's own width, since the low byte of a wider value is identical
    to the low byte of its own 32-bit view.

    An int64 target: MovSXD's own sign-extension of dst's 32-bit VIEW
    specifically, not src's full value -- even when src is already
    int64-typed (e.g. a same-type `int64(someInt64Expr)`), matching
    gen_cast_narrowing_into's own unconditional behavior there exactly
    (see its own docstring: \"correct regardless of whether the source
    was int, int8, or uint8\" -- int64-to-int64 is simply never
    exercised differently). Narrowing an int64 source down to int8/
    uint8 needs the identical 32-bit truncation first, for the same
    reason.

    A plain int target: whatever the 32-bit view already holds, no
    further change -- gen_cast_narrowing_into emits no instruction at
    all for this case."""
    if target_type == Type.INT8:
        return _wrap(src, Type.INT8)
    if target_type == Type.UINT8:
        return _wrap(src, Type.UINT8)
    if target_type == Type.INT64:
        return _wrap(_wrap(src, Type.INT), Type.INT64)
    return _wrap(src, Type.INT)  # target_type == Type.INT


def fold_constants(ir_fn: IRFunction) -> None:
    """Rewrites ir_fn.body in place, replacing every IRBinOp/IRUnOp/
    IRCast whose operand(s) are all already IRConst with an IRMove of
    the folded result -- see this module's own docstring for why an
    IRMove into the SAME dst, not deletion-plus-use-rewriting, and for
    the three width/truncation/zero-divisor rules fold_binary_op/
    fold_cast apply."""
    for i, instr in enumerate(ir_fn.body):
        if isinstance(instr, IRBinOp) and isinstance(instr.left, IRConst) and isinstance(instr.right, IRConst):
            folded = fold_binary_op(instr.op, instr.left.value, instr.right.value, instr.dst.type)
            if folded is not None:
                ir_fn.body[i] = IRMove(dst=instr.dst, src=IRConst(folded, instr.dst.type))
        elif isinstance(instr, IRUnOp) and isinstance(instr.operand, IRConst):
            folded = fold_unary_op(instr.op, instr.operand.value, instr.dst.type)
            ir_fn.body[i] = IRMove(dst=instr.dst, src=IRConst(folded, instr.dst.type))
        elif isinstance(instr, IRCast) and isinstance(instr.src, IRConst):
            folded = fold_cast(instr.dst.type, instr.src.value)
            ir_fn.body[i] = IRMove(dst=instr.dst, src=IRConst(folded, instr.dst.type))
