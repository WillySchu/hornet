"""optimize/address_folding.py: `t = base + c` used as an address becomes the access's offset."""

from ir.ir import (
    IRBinOp, IRCall, IRConst, IRCopy, IRFunction, IRJump, IRLabel, IRLoad, IRMove, IRReturn, IRStore, Temp,
)
from ops import BinaryOp
from optimize.address_folding import fold_addresses
from optimize.optimizer import optimize_function
from tests.test_compiler import GCC_SKIP, assert_program_stdout
from typesys import Type

P = Type.INT64


def t(n: int, ty: Type = P) -> Temp:
    return Temp(id=n, type=ty)


def add(dst, a, k):
    return IRBinOp(dst=dst, op=BinaryOp.ADD, left=a, right=IRConst(k, Type.INT64))


def fn(*body) -> IRFunction:
    return IRFunction(name='f', body=list(body))


def test_a_load_through_base_plus_constant_gets_the_offset_and_the_addition_goes():
    f = fn(add(t(1), t(0), 8), IRLoad(dst=t(2, Type.INT), address=t(1)), IRReturn(value=t(2, Type.INT)))
    optimize_function(f)
    assert f.body == [IRLoad(dst=t(2, Type.INT), address=t(0), offset=8), IRReturn(value=t(2, Type.INT))]


def test_offsets_accumulate_through_a_chain_and_apply_to_stores_and_copies():
    f = fn(add(t(1), t(0), 16), add(t(2), t(1), 8),
           IRStore(address=t(2), value=IRConst(5, Type.INT), value_type=Type.INT),
           IRCopy(dst_address=t(1), src_address=t(2), value_type=Type.INT),
           IRReturn(value=None))
    fold_addresses(f, set())
    assert f.body[2] == IRStore(address=t(0), value=IRConst(5, Type.INT), value_type=Type.INT, offset=24)
    assert f.body[3] == IRCopy(dst_address=t(0), src_address=t(0), value_type=Type.INT, dst_offset=16, src_offset=24)


def test_subtracting_a_constant_gives_a_negative_offset():
    f = fn(IRBinOp(dst=t(1), op=BinaryOp.SUBTRACT, left=t(0), right=IRConst(8, Type.INT64)),
           IRLoad(dst=t(2, Type.INT), address=t(1)), IRReturn(value=t(2, Type.INT)))
    fold_addresses(f, set())
    assert f.body[1] == IRLoad(dst=t(2, Type.INT), address=t(0), offset=-8)


def test_nothing_is_folded_across_a_label():
    body = [add(t(1), t(0), 8), IRJump('next'), IRLabel('next'),
            IRLoad(dst=t(2, Type.INT), address=t(1)), IRReturn(value=t(2, Type.INT))]
    f = fn(*body)
    fold_addresses(f, set())
    assert f.body == body


def test_nothing_is_folded_once_the_base_or_the_address_changes():
    body = [add(t(1), t(0), 8), IRMove(dst=t(0), src=t(3)), IRLoad(dst=t(2, Type.INT), address=t(1)),
            add(t(4), t(5), 8), IRCall(dst=t(4), name='g', args=[]), IRLoad(dst=t(6, Type.INT), address=t(4)),
            IRReturn(value=t(2, Type.INT))]
    f = fn(*body)
    fold_addresses(f, set())
    assert f.body == body


def test_a_pinned_base_is_not_folded():
    """A temp whose home slot's address is taken can change through memory."""
    body = [add(t(1), t(0), 8), IRLoad(dst=t(2, Type.INT), address=t(1)), IRReturn(value=t(2, Type.INT))]
    f = fn(*body)
    fold_addresses(f, {0})
    assert f.body == body


@GCC_SKIP
def test_fields_beyond_a_load_or_store_offsets_reach():
    """Fields at offsets past AArch64's direct load/store reach (255) and add immediate (4095)."""
    assert_program_stdout(
        "type Big struct:\n"
        "    [40]int a\n"
        "    int near\n"
        "    [600]int b\n"
        "    int far\n"
        "def int main():\n"
        "    *Big p = &Big(near=1, far=2)\n"
        "    p.near += 10\n"
        "    p.far += 20\n"
        "    Big copy = *p\n"
        "    print(copy.near + copy.far)\n"
        "    return 0\n",
        "33\n",
    )
