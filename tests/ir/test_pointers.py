"""Pointers in the IR: address-of, automatic dereference in field access, `none`, and comparison.
Built through the whole pipeline (parse, analyze, build_ir_program), like test_sum_types.py."""

import tempfile
from pathlib import Path

import parser
import semantic
from lexer import lex
from ir.ir import IRBinOp, IRConst, IRLoad, IRLocalAddress, IRStore
from ir.program_builder import build_ir_program
from semantic import Type, TypeKind


def _build(source: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "program.ht"
        src_path.write_text(source)
        tokens = lex(str(src_path))
        ast = parser.Parser(tokens).parse_program()
        return build_ir_program(semantic.analyze(ast))


def _fn(ir_program, name):
    return [f for f in ir_program.functions if f.name == name][0]


_CIRCLE_DECL = "type Circle struct:\n    int radius\n\n"


def test_address_of_computes_the_variables_own_slot_address():
    ir_program = _build(
        "def int main():\n"
        "    int x = 5\n"
        "    *int p = &x\n"
        "    return 0\n"
    )
    fn = _fn(ir_program, 'main')
    local_addrs = [instr for instr in fn.body if isinstance(instr, IRLocalAddress)]
    # Exactly one: x's own slot address, computed for &x -- p itself
    # (a plain 8-byte scalar) never gets its own IRLocalAddress at all.
    assert len(local_addrs) == 1


def test_dereference_of_a_scalar_pointee_is_an_ordinary_load():
    ir_program = _build(
        "def int main():\n"
        "    int x = 5\n"
        "    *int p = &x\n"
        "    int y = *p\n"
        "    return y\n"
    )
    fn = _fn(ir_program, 'main')
    loads = [instr for instr in fn.body if isinstance(instr, IRLoad)]
    assert any(load.dst.type == Type.INT for load in loads)


def test_auto_deref_field_read_on_a_bare_pointer_variable():
    """p.field, p a pointer variable: the field is read through p's value, and p's own slot is
    never addressed. Both IRLocalAddress instructions are for c (building it, and &c)."""
    ir_program = _build(
        _CIRCLE_DECL +
        "def int main():\n"
        "    Circle c = Circle(5)\n"
        "    *Circle p = &c\n"
        "    int r = p.radius\n"
        "    return r\n"
    )
    fn = _fn(ir_program, 'main')
    local_addrs = [instr for instr in fn.body if isinstance(instr, IRLocalAddress)]
    assert len(local_addrs) == 2
    assert len({addr.slot for addr in local_addrs}) == 1  # both target c's own slot


def test_auto_deref_field_read_through_a_chained_field_access():
    """b.next.value: b.next is itself a pointer-typed field, dereferenced the same way."""
    ir_program = _build(
        "type Node struct:\n"
        "    int value\n"
        "    *Node next\n"
        "\n"
        "def int main():\n"
        "    Node c = Node(3, none)\n"
        "    Node b = Node(2, &c)\n"
        "    return b.next.value\n"
    )
    fn = _fn(ir_program, 'main')
    loads = [instr for instr in fn.body if isinstance(instr, IRLoad)]
    # The final read of `value` (an int) through b.next's own address.
    assert any(load.dst.type == Type.INT for load in loads)


def test_none_literal_produces_a_plain_zero_constant():
    """A null pointer is the address 0, moved into p's Temp: no store, since p is a scalar."""
    ir_program = _build(
        _CIRCLE_DECL +
        "def int main():\n"
        "    *Circle p = none\n"
        "    return 0\n"
    )
    fn = _fn(ir_program, 'main')
    from ir.ir import IRMove
    zero_int64_moves = [
        instr for instr in fn.body
        if isinstance(instr, IRMove) and isinstance(instr.src, IRConst)
        and instr.src.value == 0 and instr.src.type == Type.INT64
    ]
    assert len(zero_int64_moves) >= 1


def test_pointer_equality_is_an_ordinary_binop():
    """p == q is an ordinary scalar comparison."""
    ir_program = _build(
        _CIRCLE_DECL +
        "def int main():\n"
        "    Circle c = Circle(5)\n"
        "    *Circle p = &c\n"
        "    *Circle q = &c\n"
        "    if p == q:\n"
        "        return 1\n"
        "    return 0\n"
    )
    fn = _fn(ir_program, 'main')
    from parser import BinaryOp
    equal_ops = [instr for instr in fn.body if isinstance(instr, IRBinOp) and instr.op == BinaryOp.EQUAL]
    assert len(equal_ops) >= 1
