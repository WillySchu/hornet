"""Tests for pointer IR construction (ir/pointers.py, and every other
place teaching the IR builder about pointers touched -- ir/structs.py's
auto-deref in _ir_struct_address/_ir_field_address/_check_struct_and_
field_type, ir/dispatch.py's NoneLiteral fallback, ir/statements.py's
VarDecl no-longer-excluding-NoneLiteral-for-pointers case).

Deliberately INTEGRATION-level, building a real Program through the
full lex -> parse -> desugar -> analyze -> build_ir_program pipeline,
matching test_sum_types.py's own reasoning for why: every one of the
real bugs found while building this feature was about how separate
parts of the pipeline disagreed about a pointer-typed value's own
address vs. its own value (auto-deref for a bare Variable vs. a Field/
Index base being two SEPARATE fixes, a synthesized Field node with no
resolved_type breaking the second one), which an isolated unit test of
one function alone would never have caught."""

import tempfile
from pathlib import Path

import desugar
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
        desugar.desugar_methods(ast)
        semantic.analyze(ast)
        return build_ir_program(ast)


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
    """p.field (p a bare Variable) -- _ir_struct_address's own Variable
    case auto-dereferences, never computing p's own slot address at
    all. Both IRLocalAddress instructions present target c's own slot
    (one for constructing c itself, one for &c) -- neither is p's own
    slot, which auto-deref must never address."""
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
    """b.next.value -- the case that actually found a real bug: b.next
    is ITSELF a pointer-typed Field access (not a bare Variable), and
    needs the identical auto-deref treatment _ir_struct_address's own
    Field/Index case provides, separately from the Variable case just
    above."""
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
    """gen_expr_ir's own NoneLiteral fallback (ir/dispatch.py) -- a
    null pointer is just the address 0, no descriptor to build the
    way a nil slice needs. Written via a plain IRMove into p's own
    Temp (an ordinary scalar VarDecl, no address involved at all),
    not IRStore -- p is a pointer, not a composite value addressed
    through a slot."""
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
    """p == q -- falls all the way through to the generic scalar
    equality path (_ir_binary), no dedicated pointer-comparison IR
    concept needed at all, exactly like the design discussion
    predicted."""
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
