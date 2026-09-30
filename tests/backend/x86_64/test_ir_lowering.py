"""Tests for ir_lowering.py's InstructionSelector: compare/branch fusion."""

from backend.x86_64.codegen import CodeGenerator
from backend.x86_64.ir_lowering import InstructionSelector
from ir.id_allocator import IdAllocator
from ir.ir import IRFunction, IRProgram, Temp
from semantic import Type


def _selector_with(register_assignment: dict) -> InstructionSelector:
    """Selector whose temps all have the given registers."""
    host = CodeGenerator()
    host.ir_program = IRProgram(ids=IdAllocator())
    host._register_assignment = register_assignment
    return InstructionSelector(host, IRFunction(name='f'))


# -- compare/branch fusion ----------------------------------------------------

def _cmp_branch_ir(extra_use: bool = False, gap: bool = False) -> list:
    from ir.ir import IRBinOp, IRBranch, IRMove
    from ops import BinaryOp
    a, b, c, d = Temp(10, Type.INT), Temp(11, Type.INT), Temp(12, Type.BOOL), Temp(13, Type.BOOL)
    ir = [IRBinOp(dst=c, op=BinaryOp.LESS_THAN, left=a, right=b)]
    if gap:
        ir.append(IRMove(dst=d, src=c))
    ir.append(IRBranch(cond=c, true_label='.T', false_label='.F'))
    if extra_use and not gap:
        ir.append(IRMove(dst=d, src=c))
    return ir


def test_comparison_feeding_only_the_next_branch_is_fused():
    from backend.x86_64.assembly_ast import CmpQ, JCC, Jmp, SetCC
    out = _selector_with({10: 'r10d', 11: 'r11d', 12: 'r12d'}).lower_ir(_cmp_branch_ir())
    assert not any(isinstance(i, SetCC) for i in out)
    assert any(isinstance(i, CmpQ) for i in out)
    assert out[-2:] == [JCC('ge', '.F'), Jmp('.T')]


def test_comparison_with_another_use_is_not_fused():
    from backend.x86_64.assembly_ast import SetCC
    out = _selector_with({10: 'r10d', 11: 'r11d', 12: 'r12d', 13: 'ebx'}).lower_ir(_cmp_branch_ir(extra_use=True))
    assert any(isinstance(i, SetCC) for i in out)


def test_comparison_not_directly_before_branch_is_not_fused():
    from backend.x86_64.assembly_ast import SetCC
    out = _selector_with({10: 'r10d', 11: 'r11d', 12: 'r12d', 13: 'ebx'}).lower_ir(_cmp_branch_ir(gap=True))
    assert any(isinstance(i, SetCC) for i in out)
