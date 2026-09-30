"""Tests for codegen/peephole.py."""

from backend.x86_64.assembly_ast import Imm, JCC, Jae, Je, Jmp, Label, Memory, Mov, MovQ, Register
from backend.x86_64.peephole import optimize_asm

rax, rbx, eax, ebx = Register('rax'), Register('rbx'), Register('eax'), Register('ebx')
slot = Memory(base='rbp', offset=-16)


def test_conditional_jump_over_unconditional_is_inverted():
    assert optimize_asm([JCC('l', '.A'), Jmp('.B'), Label('.A')]) == [JCC('ge', '.B'), Label('.A')]


def test_inversion_handles_fixed_jump_classes():
    assert optimize_asm([Je('.A'), Jmp('.B'), Label('.A')]) == [JCC('ne', '.B'), Label('.A')]
    assert optimize_asm([Jae('.A'), Jmp('.B'), Label('.A')]) == [JCC('b', '.B'), Label('.A')]


def test_no_inversion_when_target_is_not_next():
    instrs = [JCC('l', '.A'), Jmp('.B'), Label('.C')]
    assert optimize_asm(instrs) == instrs


def test_jump_to_next_label_is_dropped():
    assert optimize_asm([Jmp('.A'), Label('.A'), MovQ(src=rax, dst=rbx)]) == [Label('.A'), MovQ(src=rax, dst=rbx)]


def test_jump_to_other_label_is_kept():
    instrs = [Jmp('.B'), Label('.A')]
    assert optimize_asm(instrs) == instrs


def test_fused_branch_chain_collapses_to_one_jump():
    instrs = [JCC('l', '.then'), Jmp('.else'), Label('.then'), Jmp('.end'), Label('.else'), Label('.end')]
    assert optimize_asm(instrs) == [JCC('ge', '.else'), Label('.then'), Jmp('.end'), Label('.else'), Label('.end')]


def test_reverse_movq_pair_drops_second():
    assert optimize_asm([MovQ(src=rax, dst=rbx), MovQ(src=rbx, dst=rax)]) == [MovQ(src=rax, dst=rbx)]
    assert optimize_asm([MovQ(src=rax, dst=slot), MovQ(src=slot, dst=rax)]) == [MovQ(src=rax, dst=slot)]


def test_reverse_pair_across_label_is_kept():
    instrs = [MovQ(src=rax, dst=rbx), Label('.A'), MovQ(src=rbx, dst=rax)]
    assert optimize_asm(instrs) == instrs


def test_movq_self_move_is_dropped():
    assert optimize_asm([MovQ(src=rax, dst=rax)]) == []


def test_32_bit_moves_are_never_dropped():
    """movl clears the upper half of its destination, so neither is a no-op."""
    instrs = [Mov(src=eax, dst=eax), Mov(src=eax, dst=ebx), Mov(src=ebx, dst=eax)]
    assert optimize_asm(instrs) == instrs


def test_immediate_moves_are_kept():
    instrs = [MovQ(src=Imm(1), dst=rax), MovQ(src=Imm(1), dst=rax)]
    assert optimize_asm(instrs) == instrs
