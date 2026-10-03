"""backend/x86_64/calling_convention.py."""

from backend.x86_64.calling_convention import ALLOCATABLE_REGISTERS, CALLEE_SAVED_POOL, CALLEE_SAVED_REGISTERS


def test_allocatable_registers_are_caller_saved_first():
    assert ALLOCATABLE_REGISTERS == ['r10d', 'r11d', 'edi', 'esi', 'r8d', 'r9d', 'ebx', 'r12d', 'r13d', 'r14d', 'r15d']


def test_callee_saved_pool_matches_the_registers_the_prologue_saves():
    assert [r.replace('d', '').replace('e', 'r') for r in CALLEE_SAVED_POOL] == CALLEE_SAVED_REGISTERS
