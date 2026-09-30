"""x86-64 SysV register roles."""

# Allocatable registers: no SysV argument role, no implicit instruction role, not scratch
# (%rax, %rcx, %rdx, %r8, %r9). Caller-saved come first so values not live across a call
# leave the callee-saved ones free. Callee-saved registers are saved by the prologue when used.
CALLER_SAVED_POOL = ['r10d', 'r11d']
CALLEE_SAVED_POOL = ['ebx', 'r12d', 'r13d', 'r14d', 'r15d']
ALLOCATABLE_REGISTERS = CALLER_SAVED_POOL + CALLEE_SAVED_POOL

# SysV callee-saved registers the allocator may use (%rbp is the frame pointer).
CALLEE_SAVED_REGISTERS = ['rbx', 'r12', 'r13', 'r14', 'r15']
