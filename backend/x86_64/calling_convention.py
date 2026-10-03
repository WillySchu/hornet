"""x86-64 SysV register roles."""

# Allocatable registers: every general-purpose register but the scratch ones (%rax, %rcx, %rdx,
# which also have implicit roles in division and shifts) and %rsp/%rbp. %rdi, %rsi, %r8, and %r9
# also pass arguments, so incoming parameters and outgoing arguments are moved in parallel
# (ir_lowering._parallel_moves); being caller-saved, they never hold a value live across a call.
# Caller-saved come first so values not live across a call leave the callee-saved ones free.
# Callee-saved registers are saved by the prologue when used.
CALLER_SAVED_POOL = ['r10d', 'r11d', 'edi', 'esi', 'r8d', 'r9d']
CALLEE_SAVED_POOL = ['ebx', 'r12d', 'r13d', 'r14d', 'r15d']
ALLOCATABLE_REGISTERS = CALLER_SAVED_POOL + CALLEE_SAVED_POOL

# SysV callee-saved registers the allocator may use (%rbp is the frame pointer).
CALLEE_SAVED_REGISTERS = ['rbx', 'r12', 'r13', 'r14', 'r15']
