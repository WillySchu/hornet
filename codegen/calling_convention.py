"""Calling-convention constants."""

# SysV callee-saved registers the allocator may use (%rbp is the frame pointer).
CALLEE_SAVED_REGISTERS = ['rbx', 'r12', 'r13', 'r14', 'r15']
