"""AAPCS64 register roles."""

from backend.aarch64.assembly import Reg

ARG_REGISTERS = [Reg(f'x{i}') for i in range(8)]  # also the result register (x0)

# Scratch for instruction sequences; never allocated. x17 is reserved for reaching frame
# slots too far from the frame pointer (backend/aarch64/legalize.py).
SCRATCH_A, SCRATCH_B, SCRATCH_RESULT, SCRATCH_ADDRESS = Reg('x16'), Reg('x17'), Reg('x9'), Reg('x17')

# Allocatable: caller-saved first, so values not live across a call leave callee-saved ones free.
# x18 (reserved on macOS) is never used; x8 (C's struct-return register) is scratch in lowering.py.
CALLER_SAVED_POOL = [f'x{i}' for i in range(10, 16)]
CALLEE_SAVED_POOL = [f'x{i}' for i in range(19, 29)]
ALLOCATABLE_REGISTERS = CALLER_SAVED_POOL + CALLEE_SAVED_POOL

MAX_REGISTER_ARGS = 8
